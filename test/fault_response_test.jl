using EarthquakeDiffinitive
using EarthquakeDiffinitive.Elasticity
using EarthquakeDiffinitive.ElasticitySplitNode
using EarthquakeDiffinitive.FaultResponse
using Diffinitive.Grids
using Diffinitive.SbpOperators
using LinearAlgebra, SparseArrays, StaticArrays
using Test

const λ_fr = 2.0
const μ_fr = 1.0

fr_stencil_set() =
    read_stencil_set(SbpOperators.sbp_operators_path() * "standard_diagonal.toml", order=4)

fr_grids(n; L=1.0) = (equidistant_grid((-L, -L, -L), (0.0, L, L), n, n, n),
                      equidistant_grid((0.0, -L, -L), (L, L, L), n, n, n))

# Assembling and factorizing a 3D elastic system is the expensive part, so the
# testsets that only need one share it. n=13 over L=1.2 gives spacing 0.2, so
# l_f=0.6 lands exactly on nodes and Ω_f is 7×7.
const FE_SET = fr_stencil_set()
const FE_GM, FE_GP = fr_grids(13; L=1.2)
const FE = FaultElasticity(FE_GM, FE_GP, λ_fr, μ_fr, FE_SET; l_f=0.6)
const FE_NF = frictional_node_count(FE)
const FE_K = fault_stiffness(FE)

@testset "FaultResponse" begin
    @testset "boundary_selection agrees with the boundary grid" begin
        set = fr_stencil_set()
        # Deliberately unequal dimensions: an ordering bug that happens to
        # cancel on a cube shows up here.
        g = equidistant_grid((-1.0, -2.0, -3.0), (0.0, 2.0, 3.0), 9, 11, 13)
        bid = CartesianBoundary{1,UpperBoundary}()
        sel = FaultResponse.boundary_selection(g, set, bid)
        bg = boundary_grid(g, bid)
        ci = CartesianIndices(size(g))
        bci = CartesianIndices(size(bg))
        @test length(sel) == prod(size(bg))
        @test all(i -> g[ci[sel[i]]] == bg[bci[i]], eachindex(sel))
    end

    @testset "Ω_f nodes land on ±l_f and are ordered x2-fastest" begin
        x2, x3 = fault_grid_axes(FE)
        @test x2[1] ≈ -0.6 && x2[end] ≈ 0.6
        @test x3[1] ≈ -0.6 && x3[end] ≈ 0.6
        @test FE_NF == length(x2) * length(x3)
        @test issorted(x2) && issorted(x3)
        # A spacing that misses ±l_f must be rejected rather than silently
        # snapped to a nearby node.
        @test_throws ErrorException FaultElasticity(FE_GM, FE_GP, λ_fr, μ_fr, FE_SET; l_f=0.55)
    end

    @testset "stiffness reproduces a direct solve" begin
        # An arbitrary slip distribution: K*[s2;s3] must equal the traction
        # from actually solving the elastic system for that slip.
        s2 = [0.01 * sin(3k) for k in 1:FE_NF]
        s3 = [0.02 * cos(2k) for k in 1:FE_NF]
        Δτ2, Δτ3 = shear_traction(FE, s2, s3)
        stacked = FE_K * vcat(s2, s3)
        @test stacked[1:FE_NF] ≈ Δτ2 rtol = 1e-8
        @test stacked[FE_NF+1:2FE_NF] ≈ Δτ3 rtol = 1e-8
    end

    @testset "slip relieves the stress driving it" begin
        @test all(<(0), diag(FE_K))
        # A positive slip patch must reduce, not raise, the shear traction at
        # its centre — the sign that determines whether the coupled system is
        # a negative feedback or a runaway.
        Δτ2, _ = shear_traction(FE, ones(FE_NF), zeros(FE_NF))
        @test maximum(Δτ2) < 0
    end

    @testset "no slip gives no traction change" begin
        Δτ2, Δτ3 = shear_traction(FE, zeros(FE_NF), zeros(FE_NF))
        @test all(iszero, Δτ2)
        @test all(iszero, Δτ3)
    end

    @testset "K's sign convention" begin
        set = fr_stencil_set()
        gm, gp = fr_grids(11)
        fe = FaultElasticity(gm, gp, λ_fr, μ_fr, set; l_f=0.4)
        K = fault_stiffness(fe)
        @test elastic_solver_report(fe).unconverged == 0
        # Slip must relieve the stress driving it.
        @test maximum(K[i, i] for i in 1:size(K, 1)) < 0
    end

    # The threaded K build must be bit-identical to the serial one: the columns
    # are independent, so there is no reordering and no excuse for a difference.
    # Anything nonzero here is a data race.
    #
    # This only has teeth with more than one thread — run the suite as
    # `julia -t auto --project=. -e 'using Pkg; Pkg.test()'` (CI does). It is
    # kept unconditional so it still checks the serial fallback path otherwise.
    # It exists because the first threaded implementation shared its `χ`/slip
    # buffers across tasks — `if`/`else` and `begin` do not open a scope in
    # Julia — and produced a K that was 2.35 relative off.
    @testset "threaded K build matches serial" begin
        Threads.nthreads() == 1 &&
            @info "only 1 thread: the threaded K test cannot detect a data race here"
        set = fr_stencil_set()
        gm, gp = fr_grids(11)
        fe_ser = FaultElasticity(gm, gp, λ_fr, μ_fr, set; l_f=0.4)
        fe_thr = FaultElasticity(gm, gp, λ_fr, μ_fr, set; l_f=0.4)
        K_ser = fault_stiffness(fe_ser; threaded=false)
        K_thr = fault_stiffness(fe_thr; threaded=true)
        @test K_thr == K_ser
        # Same work done either way — a race usually perturbs iteration counts.
        @test solver_report(fe_thr.rs).iterations == solver_report(fe_ser.rs).iterations
    end
end

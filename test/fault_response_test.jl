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

    # `fault_stiffness_toeplitz` rebuilds K from ONE centre source instead of
    # 2*N_Ωf, using the whole-space kernel's translation invariance. It is an
    # approximation, so this pins two things: that it reproduces the exact build
    # closely, and — the part that actually broke in development — that its
    # centre column is reproduced EXACTLY, since that column is measured rather
    # than inferred and is what the near-field physics rides on.
    #
    # The accuracy claim that matters (~0.03% in V_max end-to-end) is measured
    # by scripts/k_toeplitz_validate.jl, not here; this test is a cheap guard
    # against the index arithmetic silently regressing.
    @testset "Toeplitz K build approximates the exact one" begin
        set = fr_stencil_set()
        gm, gp = fr_grids(11)
        fe_e = FaultElasticity(gm, gp, λ_fr, μ_fr, set; l_f=0.4)
        fe_t = FaultElasticity(gm, gp, λ_fr, μ_fr, set; l_f=0.4)
        K_exact = fault_stiffness(fe_e)
        K_toep = fault_stiffness_toeplitz(fe_t)

        @test size(K_toep) == size(K_exact)
        # Whole-matrix agreement. Loose because the far-field entries the centre
        # source cannot reach are left at zero by construction.
        @test norm(K_toep - K_exact) / norm(K_exact) < 0.05

        # The centre column is solved for, not inferred, so it must be exact to
        # solver tolerance. If the separation indexing is wrong this is the
        # first thing that breaks, and it breaks loudly.
        n2, n3 = length(fe_e.x2f), length(fe_e.x3f)
        nf = frictional_node_count(fe_e)
        c = ((n3 + 1) ÷ 2 - 1) * n2 + (n2 + 1) ÷ 2
        @test isapprox(K_toep[1:nf, c], K_exact[1:nf, c]; rtol=1e-6)
        @test isapprox(K_toep[nf+1:2nf, nf+c], K_exact[nf+1:2nf, nf+c]; rtol=1e-6)

        # Self-stiffness sets the instability threshold, so it must stay
        # negative — slip relieves the stress driving it — at every node.
        @test all(<(0), [K_toep[i, i] for i in 1:nf])

        # At the CENTRE it is measured, so it is exact.
        @test isapprox(K_toep[c, c], K_exact[c, c]; rtol=1e-6)

        # Away from the centre it is NOT: a Toeplitz K gives every node the
        # centre's self-stiffness, while the true one varies because nodes near
        # the Ω_f edge sit closer to the truncation boundary. That difference is
        # the approximation, not a defect — this pins its size so a regression
        # that inflates it gets noticed. (K[1,1] is the corner node.)
        @test !isapprox(K_toep[1, 1], K_exact[1, 1]; rtol=1e-6)
        @test isapprox(K_toep[1, 1], K_exact[1, 1]; rtol=0.05)

        # 5 sources = 10 solves, not 2*N_Ωf. The count is pinned because both
        # directions are known regressions: dropping to the centre alone gave
        # 138% worst-case V_max error over 30 days, and AVERAGING sources
        # instead of prioritising them gave ~500× worse.
        @test solver_report(fe_t.rs).solves == 10
        @test solver_report(fe_e.rs).solves == 2nf
    end
end

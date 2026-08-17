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

    @testset "factorize_reduced matches the one-shot solve" begin
        set = fr_stencil_set()
        gm, gp = fr_grids(11)
        A, HP_DSAT, P = split_node_system(gm, gp, λ_fr, μ_fr, set)
        χ = build_chi(gm, gp, (x2, x3) -> (exp(-(x2^2 + x3^2) / 0.2), 0.0))
        rhs = HP_DSAT * χ
        rs = factorize_reduced(A, P)
        # Reusing a factorization across steps must give the same answer as
        # factorizing every time.
        @test reduced_solve(rs, rhs) ≈ reduced_solve(A, rhs, P) rtol = 1e-8
        # And it must be reusable for a different right-hand side.
        χ2 = build_chi(gm, gp, (x2, x3) -> (0.0, exp(-(x2^2 + x3^2) / 0.3)))
        @test reduced_solve(rs, HP_DSAT * χ2) ≈ reduced_solve(A, HP_DSAT * χ2, P) rtol = 1e-8
    end
end

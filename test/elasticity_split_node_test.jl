using EarthquakeDiffinitive
using EarthquakeDiffinitive.Elasticity
using EarthquakeDiffinitive.ElasticitySplitNode
using Diffinitive.Grids
using Diffinitive.SbpOperators
using StaticArrays, LinearAlgebra, SparseArrays
using Test

const λ_sn = 2.0
const μ_sn = 1.0

# Algebraic self-consistency check: rather than deriving a manufactured
# solution's true forcing/boundary data from continuum PDE theory (easy to
# get subtly wrong — see src/Elasticity.jl's module docstring for exactly
# such a mistake found this session), pick ARBITRARY smooth, localized
# (Gaussian, decaying to ~0 at the far-field boundaries so they don't fight
# P's far-field zeroing) fields on each side, and derive the forcing that
# makes them an exact solution *directly from the assembled discrete
# operator itself*: for U = P*u + χ, the governing equation P(D+SAT)U = f
# gives A*u = HP_DSAT*(χ - U) (see derivation in PR/commit notes). Solving
# and reconstructing U should then exactly recover the manufactured field,
# for ANY smooth field — this tests that D, SAT, P, χ, and the solve are
# all self-consistent, without needing any continuum-mechanics derivation.

gaussian(x, x0, σ) = exp(-sum(abs2, x - x0) / (2σ^2))

u_plus_true(x) = SVector(
    0.10 * gaussian(x, SVector(0.3, 0.2, 0.1), 0.15),
    0.08 * gaussian(x, SVector(0.2, -0.1, 0.2), 0.15),
    0.05 * gaussian(x, SVector(0.4, 0.1, -0.2), 0.15),
)
u_minus_true(x) = SVector(
    0.12 * gaussian(x, SVector(-0.3, 0.15, -0.1), 0.15),
    0.07 * gaussian(x, SVector(-0.25, 0.05, 0.15), 0.15),
    0.06 * gaussian(x, SVector(-0.35, -0.1, 0.2), 0.15),
)

function split_node_stencil_set()
    return read_stencil_set(SbpOperators.sbp_operators_path() * "standard_diagonal.toml", order=4)
end

@testset "ElasticitySplitNode" begin
    @testset "self-consistent: recovers an arbitrary smooth two-sided field" begin
        set = split_node_stencil_set()
        n = 11
        g_minus = equidistant_grid((-1.0, -1.0, -1.0), (0.0, 1.0, 1.0), n, n, n)
        g_plus = equidistant_grid((0.0, -1.0, -1.0), (1.0, 1.0, 1.0), n, n, n)
        Nm, Np = length(g_minus), length(g_plus)

        U_minus = map(u_minus_true, g_minus)
        U_plus = map(u_plus_true, g_plus)
        U_true = vcat(flatten(U_minus), flatten(U_plus))

        # slip = actual jump of the manufactured fields at the fault.
        bid_minus = CartesianBoundary{1,UpperBoundary}()
        bid_plus = CartesianBoundary{1,LowerBoundary}()
        slip_fn(x2, x3) = begin
            up = u_plus_true(SVector(0.0, x2, x3))
            um = u_minus_true(SVector(0.0, x2, x3))
            (up[2] - um[2], up[3] - um[3])
        end

        A, HP_DSAT, P = split_node_system(g_minus, g_plus, λ_sn, μ_sn, set)
        χ = build_chi(g_minus, g_plus, slip_fn)

        rhs = HP_DSAT * (χ - U_true)
        u_sol = reduced_solve(A, rhs, P)
        U_reconstructed = reconstruct_U(P, u_sol, χ)

        # A direct solve on the (still somewhat ill-conditioned, given H's
        # small quadrature weights) reduced system doesn't reach full
        # machine precision — check relative to the field's own scale.
        scale = maximum(abs.(U_true))
        @test maximum(abs.(U_reconstructed - U_true)) / scale < 1e-3
    end
end

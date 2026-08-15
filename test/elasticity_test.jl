using EarthquakeDiffinitive
using EarthquakeDiffinitive.Elasticity
using Diffinitive.Grids
using Diffinitive.SbpOperators
using StaticArrays, LinearAlgebra, SparseArrays
using ForwardDiff
using Test

const λ_test = 2.0
const μ_test = 1.0

# Smooth manufactured field, used for accuracy checks that don't depend on
# any boundary-value problem (those moved to ElasticitySplitNode, see
# src/Elasticity.jl's module docstring for why the earlier half-space BVP
# was removed).
ua(x) = SVector(
    sin(π * x[1]) * cos(π * x[2]) * cos(π * x[3]),
    cos(π * x[1]) * sin(π * x[2]) * cos(π * x[3]),
    cos(π * x[1]) * cos(π * x[2]) * sin(π * x[3]),
)

# σ_{i,dim}[ua] via ForwardDiff Jacobian (with respect to the fixed +dim axis).
function traction_analytic(x::SVector{3}, dim)
    J = ForwardDiff.jacobian(ua, x)
    return SVector(ntuple(3) do i
        if i == dim
            λ_test * tr(J) + 2μ_test * J[i, dim]
        else
            μ_test * (J[i, dim] + J[dim, i])
        end
    end)
end

function stencil_set()
    return read_stencil_set(SbpOperators.sbp_operators_path() * "standard_diagonal.toml", order=4)
end

@testset "Elasticity" begin
    @testset "operator accuracy on polynomials" begin
        set = stencil_set()
        g = equidistant_grid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 21, 21, 21)
        # u = [x, y², xz] with λ=μ=1: ∇·σ = [2,6,0] (matches elastic.jl's own test)
        u = map(x -> SVector(x[1], x[2]^2, x[1] * x[3]), g)
        u_true = map(x -> SVector(2.0, 6.0, 0.0), g)
        E = elastic_operator(g, 1.0, 1.0, set)
        u_res = E * u
        @test u_res ≈ u_true
    end

    @testset "elastic_blocks (sparse) matches elastic_operator (LazyTensor)" begin
        set = stencil_set()
        g = equidistant_grid((0.0, -1.0, -1.0), (1.0, 1.0, 1.0), 11, 11, 11)
        u = map(ua, g)

        E = elastic_operator(g, λ_test, μ_test, set)
        u_lazy = flatten(E * u)

        M = elastic_blocks(g, λ_test, μ_test, set)
        ncomp = 3
        blocks = [sparse(M[j][k]) for j in 1:ncomp, k in 1:ncomp]
        A = reduce(vcat, [reduce(hcat, blocks[j, :]) for j in 1:ncomp])
        u_sparse = A * flatten(u)

        @test u_sparse ≈ u_lazy
    end

    @testset "traction_blocks matches manufactured traction" begin
        set = stencil_set()
        g = equidistant_grid((0.0, -1.0, -1.0), (1.0, 1.0, 1.0), 21, 21, 21)
        u = map(ua, g)
        fault_bid = CartesianBoundary{1,LowerBoundary}()

        T = traction_blocks(g, λ_test, μ_test, set, fault_bid)
        v = flatten(u)
        N = length(g)
        τ_num = [sum(T[i, k] * v[(k-1)*N+1:k*N] for k in 1:3) for i in 1:3]

        τ_true = map(x -> traction_analytic(x, 1), boundary_grid(g, fault_bid))
        for i in 1:3
            τ_true_i = vec(getindex.(τ_true, i))
            scale = max(maximum(abs.(τ_true_i)), 1.0)
            @test maximum(abs.(τ_num[i] .- τ_true_i)) / scale < 0.05
        end
    end
end

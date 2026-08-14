using EarthquakeDiffinitive
using EarthquakeDiffinitive.Elasticity
using Diffinitive.Grids
using Diffinitive.SbpOperators
using StaticArrays, LinearAlgebra, SparseArrays
using ForwardDiff
using Test

const λ_test = 2.0
const μ_test = 1.0

# Manufactured solution for the full half-space BVP.
ua(x) = SVector(
    sin(π * x[1]) * cos(π * x[2]) * cos(π * x[3]),
    cos(π * x[1]) * sin(π * x[2]) * cos(π * x[3]),
    cos(π * x[1]) * cos(π * x[2]) * sin(π * x[3]),
)

# ∇·σ[ua], i.e. μ∇²u_j + (λ+μ)∂_j(∇·u), via ForwardDiff Hessians.
function navier_force(x::SVector{D}) where D
    hess = ntuple(j -> ForwardDiff.hessian(y -> ua(y)[j], x), D)
    return SVector(ntuple(D) do j
        val = μ_test * sum(hess[j][i, i] for i in 1:D)
        for k in 1:D
            val += (λ_test + μ_test) * hess[k][j, k]
        end
        val
    end)
end

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

function solve_mms(n)
    set = stencil_set()
    g = equidistant_grid((0.0, -1.0, -1.0), (1.0, 1.0, 1.0), n, n, n)
    D = 3
    N = length(g)

    A = halfspace_system(g, λ_test, μ_test, set)
    rhs = flatten(map(navier_force, g))

    fault_bid = CartesianBoundary{1,LowerBoundary}()
    # normal_derivative (and hence sat_tensors' NeumannCondition data) uses the
    # OUTWARD normal convention, which is -∂/∂x1 at this lower-x1 boundary.
    neumann_data = map(x -> -ForwardDiff.jacobian(ua, x)[1, 1], boundary_grid(g, fault_bid))
    rhs[1:N] .+= fault_neumann_rhs_correction(g, λ_test, μ_test, set, neumann_data)

    rows = Int[]
    values = Float64[]

    far_field_bids = (
        CartesianBoundary{1,UpperBoundary}(),
        CartesianBoundary{2,LowerBoundary}(), CartesianBoundary{2,UpperBoundary}(),
        CartesianBoundary{3,LowerBoundary}(), CartesianBoundary{3,UpperBoundary}(),
    )
    for bid in far_field_bids
        for I in boundary_indices(g, bid)
            utrue = ua(g[I])
            for comp in 1:D
                push!(rows, dof_index(g, comp, I))
                push!(values, utrue[comp])
            end
        end
    end
    for I in boundary_indices(g, fault_bid)
        utrue = ua(g[I])
        for comp in 2:3
            push!(rows, dof_index(g, comp, I))
            push!(values, utrue[comp])
        end
    end

    inject_dirichlet!(A, rhs, rows, values)

    u_num = unflatten(A \ rhs, g, D)
    u_true = map(ua, g)
    return g, set, u_num, u_true
end

function max_error(n)
    _, _, u_num, u_true = solve_mms(n)
    return maximum(norm.(u_num .- u_true))
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

    @testset "full BVP matches manufactured solution" begin
        @test max_error(11) < 0.1
    end

    @testset "error decreases under grid refinement" begin
        e1 = max_error(9)
        e2 = max_error(15)
        @test e2 < e1
    end

    @testset "traction matches manufactured solution" begin
        g, set, u_num, _ = solve_mms(15)
        fault_bid = CartesianBoundary{1,LowerBoundary}()
        T = traction_blocks(g, λ_test, μ_test, set, fault_bid)
        v = flatten(u_num)
        N = length(g)
        τ_num = [sum(T[i, k] * v[(k-1)*N+1:k*N] for k in 1:3) for i in 1:3]

        τ_true = map(x -> traction_analytic(x, 1), boundary_grid(g, fault_bid))
        for i in 1:3
            τ_true_i = vec(getindex.(τ_true, i))
            # Relative error against a floor (τ2,τ3 are analytically ~0 for
            # this manufactured field, so a purely relative test would divide
            # by ~0).
            scale = max(maximum(abs.(τ_true_i)), 1.0)
            @test maximum(abs.(τ_num[i] .- τ_true_i)) / scale < 0.15
        end
    end
end

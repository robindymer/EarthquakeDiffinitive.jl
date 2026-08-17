# This module used to also provide `halfspace_system`, a single-grid
# reduction of the fault problem to x1≥0 exploiting the whole-space
# reflection symmetry — since removed. That reduction's `∂u1/∂x1=0` fault
# condition assumed the tangential displacement gradients (∂2u2, ∂3u3)
# vanish at the fault, which is only true for spatially uniform slip.
# The correct statement is that σ11 *vanishes* on the fault: the mirror
# symmetry x1 → -x1 (u1 even, u2/u3 odd) makes σ11 odd across the fault,
# and BP8 eq. 6a requires it continuous, so it must be zero — which needs
# a nonzero ∂1u1 cancelling the λ(∂2u2+∂3u3) term. Forcing ∂1u1=0 instead
# left σ11 ≈ λ(∂2s2+∂3s3)/2, i.e. the spurious normal stress that surfaced
# in cross-validation was the *symptom* of the bad BC, not real physics.
# σ11 → 0 on the fault is now a regression test, and it is what caught the
# split-node SAT sign error. See `ElasticitySplitNode.jl` (matching
# `context/SEAS_benchmark.pdf`'s two-sided P+SAT+χ formulation) for the
# correct treatment. The building blocks below (`elastic_operator`,
# `elastic_blocks`, `traction_blocks`) are unaffected — the error was
# specifically in how the fault boundary condition was derived, not in the
# operators themselves — and `ElasticitySplitNode` reuses them directly.
module Elasticity

using Diffinitive
using Diffinitive.Grids
using Diffinitive.SbpOperators
using Diffinitive.LazyTensors
using SparseArrays
using Tokens # loads DiffinitiveSparseArraysExt, which defines sparse(::LazyTensor)
using StaticArrays

export elastic_operator, elastic_blocks, traction_blocks,
       flatten, unflatten, dof_index, inject_dirichlet!

# ==============================================================================
# Constant-coefficient isotropic elastic (Navier) operator, adapted from
# context/notebooks/elastic.jl's variable-coefficient IsotropicElasticOperator.
#
# The λ (Hessian-of-divergence) terms ALWAYS use the "wide" sandwich form
# Dᵢ∘Dⱼ (composed first derivatives), including the diagonal i=j entries —
# this is not just a variable-coefficient necessity: the notebook's own
# comment is explicit that the diagonal must stay wide "to avoid dispersion
# errors" between the λ- and μ-driven parts of the operator, and that holds
# for constant coefficients too. Only μ's *diagonal* (same-direction, i=j)
# term gets the compact "narrow" native `second_derivative` — it's a
# genuinely separate physical piece (the shear-Laplacian term), not part of
# the λ Hessian structure, so there's nothing to stay consistent with.
# μ's off-diagonal terms have no narrow equivalent (no mixed-partial SBP
# operator exists) and use the same wide sandwich as λ.
# ==============================================================================

struct IsotropicElasticOperator{D,TL<:NTuple{D,NTuple{D,LazyTensor{D,D}}},TM<:NTuple{D,NTuple{D,LazyTensor{D,D}}}} <: LazyTensor{D,D}
    lambda::TL   # lambda[i][j] = λ*(Dᵢ∘Dⱼ), for all i,j
    mu::TM       # mu[i][j] = μ*Dᵢᵢ (narrow) if i==j, else μ*(Dᵢ∘Dⱼ) (wide)
    size::NTuple{D,Int}
end

"""
    isotropic_lambda_mu(g, λ, μ, stencil_set)

Builds the `D×D` tuples of λ- and μ-weighted second-derivative operators
(see module notes above for the wide/narrow split) used by both
`elastic_operator` and `elastic_blocks`.
"""
function isotropic_lambda_mu(g, λ, μ, stencil_set)
    D = ndims(g)
    D1 = ntuple(i -> first_derivative(g, stencil_set, i), D)
    D2 = ntuple(i -> second_derivative(g, stencil_set, i), D)
    lambda = ntuple(Val(D)) do i
        ntuple(Val(D)) do j
            λ * (D1[i] ∘ D1[j])
        end
    end
    mu = ntuple(Val(D)) do i
        ntuple(Val(D)) do j
            i == j ? μ * D2[i] : μ * (D1[i] ∘ D1[j])
        end
    end
    return lambda, mu
end

"""
    elastic_operator(g, λ, μ, stencil_set)

The constant-coefficient isotropic Navier operator as a `LazyTensor` acting
on vector-valued (`SVector{D}`) grid functions.
"""
function elastic_operator(g, λ, μ, stencil_set)
    lambda, mu = isotropic_lambda_mu(g, λ, μ, stencil_set)
    return IsotropicElasticOperator(lambda, mu, size(g))
end

"""
    elastic_blocks(g, λ, μ, stencil_set)

The `D×D` tuple of scalar operators `M[j][k]` (coefficient of `u_k` in
equation `j`) for the assembled Navier operator, combining the λ/μ pieces
from `isotropic_lambda_mu`: `M[j][j] = μ∇² + μ*Dⱼⱼ + λ*(Dⱼ∘Dⱼ)`,
`M[j][k] = μ*(Dⱼ∘Dₖ) + λ*(Dⱼ∘Dₖ)` for `k≠j`.
"""
function elastic_blocks(g, λ, μ, stencil_set)
    D = ndims(g)
    lambda, mu = isotropic_lambda_mu(g, λ, μ, stencil_set)
    Δμ = μ * laplace(g, stencil_set)
    return ntuple(Val(D)) do j
        ntuple(Val(D)) do k
            j == k ? Δμ + mu[j][j] + lambda[j][j] : mu[j][k] + lambda[j][k]
        end
    end
end

LazyTensors.range_size(op::IsotropicElasticOperator) = op.size
LazyTensors.domain_size(op::IsotropicElasticOperator) = op.size

# General D-dimensional apply, kept as a correctness reference (mirrors
# elastic.jl's general case) — allocates due to Julia/Diffinitive inference
# limitations on the generic `ntuple`+loop form (documented in the notebook
# as a known issue), which is why the 3D case below is hand-specialized.
function LazyTensors.apply(op::IsotropicElasticOperator{D}, u::AbstractArray{Tu,D}, I...) where {D,Tu}
    S = eltype(Tu)
    return ntuple(Val(D)) do j
        uⱼ = componentview(u, j)
        res = S(2) * apply(op.mu[j][j], uⱼ, I...)
        for i in 1:D
            res += apply(op.lambda[j][i], componentview(u, i), I...)
        end
        for i in Iterators.flatten((1:j-1, j+1:D))
            uᵢ = componentview(u, i)
            res += apply(op.mu[i][i], uⱼ, I...) + apply(op.mu[i][j], uᵢ, I...)
        end
        res
    end |> SVector
end

# Non-allocating specialized 3D apply (this project is always 3D), directly
# mirroring elastic.jl's hand-unrolled 3D specialization.
@inline function LazyTensors.apply(op::IsotropicElasticOperator{3}, u::AbstractArray{Tu,3}, I...) where Tu
    u1 = componentview(u, 1)
    u2 = componentview(u, 2)
    u3 = componentview(u, 3)

    D₁₁λ = op.lambda[1][1]; D₁₂λ = op.lambda[1][2]; D₁₃λ = op.lambda[1][3]
    D₂₁λ = op.lambda[2][1]; D₂₂λ = op.lambda[2][2]; D₂₃λ = op.lambda[2][3]
    D₃₁λ = op.lambda[3][1]; D₃₂λ = op.lambda[3][2]; D₃₃λ = op.lambda[3][3]

    D₁₁μ = op.mu[1][1]; D₁₂μ = op.mu[1][2]; D₁₃μ = op.mu[1][3]
    D₂₁μ = op.mu[2][1]; D₂₂μ = op.mu[2][2]; D₂₃μ = op.mu[2][3]
    D₃₁μ = op.mu[3][1]; D₃₂μ = op.mu[3][2]; D₃₃μ = op.mu[3][3]

    res1 = apply(D₁₁λ, u1, I...) + apply(D₁₂λ, u2, I...) + apply(D₁₃λ, u3, I...) +
         2 * apply(D₁₁μ, u1, I...) + apply(D₂₂μ, u1, I...) + apply(D₂₁μ, u2, I...) +
         apply(D₃₁μ, u3, I...) + apply(D₃₃μ, u1, I...)

    res2 = apply(D₂₁λ, u1, I...) + apply(D₂₂λ, u2, I...) + apply(D₂₃λ, u3, I...) +
         apply(D₁₁μ, u2, I...) + apply(D₁₂μ, u1, I...) +
         2 * apply(D₂₂μ, u2, I...) + apply(D₃₂μ, u3, I...) + apply(D₃₃μ, u2, I...)

    res3 = apply(D₃₁λ, u1, I...) + apply(D₃₂λ, u2, I...) + apply(D₃₃λ, u3, I...) +
         apply(D₁₁μ, u3, I...) + apply(D₁₃μ, u1, I...) +
         apply(D₂₂μ, u3, I...) + apply(D₂₃μ, u2, I...) +
         2 * apply(D₃₃μ, u3, I...)

    return SVector{3,eltype(Tu)}(res1, res2, res3)
end

# Non-allocating specialized 2D apply, mirroring elastic.jl's 2D
# specialization (used e.g. for 2D elastic wave simulations).
@inline function LazyTensors.apply(op::IsotropicElasticOperator{2}, u::AbstractArray{Tu,2}, I...) where Tu
    u1 = componentview(u, 1)
    u2 = componentview(u, 2)

    D₁₁λ = op.lambda[1][1]; D₁₂λ = op.lambda[1][2]
    D₂₁λ = op.lambda[2][1]; D₂₂λ = op.lambda[2][2]

    D₁₁μ = op.mu[1][1]; D₁₂μ = op.mu[1][2]
    D₂₁μ = op.mu[2][1]; D₂₂μ = op.mu[2][2]

    res1 = apply(D₁₁λ, u1, I...) + apply(D₁₂λ, u2, I...) +
         2 * apply(D₁₁μ, u1, I...) + apply(D₂₂μ, u1, I...) + apply(D₂₁μ, u2, I...)

    res2 = apply(D₂₁λ, u1, I...) + apply(D₂₂λ, u2, I...) +
         2 * apply(D₂₂μ, u2, I...) + apply(D₁₁μ, u2, I...) + apply(D₁₂μ, u1, I...)

    return SVector{2,eltype(Tu)}(res1, res2)
end

"""
    traction_blocks(g, λ, μ, stencil_set, bid)

The `D × D` matrix of sparse operators `T[i,k]` such that the traction
component `τᵢ = σᵢ,dim` (with respect to the fixed `+dim` axis direction,
`dim = grid_id(bid)` — NOT the outward-normal sign convention) on boundary
`bid` is `τᵢ = Σₖ T[i,k]*uₖ`.

**The normal-direction μ derivatives use the boundary derivative, not
`first_derivative`**, mirroring `context/notebooks/elastic_clean.jl`'s
`IsotropicTractionOperator`. This is required for `T` to be the operator that
appears in `elastic_blocks`' own discrete SBP identity, i.e. for

    (v, Eu)_Ω - (Ev, u)_Ω = (v, Tu)_∂Ω - (Tv, u)_∂Ω

to hold — see the `"SBP property: E and T are compatible"` test. The reason is
that the two schemes' SBP identities carry *different* boundary operators:
`H·D₂ = -D₁ᵀHD₁ - R + e'Hᵧ·D̂` for the narrow `second_derivative`, versus
`H·D₁∘D₁ = -D₁ᵀHD₁ + e'Hᵧ·(e∘D₁)` for the wide sandwich. So λ terms and μ's
*tangential* terms pair with `e∘first_derivative`, while μ's normal-direction
terms — which come from the narrow operators (`laplace` and `second_derivative`
in `elastic_blocks`) — pair with `D̂`. Using `first_derivative` throughout left
`-HP(D+SAT)P` ~14% asymmetric and made CG unusable; see `SYMMETRIC_SAT.md` for
the full diagnosis and `scripts/symmetry_decomposition.jl` to reproduce it.

`D̂ = s·normal_derivative`, undoing Diffinitive's outward sign to get the
fixed-`+dim`-axis convention used here. (`Grids._boundary_sign` is private
Diffinitive API; the notebook uses it the same way.)
"""
function traction_blocks(g, λ, μ, stencil_set, bid)
    D = ndims(g)
    dim = grid_id(bid)
    e = boundary_restriction(g, stencil_set, bid)
    Dk = ntuple(k -> first_derivative(g, stencil_set, k), D)
    # Boundary derivative on the fixed +dim axis, i.e. `normal_derivative` with
    # its outward sign removed.
    s = Grids._boundary_sign(component_type(g), bid)
    d̂ = s * sparse(normal_derivative(g, stencil_set, bid))
    Nb = prod(size(boundary_grid(g, bid)))
    N = length(g)

    T = [spzeros(Nb, N) for _ in 1:D, _ in 1:D]
    # σ_nn = λ·Σₖ∂ₖuₖ + 2μ·∂ₙuₙ: the λ part is wide, the 2μ part narrow.
    T[dim, dim] = sparse(λ * (e ∘ Dk[dim])) + 2μ * d̂
    for k in 1:D
        k == dim && continue
        T[dim, k] = sparse(λ * (e ∘ Dk[k]))
    end
    # σ_in = μ(∂ₙuᵢ + ∂ᵢuₙ): the ∂ₙ part is narrow, the tangential ∂ᵢ is not.
    for i in 1:D
        i == dim && continue
        T[i, dim] = sparse(μ * (e ∘ Dk[i]))
        T[i, i] = μ * d̂
    end
    return T
end

# ==============================================================================
# Vector-valued grid function <-> flat component-major vector utilities.
# ==============================================================================

"""
    dof_index(g, component, I::CartesianIndex)

Linear index into the flattened, component-major `D*length(g)`-vector
representation of a `D`-component vector-valued grid function on `g`.
"""
dof_index(g, component, I::CartesianIndex) = (component - 1) * length(g) + LinearIndices(size(g))[I]

"""
    flatten(u::AbstractArray{<:SVector{D}})

Flattens a `D`-component vector-valued grid function into a component-major
vector of length `D*length(u)`, matching `dof_index`/`halfspace_system`'s
ordering.
"""
function flatten(u::AbstractArray{<:SVector{D}}) where D
    N = length(u)
    v = Vector{Float64}(undef, D * N)
    for comp in 1:D
        v[(comp-1)*N+1:comp*N] .= vec(collect(componentview(u, comp)))
    end
    return v
end

"""
    unflatten(v::AbstractVector, g, D)

Inverse of `flatten`: reshapes a component-major flat vector back into a
`D`-component vector-valued grid function on `g`.
"""
function unflatten(v::AbstractVector, g, D)
    li = LinearIndices(size(g))
    N = length(g)
    return map(CartesianIndices(size(g))) do I
        SVector(ntuple(comp -> v[(comp-1)*N+li[I]], D))
    end
end

"""
    inject_dirichlet!(A, rhs, rows, values)

Strongly enforces `rhs[rows] .= values` by zeroing those rows of `A` and
setting the diagonal to `1` (no Diffinitive helper exists for this — it's
only needed for strong/injection boundary conditions, not SAT).
"""
function inject_dirichlet!(A::SparseMatrixCSC, rhs::AbstractVector, rows, values)
    for (r, v) in zip(rows, values)
        A[r, :] .= 0.0
        A[r, r] = 1.0
        rhs[r] = v
    end
    return A, rhs
end

end # module Elasticity

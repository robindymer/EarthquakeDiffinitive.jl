module ElasticitySplitNode

using Diffinitive
using Diffinitive.Grids
using Diffinitive.SbpOperators
using Diffinitive.LazyTensors
using SparseArrays
using LinearAlgebra: I, Diagonal, diag, mul!
import LinearAlgebra
using Krylov: CgWorkspace, cg!
using Tokens
using StaticArrays
using ..Elasticity: elastic_blocks, traction_blocks

export split_node_system, dof_index_minus, dof_index_plus, build_chi,
       reconstruct_U, fault_node_pairs,
       SplitNodeOperator, AssembledSplitNode, split_node_operator,
       apply_P!, hp_dsat!, duplicate_operator,
       CGSolver, CG_DEFAULTS, split_node_solve, jacobi_preconditioner,
       solver_report, duplicate, merge_stats!

# Two-sided (split-node) SBP-SAT elastic system, per context/SEAS_benchmark.pdf:
#
#   -H P (D + SAT) P u = H P (D + SAT) χ(s)
#
# D    block-diagonal elastic operator (`Elasticity.elastic_blocks`) per side.
# SAT  interface traction coupling: the scalar Neumann-SAT pattern with penalty
#      -H⁻¹∘e'∘Hᵧ, the normal derivative replaced by the traction operator, and
#      "data" the other side's traction. Signs: see `split_node_system`.
# P    averages all three fault DOF pairs (u1 too — that is how no-opening,
#      BP8 eq. 3, is imposed), zeroes far-field DOFs, identity elsewhere.
# χ(s) prescribed slip, ±s_j/2 at the fault.
#
# DOF layout (length 3*(N₋+N₊)): component-major within a side, sides
# concatenated [u₋ ; u₊]. See `dof_index_minus`/`dof_index_plus`.

dof_index_minus(g_minus, component, I) = (component - 1) * length(g_minus) + LinearIndices(size(g_minus))[I]
dof_index_plus(g_minus, g_plus, component, I) = 3 * length(g_minus) + (component - 1) * length(g_plus) + LinearIndices(size(g_plus))[I]

_to_sparse_matrix(M) = reduce(vcat, [reduce(hcat, [sparse(M[j][k]) for k in 1:length(M)]) for j in 1:length(M)])

# Appends `B`'s stored entries to the (I,J,V) lists with B[1,1] at
# (row0+1, col0+1), so `SATmat` needs one `sparse(I,J,V,…)` call. Repeated
# `SATmat[rows,cols] .+= B` rewrites the CSC structure each time and made
# assembly quadratic (199× slower by n=21) — `PERFORMANCE.md` §2.2.
function _append_block!(I, J, V, B::AbstractSparseMatrix, row0, col0)
    rows = rowvals(B)
    vals = nonzeros(B)
    for col in 1:size(B, 2), idx in nzrange(B, col)
        push!(I, row0 + rows[idx])
        push!(J, col0 + col)
        push!(V, vals[idx])
    end
    return nothing
end

# Prolong = -H⁻¹∘e'∘Hᵧ, the SAT penalty prefactor.
function _prolongation(g, stencil_set, bid)
    H_inv = inverse_inner_product(g, stencil_set)
    e = boundary_restriction(g, stencil_set, bid)
    Hb = inner_product(boundary_grid(g, bid), stencil_set)
    return -sparse(H_inv ∘ e' ∘ Hb)
end

"""
    fault_node_pairs(g_minus, g_plus)

Pairs of `(I_minus, I_plus)` `CartesianIndex`es on the two fault-facing
boundaries sharing an `(x2,x3)` location, matched by coordinate value so the
two grids' `boundary_indices` order does not matter.

The ring where the fault meets the truncated domain's sides is **excluded**:
those nodes carry the far-field Dirichlet `u=0` for all three components, so
they must be neither averaged by `P` nor given slip by `χ`. They also sit
outside `Ω_f`, where BP8 eq. 13 gives zero slip anyway.
"""
function fault_node_pairs(g_minus, g_plus)
    bid_minus = CartesianBoundary{1,UpperBoundary}()
    bid_plus = CartesianBoundary{1,LowerBoundary}()
    far_field = Set(Iterators.flatten(
        boundary_indices(g_minus, bid) for bid in boundary_identifiers(g_minus) if bid != bid_minus))
    lookup = Dict(Tuple(g_plus[I][2:3]) => I for I in boundary_indices(g_plus, bid_plus))
    return [(Im, lookup[Tuple(g_minus[Im][2:3])])
            for Im in boundary_indices(g_minus, bid_minus) if Im ∉ far_field]
end

"""
    sat_matrix(g_minus, g_plus, λ, μ, stencil_set) -> SparseMatrixCSC

The interface SAT block, `Ntot × Ntot`, half-weighted on both sides — signs in
[`split_node_system`](@ref). Boundary-local (nonzeros only in the few node
layers the prolongation reaches), hence cheap to assemble and small enough for
the matrix-free [`SplitNodeOperator`](@ref) to keep as a sparse matrix.
"""
function sat_matrix(g_minus, g_plus, λ, μ, stencil_set)
    D = 3
    Nm, Np = length(g_minus), length(g_plus)
    Ntot = D * (Nm + Np)
    bid_minus = CartesianBoundary{1,UpperBoundary}()
    bid_plus = CartesianBoundary{1,LowerBoundary}()
    Tm = traction_blocks(g_minus, λ, μ, stencil_set, bid_minus)
    Tp = traction_blocks(g_plus, λ, μ, stencil_set, bid_plus)
    Prolong_p = _prolongation(g_plus, stencil_set, bid_plus)
    Prolong_m = _prolongation(g_minus, stencil_set, bid_minus)

    # Triplets, assembled in one `sparse` call (see `_append_block!`). Column
    # offset 0 is the `-` side's block, `D*Nm` the `+` side's.
    Isat, Jsat, Vsat = Int[], Int[], Float64[]
    for j in 1:D
        row0_m = (j-1) * Nm
        row0_p = D * Nm + (j-1) * Np
        Tp_j = reduce(hcat, Tp[j, :])   # Nb × Np
        Tm_j = reduce(hcat, Tm[j, :])   # Nb × Nm

        # - side (fault = upper boundary, outward = +x₁):  ½ Prolong₋ (τ₋ - τ₊)
        _append_block!(Isat, Jsat, Vsat, 0.5 .* (Prolong_m * Tm_j), row0_m, 0)
        _append_block!(Isat, Jsat, Vsat, -0.5 .* (Prolong_m * Tp_j), row0_m, D * Nm)

        # + side (fault = lower boundary, outward = -x₁): -½ Prolong₊ (τ₊ - τ₋)
        _append_block!(Isat, Jsat, Vsat, -0.5 .* (Prolong_p * Tp_j), row0_p, D * Nm)
        _append_block!(Isat, Jsat, Vsat, 0.5 .* (Prolong_p * Tm_j), row0_p, 0)
    end
    return sparse(Isat, Jsat, Vsat, Ntot, Ntot)
end

"""
    projection_parts(g_minus, g_plus) -> (mask, pm, pp)

`P` as data rather than a matrix: `mask[r]` is `0.0` on far-field DOFs and
`1.0` elsewhere; `(pm[k], pp[k])` are the `-`/`+` DOFs of the `k`-th averaged
fault pair, all three components (averaging `u1` *is* no-opening, BP8 eq. 3;
leaving it unaveraged decouples the fault-normal DOFs). `fault_node_pairs`
excludes the fault ∩ far-field ring, so no DOF is both masked and paired.

`P v` is `mask .* v` with `v[pm]`, `v[pp]` replaced by their mean
([`apply_P!`](@ref)). [`projection_matrix`](@ref) builds the explicit matrix
from the same data, so the two agree by construction.
"""
function projection_parts(g_minus, g_plus)
    D = 3
    Nm, Np = length(g_minus), length(g_plus)
    Ntot = D * (Nm + Np)
    bid_minus = CartesianBoundary{1,UpperBoundary}()
    bid_plus = CartesianBoundary{1,LowerBoundary}()
    mask = ones(Ntot)
    for bid in filter(!=(bid_minus), boundary_identifiers(g_minus)),
        I in boundary_indices(g_minus, bid), comp in 1:D
        mask[dof_index_minus(g_minus, comp, I)] = 0.0
    end
    for bid in filter(!=(bid_plus), boundary_identifiers(g_plus)),
        I in boundary_indices(g_plus, bid), comp in 1:D
        mask[dof_index_plus(g_minus, g_plus, comp, I)] = 0.0
    end
    pm, pp = Int[], Int[]
    for (Im, Ip) in fault_node_pairs(g_minus, g_plus), comp in 1:D
        push!(pm, dof_index_minus(g_minus, comp, Im))
        push!(pp, dof_index_plus(g_minus, g_plus, comp, Ip))
    end
    return mask, pm, pp
end

"""
    projection_matrix(mask, pm, pp) -> SparseMatrixCSC

The explicit `P` from [`projection_parts`](@ref), from triplets in one
`sparse` call. The old `sparse(I)` + per-row `P[r, :] .= 0.0` form rewrote the
CSC structure once per row and scaled as DOF^1.7 (~27-37% of assembly at 1.2 M
DOF). Same matrix, linear cost.
"""
function projection_matrix(mask, pm, pp)
    Ntot = length(mask)
    paired = falses(Ntot)
    paired[pm] .= true
    paired[pp] .= true
    Ip, Jp, Vp = Int[], Int[], Float64[]
    for r in 1:Ntot
        (mask[r] != 0.0 && !paired[r]) || continue
        push!(Ip, r); push!(Jp, r); push!(Vp, 1.0)
    end
    for (rm, rp) in zip(pm, pp), r in (rm, rp)
        push!(Ip, r); push!(Jp, rm); push!(Vp, 0.5)
        push!(Ip, r); push!(Jp, rp); push!(Vp, 0.5)
    end
    return sparse(Ip, Jp, Vp, Ntot, Ntot)
end

"""
    split_node_system(g_minus, g_plus, λ, μ, stencil_set)

Assembles the two-sided split-node elastic system `A = -H*P*(D+SAT)*P` per
`context/SEAS_benchmark.pdf`. `g_minus`/`g_plus` must share the same
`(x2,x3)` discretization. Fault boundary is `g_minus`'s `UpperBoundary` /
`g_plus`'s `LowerBoundary` in dimension 1. Returns `(A, HP_DSAT, P)`:
`HP_DSAT = H*P*(D+SAT)` (needed again, applied to `χ(s)`, to build the RHS
for a given slip — see `build_chi`); `P` is needed afterward to reconstruct
the true field `U = P*u + χ(s)` (see `reconstruct_U`).
"""
function split_node_system(g_minus, g_plus, λ, μ, stencil_set)
    D = 3
    Nm, Np = length(g_minus), length(g_plus)
    Ntot = D * (Nm + Np)
    bid_minus = CartesianBoundary{1,UpperBoundary}()
    bid_plus = CartesianBoundary{1,LowerBoundary}()

    # ---- D: block-diagonal elastic operator ----
    # Assembled, not lazy: Diffinitive's lazy composition measured 41-67× slower
    # than `mul!` (`PERFORMANCE.md` §1). The matrix-free route that does work is
    # `SplitNodeOperator` below. This sparsification is ~60-70% of assembly time.
    Dm = _to_sparse_matrix(elastic_blocks(g_minus, λ, μ, stencil_set))
    Dp = _to_sparse_matrix(elastic_blocks(g_plus, λ, μ, stencil_set))
    Dmat = blockdiag(Dm, Dp)

    # ---- SAT: interface traction coupling, half-weighted on both sides ----
    # The benchmark also allows applying it fully to one side; that measured
    # asymmetric and non-PSD, so both sides get half.
    #
    # SIGNS. `-H⁻¹∘e'∘Hᵧ ∘ (t_out - data)` is in terms of the OUTWARD traction,
    # and `traction_blocks` is fixed-`+x₁`-axis: t_out = +T on `g_minus` (fault
    # is its upper boundary) and -T on `g_plus`. The data is the other side's
    # outward traction negated, since traction balance (BP8 eq. 6) is
    # t_out⁻ + t_out⁺ = 0. So both sides penalize the same (τ₋ - τ₊) and the `+`
    # side carries one extra minus. Wrong signs leave the shear tractions
    # discontinuous and inflate the solution by orders of magnitude.
    SATmat = sat_matrix(g_minus, g_plus, λ, μ, stencil_set)

    DSAT = Dmat + SATmat

    # ---- P: projection (average fault DOF pairs, zero far-field) ----
    # Explicit because `A` is being assembled: a lazy `P` forces `A` into a
    # composite, 1.66× slower per mat-vec (`PERFORMANCE.md` §1). The matrix-free
    # operator applies the same `(mask, pm, pp)` data directly via `apply_P!`.
    P = projection_matrix(projection_parts(g_minus, g_plus)...)

    # ---- H: block-diagonal volume inner product (for symmetrization) ----
    Hm = sparse(inner_product(g_minus, stencil_set))
    Hp = sparse(inner_product(g_plus, stencil_set))
    H = blockdiag(blockdiag(fill(Hm, D)...), blockdiag(fill(Hp, D)...))

    HP_DSAT = H * P * DSAT
    A = -HP_DSAT * P

    return A, HP_DSAT, P
end

"""
    build_chi(g_minus, g_plus, slip_fn)

`χ(s)`: zero except on the fault, zero for `u1`, and `±slip_fn(x2,x3)/2` for
`u2,u3` on the +/- sides. `slip_fn(x2,x3) -> (s2,s3)`.
"""
function build_chi(g_minus, g_plus, slip_fn)
    Nm, Np = length(g_minus), length(g_plus)
    χ = zeros(3 * (Nm + Np))
    for (Im, Ip) in fault_node_pairs(g_minus, g_plus)
        x = g_plus[Ip]
        s2, s3 = slip_fn(x[2], x[3])
        χ[dof_index_plus(g_minus, g_plus, 2, Ip)] = s2 / 2
        χ[dof_index_plus(g_minus, g_plus, 3, Ip)] = s3 / 2
        χ[dof_index_minus(g_minus, 2, Im)] = -s2 / 2
        χ[dof_index_minus(g_minus, 3, Im)] = -s3 / 2
    end
    return χ
end

"""
    reconstruct_U(P, u, χ)

The true displacement field `U = P*u + χ`. Tractions must come from `U`, not
the raw solve variable `u`.
"""
reconstruct_U(P, u, χ) = P * u + χ

# Matrix-free split-node operator. This is the production path.
#
# On an equidistant `TensorGrid` Diffinitive's 3D `D1`/`D2` are exactly
# Kronecker products of the 1D operators (`max|diff| = 0.0`), so one side's
# Navier operator is six `n×n` 1D matrices plus the 1D weights.
# `A v = -H P (D + SAT) P v` is applied as
#
#   P     mask + pair averaging              (`apply_P!`)
#   D     18 axis passes per side            (`navier!` / `axpass!`)
#   SAT   the boundary-local sparse block    (`sat_matrix`)
#   H     Kronecker weights h1[i]h2[j]h3[k]  (`scale_H!`)
#
# Nothing of size `nnz(A)` is formed. At Δz = 10 m on (1600, 1600) that saves
# ~85 GB of device memory, ~73 h of host assembly and ~3× the memory traffic
# per mat-vec. Agrees with the assembled `A` to 3e-16 with identical CG
# iteration counts. See `MATRIX_FREE_PLAN.md`.
#
# The Navier block, arranged to minimise passes, with `w_k = D1_k u_k`:
#
#   out_j = μ Σ_i D2_i u_j + μ D2_j u_j + D1_j[(λ+μ) div − μ w_j],  div = Σ_k w_k
#
# — `elastic_blocks`' wide sandwiches folded into one `D1_j` pass over a
# combined field. Same operator term for term, different summation order.
#
# Fields are flat vectors plus an offset, not reshaped views, so one `mul!`
# runs on `Vector` and (via the CUDA extension's `axpass!`/`scale_H!`) on
# `CuVector` with no array wrappers.

"""
    Rows1D(M::SparseMatrixCSC) -> Rows1D

A 1D SBP operator stored by rows (CSR), the access pattern an axis pass needs:
output point `r` reads row `r`, so interior stencil and boundary closures need
no special-casing. Index vectors are `Int` on the host, `Int32` on the device.
"""
struct Rows1D{VI<:AbstractVector{<:Integer},VV<:AbstractVector{Float64}}
    n::Int
    rowptr::VI
    colind::VI
    val::VV
end
function Rows1D(M::SparseMatrixCSC)
    size(M, 1) == size(M, 2) || error("Rows1D: 1D operator must be square")
    Mt = sparse(transpose(M))
    return Rows1D(size(M, 1), copy(SparseArrays.getcolptr(Mt)), copy(rowvals(Mt)), copy(nonzeros(Mt)))
end

"""
    SideOps

One half-space: grid size, the three 1D first- and second-derivative
operators, and the three 1D inner-product weight vectors. `H` on this side is
`h[1][i] * h[2][j] * h[3][k]`.
"""
struct SideOps{R<:Rows1D,VV<:AbstractVector{Float64}}
    n::NTuple{3,Int}
    d1::NTuple{3,R}
    d2::NTuple{3,R}
    h::NTuple{3,VV}
end
function SideOps(g, stencil_set)
    # The TensorGrid's 1D factor grids, axis order = index order — the field
    # Diffinitive's `first_derivative(::TensorGrid, set, dim)` inflates from.
    gs = g.grids
    d1 = ntuple(d -> Rows1D(sparse(first_derivative(gs[d], stencil_set))), 3)
    d2 = ntuple(d -> Rows1D(sparse(second_derivative(gs[d], stencil_set))), 3)
    h = ntuple(d -> Vector{Float64}(diag(sparse(inner_product(gs[d], stencil_set)))), 3)
    return SideOps(size(g), d1, d2, h)
end
Base.length(s::SideOps) = prod(s.n)

"""
    SplitNodeOperator

`A = -H P (D + SAT) P` applied matrix-free (see the notes above). Holds both
sides' 1D operators, `P` as `(mask, pm, pp)`, the sparse SAT block, and the
scratch one mat-vec needs, so `mul!` allocates nothing.

That scratch makes the operator **stateful per call**: two solves must not
share an instance concurrently. [`duplicate_operator`](@ref) gives a copy with
its own scratch and everything else shared; `duplicate(::CGSolver)` uses it.

Build with [`split_node_operator`](@ref); use via `mul!`, [`apply_P!`](@ref),
[`hp_dsat!`](@ref), or hand it to [`CGSolver`](@ref) like a matrix.
"""
struct SplitNodeOperator{S<:SideOps,VV<:AbstractVector{Float64},VI<:AbstractVector{<:Integer},TS}
    minus::S
    plus::S
    λ::Float64
    μ::Float64
    mask::VV
    pm::VI
    pp::VI
    SAT::TS
    Ntot::Int
    # scratch: two full-length vectors, one pair-length, and five field-length
    pv::VV
    dsat::VV
    pavg::VV
    w::NTuple{3,VV}
    div::VV
    tmp::VV
end

"""
    split_node_operator(g_minus, g_plus, λ, μ, stencil_set) -> SplitNodeOperator

The matrix-free counterpart of [`split_node_system`](@ref): the same `A`,
`HP_DSAT` and `P` as functions rather than matrices. Seconds to build (the SAT
block dominates) where assembly takes hours, and holds no `nnz(A)`-sized data.
"""
function split_node_operator(g_minus, g_plus, λ, μ, stencil_set)
    minus = SideOps(g_minus, stencil_set)
    plus = SideOps(g_plus, stencil_set)
    Ntot = 3 * (length(minus) + length(plus))
    mask, pm, pp = projection_parts(g_minus, g_plus)
    SAT = sat_matrix(g_minus, g_plus, λ, μ, stencil_set)
    N = max(length(minus), length(plus))
    return SplitNodeOperator(minus, plus, Float64(λ), Float64(μ), mask, pm, pp, SAT, Ntot,
                             zeros(Ntot), zeros(Ntot), zeros(length(pm)),
                             ntuple(_ -> zeros(N), 3), zeros(N), zeros(N))
end

"""
    duplicate_operator(A) -> A′

An operator safe to use concurrently with `A`: for a matrix, `A` itself (it is
only ever read); for a [`SplitNodeOperator`](@ref), a copy sharing the 1D
operators, `P` data and `SAT` but with its own scratch.
"""
duplicate_operator(A) = A
function duplicate_operator(op::SplitNodeOperator)
    return SplitNodeOperator(op.minus, op.plus, op.λ, op.μ, op.mask, op.pm, op.pp, op.SAT, op.Ntot,
                             similar(op.pv), similar(op.dsat), similar(op.pavg),
                             map(similar, op.w), similar(op.div), similar(op.tmp))
end

Base.size(op::SplitNodeOperator) = (op.Ntot, op.Ntot)
Base.size(op::SplitNodeOperator, i::Integer) = i <= 2 ? op.Ntot : 1
Base.eltype(::SplitNodeOperator) = Float64
LinearAlgebra.issymmetric(::SplitNodeOperator) = true
LinearAlgebra.ishermitian(::SplitNodeOperator) = true

"""
    axpass!(out, off_out, M::Rows1D, u, off_u, n, d, α)

`out[off_out + p] += α * (M along axis d of u[off_u + …])[p]` for the
`n = (n1,n2,n3)` field stored column-major at those offsets. The CUDA
extension adds the `CuVector` method with the same signature.
"""
function axpass!(out::Vector{Float64}, off_out::Int, M::Rows1D, u::Vector{Float64}, off_u::Int,
                 n::NTuple{3,Int}, d::Int, α::Float64)
    n1, n2, n3 = n
    rp, ci, val = M.rowptr, M.colind, M.val
    if d == 1
        Threads.@threads for k in 1:n3
            @inbounds for j in 1:n2
                base_u = off_u + (j-1)*n1 + (k-1)*n1*n2
                base_o = off_out + (j-1)*n1 + (k-1)*n1*n2
                for i in 1:n1
                    s = 0.0
                    for idx in rp[i]:(rp[i+1]-1)
                        s += val[idx] * u[base_u + ci[idx]]
                    end
                    out[base_o + i] += α * s
                end
            end
        end
    elseif d == 2
        Threads.@threads for k in 1:n3
            @inbounds for j in 1:n2
                base_o = off_out + (j-1)*n1 + (k-1)*n1*n2
                for idx in rp[j]:(rp[j+1]-1)
                    c = α * val[idx]
                    base_u = off_u + (ci[idx]-1)*n1 + (k-1)*n1*n2
                    @simd for i in 1:n1
                        out[base_o + i] += c * u[base_u + i]
                    end
                end
            end
        end
    else
        Threads.@threads for k in 1:n3
            @inbounds for idx in rp[k]:(rp[k+1]-1)
                c = α * val[idx]
                kk = ci[idx]
                base_o = off_out + (k-1)*n1*n2
                base_u = off_u + (kk-1)*n1*n2
                @simd for q in 1:n1*n2
                    out[base_o + q] += c * u[base_u + q]
                end
            end
        end
    end
    return out
end

"""
    scale_H!(y, off, side::SideOps, α)

`y[off + (i,j,k)] *= α * h1[i] h2[j] h3[k]` over one side's three components
(`3 * length(side)` entries from `off`). The CUDA extension adds the
`CuVector` method.
"""
function scale_H!(y::Vector{Float64}, off::Int, side::SideOps{<:Any,Vector{Float64}}, α::Float64)
    n1, n2, n3 = side.n
    h1, h2, h3 = side.h
    N = n1 * n2 * n3
    for comp in 0:2
        Threads.@threads for k in 1:n3
            @inbounds for j in 1:n2
                c = α * h2[j] * h3[k]
                base = off + comp * N + (j-1)*n1 + (k-1)*n1*n2
                @simd for i in 1:n1
                    y[base + i] *= c * h1[i]
                end
            end
        end
    end
    return y
end

# The Navier operator of one side, `out[off_out + …] = D u[off_u + …]` over the
# three components, using the operator's field scratch.
function navier!(out, off_out, u, off_u, side::SideOps, op::SplitNodeOperator)
    N = length(side)
    λ, μ = op.λ, op.μ
    for k in 1:3
        fill!(op.w[k], 0.0)
        axpass!(op.w[k], 0, side.d1[k], u, off_u + (k-1)*N, side.n, k, 1.0)
    end
    op.div .= op.w[1] .+ op.w[2] .+ op.w[3]
    for j in 1:3
        oj = off_out + (j-1)*N
        uj = off_u + (j-1)*N
        for i in 1:3
            axpass!(out, oj, side.d2[i], u, uj, side.n, i, μ)
        end
        axpass!(out, oj, side.d2[j], u, uj, side.n, j, μ)
        op.tmp .= (λ + μ) .* op.div .- μ .* op.w[j]
        axpass!(out, oj, side.d1[j], op.tmp, 0, side.n, j, 1.0)
    end
    return out
end

"""
    apply_P!(y, op, v) -> y

`y = P v`: far-field DOFs zeroed, each fault pair replaced by its mean.
Broadcast/gather/scatter only, so it runs on host and device vectors alike.
"""
function apply_P!(y::AbstractVector, op::SplitNodeOperator, v::AbstractVector)
    y .= op.mask .* v
    op.pavg .= 0.5 .* (view(v, op.pm) .+ view(v, op.pp))
    view(y, op.pm) .= op.pavg
    view(y, op.pp) .= op.pavg
    return y
end

"""
    hp_dsat!(y, op, x) -> y

`y = H P (D + SAT) x` — the `HP_DSAT` of [`split_node_system`](@ref) applied to
`x`. With `x = χ(s)` this is the right-hand side of the split-node solve.
"""
function hp_dsat!(y::AbstractVector, op::SplitNodeOperator, x::AbstractVector)
    Nm = length(op.minus)
    fill!(op.dsat, 0.0)
    navier!(op.dsat, 0, x, 0, op.minus, op)
    navier!(op.dsat, 3Nm, x, 3Nm, op.plus, op)
    mul!(op.dsat, op.SAT, x, 1.0, 1.0)
    apply_P!(y, op, op.dsat)
    scale_H!(y, 0, op.minus, 1.0)
    scale_H!(y, 3Nm, op.plus, 1.0)
    return y
end

"""
    mul!(y, op::SplitNodeOperator, v) -> y

`y = A v = -H P (D + SAT) P v`.
"""
function LinearAlgebra.mul!(y::AbstractVector, op::SplitNodeOperator, v::AbstractVector)
    apply_P!(op.pv, op, v)
    hp_dsat!(y, op, op.pv)
    y .*= -1.0
    return y
end
Base.:*(op::SplitNodeOperator, v::AbstractVector) = mul!(similar(v, op.Ntot), op, v)

"""
    AssembledSplitNode(A, HP_DSAT, P)

[`split_node_system`](@ref)'s three matrices behind the same interface as
[`SplitNodeOperator`](@ref), so `FaultElasticity` can be built either way and
the two compared.
"""
struct AssembledSplitNode{TA,TH,TP}
    A::TA
    HP_DSAT::TH
    P::TP
end
Base.size(w::AssembledSplitNode) = size(w.A)
Base.size(w::AssembledSplitNode, i::Integer) = size(w.A, i)
Base.eltype(w::AssembledSplitNode) = eltype(w.A)
LinearAlgebra.issymmetric(::AssembledSplitNode) = true
LinearAlgebra.ishermitian(::AssembledSplitNode) = true
LinearAlgebra.diag(w::AssembledSplitNode) = diag(w.A)
LinearAlgebra.mul!(y::AbstractVector, w::AssembledSplitNode, v::AbstractVector) = mul!(y, w.A, v)
Base.:*(w::AssembledSplitNode, v::AbstractVector) = w.A * v
apply_P!(y::AbstractVector, w::AssembledSplitNode, v::AbstractVector) = mul!(y, w.P, v)
hp_dsat!(y::AbstractVector, w::AssembledSplitNode, x::AbstractVector) = mul!(y, w.HP_DSAT, x)

# Iterative solver: CG straight onto the singular A.

"""
    CGStats

Running totals across every solve a [`CGSolver`](@ref) has performed.
Iteration count is what decides the cost of a `K` build, and it grows with
problem size.
"""
mutable struct CGStats
    solves::Int
    iterations::Int
    max_iterations::Int
    unconverged::Int
end
CGStats() = CGStats(0, 0, 0, 0)

"""
    CGSolver(A; rtol=1e-10, atol=0.0, itmax=0)

Conjugate gradients on `A = -H*P*(D+SAT)*P` directly — no reduction, no
factorization. Memory is a handful of vectors instead of a sparse factor, and
that is the point: fill-in is what used to cap resolution.

## Why the singularity needs no special handling

`A` annihilates `null(P)` (far-field DOFs plus the antisymmetric half of every
fault pair, ~40% of the system), which would normally rule out CG. Three
things make it safe, all holding exactly rather than approximately:

 1. **The system is consistent**, `b ⊥ null(A)`. For `v ∈ null(P) ⊆ null(A)`,
    `vᵀb = vᵀHP(D+SAT)χ = (HPv)ᵀ(D+SAT)χ = 0`, using `HP = PH` and `P = Pᵀ`.
    Measured `‖(I-P)b‖/‖b‖ = 0.0`.
 2. **Iterates never leave `range(A)`.** From `x₀ = 0` every Krylov vector is
    in `span{b, Ab, A²b, …} ⊆ range(A)`; measured null-space content of `u`
    after ~100 iterations is exactly 0.
 3. **Any leak is projected away** by the only use of `u`, `U = P*u + χ`.

On `range(A)` it is positive definite (κ = 118 at n=9), so CG converges at the
normal rate and matches a direct solve's tractions to 1.4e-11.

All of this needs `A` **symmetric**, which required fixing `traction_blocks` —
before that CG stalled at residual 3.7e-2 after 5000 iterations
(`SYMMETRIC_SAT.md`).

`itmax=0` lets Krylov pick its own cap. A solve that hits it is counted in
[`solver_report`](@ref)'s `unconverged` and warned about once: silently
returning an unconverged `u` would corrupt `K` unnoticed.

`precond` selects `M⁻¹` — `:none` (default) or `:jacobi`. A diagonal `M`
commutes with `P` exactly, so any SPD preconditioner is safe here, but Jacobi
measures 0.92× (a regression) and is kept only to record that. `PERFORMANCE.md`
§6.
"""
struct CGSolver{TA,TM}
    A::TA
    M::TM                  # applies M⁻¹; `I` for unpreconditioned
    ldiv::Bool             # true ⇒ Krylov calls ldiv!(M, r) rather than M*r
    precond::Symbol
    workspace::CgWorkspace{Float64,Float64,Vector{Float64}}
    rtol::Float64
    atol::Float64
    itmax::Int
    stats::CGStats
end

"""
    jacobi_preconditioner(A) -> Diagonal

`M⁻¹ = diag(A)⁻¹`, with zero diagonal entries floored to 1.

The floor is required, not defensive: `P` zeroes every far-field DOF's row, so
`diag(A)` contains exact zeros there (~36% of entries) and `1 ./ diag(A)` would
give `Inf`. The value is arbitrary — those DOFs lie in `null(P)` and are
annihilated by `U = P*u + χ` — it only has to keep `M` positive definite. Every
nonzero entry is positive, so nothing else needs guarding.
"""
jacobi_preconditioner(::SplitNodeOperator) =
    error("precond=:jacobi needs diag(A), which the matrix-free SplitNodeOperator does " *
          "not form; use representation=:assembled (it is measured useless anyway, " *
          "PERFORMANCE.md §6)")
function jacobi_preconditioner(A)
    d = diag(A)
    any(x -> x < 0, d) && error("diag(A) has negative entries; A is not PSD and " *
                                "a Jacobi preconditioner would not be SPD")
    return Diagonal([x == 0 ? 1.0 : inv(x) for x in d])
end

function build_preconditioner(A, precond::Symbol)
    precond === :none   && return I, false
    precond === :jacobi && return jacobi_preconditioner(A), false
    error("precond must be :none or :jacobi, got $precond")
end

"""
    CG_DEFAULTS

The default `CGSolver` keywords in one place, so `StiffnessCache` can fold
the solver settings into a cache key without duplicating the literals.
"""
const CG_DEFAULTS = (rtol=1e-10, atol=0.0, itmax=0, precond=:none)

function CGSolver(A; rtol=CG_DEFAULTS.rtol, atol=CG_DEFAULTS.atol,
                  itmax=CG_DEFAULTS.itmax, precond::Symbol=CG_DEFAULTS.precond)
    n = size(A, 2)
    M, ldiv = build_preconditioner(A, precond)
    return CGSolver(A, M, ldiv, precond, CgWorkspace(n, n, Vector{Float64}),
                    rtol, atol, itmax, CGStats())
end

"""
    split_node_solve(solver, rhs) -> u

Solves `A u = rhs` by CG. Only `P*u` is meaningful; `U = P*u + χ` discards
whatever the iterates carry on `null(P)`.
"""
function split_node_solve(s::CGSolver, rhs)
    cg!(s.workspace, s.A, rhs; M=s.M, ldiv=s.ldiv,
        rtol=s.rtol, atol=s.atol, itmax=s.itmax)
    st = s.workspace.stats
    t = s.stats
    t.solves += 1
    t.iterations += st.niter
    t.max_iterations = max(t.max_iterations, st.niter)
    if !st.solved
        t.unconverged += 1
        t.unconverged == 1 && @warn """
            CG did not converge on a split-node solve (status "$(st.status)") \
            after $(st.niter) iterations. The returned displacement is not a \
            solution, and `fault_stiffness` would fold it into `K` silently. \
            Raise `itmax` or loosen `rtol`.""" rtol = s.rtol itmax = s.itmax
    end
    # The workspace buffer is reused by the next solve, so hand back a copy.
    return copy(s.workspace.x)
end

"""
    duplicate(solver) -> solver

An independent solver for another thread. The per-solve state is the
`CgWorkspace` (a few vectors); a matrix `A` is shared since it is only read,
and a `SplitNodeOperator` goes through [`duplicate_operator`](@ref) for its
scratch. This is what makes `fault_stiffness` embarrassingly parallel — its
`2·N_Ωf` columns are independent right-hand sides against one `A`.
"""
function duplicate(s::CGSolver)
    n = size(s.A, 2)
    # `M` is shared, not rebuilt — valid only while it is stateless under
    # application, true for `I` and `Diagonal`. A preconditioner with internal
    # scratch must be duplicated here instead or the threaded build races.
    return CGSolver(duplicate_operator(s.A), s.M, s.ldiv, s.precond, CgWorkspace(n, n, Vector{Float64}),
                    s.rtol, s.atol, s.itmax, CGStats())
end

"""
    merge_stats!(into, from)

Folds a duplicated solver's counters back into the original, so
[`solver_report`](@ref) totals the whole threaded build.
"""
function merge_stats!(into::CGSolver, from::CGSolver)
    a, b = into.stats, from.stats
    a.solves += b.solves
    a.iterations += b.iterations
    a.max_iterations = max(a.max_iterations, b.max_iterations)
    a.unconverged += b.unconverged
    return into
end

"""
    solver_report(solver::CGSolver) -> NamedTuple

What the solves have cost so far: totals and worst case over every solve, so
a `K` build reports across all `2·N_Ωf` right-hand sides.
"""
function solver_report(s::CGSolver)
    t = s.stats
    return (; kind=:cg, precond=s.precond, t.solves, t.iterations,
            mean_iterations=t.solves == 0 ? 0.0 : t.iterations / t.solves,
            t.max_iterations, t.unconverged)
end

end # module ElasticitySplitNode

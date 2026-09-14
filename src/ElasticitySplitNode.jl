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

# ==============================================================================
# Two-sided (split-node) SBP-SAT elastic system, matching the formulation in
# context/SEAS_benchmark.pdf:
#
#   -H P (D + SAT) P u = H P (D + SAT) χ(s)
#
# D: block-diagonal elastic operator (Elasticity.elastic_blocks, unchanged,
#    on each side's own grid).
# SAT: interface traction coupling — the usual scalar Neumann-SAT pattern
#    penalty = -H⁻¹∘e'∘Hᵧ, but with the scalar normal-derivative operator
#    replaced by the traction operator, and "data" being the OTHER side's
#    traction. See the sign discussion in `split_node_system`.
# P: projection — averages ALL THREE fault DOF pairs (u1 too: that is how the
#    no-opening condition u1(0⁺)=u1(0⁻), BP8 eq. 3, gets imposed), zeroes
#    far-field DOFs, identity elsewhere.
# χ(s): forcing vector encoding the prescribed slip, ±s_j/2 at the fault.
#
# DOF layout (flat vector of length 3*(N₋+N₊)): component-major within each
# side, sides concatenated as [u₋ (3N₋) ; u₊ (3N₊)] — see dof_index_minus/
# dof_index_plus.
# ==============================================================================

dof_index_minus(g_minus, component, I) = (component - 1) * length(g_minus) + LinearIndices(size(g_minus))[I]
dof_index_plus(g_minus, g_plus, component, I) = 3 * length(g_minus) + (component - 1) * length(g_plus) + LinearIndices(size(g_plus))[I]

_to_sparse_matrix(M) = reduce(vcat, [reduce(hcat, [sparse(M[j][k]) for k in 1:length(M)]) for j in 1:length(M)])

# Appends block `B`'s stored entries to the (I,J,V) triplet lists, shifted so
# that B[1,1] lands at (row0+1, col0+1). Lets `SATmat` be built in one
# `sparse(I,J,V,…)` call instead of repeated `SATmat[rows,cols] .+= B`, which
# rewrites the whole CSC structure each time and made assembly quadratic
# (199× slower by n=21). See `PERFORMANCE.md` §2.2.
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

Pairs of `(I_minus, I_plus)` full-grid `CartesianIndex`es on the two grids'
fault-facing boundaries that share the same `(x2,x3)` location, matched by
coordinate value (robust to any difference in `boundary_indices` iteration
order between the two grids).

Nodes that lie on a far-field boundary as well as on the fault — the ring
where the fault plane meets the truncated domain's sides — are **excluded**:
those carry the far-field Dirichlet condition `u=0` (which the reference note
applies to all of `j=1,2,3`), so they must be neither averaged by `P` nor
given slip by `χ`. Physically consistent too: they sit outside the frictional
domain `Ω_f`, where BP8 eq. 13 gives zero slip anyway.
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

The interface SAT block of the split-node system, `Ntot × Ntot`, half-weighted
on both sides — see the sign discussion in [`split_node_system`](@ref). It is
boundary-local (nonzeros only in the rows of the few node layers the SAT
prolongation reaches), so it is cheap to assemble and small to store, which is
why the matrix-free [`SplitNodeOperator`](@ref) keeps it as a sparse matrix.
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

    # Accumulated as triplets and assembled in one `sparse` call — see
    # `_append_block!` for why the obvious `SATmat[rows, cols] .+= …` form is
    # not used. Column offset 0 addresses the `-` side's block, `D*Nm` the `+`
    # side's.
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

The projection `P` as data rather than a matrix: `mask[r]` is `0.0` on every
far-field DOF (rows `P` zeroes) and `1.0` elsewhere; `(pm[k], pp[k])` are the
`-`/`+` side DOFs of the `k`-th averaged fault pair, for all three components
(`u1` too — averaging it IS the no-opening condition `u1(0⁺)=u1(0⁻)`, BP8
eq. 3; leaving it unaveraged decouples the two fault-normal DOFs entirely).
Pairs on the fault ∩ far-field ring are excluded by `fault_node_pairs`, so a
DOF is never both masked and paired.

`P v` is then `mask .* v` with `v[pm]` and `v[pp]` both replaced by their mean —
[`apply_P!`](@ref) — and [`projection_matrix`](@ref) builds the explicit
matrix from the same data, so the two are identical by construction.
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

The explicit `P` from [`projection_parts`](@ref), assembled from triplets in
one `sparse` call. The previous construction — `sparse(I)` followed by
`P[r, :] .= 0.0` for every far-field row — rewrote the CSC structure once per
row and scaled as DOF^1.7 (52 s at 634 k DOF, 158 s at 1.2 M, ~27-37% of the
whole assembly). This is linear and gives the same matrix.
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
    # Assembled here. Applying Diffinitive's *lazy* composition instead was
    # measured 41-67× slower than `mul!` (`PERFORMANCE.md` §1); the matrix-free
    # route that does work is `SplitNodeOperator` below, which applies the same
    # `D` as Kronecker products of the 1D operators — see `MATRIX_FREE_PLAN.md`.
    # This sparsification is ~60-70% of the assembly time at production size.
    Dm = _to_sparse_matrix(elastic_blocks(g_minus, λ, μ, stencil_set))
    Dp = _to_sparse_matrix(elastic_blocks(g_plus, λ, μ, stencil_set))
    Dmat = blockdiag(Dm, Dp)

    # ---- SAT: interface traction coupling, half-weighted on both sides.
    # (The note also allows applying it fully to one side only; that
    # produced an asymmetric, non-PSD system empirically, so using the
    # symmetric half-on-both-sides construction instead.)
    #
    # SIGNS. The Neumann/traction SAT `-H⁻¹∘e'∘Hᵧ ∘ (t_out - data)` is written
    # in terms of the OUTWARD-normal traction: in Diffinitive's own
    # `sat_tensors(::NeumannCondition)` the penalty prefactor `-H⁻¹∘e'∘Hᵧ` is
    # side-independent and all of the side-dependence lives in
    # `normal_derivative`'s outward sign. `traction_blocks` is deliberately
    # fixed-`+x₁`-axis, so t_out = +T on `g_minus` (whose fault is its UPPER
    # boundary, outward = +x₁) but t_out = -T on `g_plus` (LOWER boundary,
    # outward = -x₁). The data is the other side's outward traction negated,
    # since traction balance across the interface (BP8 eq. 6) reads
    # t_out⁻ + t_out⁺ = 0. Both sides therefore penalize the same fixed-axis
    # difference (τ₋ - τ₊), but the `+` side carries an extra overall minus
    # from its outward normal. Getting this wrong leaves the shear tractions
    # discontinuous and inflates the solution by orders of magnitude.
    SATmat = sat_matrix(g_minus, g_plus, λ, μ, stencil_set)

    DSAT = Dmat + SATmat

    # ---- P: projection (average tangential fault DOFs, zero far-field) ----
    # An explicit matrix here because `A` is being assembled; a *lazy* `P` would
    # force `A` into a composite that measured 1.66× slower per mat-vec
    # (`PERFORMANCE.md` §1). The matrix-free operator instead applies `P` from
    # the same `(mask, pm, pp)` data directly — `apply_P!`, no matrix at all.
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

Builds `χ(s)`: zero everywhere except the fault, zero for `u1`, and
`±slip_fn(x2,x3)/2` for `u2,u3` on the +/- sides, per
`context/SEAS_benchmark.pdf`. `slip_fn(x2,x3) -> (s2,s3)`.
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

The true displacement field `U = P*u + χ`; tractions must be computed from
`U`, not the raw solve variable `u` (per `context/SEAS_benchmark.pdf`).
"""
reconstruct_U(P, u, χ) = P * u + χ

# ==============================================================================
# Matrix-free split-node operator.
#
# On an equidistant `TensorGrid` Diffinitive's 3D `D1`/`D2` are *exactly*
# Kronecker products of the 1D operators (measured `max|diff| = 0.0`,
# `scripts/matrix_free_prototype.jl`), so the whole Navier block operator on
# one side is six small `n×n` 1D matrices plus the 1D inner-product weights.
# `A v = -H P (D + SAT) P v` is then applied as
#
#   P            mask + pair averaging            (`apply_P!`)
#   D            18 axis passes per side           (`navier!` / `axpass!`)
#   SAT          the small boundary-local sparse   (`sat_matrix`)
#   H            Kronecker weights h1[i]h2[j]h3[k]  (`scale_H!`)
#
# Nothing of size `nnz(A)` is ever formed. At Δz = 10 m on (1600, 1600) that
# removes ~85 GB of device memory, ~73 h of host assembly and ~3× of memory
# traffic per mat-vec (~530 B/point against ~1590 B/point for the CSR SpMV).
# Results agree with the assembled `A` to round-off (3e-16 on `A v`, identical
# CG iteration counts). See `MATRIX_FREE_PLAN.md`.
#
# The Navier block, arranged to minimise passes, with `w_k = D1_k u_k`:
#
#   out_j = μ Σ_i D2_i u_j + μ D2_j u_j + D1_j[(λ+μ) div − μ w_j],   div = Σ_k w_k
#
# — the wide `λ D1_j∘D1_j` and `(λ+μ) D1_j∘D1_k` sandwiches of `elastic_blocks`
# folded into one `D1_j` pass over a combined field. Same operator, term for
# term; only the summation order differs.
#
# All fields are passed as flat vectors plus an offset (`u[off + i + (j-1)n1 +
# (k-1)n1n2]`) rather than as reshaped views, so the very same `mul!` runs on
# `Vector` and, through the CUDA extension's `axpass!`/`scale_H!` methods, on
# `CuVector`, with no array wrappers in either kernel.
# ==============================================================================

"""
    Rows1D(M::SparseMatrixCSC) -> Rows1D

A 1D SBP operator stored **by rows** (CSR: `rowptr`, `colind`, `val`), which is
the access pattern an axis pass needs: output point `r` along the axis reads
row `r`. Interior stencil and boundary closures are just different rows, so a
pass needs no special-casing. Index vectors are `Int` on the host and `Int32`
on the device (see the CUDA extension).
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

One elastic half-space of the split-node system: its grid size, the three 1D
first- and second-derivative operators and the three 1D inner-product weight
vectors. `H` on this side is `h[1][i] * h[2][j] * h[3][k]`.
"""
struct SideOps{R<:Rows1D,VV<:AbstractVector{Float64}}
    n::NTuple{3,Int}
    d1::NTuple{3,R}
    d2::NTuple{3,R}
    h::NTuple{3,VV}
end
function SideOps(g, stencil_set)
    # The TensorGrid's 1D factor grids, axis order = index order — the same field
    # Diffinitive's own `first_derivative(::TensorGrid, set, dim)` inflates from.
    gs = g.grids
    d1 = ntuple(d -> Rows1D(sparse(first_derivative(gs[d], stencil_set))), 3)
    d2 = ntuple(d -> Rows1D(sparse(second_derivative(gs[d], stencil_set))), 3)
    h = ntuple(d -> Vector{Float64}(diag(sparse(inner_product(gs[d], stencil_set)))), 3)
    return SideOps(size(g), d1, d2, h)
end
Base.length(s::SideOps) = prod(s.n)

"""
    SplitNodeOperator

`A = -H P (D + SAT) P` applied matrix-free — see the module notes above. Holds
the two sides' 1D operators, `P` as `(mask, pm, pp)` from
[`projection_parts`](@ref), the sparse SAT block, and the scratch vectors one
mat-vec needs, so `mul!` allocates nothing. Scratch makes an operator
**stateful per call**: two solves must not share one instance concurrently —
[`duplicate_operator`](@ref) gives a copy with its own scratch and everything
else shared, and `duplicate(::CGSolver)` uses it.

Build with [`split_node_operator`](@ref); use with `mul!`, [`apply_P!`](@ref),
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
`HP_DSAT` and `P` as functions rather than matrices. Takes seconds (the SAT block dominates) where
the assembly takes hours, and holds no `nnz(A)`-sized data.
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

`out[off_out + p] += α * (M applied along axis d of u[off_u + …])[p]` for the
`n = (n1,n2,n3)` field stored column-major at those offsets. The host version;
the CUDA extension adds the `CuVector` method with the same signature.
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
(`3 * length(side)` entries from `off`). Host version; the CUDA extension adds
the `CuVector` method.
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

`y = P v`: far-field DOFs zeroed, each fault pair replaced by its mean. Works
on host and device vectors alike (broadcast, gather, scatter).
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

The three matrices of [`split_node_system`](@ref) behind the same interface as
[`SplitNodeOperator`](@ref) (`mul!`, [`apply_P!`](@ref), [`hp_dsat!`](@ref)),
so `FaultElasticity` can be built either way and the two compared.
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

# ==============================================================================
# Iterative solver: CG straight onto the singular A.
# ==============================================================================

"""
    CGStats

Running totals across every solve a [`CGSolver`](@ref) has performed. Iteration
count is the thing to watch: it grows with problem size, and it is what decides
how expensive a build of `K` is.
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

Conjugate gradients on `A = -H*P*(D+SAT)*P` **directly**, with no reduction and
no factorization. Memory is a handful of vectors rather than a sparse factor,
which is the entire point: fill-in is what caps resolution (PROGRESS.md "Known
limitations" 2), and this has none.

## Why the singularity needs no special handling

`A` is singular by construction — it annihilates `P`'s null space (far-field
DOFs, and the antisymmetric half of every fault pair, together ~40% of the
system). That would normally rule out CG. It does not here, for three reasons
that hold *exactly* rather than approximately:

 1. **The system is consistent**, `b ⊥ null(A)`. Since `null(P) ⊆ null(A)`, take
    any `v ∈ null(P)`: `vᵀb = vᵀHP(D+SAT)χ = (HPv)ᵀ(D+SAT)χ = 0`, using
    `HP = PH` (measured exact) and `P = Pᵀ`. Measured directly:
    `‖(I-P)b‖/‖b‖ = 0.0`, i.e. `b ∈ range(P)` to the last bit.
 2. **The iterates never leave `range(A)`.** Started from `x₀ = 0`, every Krylov
    vector lies in `span{b, Ab, A²b, …} ⊆ range(A)`. Measured null-space content
    of the returned `u` after ~100 iterations: `0.0` exactly.
 3. **Anything that did leak would be projected away.** The only use of `u` is
    `U = P*u + χ`, and `P` annihilates `null(P)`. So null-space content is not
    merely small, it is irrelevant.

`A` restricted to `range(A)` is positive definite (`λ ∈ [0.101, 11.9]`, κ = 118
at n=9), so CG converges there at the normal rate — 71 iterations at n=9 and 108
at n=13, agreeing with a reference direct solve's tractions to 1.4e-11 (measured
when this module also carried a Cholesky/LU path; see git history).

This all depends on `A` being **symmetric**, which it only became once
`traction_blocks` was fixed; before that CG stalled at residual 3.7e-2 after
5000 iterations. See SYMMETRIC_SAT.md.

## Caveats

`itmax=0` lets Krylov pick its own default cap. A solve that hits the cap is
counted in [`solver_report`](@ref)'s `unconverged` and warned about once —
silently returning an unconverged `u` would corrupt `K` in a way nothing
downstream would notice.

## Preconditioning

`precond` selects `M⁻¹`. `:none` (default) is plain CG; `:jacobi` is
[`jacobi_preconditioner`](@ref).

The safety question the earlier version of this docstring left open is now
**measured and closed**: a diagonal `M` commutes with `P` *exactly*, not
approximately. Rows `rm` and `rp` of `P` are identical (both `0.5e_rm + 0.5e_rp`),
so rows `rm` and `rp` of `A = -HP·DSAT·P` are identical, and symmetry then forces
`A[rm,rm] = A[rp,rp]`. Measured at n1=9, n23=13: worst relative disagreement
within a merged pair is `0.0`, and `‖MP − PM‖/‖MP‖ = 0.0`.

**Jacobi does not help, and is kept only so that stays visible.** 86 iterations
against plain CG's 79 — a 0.92× *regression*. `diag(A)`'s nonzero entries span
only 13.5×, so there is almost no diagonal scaling to remove. The reconstructed
`U` is unchanged to 2.5e-11, so the option is correct; it is simply not worth
selecting. See `PERFORMANCE.md` §6.
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

`M⁻¹ = diag(A)⁻¹`, with the **zero** diagonal entries floored to 1.

The floor is not defensive coding, it is required. `P` zeroes every far-field
DOF's row, and `A = -HP·DSAT·P` inherits that, so `A` has entirely zero rows and
columns there and `diag(A)` contains *exact* zeros — measured at n1=9, n23=13:
3,318 of 9,126 entries, and that set is precisely the far-field DOF set. Naive
`1 ./ diag(A)` gives `Inf`.

The floor *value* is arbitrary: those DOFs lie in `null(P)`, so whatever the
solve puts there is annihilated by `U = P*u + χ`. It only has to keep `M`
positive definite, which CG requires. Every nonzero entry is positive (measured),
so no other entry needs guarding.
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

The default `CGSolver` keyword values, in one place so that anything needing to
know them — notably `StiffnessCache`, which has to fold the solver settings into
a cache key and cannot afford to guess — reads them from here rather than
duplicating the literals.
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

Solves `A u = rhs` by CG. Only `P*u` is meaningful — whatever CG's iterates
carry on `P`'s null space is discarded by the `U = P*u + χ` reconstruction.
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

An independent solver for use on another thread. `CGSolver`'s per-solve state
is its `CgWorkspace`, which is cheap to duplicate (a few vectors); a matrix
`A` is only ever read concurrently and is shared, while a `SplitNodeOperator`
carries scratch and is handed over via [`duplicate_operator`](@ref).
This is what makes the `K` build in `fault_stiffness` embarrassingly
parallel: its `2·N_Ωf` columns are independent right-hand sides against the
same `A`.
"""
function duplicate(s::CGSolver)
    n = size(s.A, 2)
    # `M` is SHARED, not rebuilt: rebuilding is pointless for a Diagonal and
    # would be expensive for anything with a setup phase. That is only valid
    # while `M` is stateless under application — true for `I` and `Diagonal`.
    # A preconditioner carrying internal scratch (AMG's cycle temporaries) must
    # be duplicated here instead, or the threaded build races exactly the way
    # the shared per-task buffers did (see `fault_stiffness`).
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

What the solve has cost so far: total and worst-case iteration counts,
accumulated over every solve — so a build of `K` reports the totals over all
`2·N_Ωf` right-hand sides.
"""
function solver_report(s::CGSolver)
    t = s.stats
    return (; kind=:cg, precond=s.precond, t.solves, t.iterations,
            mean_iterations=t.solves == 0 ? 0.0 : t.iterations / t.solves,
            t.max_iterations, t.unconverged)
end

end # module ElasticitySplitNode

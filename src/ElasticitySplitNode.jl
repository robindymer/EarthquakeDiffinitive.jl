module ElasticitySplitNode

using Diffinitive
using Diffinitive.Grids
using Diffinitive.SbpOperators
using Diffinitive.LazyTensors
using SparseArrays
using LinearAlgebra: I, lu, cholesky, Symmetric, PosDefException, norm
using Tokens
using StaticArrays
using ..Elasticity: elastic_blocks, traction_blocks

export split_node_system, dof_index_minus, dof_index_plus, build_chi,
       reconstruct_U, reduced_solve, fault_node_pairs, ReducedSystem,
       factorize_reduced, prolongation

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
    Tm = traction_blocks(g_minus, λ, μ, stencil_set, bid_minus)
    Tp = traction_blocks(g_plus, λ, μ, stencil_set, bid_plus)
    Prolong_p = _prolongation(g_plus, stencil_set, bid_plus)
    Prolong_m = _prolongation(g_minus, stencil_set, bid_minus)

    SATmat = spzeros(Ntot, Ntot)
    for j in 1:D
        rows_p = D * Nm .+ ((j-1)*Np+1:j*Np)
        rows_m = (j-1)*Nm+1:j*Nm
        Tp_j = reduce(hcat, Tp[j, :])   # Nb × Np
        Tm_j = reduce(hcat, Tm[j, :])   # Nb × Nm

        # - side (fault = upper boundary, outward = +x₁):  ½ Prolong₋ (τ₋ - τ₊)
        SATmat[rows_m, 1:D*Nm] .+= 0.5 .* (Prolong_m * Tm_j)
        SATmat[rows_m, D*Nm+1:Ntot] .+= -0.5 .* (Prolong_m * Tp_j)

        # + side (fault = lower boundary, outward = -x₁): -½ Prolong₊ (τ₊ - τ₋)
        SATmat[rows_p, D*Nm+1:Ntot] .+= -0.5 .* (Prolong_p * Tp_j)
        SATmat[rows_p, 1:D*Nm] .+= 0.5 .* (Prolong_p * Tm_j)
    end

    DSAT = Dmat + SATmat

    # ---- P: projection (average tangential fault DOFs, zero far-field) ----
    P = sparse(1.0I, Ntot, Ntot)

    far_field_minus = filter(!=(bid_minus), boundary_identifiers(g_minus))
    far_field_plus = filter(!=(bid_plus), boundary_identifiers(g_plus))
    for bid in far_field_minus, I in boundary_indices(g_minus, bid), comp in 1:D
        P[dof_index_minus(g_minus, comp, I), :] .= 0.0
    end
    for bid in far_field_plus, I in boundary_indices(g_plus, bid), comp in 1:D
        P[dof_index_plus(g_minus, g_plus, comp, I), :] .= 0.0
    end

    # All three components are averaged: u2,u3 so that χ supplies the slip
    # jump, and u1 because averaging it IS the no-opening condition
    # u1(0⁺)=u1(0⁻) (BP8 eq. 3). Leaving u1 unaveraged decouples the two
    # fault-normal DOFs entirely — nothing else in the system constrains
    # their difference.
    for (Im, Ip) in fault_node_pairs(g_minus, g_plus)
        for comp in 1:3
            rm = dof_index_minus(g_minus, comp, Im)
            rp = dof_index_plus(g_minus, g_plus, comp, Ip)
            for r in (rm, rp)
                P[r, :] .= 0.0
                P[r, rm] = 0.5
                P[r, rp] = 0.5
            end
        end
    end

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

"""
    ReducedSystem

A factorized, non-redundant form of `A = -H*P*(D+SAT)*P`, produced by
[`factorize_reduced`](@ref) and applied by [`reduced_solve`](@ref). Holds the
factorization of the Galerkin reduction `SᵀAS` plus the bookkeeping — `S`
itself, and the `keep`/`merge_pairs` it was derived from — needed to move
between reduced unknowns and the full DOF vector.

`λ` and `μ` are constant in this benchmark, so `A` never changes as slip
evolves — only `χ(s)` and hence the right-hand side do. Factorizing once and
reusing the factorization across every time step / RK stage is the whole
reason for going constant-coefficient.
"""
struct ReducedSystem{F}
    keep::Vector{Int}                   # DOFs retained as unknowns
    merge_pairs::Vector{Pair{Int,Int}}  # dropped DOF => the representative it equals
    S::SparseMatrixCSC{Float64,Int}     # prolongation, reduced unknowns → full DOFs
    fact::F                             # factorization of the Galerkin form SᵀAS
    Ntot::Int
end

"""
    prolongation(rs::ReducedSystem) -> S

The `Ntot × length(rs.keep)` sparse map from reduced unknowns back onto the
full DOF vector: identity on kept DOFs, plus a 1 setting each merged partner
equal to its representative. Dropped (far-field) DOFs get an all-zero row,
which is what zeroes them in [`reduced_solve`](@ref)'s output.

`S` is also the congruence transform the reduction itself is built from — see
[`factorize_reduced`](@ref). (Not to be confused with the module-private
`_prolongation(g, stencil_set, bid)`, which is the SAT penalty `-H⁻¹∘e'∘Hᵧ`.)
"""
prolongation(rs::ReducedSystem) = rs.S

"""
    factorize_reduced(A, P; method=:cholesky) -> ReducedSystem

`A = -H*P*(D+SAT)*P` is singular by construction — `P`'s null space
(far-field DOFs, and the antisymmetric half of each fault pair) is a large
fraction of the system, which makes plain Krylov solvers (GMRES) stall well
short of convergence. This reduces the system to the non-redundant DOFs
implied by `P`'s own structure (derived directly from `P`, not re-derived
from the grids) — regular DOFs kept as-is, far-field DOFs dropped, and each
averaging pair merged into a single unknown — giving a genuinely non-singular
system, which is then factorized.

The reduction is the **Galerkin** (congruence) form `SᵀAS`, with `S` the
[`prolongation`](@ref). This is load-bearing rather than cosmetic. The obvious
alternative `E·A·S` — which this function used to build — sums columns over the
merge pairs but only *selects* rows, so it is not a congruence transform and
comes out nonsymmetric even when `A` is symmetric. `SᵀAS` is the only reduction
that inherits `A`'s symmetry. The two are the same linear system up to a factor
of 2 on merged rows (`P` makes each merged pair's two rows of `A` identical), so
they have the same solution — measured agreement 2.7e-15 at n=13 — and `SᵀAS`
is simply the form that can be Cholesky-factorized.

`method` picks the factorization:

  * `:cholesky` (default) exploits the symmetry for a factor ≈1.9× fewer
    nonzeros than LU at these sizes (10.6M vs 19.9M at n=13), and fill-in is
    the binding resource — see PROGRESS.md "Known limitations" 2.
  * `:lu` is the previous behaviour, kept for comparison.

Cholesky is admissible only because `traction_blocks` pairs each of
`elastic_blocks`' two SBP schemes with its own boundary operator, which makes
`A` symmetric and `SᵀAS` SPD. It *fails* on the pre-fix operator, so a
`PosDefException` here is a genuine regression signal, not a tuning problem —
hence the fallback below warns loudly rather than degrading silently.
"""
function factorize_reduced(A, P; method=:cholesky)
    P = dropzeros(P) # explicit zeros (from `P[r,:] .= 0.0`) would otherwise
    # remain stored in the sparsity pattern, breaking the emptiness check below.
    Ntot = size(P, 1)
    keep = Int[]
    merge_pairs = Pair{Int,Int}[]
    seen = falses(Ntot)

    for i in 1:Ntot
        seen[i] && continue
        rng = nzrange(P, i)
        rows = P.rowval[rng]
        if isempty(rows)
            seen[i] = true
        elseif length(rows) == 1 && rows[1] == i
            push!(keep, i)
            seen[i] = true
        elseif length(rows) == 2
            partner = rows[1] == i ? rows[2] : rows[1]
            rep, other = max(i, partner), min(i, partner)
            seen[rep] || push!(keep, rep)
            push!(merge_pairs, other => rep)
            seen[i] = true
            seen[partner] = true
        else
            error("Unexpected P column structure at row $i: $rows")
        end
    end

    # S: identity on kept DOFs, and each merged partner reading its
    # representative's value. Zero rows for the dropped far-field DOFs.
    pos = Dict(k => c for (c, k) in enumerate(keep))
    rows = collect(keep)
    cols = collect(1:length(keep))
    for (other, rep) in merge_pairs
        push!(rows, other)
        push!(cols, pos[rep])
    end
    S = sparse(rows, cols, 1.0, Ntot, length(keep))

    return ReducedSystem(keep, merge_pairs, S, _factorize_galerkin(S' * A * S, method), Ntot)
end

function _factorize_galerkin(M, method)
    method === :lu && return lu(M)
    method === :cholesky ||
        throw(ArgumentError("unknown factorization method $method; use :cholesky or :lu"))
    try
        # `Symmetric` picks the upper triangle rather than averaging; legitimate
        # here only because the two halves agree to round-off (measured
        # ‖M-Mᵀ‖/‖M‖ ≈ 9e-17). The asymmetry is reported below if this fails.
        return cholesky(Symmetric(M))
    catch e
        e isa PosDefException || rethrow()
        @warn """
              SᵀAS is not positive definite — falling back to LU, which costs \
              ≈1.9× the factor nonzeros. This should not happen with the \
              current operators: it is the signature of `traction_blocks` and \
              `elastic_blocks` disagreeing about which boundary operator pairs \
              with which SBP scheme, which is what made the pre-fix system \
              non-SPD. See SYMMETRIC_SAT.md and \
              `test/elasticity_test.jl`'s "SBP property" test.\
              """ asymmetry = norm(M - M') / norm(M)
        return lu(M)
    end
end

"""
    reduced_solve(rs::ReducedSystem, rhs)
    reduced_solve(A, rhs, P)

Solves the reduced system for `rhs`. Dropped/far-field entries of the
returned `u` are set to `0` (their value is irrelevant — `P` discards them
regardless). The three-argument form factorizes on every call; prefer
[`factorize_reduced`](@ref) plus this two-argument form when solving
repeatedly with the same operator.

Restriction and expansion are both `S` (see [`prolongation`](@ref)): the
Galerkin system is `SᵀAS w = Sᵀrhs`, and `u = S w` both scatters `w` back to
the kept DOFs and copies each representative onto its merged partner.
"""
reduced_solve(rs::ReducedSystem, rhs) = rs.S * (rs.fact \ (rs.S' * rhs))

reduced_solve(A, rhs, P; kwargs...) = reduced_solve(factorize_reduced(A, P; kwargs...), rhs)

end # module ElasticitySplitNode

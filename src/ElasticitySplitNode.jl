module ElasticitySplitNode

using Diffinitive
using Diffinitive.Grids
using Diffinitive.SbpOperators
using Diffinitive.LazyTensors
using SparseArrays
using LinearAlgebra: I, Diagonal, diag
using Krylov: CgWorkspace, cg!
using Tokens
using StaticArrays
using ..Elasticity: elastic_blocks, traction_blocks

export split_node_system, dof_index_minus, dof_index_plus, build_chi,
       reconstruct_U, fault_node_pairs,
       CGSolver, split_node_solve, jacobi_preconditioner,
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
    # Assembled rather than applied matrix-free, which is the measured choice,
    # not an oversight: lazy composition re-expands the stencil at every point
    # and benchmarks 41-67× slower than `mul!`. See `PERFORMANCE.md` §1.
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

    # Accumulated as triplets and assembled in one `sparse` call — see
    # `_append_block!` for why the obvious `SATmat[rows, cols] .+= …` form is
    # not used. Column offset 0 addresses the `-` side's block, `D*Nm` the `+`
    # side's, matching the `1:D*Nm` / `D*Nm+1:Ntot` ranges this replaces.
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
    SATmat = sparse(Isat, Jsat, Vsat, Ntot, Ntot)

    DSAT = Dmat + SATmat

    # ---- P: projection (average tangential fault DOFs, zero far-field) ----
    # An explicit matrix deliberately: `P` is a near-identity applied once per
    # solve, and making it lazy would force `A` into a composite that measured
    # 1.66× slower per mat-vec. See `PERFORMANCE.md` §1.
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

    # `P[r,:] .= 0.0` on a CSC zeroes the value but KEEPS the structural entry,
    # and those stored zeros multiply into full rows of `DSAT`, inflating
    # `HP_DSAT` and `A` by 29-52%. Same matrix either way — see `PERFORMANCE.md`
    # §2.1.
    dropzeros!(P)

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

function CGSolver(A; rtol=1e-10, atol=0.0, itmax=0, precond::Symbol=:none)
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

An independent solver sharing the same operator `A`, for use on another
thread. `CGSolver`'s per-solve state is its `CgWorkspace`, which is cheap to
duplicate (a few vectors) while `A` itself is only ever read concurrently.
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
    return CGSolver(s.A, s.M, s.ldiv, s.precond, CgWorkspace(n, n, Vector{Float64}),
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

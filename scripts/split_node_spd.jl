# Is the split-node system symmetric positive definite, and can CG solve it?
#
# `context/SEAS_benchmark.pdf` asserts `A = -H P (D+SAT) P` is SPD and on that
# basis suggests CG. `reduced_solve` does a direct LU instead, and the fill-in
# of that factorization is what caps resolution at Δz ≈ 50 m (PROGRESS.md
# "Known limitations" 2) — so whether CG is admissible decides whether this
# approach can reach the benchmark's Δz = 10 m.
#
# WHICH MATRIX. `A` itself is singular by construction: `P`'s null space
# (far-field DOFs, and the antisymmetric half of every fault pair) is
# annihilated, so definiteness cannot even be asked of it. The question has to
# be asked of a reduction to the non-redundant DOFs, and there are TWO, which
# is the subtlety this script exists to expose. With `S` the prolongation from
# reduced unknowns to full DOFs (identity on kept DOFs, and a 1 in the
# representative's column for each merged partner):
#
#   PETROV   `E·A·S` — columns summed over merge pairs, rows merely SELECTED.
#            This is what `factorize_reduced` builds and LU-factorizes today.
#            It is a valid square solve, but it is NOT a congruence transform,
#            so it is nonsymmetric even when `A` is — CG cannot use it, and
#            feeding it to CG silently converges to a different vector.
#
#   GALERKIN `Sᵀ·A·S` — rows AND columns summed. The actual congruence
#            transform, hence the only reduction that can inherit symmetry
#            from `A`. Not currently built anywhere in the package; this is
#            the candidate CG would have to run on.
#
# So the script reports symmetry for both, and runs CG on the Galerkin form
# while cross-checking the result against the production LU path. The
# cross-check is the load-bearing test: no theorem covers CG on a
# nonsymmetric matrix, so agreement with the direct solve is what stands in
# for one. Hand-rolled CG — 25 lines, versus a new dependency.
#
# Run with:  julia --project=. scripts/split_node_spd.jl [n ...]
using EarthquakeDiffinitive
using EarthquakeDiffinitive.ElasticitySplitNode
using Diffinitive.Grids
using Diffinitive.SbpOperators
using LinearAlgebra, SparseArrays, Printf

const λ_ = 2.0    # ν = 1/4, as in BP8 (only the ratio matters for symmetry)
const μ_ = 1.0

asymmetry(M) = norm(M - M') / norm(M)

# Dense eigendecomposition is O(N³) and dominates above a few thousand reduced
# DOFs. Symmetry and CG are cheap, so past this size report only those — the
# spectrum is a fixed structural property, while CG's iteration count is the
# part that has to be measured as n grows.
const EIG_MAX = 5000

# The right-hand side must be a FIXED PHYSICAL field, not an index pattern:
# CG's iteration count depends on the RHS, so anything whose shape changes with
# `n` (a stride like `b[1:7:end]`) contaminates the scaling measurement with
# RHS variation. A Gaussian slip patch of fixed physical width is the same
# field at every resolution, and is what the solver actually sees in
# production — `fault_stiffness` feeds it exactly this via `HP_DSAT*χ(s)`.
const SLIP_WIDTH = 0.25
gaussian_slip(x2, x3) = (exp(-(x2^2 + x3^2) / (2 * SLIP_WIDTH^2)), 0.0)

"""
    cg_solve(A, b; tol, maxiter) -> (x, iters, recres, trueres, breakdown)

Textbook CG, no preconditioner. Returns after `maxiter` regardless of
convergence, so a stall is visible in the residual rather than thrown.

Reports the recursively-updated residual AND `‖b-Ax‖/‖b‖` recomputed from
scratch. **They only agree if A is symmetric** — the recursion `r ← r - αAp`
is derived from that assumption, so on a nonsymmetric matrix CG can report
convergence it has not achieved. Comparing the two is the point.
"""
function cg_solve(A, b; tol=1e-10, maxiter=5000)
    x = zeros(length(b))
    r = copy(b)
    p = copy(r)
    rr = dot(r, r)
    nb = norm(b)
    breakdown = false
    it = 0
    for k in 1:maxiter
        it = k
        Ap = A * p
        pAp = dot(p, Ap)
        if pAp <= 0                  # pᵀAp ≤ 0: A is not positive definite
            breakdown = true
            break
        end
        α = rr / pAp
        @. x += α * p
        @. r -= α * Ap
        rr_new = dot(r, r)
        sqrt(rr_new) / nb < tol && (rr = rr_new; break)
        @. p = r + (rr_new / rr) * p
        rr = rr_new
    end
    return x, it, sqrt(rr) / nb, norm(b - A * x) / nb, breakdown
end

"""
    prolongation(rs::ReducedSystem) -> S

The `Ntot × length(keep)` sparse map from reduced unknowns back to full DOFs:
identity on kept DOFs, plus a 1 putting each merged partner's value equal to
its representative's. Dropped (far-field) DOFs get an all-zero row.

`reduced_solve`'s expansion step *is* multiplication by this `S`, so
`E·A·S = (A*S)[keep,:]` is exactly the matrix `factorize_reduced` factorizes.
"""
function prolongation(rs::ReducedSystem)
    pos = Dict(k => c for (c, k) in enumerate(rs.keep))
    rows = collect(rs.keep)
    cols = collect(1:length(rs.keep))
    for (other, rep) in rs.merge_pairs
        push!(rows, other)
        push!(cols, pos[rep])
    end
    return sparse(rows, cols, 1.0, rs.Ntot, length(rs.keep))
end

"""
    bulk_only(A, P)

`A` restricted to DOFs that are neither far-field nor on the fault. The
interface SAT touches only the fault rows/columns, so if the symmetry gap is
in the SAT this submatrix is symmetric to round-off while the full system is
not — which localizes the defect without any continuum derivation.
"""
function bulk_only(A, P)
    P = dropzeros(P)
    interior = [i for i in 1:size(P, 2) if length(nzrange(P, i)) == 1]
    return A[interior, interior]
end

function report(n; order=4)
    set = read_stencil_set(SbpOperators.sbp_operators_path() * "standard_diagonal.toml"; order)
    g_minus = equidistant_grid((-1.0, -1.0, -1.0), (0.0, 1.0, 1.0), n, n, n)
    g_plus = equidistant_grid((0.0, -1.0, -1.0), (1.0, 1.0, 1.0), n, n, n)

    A, HP_DSAT, P = split_node_system(g_minus, g_plus, λ_, μ_, set)
    t_lu = @elapsed rs = factorize_reduced(A, P)
    S = prolongation(rs)
    A_pet = (A*S)[rs.keep, :]      # what factorize_reduced actually factorizes
    A_gal = S' * A * S             # the congruence transform; CG's candidate
    nred = size(A_gal, 1)

    # --- P itself: the projection the whole construction rests on ---
    P_sym = asymmetry(P)
    P_idem = norm(P * P - P) / norm(P)

    # --- symmetry, on all three operators ---
    a_full = asymmetry(A)
    a_bulk = asymmetry(bulk_only(A, P))
    a_pet = asymmetry(A_pet)
    a_gal = asymmetry(A_gal)

    # --- definiteness of the Galerkin form (skipped above EIG_MAX) ---
    do_eig = nred <= EIG_MAX
    n_neg_sym, n_neg_re, λ_min, λ_max, κ = -1, -1, NaN, NaN, NaN
    if do_eig
        # pᵀAp = pᵀ½(A+Aᵀ)p, so the symmetric part's sign pattern is what
        # CG's positivity test actually sees.
        ev_s = eigvals(Symmetric(Matrix(0.5 * (A_gal + A_gal'))))
        scale = maximum(abs, ev_s)
        n_neg_sym = count(<(-1e-12 * scale), ev_s)
        λ_min, λ_max = minimum(ev_s), maximum(ev_s)
        κ = scale / minimum(abs, ev_s)
        n_neg_re = count(λ -> real(λ) < -1e-12 * scale, eigvals(Matrix(A_gal)))
    end

    # --- CG on the Galerkin form vs the production LU, same physical RHS ---
    χ = build_chi(g_minus, g_plus, gaussian_slip)
    rhs = HP_DSAT * χ
    t_bs = @elapsed x_lu = rs.fact \ rhs[rs.keep]     # one back-substitution
    t_cg = @elapsed (x_cg, iters, recres, trueres, breakdown) =
        cg_solve(A_gal, S' * rhs)
    # Both live in the reduced basis, so compare directly. This is the real
    # correctness evidence for CG here.
    agree = norm(x_cg - x_lu) / norm(x_lu)
    # Control: solve the SAME Galerkin system DIRECTLY. `P` makes each merged
    # pair's two rows identical, so `Sᵀ` (summing them) and `E` (picking one)
    # differ only by a factor of 2 on those rows — the two reductions are the
    # same linear system up to row scaling, and must have the same solution.
    # If this agrees with LU but CG does not, the failure is CG's alone and
    # not a defect in either reduction.
    agree_direct = norm((Matrix(A_gal) \ (S' * rhs)) - x_lu) / norm(x_lu)

    @printf("\n%s  n = %d  (%d DOF total, %d reduced)\n", "="^58, n, size(A, 1), nred)
    @printf("  P:  ‖P-Pᵀ‖/‖P‖ = %.2e     ‖P²-P‖/‖P‖ = %.2e\n", P_sym, P_idem)
    @printf("  asymmetry ‖M-Mᵀ‖/‖M‖:  A %.3f   bulk(A) %.2e   E·A·S %.3f   SᵀAS %.3f\n",
            a_full, a_bulk, a_pet, a_gal)
    if do_eig
        @printf("  SᵀAS spectrum:  %d/%d negative in ½(M+Mᵀ),  %d with Re λ < 0,  λ ∈ [%.2e, %.2e],  κ = %.2e\n",
                n_neg_sym, nred, n_neg_re, λ_min, λ_max, κ)
    else
        @printf("  SᵀAS spectrum: skipped (%d > EIG_MAX = %d)\n", nred, EIG_MAX)
    end
    @printf("  CG on SᵀAS, %d iterations:  recursive %.2e,  TRUE ‖b-Ax‖/‖b‖ = %.2e%s\n",
            iters, recres, trueres, breakdown ? "   [pᵀAp ≤ 0 BREAKDOWN]" : "")
    @printf("  CG vs LU:  ‖x_cg - x_lu‖/‖x_lu‖ = %.2e     (direct SᵀAS vs LU: %.2e)\n",
            agree, agree_direct)
    # `fault_stiffness` needs 2·N_Ωf right-hand sides, so the per-RHS cost
    # after the one-off setup is what matters, not the setup itself.
    @printf("  timing:  LU %.2f s once + %.4f s/RHS   vs   CG %.3f s/RHS   (break-even %.0f RHS)\n",
            t_lu, t_bs, t_cg, t_lu / max(t_cg - t_bs, eps()))
    return (; n, dofs=size(A, 1), reduced=nred, a_full, a_bulk, a_pet, a_gal,
            n_neg_sym, n_neg_re, λ_min, λ_max, κ,
            cg_iters=iters, recres, trueres, breakdown, agree, agree_direct,
            t_lu, t_bs, t_cg)
end

ns = isempty(ARGS) ? [9, 13] : parse.(Int, ARGS)
results = [report(n) for n in ns]

println("\n", "="^108)
println("SUMMARY")
println("="^108)
@printf("%4s %8s %8s %8s %10s %8s %8s %7s %10s %10s %8s %8s\n",
        "n", "DOF", "reduced", "asym(A)", "asym bulk", "E·A·S", "SᵀAS",
        "CG its", "CG res", "cg vs lu", "LU s", "CG s")
for r in results
    @printf("%4d %8d %8d %8.3f %10.1e %8.3f %8.1e %7d %10.1e %10.1e %8.2f %8.3f\n",
            r.n, r.dofs, r.reduced, r.a_full, r.a_bulk, r.a_pet, r.a_gal,
            r.cg_iters, r.trueres, r.agree, r.t_lu, r.t_cg)
end
println("""
Reading this:
  `asym(A)` ≫ 0 with `asym bulk` ≈ round-off localizes the gap to the interface
  SAT, not to `elastic_blocks`. Whether it DECAYS with n says whether the
  missing `T'·displacement-jump` term is a consistency-order effect or a
  structural error.
  `E·A·S` is what `factorize_reduced` factorizes today; it is nonsymmetric BY
  CONSTRUCTION (rows selected, columns summed) regardless of `A`, so it is not
  a CG candidate. `SᵀAS` is the congruence transform, and inherits whatever
  symmetry `A` has — compare the two columns to see the difference.
  `CG res` is recomputed from scratch; CG's own recursive residual is only
  valid for A = Aᵀ.
  `cg vs lu` is decisive: at round-off, CG agrees with the production direct
  solve. Anything larger means CG converged to a different vector, and its
  small residual is meaningless.
  `LU s` vs `CG s`: CG's advantage is MEMORY, not time. The LU is paid once and
  then serves 2·N_Ωf back-substitutions nearly free — exactly the access
  pattern `fault_stiffness` has.""")

# Is the split-node system symmetric positive definite, and can CG solve it?
#
# `context/SEAS_benchmark.pdf` asserts `A = -H P (D+SAT) P` is SPD and on that
# basis suggests CG. `reduced_solve` does a direct sparse factorization instead,
# and its fill-in is what caps resolution at Δz ≈ 50 m (PROGRESS.md "Known
# limitations" 2) — so whether CG is admissible decides whether this approach
# can reach the benchmark's Δz = 10 m.
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
#            A valid square solve, but NOT a congruence transform, so it is
#            nonsymmetric even when `A` is — CG cannot use it, and feeding it
#            to CG silently converges to a different vector.
#
#   GALERKIN `Sᵀ·A·S` — rows AND columns summed. The actual congruence
#            transform, hence the only reduction that can inherit symmetry
#            from `A`.
#
# `factorize_reduced` built the PETROV form when this script was written, which
# is why the two are still reported side by side; it now builds GALERKIN and
# Cholesky-factorizes it, both of which this script's measurements are what
# motivated. The Petrov column is kept because it is the reason a symmetric `A`
# alone would NOT have been enough — two changes were needed, not one.
#
# So the script reports symmetry for both, and runs CG on the Galerkin form
# while cross-checking the result against the production direct path. The
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

# Measuring how much smaller the Cholesky factor is than an LU one costs a
# second factorization, and LU is the bigger of the two — at n = 21 that is
# ~2.4 GB on its own. Above this many reduced DOFs, report Cholesky's fill-in
# alone rather than risking an OOM kill in a diagnostic script.
const LU_COMPARE_MAX = 25_000

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
    t_fact = @elapsed rs = factorize_reduced(A, P)   # Galerkin + Cholesky
    S = prolongation(rs)
    A_pet = (A*S)[rs.keep, :]      # the Petrov form, for the symmetry contrast
    A_gal = S' * A * S             # what factorize_reduced factorizes; CG's candidate
    nred = size(A_gal, 1)

    # --- fill-in: the resource that actually caps resolution ---
    # Cholesky's factor is free to measure (we already have it); LU's costs a
    # second factorization, so it is skipped once that would double an already
    # multi-GB peak.
    nnz_chol = rs.fact isa SparseArrays.CHOLMOD.Factor ? nnz(sparse(rs.fact.L)) : -1
    nnz_lu = nred <= LU_COMPARE_MAX ?
             (F = factorize_reduced(A, P; method=:lu).fact; nnz(F.L) + nnz(F.U)) : -1

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

    # --- CG on the Galerkin form vs the production direct solve, same RHS ---
    χ = build_chi(g_minus, g_plus, gaussian_slip)
    rhs = HP_DSAT * χ
    b = S' * rhs                                     # the Galerkin right-hand side
    t_bs = @elapsed x_dir = rs.fact \ b              # one back-substitution
    t_cg = @elapsed (x_cg, iters, recres, trueres, breakdown) = cg_solve(A_gal, b)
    # Both live in the reduced basis, so compare directly. This is the real
    # correctness evidence for CG here.
    agree = norm(x_cg - x_dir) / norm(x_dir)
    # Control: solve the SAME system with a DENSE factorization. If this agrees
    # with the sparse direct solve but CG does not, the failure is CG's alone
    # and not a defect in the reduction.
    agree_direct = norm((Matrix(A_gal) \ b) - x_dir) / norm(x_dir)

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
    @printf("  CG vs direct:  ‖x_cg - x_dir‖/‖x_dir‖ = %.2e     (dense SᵀAS vs sparse: %.2e)\n",
            agree, agree_direct)
    @printf("  factor nonzeros:  Cholesky %s   LU %s   ratio %s\n",
            nnz_chol < 0 ? "n/a" : string(nnz_chol),
            nnz_lu < 0 ? @sprintf("skipped (%d > LU_COMPARE_MAX = %d)", nred, LU_COMPARE_MAX) : string(nnz_lu),
            (nnz_chol > 0 && nnz_lu > 0) ? @sprintf("%.2fx", nnz_lu / nnz_chol) : "-")
    # `fault_stiffness` needs 2·N_Ωf right-hand sides, so the per-RHS cost
    # after the one-off setup is what matters, not the setup itself.
    @printf("  timing:  factorize %.2f s once + %.4f s/RHS   vs   CG %.3f s/RHS   (break-even %.0f RHS)\n",
            t_fact, t_bs, t_cg, t_fact / max(t_cg - t_bs, eps()))
    return (; n, dofs=size(A, 1), reduced=nred, a_full, a_bulk, a_pet, a_gal,
            n_neg_sym, n_neg_re, λ_min, λ_max, κ, nnz_chol, nnz_lu,
            cg_iters=iters, recres, trueres, breakdown, agree, agree_direct,
            t_fact, t_bs, t_cg)
end

ns = isempty(ARGS) ? [9, 13] : parse.(Int, ARGS)
results = [report(n) for n in ns]

println("\n", "="^108)
println("SUMMARY")
println("="^108)
@printf("%4s %8s %8s %8s %10s %8s %8s %7s %10s %10s %9s %8s %8s\n",
        "n", "DOF", "reduced", "asym(A)", "asym bulk", "E·A·S", "SᵀAS",
        "CG its", "CG res", "cg vs dir", "LU/chol", "fact s", "CG s")
for r in results
    @printf("%4d %8d %8d %8.3f %10.1e %8.3f %8.1e %7d %10.1e %10.1e %9s %8.2f %8.3f\n",
            r.n, r.dofs, r.reduced, r.a_full, r.a_bulk, r.a_pet, r.a_gal,
            r.cg_iters, r.trueres, r.agree,
            (r.nnz_chol > 0 && r.nnz_lu > 0) ? @sprintf("%.2fx", r.nnz_lu / r.nnz_chol) : "-",
            r.t_fact, r.t_cg)
end
println("""
Reading this:
  `asym(A)` ≫ 0 with `asym bulk` ≈ round-off localizes the gap to the interface
  SAT, not to `elastic_blocks`. Whether it DECAYS with n says whether the
  missing `T'·displacement-jump` term is a consistency-order effect or a
  structural error. Post-fix both columns should be at round-off.
  `E·A·S` is the reduction `factorize_reduced` USED to build; it is nonsymmetric
  BY CONSTRUCTION (rows selected, columns summed) regardless of `A`, so it was
  never a CG candidate and could not be Cholesky-factorized either. `SᵀAS` is
  the congruence transform it builds now, and inherits whatever symmetry `A`
  has — compare the two columns to see why both changes were needed.
  `CG res` is recomputed from scratch; CG's own recursive residual is only
  valid for A = Aᵀ.
  `cg vs dir` is decisive: at round-off, CG agrees with the production direct
  solve. Anything larger means CG converged to a different vector, and its
  small residual is meaningless.
  `LU/chol` is how many times bigger an LU factor would be than the Cholesky
  factor now used — the memory saved by exploiting symmetry. Fill-in scales
  ≈ n^5.6 here, so this buys grid spacing only as its 1/5.6 power.
  `fact s` vs `CG s`: per right-hand side, CG is slower — the factorization is
  paid once and then serves 2·N_Ωf back-substitutions nearly free, which is
  exactly the access pattern `fault_stiffness` has. CG's advantages are MEMORY
  (no factor at all: 0.38 GB vs 2.45 GB of solver footprint at production size)
  and PARALLELISM (`fault_stiffness`'s columns are independent and thread,
  measured 5.1x on 8 threads; a sparse factorization cannot be shared across
  threads at all). Both are chosen with `solver=:cg` — see
  `ElasticitySplitNode.CGSolver`, which runs CG on the singular `A` directly.""")

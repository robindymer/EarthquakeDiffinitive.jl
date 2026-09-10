# Performance: what costs what, and why the operators are assembled

Reference material for the cost side of the elastic solve, in the same spirit
as `SYMMETRIC_SAT.md`. Read it when a code comment cites it, or when deciding
what to optimize next. Everything here is measured, not estimated, unless a
line says otherwise.

Measurements were taken on a 12-core workstation, Julia 1.12, SBP order 4,
against `Diffinitive` at the pinned rev in `Project.toml`.

## 1. Why the operators are assembled rather than applied matrix-free

Diffinitive is a matrix-free library, so assembling its operators into
`SparseMatrixCSC` looks like it works against the design. It does not — it is
the intended trade, and the margin is large.

`Diffinitive`'s operators **compose**: `isotropic_lambda_mu` builds terms like
`D1[i] ∘ D1[j]`. Applying a composition at a point re-evaluates the inner
derivative across the whole outer stencil — roughly 25 evaluations at order 4
where the collapsed product stencil is ~9 points. `sparse()` does that
collapsing **once, at assembly**; the lazy form redoes it at **every point of
every application**. CG then performs ~10⁵ applications per `K` build.

Measured on `Dmat`, both paths agreeing numerically, both non-allocating:

| n | DOF | matrix-free | sparse `mul!` | ratio |
|---|---|---|---|---|
| 11 | 3,993 | 7.29 ms | 0.115 ms | 64× slower |
| 15 | 10,125 | 17.5 ms | 0.261 ms | 67× slower |
| 21 | 27,783 | 47.2 ms | 0.957 ms | 49× slower |
| 27 | 59,049 | 99.7 ms | 2.44 ms | 41× slower |

The gap narrows with `n` but never approaches parity, so no amount of grid
growth rescues it.

Two corollaries worth recording, because both are tempting and both are wrong:

- **A matrix-free `P` is not worth it.** `P` is a near-identity (15,210
  nonzeros at n=15, 2.7% of `A`) and is applied once per *solve*, not once per
  CG *iteration*. Worse, `A = -HP_DSAT*P` is what `cg!` consumes, so a lazy `P`
  forces `A` into a composite; applying `P` then `HP_DSAT` separately measured
  **1.66× slower** per mat-vec than the assembled `A` (0.50 → 0.84 ms at n=15).
- **"Matrix-free saves memory bandwidth" does not apply here.** It would, if
  matrix-free meant doing *less* arithmetic per point. With lazy composition it
  does far more, and bandwidth never becomes the limit.

Genuine matrix-free here would mean hand-writing the collapsed elastic stencil
as an explicit kernel, and doing the same for the interface SAT. That is a
different project from using the library's composition, and it is the only
route that would change this conclusion.

`Krylov.cg!` also needs `mul!`/`size`/`eltype`, none of which `LazyTensor`
defines — so even the composite route needs a hand-written adapter first.

## 2. Two fixes applied

Both were sparse-matrix mutation antipatterns, both are pure performance work,
and both were verified to leave the operators **numerically identical** (`==`,
not a tolerance) with the suite green at 179/179.

### 2.1 Stored zeros in `P`

`P[r, :] .= 0.0` on a `SparseMatrixCSC` zeroes the value but **keeps the
structural entry**. The construction therefore left one stored zero per
far-field row. Those are not inert: a stored zero at `P[r,r]` multiplies into a
full stencil-width row of `DSAT`, so they propagate into `HP_DSAT` and `A` and
are then carried through every CG mat-vec.

| | nnz(`HP_DSAT`) | nnz(`A`) |
|---|---|---|
| before | 345,878 (n=11) / 912,190 (n=15) | 356,756 / 934,188 |
| after `dropzeros!` | 214,202 / 648,386 | 172,898 / 563,882 |
| saving | 38% / 29% | **52% / 40%** |

The share falls with `n`, since it tracks the far-field boundary fraction.

### 2.2 Quadratic SAT assembly

`SATmat[rows, cols] .+= B` rewrites the whole CSC structure each time. Done 12
times over a matrix that grows as it goes, assembly scaled **quadratically** —
by n=21 the SAT loop was 64% of all assembly time. Replaced with triplet
accumulation and a single `sparse(I,J,V,…)` call (`sparse` sums duplicate
`(i,j)` pairs, which is exactly what `.+=` was doing).

| n | DOF | before | after | speedup |
|---|---|---|---|---|
| 11 | 7,986 | 0.85 s | 0.050 s | 17× |
| 15 | 20,250 | 5.09 s | 0.092 s | 55× |
| 21 | 55,566 | 36.9 s | 0.185 s | **199×** |
| 25 | 93,750 | 105.6 s | 0.650 s | 162× |

Growth is now roughly linear.

## 3. Cost model

Measured on grids with the benchmark's real shape (`n1 × n23 × n23`,
`L_fault = 3·l_f`, `L_normal = 2·l_f`), not cubes:

| Δz | grid/side | DOF | `K` cols | nnz(`A`)/row | one CG solve | CG iters |
|---|---|---|---|---|---|---|
| 100 m | 9×25×25 | 33,750 | 162 | 33.8 | 0.131 s | 128 |
| 80 m | 11×31×31 | 63,426 | 242 | 36.4 | 0.364 s | 155 |
| 50 m | 17×49×49 | 244,902 | 578 | 40.8 | 2.946 s | 236 |

Fitted from those three points:

- **`t_solve ∝ DOF^1.56`** — from nnz(`A`) ∝ DOF and iterations ∝ DOF^0.31
- **iterations ∝ DOF^0.31 ≈ O(1/Δz)**, the usual unpreconditioned CG behaviour
- **`K` columns = 2·N_Ωf ∝ Δz⁻²**
- therefore **total `K` build ∝ Δz⁻⁶·⁷**

That exponent is the whole story: halving Δz costs ~100× more.

## 4. Tractability

**Sizing requires the domain, and the domain is much larger than the code's
defaults.** The convergence study (`PROGRESS.md` "Results") measured what the
truncation boundaries actually have to be: **`L_fault` ≥ 4·`l_f` = 1600 m and
`L_normal` ≥ 3·`l_f` = 1200 m** for ~1% in `V_max`. `build_model` defaults to
3·`l_f` and 2·`l_f`; production shipped 2·`l_f` and 1·`l_f`. Every earlier cost
estimate in this file used the defaults and was therefore **~4.5× optimistic in
compute and ~2.6× in memory**. The table below uses the measured domain.

Baseline is measured, not fitted: at Δz = 50 m with that domain (633,750 DOF)
a full `FaultElasticity` + 578-column `K` build takes **4058.6 s on 12 threads**,
i.e. 7.02 s per column. Wall-clock per column scales as **DOF^1.42** across the
three measured domain rows.

| Δz | grid/side | DOF | `A`+`HP_DSAT` | `K` cols | 12-core wall | cells per `L_b` |
|---|---|---|---|---|---|---|
| 50 m | 25×65×65 | 634 k | ~0.5 GB | 578 | **1.1 h** (measured) | 1.3 ✗ |
| 25 m | 49×129×129 | 4.9 M | ~7 GB | 2,178 | ~77 h | 2.6 ✗ |
| **20 m** | 61×161×161 | **9.5 M** | **~15 GB** | 3,362 | **~306 h** | **3.2 ✓** |
| 10 m (spec) | 121×321×321 | **74.8 M** | **~116 GB** | 13,122 | **~934 days** | 6.4 |

`L_b ≈ 64 m`; `resolution_report` calls Δz converged at `L_b/Δz ≥ 3`, i.e.
**Δz ≤ 21 m**, so Δz = 20 m is the coarsest converged grid and the nominal 10 m
is 6.7× more expensive than convergence requires.

This supersedes `PROGRESS.md` "Known limitations" 2, whose Δz = 50 m ceiling was
a property of the removed direct solver's fill-in.

**On a workstation** (15 GB): only Δz = 50 m fits, and that is not converged.
Δz = 20 m now needs ~15 GB. The workstation is out of the production picture.

**On a cluster**: Δz = 20 m is one high-memory node for ~2 weeks, or a handful
of nodes for days. Δz = 10 m needs ~120 GB **per node** (each node holds its own
copy of `A`) and ~934 node-days of 12-core work — so ~100 nodes for ~10 days,
or ~500 for ~2 days.

**These `:exact` node-day figures predate §5 item 0b (D4 symmetry) and are now
~7.8× pessimistic.** With `fault_stiffness_d4_shard` (§5 item 0b) the Δz = 10 m
`:exact` build is ~120 node-days, not ~934 — so ~12-15 nodes for ~10 days, or
~60-75 for ~2 days, at the same per-node memory. The `K` columns figures in
the table above are unchanged (they are `2·N_Ωf`, the count `:exact` would
need *without* D4); §5 item 0b's own table gives the D4-reduced solve counts
per Δz.

**Do not convert these to core-hours.** Threading efficiency is ~15% (7.02 s per
column on 12 cores against ~13 s serial-equivalent), because the sparse mat-vec
is memory-bandwidth bound. Cores within a node are nearly free of benefit past a
point; independent **nodes**, each with its own memory bandwidth, are what
scales. That is exactly why §5 item 1 is the blocker.

## 4b. `K` is block-Toeplitz, and that changes the scaling

**Measured, not conjectured** (`scripts/k_toeplitz_structure.jl`,
`scripts/k_toeplitz_validate.jl`). In a homogeneous whole-space the
slip→traction kernel is translation invariant, so `K[i,j]` should depend only on
the separation `x_i − x_j`. It very nearly does, even at the *small* production
domain where truncation is worst:

| block | best BTTB approximation error |
|---|---|
| K22, K33 | **0.010%** |
| K23 | 0.353% |

So a few source columns determine almost the whole matrix. Measured end-to-end —
running the full coupled benchmark with each `K` and comparing to the 578-solve
build — across all four combinations of domain and duration. The figure is the
**worst** relative error in `V_max(t)` over the run, which is the quantity the
benchmark actually asks for (§4.2):

| sources | solves | 100 h small | 100 h converged | 30 d small | **30 d converged** |
|---|---|---|---|---|---|
| centre only | 2 | 0.030% | 0.004% | 138% | **96.9%** |
| **centre + 4 corners (priority)** | **10** | 0.005% | 0.005% | 5.8% | **0.41%** |
| centre + 3×3 ring (priority) | 18 | 0.004% | — | 5.4% | 0.49% |
| 4 corners, *averaged* | 8 | 16.5% | 6.4% | 162% | 213% |
| 6×6 grid, *averaged* | 72 | 14.1% | 0.69% | 68.6% | 9.9% |

**Read the last column.** It is the only one describing the configuration that
will actually be run — converged domain, full 30 days — and it is what the
shipped default is chosen against. Centre-only looks superb in three of the four
regimes and is catastrophic in the one that matters.

Two effects are visible. Accuracy **improves at the converged domain** (less
truncation, better translation invariance), which also confirms the
contamination mechanism: the averaged-corner penalty falls from 16.5% to 6.4% at
100 h once corners sit further from the boundary. And accuracy **degrades over
30 days**, because injection stops at `t_off` = 100 h, `V_max` drops an order of
magnitude, and the relaxation phase is far more sensitive to the long-range
kernel than the driven phase is.

**Three results worth not forgetting, each of which cost a wrong turn.**

1. **Averaging sources is catastrophic; prioritising them is not.** The 6×6 grid
   has the *best* matrix-norm error (0.0005%) and 24× the physics error of the
   10-solve priority build. Corner and edge sources sit against the truncation
   boundary and the locked ring, so their kernels are contaminated; averaging
   lets that into the near field. Consulting them in order — centre first, and
   only for separations it cannot reach — keeps the clean near field and still
   closes the coverage gap.
2. **Matrix error is amplified ~2-3 orders of magnitude into `V_max`**, because
   `V ~ exp(τ/aσ̄)`. Slip and moment rate are not amplified (slip error stays
   ≤0.02% everywhere above). Frobenius norm is therefore the wrong acceptance
   metric, and it can point in the *opposite* direction to the truth — only an
   end-to-end run settles it.
3. **Validating through the injection phase alone is not validation.** At 100 h
   the centre source's ~44% coverage gap looks irrelevant (0.004%); over 30 days
   the same gap gives 97%. After shut-in, `V_max` falls an order of magnitude
   and the long-range kernel starts to matter. Any future change here must be
   re-checked at 30 days, not 100 h.

**Consequence.** The `K` build is ~98% of the run cost and scales as
`2·N_Ωf ∝ Δz⁻²` — 13,122 solves at Δz = 10 m. A Toeplitz build makes it **2**,
independent of resolution. That removes the dominant term from §4 entirely: the
Δz = 10 m estimate of ~934 node-days becomes a handful of hours plus assembly,
on one node. It also makes §5 item 1 (multi-node) largely moot, since the
parallel work it was meant to distribute no longer exists.

### Resolution transferability: settled, and it improves

The last open gate. `scripts/k_toeplitz_resolution.jl`, **matched pair at fixed
small domain**, full 30 days — holding the domain fixed so resolution is the only
variable, because the converged domain at Δz = 25 m is 4.9 M DOF and ~77 h:

| Δz | Ω_f nodes | exact solves | `K` Frobenius err | **`V_max(t)` worst** |
|---|---|---|---|---|
| 50 m | 17×17 | 578 | 0.0155% | **5.7767%** |
| **25 m** | 33×33 | 2,178 | 0.0034% | **0.8743%** |

**The error falls 6.6× under refinement**, and the matrix error falls 4.6× with
it — the doubt was that the centre source's ~44% coverage gap might bite harder
as the grid refines, and it does the opposite. The discrete kernel converges
toward the smooth continuum one, so translation invariance holds *better*, not
worse. Amplification from matrix error into `V_max` is comparable at both
resolutions (373× and 257×), which is the consistency check on the claim.

So the approximation improves along **both** axes that production moves along —
domain size (5.78% → 0.41% small → converged at Δz = 50 m) and resolution
(5.78% → 0.87% at 50 → 25 m). Every configuration that will actually be run is
more favourable than the ones measured.

**What this does to the production estimate.** The exact build at Δz = 25 m,
small domain took **7,643 s** for 2,178 solves (271 mean CG iterations, 0
unconverged) on 12 threads. `:toeplitz` needs 10 solves for the same `K`. At the
Δz = 20 m converged target (9.5 M DOF, §4), extrapolating `t_solve ∝ DOF^1.56`
gives ~880 s per solve, so:

| build | solves | one node |
|---|---|---|
| `:exact` | 3,362 | **~17 days** |
| `:toeplitz` | **10** | **~2.5-3 h** |

Memory (~15 GB for `A`+`HP_DSAT`), not time, becomes the binding constraint —
and it is a single-node constraint, not a cluster one. This is what removes the
multi-node requirement rather than merely deferring it.

**Caveats before relying on it.** Validated at Δz = 50 m, at both the small and
the converged domain. **Not** yet validated at a finer Δz — the centre source
covers ~56% of separations (the rest far-field, set to zero) and that fraction
is resolution-independent, but the decay argument behind it is not proven to be.
And it is an approximation — a controlled ~0.004% against a ~53% domain bias and
a much larger resolution error, so it is nowhere near the accuracy-limiting
step, but `:exact` stays the default until the finer-Δz check lands.

### Symmetrising the reconstruction: tested, rejected

Reciprocity says the whole-space kernel is even, so the exact `K` should be
symmetric — it is, to 0.2003%. The reconstruction reads the kernel off sampled
columns and does not enforce that, so `(K + Kᵀ)/2` looked like a free
improvement. It is not an improvement, and it is not free.
`scripts/k_toeplitz_symmetrise.jl`, Δz = 50 m, small domain, full 30 days:

| `K` variant | asymmetry | Frob. err | `V_max(t)` worst |
|---|---|---|---|
| Toeplitz (shipped) | 0.2004% | 0.0155% | **5.7767%** |
| Toeplitz + sym full | 0.0000% | 0.1013% | 5.7846% |
| Toeplitz + sym diag blocks only | 0.2004% | 0.0155% | 5.7767% |
| exact + sym full | 0.0000% | 0.1001% | **2.1489%** |

**The premise was wrong.** `K_toep`'s asymmetry (0.2004%) matches the exact
`K`'s (0.2003%) to four digits — the expansion preserves whatever evenness the
sampled columns carried, so there was nothing to recover. Symmetrising moves
`K_toep` 6.5× *further* from the exact `K` in Frobenius norm and leaves `V_max`
marginally worse. The diagonal-blocks-only variant is bit-identical to shipped,
i.e. `K22` and `K33` are already even and all 0.2% lives in the cross blocks.

**The last row is the durable result.** Symmetrising the *exact* `K` — no
reconstruction error in play at all — moves worst `V_max(t)` by **2.15%**. A
0.2% asymmetry amplifies ~10× into the observable, so enforcing reciprocity is
not a neutral projection: it is a 2.15% perturbation, five times the Toeplitz
error at the converged domain. Anyone re-proposing this on "it costs nothing"
grounds should read that row first.

**What this does not settle.** It says symmetrisation *perturbs* the answer, not
that it perturbs it the wrong way. If the continuous kernel is genuinely even,
`(K + Kᵀ)/2` may be the better matrix and the exact `K` the biased reference.
Deciding that needs an independent reference — an analytic whole-space solution,
or a far larger domain — and neither was in reach. Measured at the small domain
only; the converged-domain repeat was cut when the project moved to
preconditioned CG.

## 4c. `K` on disk: the same build never runs twice

`K` depends on exactly ten things — λ, μ, `l_f`, `Δz`, `L_fault`, `L_normal`,
`order`, the SBP coefficients that `order` selects, the build mode, and the CG
settings — and on nothing else in the model.
Injection variant, friction parameters, `t_f`, integrator tolerances, output
choices: none of them touch it. So the entire cost of §4's table is paid per
*configuration*, not per run, and `StiffnessCache` makes that literal by writing
`K` to `$EQD_STIFFNESS_CACHE` and reading it back.

**A hit skips `FaultElasticity` too, not just the solves.** The cache file
carries the `Ω_f` axes, which is all `build_model` needed the elastic system for
once `K` exists — so the ~15 GB, minutes-long sparse assembly at Δz = 20 m is
skipped as well, and a cached model at any resolution builds in the time it
takes to read the file. Measured at Δz = 100 m, `:exact`: 25.8 s cold, 0.024 s
warm.

| Δz | `Ω_f` nodes | `K` on disk |
|---|---|---|
| 50 m | 289 | 2.5 MB |
| 25 m | 1,089 | 36 MB |
| **20 m** | **1,681** | **86 MB** |
| 10 m | 6,561 | 1.3 GB |

Nothing about this changes the Δz⁻⁶·⁷ scaling of the first build. What it
changes is the *number of first builds*: it is what makes `:exact` usable at a
production configuration despite §4b's ~17 days at Δz = 20 m, because that price
is paid once, offline, in its own job
(`scripts/build_stiffness_cache.jl`) — after which every parameter study that
does not move the grid is free.

**The correctness risk is a hit that should have been a miss**, and it is silent:
`K` looks plausible whatever configuration produced it, and nothing downstream
would notice. The key therefore spells out all ten inputs — including a digest of the operator
coefficients themselves, since Diffinitive is pinned by git revision and could
change them under a fixed `order` — is stored in the file, and is re-compared on
load — the filename hash only has to be
unique-in-practice, since a collision produces a miss. `:exact` and `:toeplitz`
are separate entries and are distinguishable in the filename, not merely in the
hash.

Caching is off unless `EQD_STIFFNESS_CACHE` is set, so the test suite and CI
never read or write it.

## 6. Preconditioning: measured, and it does not pay

Preconditioning was the natural alternative to §4b's Toeplitz build — it is the
standard route, it needs no approximation, and it attacks the iteration count
directly. It was measured end to end and **rejected on the numbers**. Both dead
ends are kept as `precond` options rather than deleted, so neither gets
re-proposed from first principles.

### The safety question, closed

`CGSolver`'s docstring previously left diagonal preconditioning unimplemented
pending a check that `M` commutes with `P`. Two results close it:

- **A diagonal `M` commutes with `P` exactly.** Rows `rm` and `rp` of `P` are
  identical, so rows `rm` and `rp` of `A = -HP·DSAT·P` are identical, and
  symmetry forces `A[rm,rm] = A[rp,rp]`. Measured: worst in-pair disagreement
  `0.0`, `‖MP − PM‖/‖MP‖ = 0.0`.
- **`null(A) == null(P)`, so *any* SPD preconditioner is safe.** Measured at
  N = 4,374: `rank(P) = tr(P) = 2205` and `rank(A) = 2205`, with a 13-order gap
  between the largest "zero" eigenvalue (6.9e-15) and the smallest nonzero
  (8.7e-2). Confirmed end-to-end with a deliberately non-commuting SPD `M`
  (`‖MP − PM‖/‖MP‖ = 0.51`): the iterate picks up **41% null-space content** —
  the leak the old invariant ruled out — and `U` is still unchanged to 1.5e-10,
  because all of it lands in `null(P)` and `U = P·u + χ` annihilates it.

A practical trap worth recording: `P` zeroes the far-field rows, so `A` has
entirely zero rows *and* columns there and `diag(A)` contains **exact** zeros
(3,318 of 9,126, precisely the far-field DOF set). Naive Jacobi divides by zero.

### Jacobi: 0.92×

`diag(A)`'s nonzero entries span only 13.5×, so there is almost no diagonal
scaling to remove. 86 iterations against plain CG's 79 — a regression. `U`
unchanged to 2.5e-11, so the option is correct, just not worth selecting.

### AMG: valid hierarchy, no mesh-independence, 3-8× slower

AlgebraicMultigrid.jl, both `ruge_stuben` and `smoothed_aggregation`. It builds a
*working* hierarchy — symmetric to 1e-15 (so CG's recursive residual stays
valid), correct answers to ~1e-10, and it needs no null-space handling at all:
building on the full `A` and on `A[keep,keep]` gives identical iteration counts,
because AMG isolates the zero rows as their own coarse points.

| N | plain CG | RS | SA | RS wall | SA wall |
|---|---|---|---|---|---|
| 4,374 | 72 | 20 | 20 | 0.19× | 0.33× |
| 9,126 | 79 | 24 | 25 | 0.17× | 0.30× |
| 19,074 | 102 | 30 | 31 | 0.24× | 0.25× |
| 34,398 | 124 | 37 | 38 | 0.12× | 0.32× |

**There is no asymptotic gain.** Fitting `iters ∝ DOF^p`: plain CG `p = 0.264`,
RS `p = 0.298`, SA `p = 0.311`. AMG's iteration count grows *slightly faster*
than plain CG's, so the ~3.3× is a flat constant that narrows with size — the
opposite of the mesh-independence that motivates AMG. The cause is not mysterious:
coarsening heuristics assume something Laplacian-like, and an SBP order-4
operator with wide mixed-sign closures, SAT coupling and a 50% null space is not.

**Tuning the smoother cannot rescue it, and the ceiling is computable.** The
V-cycle measures 9.2 mat-vecs against an operator complexity of 1.05 that implies
~3.5, so the Gauss-Seidel implementation is ~2.5× slower than it should be. But
even at the ideal 3.5:

- preconditioned iteration ≈ 1.6 (CG) + 3.5 (V-cycle) ≈ **5.1** mat-vecs
- plain iteration ≈ **1.6** mat-vecs → cost ratio **3.2×**
- measured iteration reduction **3.3×**
- **net ≈ 1.03× — break-even.**

AMG needs roughly a 10× iteration cut to pay for a V-cycle. It delivers 3.3×,
shrinking. **AlgebraicMultigrid is therefore not a dependency of this package**;
the experiments live in a scratch environment.

### Consequence

Against `:toeplitz`'s 578 → 10 solves (~58× on the dominant term), preconditioning
offers ~1×. That is what returned the project to finishing the Toeplitz build.

## 5. Where the time goes, and what to attack

At any converged resolution, `fault_stiffness` is ~98% of the cost. Assembly is
minutes since §2.2; the time integration is a dense `K` mat-vec per RHS
evaluation and is comparatively cheap.

Ranked by expected payoff. **§4b reorders this list**: the Toeplitz result
attacks the `K` build's *scaling*, so it dominates everything below, and it
largely dissolves item 1 rather than competing with it.

0. **Implement the Toeplitz `K` build** (§4b). Turns `2·N_Ωf` solves into 2, at
   ~0.03% cost in `V_max`. Purely local work, no cluster needed.
0b. **Exploit `K`'s square symmetry (`D4`) in the `:exact` build — an exact
   discrete identity, not an approximation.** **Implemented** —
   `fault_stiffness(fe; symmetry=true)` (`src/FaultResponse.jl`), and it is
   what `build_model`'s `:exact` path uses by default now (`stiffness_matrix`
   in `src/BP8.jl`), since the domain it builds is always square and centred.
   Verified against the plain build on a 7×7 `Ω_f`: agrees to `rtol=1e-8`
   (solver tolerance), 16 solves instead of 98
   (`test/fault_response_test.jl` "D4 symmetry build agrees with the plain
   exact build"). `Ω_f` and the elastic grids are
   square and centred in the two fault-parallel directions, the medium is
   homogeneous, and all four fault-parallel far-field faces carry the same
   `u=0` condition. So the whole discretization is invariant under the eight
   symmetries of the square, acting on positions and on the slip/traction
   components together: reflecting `x2 → −x2` flips `s2` and `τ2` and leaves
   `s3`, `τ3` alone; reflecting about the diagonal swaps the two. In block form,
   with `Q` one of the eight signed permutations and `g` its action on node
   indices,

       K[g·i, g·j] = Q · K[i, j] · Qᵀ

   Rebuilding the whole `K` from one node per orbit and differencing against a
   full `:exact` build: **1.09e-16** at Δz = 100 m, **1.51e-16** at Δz = 80 m
   (Frobenius, relative; worst single entry 4.3e-16).

   **Why that is roundoff and not a small error.** `A` commutes with the
   symmetry and the symmetry is orthogonal, so CG started from zero produces
   *exactly* the mapped iterates — same Krylov space, same stopping test, same
   iteration count. The symmetry therefore survives in the CG **error**, not
   just in the converged solution. Measured by loosening the tolerance until the
   columns are visibly wrong:

   | `rtol` | `K` error vs `rtol` = 1e-12 | `D4` residual |
   |---|---|---|
   | 1e-12 | — | 1.09e-16 |
   | 1e-6 | 1.9e-7 | 1.22e-16 |
   | **1e-3** | **1.5e-4** | **1.04e-16** |

   A badly converged `K` is still symmetric to the last bit. An approximation
   would track *something* — the tolerance, `Δz`, the domain size; this tracks
   only the machine epsilon. Contrast reciprocity, `K = Kᵀ`, which sits at
   **1.8e-3**: that one is a physical near-symmetry spoiled by the interface-SAT
   asymmetry, and it is what a real approximation looks like here.

   **Contrast with §4b.** `:toeplitz` assumes *translation* invariance, which
   the truncation boundary genuinely breaks — hence its 0.41–5.8%. `D4` assumes
   only *reflection* invariance, which the truncation boundary **preserves**,
   because the boundary is itself square and centred. Same idea, and the
   difference between the two is exactly why one costs accuracy and the other
   does not.

   **The payoff.** The group does not act freely — nodes on the axes and
   diagonals have stabilizers — so the reduction is 8× only asymptotically:

   | Δz | `2·N_Ωf` | solves needed | speedup |
   |---|---|---|---|
   | 50 m | 578 | 81 | 6.5–7.1× |
   | 25 m | 2,178 | 289 | 7.5× |
   | **20 m** | **3,362** | **441** | **7.6×** |
   | 10 m | 13,122 | 1,681 | 7.8× |

   That turns §4b's ~17 days at the Δz = 20 m target into **~2.2 days**, with no
   approximation at all — and combined with §4c it is a one-off job for a `K`
   that is then reused indefinitely. It is what makes `:toeplitz` *avoidable*
   for the submission rather than merely defensible.

   **What it depends on**, in case the geometry ever changes: square and centred
   `Ω_f` *and* elastic grid (a rectangular fault or `L_fault` differing between
   `x2` and `x3` loses the diagonal reflection — 4× not 8×, still exact);
   homogeneous λ, μ; the same boundary condition on all four fault-parallel
   faces; a uniform grid with mirror-image SBP closures. The fault-normal
   direction is untouched by these maps, so `L_normal` and the `x1`
   discretization are unconstrained. Crucially it is a property of the
   **operator and geometry only** — the injection source, the initial
   conditions, and how asymmetrically slip evolves during the run are all
   irrelevant, because `K` never sees them. A **free-surface** problem (BP1/BP3
   rather than BP8's whole space) would break the depth reflection and leave
   2×; that is the change most likely to cost this.

0c. **GPU offload for the CG solve itself — implemented (2026-09-10),
   measured positive at small scale, extrapolated beyond it.** Every other
   speedup in this section attacks the *number* of solves; this attacks the
   cost of *each* solve, and composes with 0b (same D4 reduction, GPU instead
   of CPU threads underneath). `fault_stiffness_gpu`
   (`EarthquakeDiffinitiveCUDAExt`, loaded by `using CUDA`) holds `A`, `P`,
   `T2`, `T3` resident on one GPU for the whole build and runs the D4
   representatives' CG solves against it sequentially — sequentially, not
   concurrently, because the whole premise is a single bandwidth budget: two
   solves at once would contend for it rather than add to it, unlike CPU
   threads with their own caches. **No sharding** — the whole `A` must fit in
   one GPU's memory.

   **Measured**, on a consumer RTX 2060 (Turing, 336 GB/s, correctness
   checked against the CPU D4 build each time, agreement at CG-tolerance
   level ~1e-13 – 1e-12, not merely close):

   | n | DOF | CPU | GPU | speedup |
   |---|---|---|---|---|
   | 11 | 7,986 | 0.019 s | 0.010 s | 1.89× |
   | 15 | 20,250 | 0.070 s | 0.017 s | 4.13× |
   | 21 | 55,566 | 0.296 s | 0.035 s | 8.43× |

   The speedup **grows with problem size**, consistent with the
   memory-bandwidth-bound mechanism this whole document is built around (§1):
   a GPU's raw bandwidth advantage over a CPU compounds once the working set
   exceeds CPU cache. Production DOF is 100-1000× larger than n=21.

   **What is and is not validated.** Correctness is real and checked, at
   every size tested, on this hardware — that is not extrapolated. The
   *speedup number* at production DOF counts, and on any datacenter GPU
   (Hopper/Ada, not the Turing card measured), **is** extrapolated — from the
   observed size trend and from the two architectures' bandwidth ratio, not
   measured directly. cuSPARSE kernel behaviour does not necessarily scale
   linearly with raw bandwidth across architecture generations. Re-measure on
   the actual target GPU before relying on a specific number; `verbose=true`
   reports per-representative timing for exactly that purpose.

   **Fit against UPPMAX Pelle's GPUs**, using this section's own `A`+`HP_DSAT`
   memory figures (§4: ~15 GB at Δz = 20 m, ~65 GB at Δz = 10 m at the
   currently-targeted relaxed domain, ~116 GB at the nominal spec domain):

   | GPU | VRAM | bandwidth | fits Δz=10m (~65 GB)? | fits Δz=10m (~116 GB)? |
   |---|---|---|---|---|
   | H100 NVL | 94 GB | 3,900 GB/s | **yes**, ~29 GB headroom | no |
   | L40S | 48 GB | 864 GB/s | no | no |
   | T4 | 16 GB | 300 GB/s | no (same ballpark as the 15 GB workstation limit) | no |

   The H100 NVL is the only one of the three that holds the whole Δz = 10 m
   `A` at the relaxed domain on one card — which is what makes "GPU without
   sharding" a real option there rather than needing multi-GPU matrix
   splitting on top of everything else in this section.

   **Not built**: multi-GPU matrix splitting (for the ~116 GB nominal-spec
   domain, or for L40S/T4-class cards) — would need `A` itself divided across
   devices, a materially bigger project than the column-sharding this
   document already covers, and not attempted since the relaxed-domain H100
   case doesn't need it.

### Also tried on the `:exact` CG solve, and rejected (2026-09-10)

Kept here so none of these get re-proposed from first principles. All
measured on the real assembled `A`, not reasoned about in the abstract.

- **Mixed precision** (Float32 CG + Float64 iterative refinement): **0.59×**
  (slower), and did not reach `rtol=1e-10` in 10 refinement rounds. This
  matrix's conditioning hits Float32's roundoff floor too early for
  refinement rounds to amortize.
- **Block Krylov methods** (`Krylov.block_minres`, solving several columns as
  one block — no `block_cg` exists in Krylov.jl): returns **all-NaN** while
  reporting `solved=true` — a silent wrong answer, not a slow one. Confirmed
  the cause is `A`'s ~40% null space (`P`'s far-field/tangential-pair
  structure) by running the identical call on a non-singular test matrix,
  where it works correctly. Also slower even ignoring correctness: 126 block
  iterations cost 2.5× the wall-clock of 913 total single-column iterations,
  since each block iteration is far more expensive here.
- **Matrix reordering** (hand-rolled RCM, no fill-reducing/bandwidth-reducing
  package was already a dependency): **~1.0×**, no effect, despite cutting
  nominal bandwidth 17,314→2,361. A first pass showed 7×, which was a Julia
  JIT-compilation timing artifact from not warming up the timed call before
  measuring — corrected and reproduced at ~1.0× on both raw `mul!` throughput
  and full CG solve time. The SBP+SAT sparsity pattern on a structured grid
  already has enough locality that bandwidth-reducing reordering has nothing
  left to gain.
- **Warm-starting CG** from a neighbouring column's converged solution:
  **~1.0×**, no iteration reduction, on realistic `Ω_f`-node columns (an
  earlier pass showed apparent iteration blowup and huge disagreement, but
  that traced to a degenerate all-zero RHS in the synthetic test, not a real
  hazard for genuine columns). Even the null result isn't worth taking: warm
  starting forces columns to solve sequentially, which would forfeit the
  existing embarrassingly-parallel CPU threading for a measured ~0% gain.
- **Exact dimension reduction** (drop far-field DOFs and merge tangential
  pairs into one unknown before CG, via the same congruence transform the
  removed `factorize_reduced` used — but skipping its Cholesky factorization,
  which is what was actually rejected before, not the reduction itself):
  mathematically exact and verified (`P*u` agrees with the full-system CG
  answer), cuts the DOF count 1.48× — but **~1.0×** wall-clock. Far-field
  rows were already all-zero (0 stored nonzeros — nothing to save by dropping
  them), and merging tangential pairs *increases* density in the surviving
  rows enough to cancel the DOF reduction. `nnz`, not DOF count, is what
  tracks mat-vec cost here, and `nnz` barely moved (563,882 → 498,789).

1. **Multi-node parallelism — implemented, and now composes with item 0b.**
   `fault_stiffness`'s columns (or, with item 0b, its D4 orbit representatives)
   are independent right-hand sides against a shared `A`, which distributes
   without `Distributed`/MPI: each node rebuilds its own copy of `A` (minutes,
   §4c) and takes a slice of the work, coordinating through nothing fancier
   than a shared directory — `build_stiffness_cache.jl [shard] [nshards]` +
   `merge_stiffness_cache.jl`, backed by `fault_stiffness_d4_shard` so sharding
   splits *representatives* rather than raw columns and does not give up item
   0b's ~7.8× (see that function's docstring for why the shard-file format
   needed no change to support this). Within a node, threading is still only
   sublinear (2.13× on 16 threads at production, PROGRESS.md) because the
   sparse mat-vec is **memory-bandwidth bound** — independent **nodes**, each
   with their own bandwidth, are what scales, which is why this axis matters at
   all even after item 0b's per-node win.
2. **Exploit structure in `K` (untested, potentially the largest win).** In a
   homogeneous medium `K[i,j]` should depend mainly on the separation
   `x_i − x_j`, making `K` near-block-Toeplitz — one solve could populate most
   of it, collapsing thousands of solves to a handful. Far-field truncation
   breaks this exactly, which is why it needs testing rather than assuming.
   Cheaply testable: build `K` at Δz = 50 m and check how well entries collapse
   onto separation alone.
3. **A CG preconditioner.** Attacks the iteration count directly (236 already at
   Δz = 50 m, extrapolating to ~450 at Δz = 20 m). Not shipped unvalidated: it
   preserves the `range(A)` invariant CG relies on here only if it commutes with
   `P` — see `CGSolver`'s docstring.
4. **The boundary-integral route.** What most SEAS codes do: the fault-to-fault
   kernel is a convolution, `O(N log N)` with FFTs, no volume unknowns and no
   `K` build at all. A design decision, not an increment — but it is the only
   option here that changes the Δz⁻⁶·⁷ scaling rather than its constant.

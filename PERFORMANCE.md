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

**Do not convert these to core-hours.** Threading efficiency is ~15% (7.02 s per
column on 12 cores against ~13 s serial-equivalent), because the sparse mat-vec
is memory-bandwidth bound. Cores within a node are nearly free of benefit past a
point; independent **nodes**, each with its own memory bandwidth, are what
scales. That is exactly why §5 item 1 is the blocker.

## 5. Where the time goes, and what to attack

At any converged resolution, `fault_stiffness` is ~98% of the cost. Assembly is
minutes since §2.2; the time integration is a dense `K` mat-vec per RHS
evaluation and is comparatively cheap.

Ranked by expected payoff:

1. **Multi-node parallelism (the cluster blocker).** `fault_stiffness` uses
   `Threads.@spawn` only — single node. Worse, PROGRESS records threading
   scaling as sublinear (2.13× on 16 threads at production) because sparse
   mat-vec is **memory-bandwidth bound**, so piling on cores within a node buys
   little. The columns are independent right-hand sides against a shared `A`,
   which is exactly the shape that distributes well: give each node its own copy
   of `A` and a slice of the columns, and scaling is near-linear because each
   node has its own memory bandwidth. Needs `Distributed`/MPI; neither is
   present. **This is what stands between the cluster and Δz = 10 m.**
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

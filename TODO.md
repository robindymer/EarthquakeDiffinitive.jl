# TODO

Open work, in dependency order. Background for the symmetry items is in
`SYMMETRIC_SAT.md`, for the cost items in `PERFORMANCE.md`; the rest is from
`PROGRESS.md`'s "Known limitations".

## Direction (2026-08-20)

**Robin's call: go the standard route — preconditioned CG, sized for a cluster.**
The Toeplitz `K` build is judged too much work and testing for what it buys, and
the boundary-element route is explicitly ruled out. Multi-node parallelism is
back in scope, because it was only ever dissolved *by* Toeplitz.

`:toeplitz` stays in the tree as opt-in with `:exact` the default. It is written,
tested and documented, so keeping it costs nothing; no further work goes into it,
including the Δz = 25 m validation, which is dropped.

### Decisions settled

- **2 → 5 sources in `fault_stiffness_toeplitz`** — delegated back to me and
  kept. Centre-only is 0.004% through injection and 97% over 30 days; the
  5-source priority build is 0.41%. Moot for production now that `:toeplitz`
  is frozen, but the code should not ship the version that scores 97%.
- **`:toeplitz` as default** — no. Superseded by the direction above.
- **`output/` regeneration** — not needed now. It is gitignored and not present
  locally, so no biased numbers are sitting in the repo; the shipped set gets
  generated once, at the converged domain and target resolution.
- **Multi-node parallelism** — wanted, per the direction above.
- **Housekeeping:** `diffinitive_registry` is still in `~/.julia/registries`.
  Remove with `Pkg.Registry.rm("diffinitive_registry")` if unwanted.

### Why preconditioning alone is not the whole answer

Total `K` cost = (solves) × (iterations per solve) × (cost per iteration).
Preconditioning attacks **only the middle term**. The solve count is
`2·N_Ωf ∝ Δz⁻²` and is untouched — that was the term Toeplitz removed, so
dropping Toeplitz means keeping it and needing the cluster.

Iterations scale as `DOF^0.31` (measured), giving ~300 per solve at Δz = 50 m,
~710 at 20 m, ~1345 at 10 m against `PERFORMANCE.md` §4's measured baseline.

**Δz = 20 m is the target, and it is close.** `resolution_report` calls a grid
converged at `L_b/Δz ≥ 3`, i.e. Δz ≤ 21 m, so the nominal 10 m is 6.7× more
expensive than convergence requires. §4 puts Δz = 20 m at ~306 h on 12 cores; a
5× iteration cut makes that ~60 h — one node for a weekend, no multi-node
strictly required. Δz = 10 m needs the distribution.

**The unusually favourable part:** `A` is fixed for the whole run and an
expensive preconditioner setup amortizes over 3,362 (or 13,122) right-hand
sides. The per-solve economics here are the opposite of the usual ones, which is
the strongest argument for AMG over anything cheap.

## Agreed next steps (2026-08-20)

1. **Jacobi preconditioner, validated.** Cheap, and it builds the preconditioner
   plumbing whether or not Jacobi itself is worth shipping. Gate measurements
   below are done.
2. **AMG (AlgebraicMultigrid.jl) — the actual lever.** Measure iterations vs DOF
   at three sizes; the question is whether it is mesh-independent *here*. Not a
   given: this is an SBP split-node operator with SAT terms and a deliberate 40%
   null space, not a textbook elasticity discretization. Not currently a dep.
3. **Multi-node over the RHSs.** Needed only for Δz = 10 m. `Distributed`/MPI;
   neither is present.

### Preconditioner gate — measured, not assumed

`CGSolver`'s docstring left a diagonal preconditioner unimplemented pending one
check. Both parts are now measured at n1=9, n23=13 (9,126 DOF):

- **The commutation condition holds by construction, not by luck.** Rows `rm`
  and `rp` of `P` are identical (both `0.5e_rm + 0.5e_rp`), so rows `rm` and `rp`
  of `A = -HP·DSAT·P` are identical, and symmetry then forces
  `A[rm,rm] = A[rp,rp]`. Verified numerically rather than left as a derivation.
- **A second issue the docstring does not mention:** `P` has entirely zero rows
  on the far-field DOFs, so `A` has zero rows *and* columns there and `diag(A)`
  contains **exact zeros** — 3,318 of 9,126, and that set is *exactly* the
  far-field DOF set. Naive Jacobi divides by zero and needs a floor. Those DOFs
  are annihilated by `P`, so the floor value cannot affect the answer, only `M`'s
  definiteness.
- `diag(A)` varies only **13.5×** max/min over the nonzero entries, and all are
  positive. That is a narrow spread, and it is the reason to expect little from
  Jacobi: there is not much diagonal scaling to remove.

## Done

### Earlier session

- [x] **Diagnose the `-HP(D+SAT)P` asymmetry.** Root cause: `traction_blocks`
      used `first_derivative` for every term, but μ's normal-direction terms
      come from the *narrow* `second_derivative`, whose SBP identity carries a
      different boundary operator (`normal_derivative`). λ and μ's tangential
      terms were already correct. `scripts/symmetry_decomposition.jl`
      reproduces the whole decomposition; `SYMMETRIC_SAT.md` writes it up.
      Ruled out by measurement: `P`, operator ordering, the interface, the
      two-grid coupling, `elastic_blocks`, and the quadrature `H`.
- [x] **Fix `traction_blocks`** (`src/Elasticity.jl`) to match
      `context/notebooks/elastic_clean.jl`'s `IsotropicTractionOperator`.
- [x] **Add the SBP property test** (`test/elasticity_test.jl`,
      `"SBP property: E and T are compatible"`) — the notebook had it, this
      package did not, and it is what would have caught the bug immediately.

### This session

- [x] **Run the test suite.** 172/172 before the solver change, **176/176**
      after, ~2m45s. The `src/` edit and the new SBP test are exercised. Both
      of the risks flagged for this step held up: the 5% manufactured-traction
      tolerance passes, and `Grids._boundary_sign` still exists.
- [x] **Verify the traction fix against production runs** (was BLOCKING).
      All four configurations re-run and diffed against `PROGRESS.md`'s table;
      numbers in "Results" there and in `SYMMETRIC_SAT.md` "Gate 2". Peak `V`
      shifts 2.5-7.0× against a 105-406× resolution spread, final slip 5-11%,
      peak pressure bit-identical. The fix also *narrows* the resolution spread
      and makes the GS peak time resolution-consistent (2.26/0.62 d → 2.22/2.18
      d), neither of which was tuned for.
- [x] **Re-check `K`'s symmetry at production size.** 0.2003% at Δz = 50 and
      0.1827% at Δz = 100, against 0.19% before — unchanged, as predicted, and
      `diag(K) < 0` holds. Confirms the 2.8-3.3% seen at `L=1, n=11..15` was
      domain truncation, not a regression.
- [x] **Re-run `scripts/bp8_validate_pressure.jl`** as a control. Reproduces
      its entire table to the digit, as it must — pore pressure never touches
      elasticity.
- [x] **Galerkin reduction `SᵀAS` + Cholesky** in `factorize_reduced`, with
      `prolongation(rs)` exported. Solutions agree with the old LU path to
      2.7e-15; factor is 1.77×/1.88× smaller at n = 9/13. Falls back to LU with
      a loud warning on `PosDefException`, since that would mean a real
      regression. Pinned by `test/elasticity_split_node_test.jl`'s
      `"Galerkin reduction is SPD and Cholesky agrees with LU"`, which also
      asserts `E·A·S` is *still* asymmetric so the Galerkin form cannot be
      quietly swapped back.
- [x] **Reproducible environment.** `Project.toml` now has a `[sources]` entry
      pinning `Diffinitive` to upstream `92d842cf`. Verified by instantiating
      `Project.toml` + `src/` with no Manifest: it resolves, precompiles, and
      builds a symmetric operator. An explicit `Pkg.develop` still overrides it,
      so the local dev workflow is untouched.
- [x] **Domain-size validation gap.** The obvious fix — adding a
      `(L_fault=800, L_normal=400)` row to the existing sweep — is impossible:
      that sweep runs at Δz = 100 m, where `L_normal = 400` is only 5 points
      across the fault-normal direction against SBP order 4's minimum of 9.
      `L_normal = 400` is legal *only* at Δz = 50 m, which is also the
      production resolution. `scripts/bp8_domain_convergence.jl` therefore now
      runs a **second sweep** at Δz = 50 m over `L_normal ∈ {400, 600, 800}`
      with `L_fault` fixed at 800 m, isolating the fault-normal truncation at
      the resolution that actually ships. The loop and reporting were factored
      into `sweep`/`report` so both studies share them.
- [x] **Run both sweeps.** All 8 rows completed. The Δz = 100 m `L_fault` study
      reproduces its published numbers and pressure is identical across its
      rows. The new Δz = 50 m study found a real bias — see item 1 below, which
      is the one thing this session opened rather than closed.
- [x] **Iterative solver.** `build_model(...; solver=:cg)` runs CG (Krylov.jl)
      on the singular `A` directly — no reduction, no factorization, no
      null-space handling, because `b ⊥ null(A)` exactly and CG from `x₀ = 0`
      never leaves `range(A)`. Same `K` as the direct path to 1.6e-11. At
      production size it costs ~2× the wall-clock and **~6× less of the memory
      that actually binds** (0.38 GB of solver footprint against 2.45 GB, and
      no 168M-nonzero factor).
- [x] **Thread the `K` build.** Its `2·N_Ωf` columns are independent and now
      run in parallel under CG, bit-identical to serial. That brings threaded
      CG to 156 s against the direct path's 128 s at production size — a 22%
      time penalty for 2.3× less peak memory. Scaling is sublinear (5.1× on 8
      threads at n=13, 2.13× on 16 at production) because the sparse mat-vec is
      memory-bandwidth bound; more threads is not the remaining lever, fewer
      iterations is. Impossible for the direct path (CHOLMOD's solve is not
      thread-safe), so `duplicate` throws there and `fault_stiffness` falls
      back to serial.
      Caught a genuine data race doing this: `if`/`else` and `begin` do not
      open a scope in Julia, so the per-task buffers were shared and `K` came
      out 2.35 relative off. Now a regression test demanding bit-identity, and
      CI runs with `JULIA_NUM_THREADS: 4` — one thread cannot detect it.
- [x] **Diagnose "order 6 is unusable".** Broader than recorded: order 6 in
      `standard_diagonal.toml` has only `H`, `e`, `d1` and `D2.positivity` —
      no `D1` at all and no `D2` stencils. Upstream Diffinitive data gap, not
      fixable here. Recorded as `PROGRESS.md` limitation 5.

### Performance session (2026-08-19)

Full write-up in `PERFORMANCE.md`; only the outcomes are listed here.

- [x] **Answer "should this be matrix-free?"** No — and by a wide margin.
      Diffinitive's lazy composition re-expands the stencil at every point,
      measuring **41-67× slower** than sparse `mul!` on the same operator.
      A matrix-free `P` is worse still (forces `A` into a composite, 1.66×
      slower per mat-vec). Both TODO notes in `ElasticitySplitNode.jl` are
      now answered in place with pointers to `PERFORMANCE.md` §1.
- [x] **Remove stored zeros from `P`.** `P[r,:] .= 0.0` keeps the CSC
      structural entry; those zeros multiplied into full `DSAT` rows and
      inflated `HP_DSAT`/`A` by **29-52%**. One `dropzeros!`. Operators
      bit-identical, 179/179.
- [x] **Fix quadratic SAT assembly.** `SATmat[rows,cols] .+= B` rewrote the
      whole CSC each of 12 times; assembly scaled quadratically and was 64% of
      all assembly by n=21. Triplet accumulation + one `sparse()` call:
      **199× faster** at n=21, growth now linear, matrix identical, 179/179.
- [x] **Establish the cost model.** `t_solve ∝ DOF^1.56`, CG iterations
      ∝ DOF^0.31, columns ∝ Δz⁻², so the `K` build scales as **Δz⁻⁶·⁷**.
      Measured at Δz = 100/80/50 m on benchmark-shaped grids.
- [x] **Confirm the assembly changes are physics-neutral end-to-end.** All six
      configurations of `bp8_domain_convergence.jl` with published values
      reproduce **to every digit** on full coupled 100 h runs, at two
      resolutions — a far stronger check than the unit tests, since it exercises
      the whole elastic → friction → pressure chain.

**Stale references below.** The `Done` entries above this block mention
`factorize_reduced`, `prolongation(rs)`, `solver=:cg` and the CHOLMOD fallback;
none of those exist in `src/` any more (commit `02b6c91` "Only CG"). They are
kept as history — do not treat them as API.

## 1. Act on the domain-convergence result — `L_normal = 400` is not innocent

**The sweep now runs to convergence** (2026-08-19): `L_normal ∈ {400, 600, 800,
1000, 1200, 1600}` at Δz = 50 m, `L_fault` = 800 m. Full table and discussion in
`PROGRESS.md` "Results".

`V_max` **does** settle, in both directions — but only well beyond any domain
previously used. **Both sweeps are now done** (fault-normal and the `L_fault`
mirror), and the combined answer is:

> **Required domain: `L_fault` ≥ 4·`l_f` = 1600 m, `L_normal` ≥ 3·`l_f` = 1200 m**
> for ~1% in `V_max`. The shipped configuration (2·`l_f`, 1·`l_f`) is **~53%
> low**. `build_model`'s defaults (3·`l_f`, 2·`l_f`) are also short.

Successive estimates were 42% → 45.3% → 53%, each growing as the reference
improved — the signature of measuring against an unconverged reference. The 53%
is the first bounded in both directions. Convergence is slow for a physical
reason: the elastostatic kernel decays as 1/r³, so doubling the domain cuts the
error only ~8×. `K_self` and slip were converged throughout (≤0.3%); this is
entirely a peak-slip-rate problem — and `V_max(t)` is one of the two global
source parameters the benchmark requires, not a diagnostic.

**This inflates every cost estimate**: ~4.5× compute and ~2.6× memory at each
resolution. `PERFORMANCE.md` §4 is rewritten accordingly — Δz = 20 m is no
longer a workstation job (~15 GB), and Δz = 10 m needs ~120 GB per node.

So the answer to the original question is *no*: the halving was not free for
peak `V`. It is free for slip and stiffness.

- [x] **Push the fault-normal sweep past 800 m.** Done — `V_max` converges by
      1600 m (Richardson limit 3.048e-7, 0.13% beyond that row). The rows were
      previously thought not to fit; that was the direct solver's fill-in.
- [x] **Mirror sweep: vary `L_fault` at large fixed `L_normal`.** Done, and it
      was needed — `V_max` moved 12.8% over `L_fault` 2·`l_f` → 3·`l_f`, so the
      fault-normal sweep's limit *was* set by its pinned `L_fault`. Run it with
      `scripts/bp8_domain_convergence.jl mirror`.
- [ ] **Re-run production at the converged domain.** Every number in `output/`
      carries the ~53% `V_max` bias. Not yet done deliberately: at the required
      domain Δz = 50 m is 634k DOF (~1.1 h) but still not resolution-converged,
      so a regeneration now gets superseded by §2's resolution work. Decide
      whether to ship a corrected-domain Δz = 50 m set as an interim, or wait
      and regenerate once at the target resolution.
- [x] **Report the computational domain in the output headers.** Already done —
      `domain_line` (`src/BP8.jl`) emits `# elastic_domain=…` with `L_normal`,
      `L_fault`, SBP order and DOF count into both the time-series and profile
      headers. (Listed as open in an earlier draft of this file; that was an
      assumption, not a check.)
- [ ] **Do not assume these domain requirements transfer to finer Δz.** The
      same `L_normal` 800→1200 step moves `V_max` +0.324% at Δz = 100 m but
      +5.17% at Δz = 50 m. That comparison is confounded (different `L_fault`,
      and Δz = 100 m is badly under-resolved) so it proves nothing either way —
      but it removes the grounds for assuming transferability. A matched
      two-resolution sweep would settle it; at Δz = 25 m that is ~12 h.
      **If the requirement does grow with resolution, §2 should have come
      first** and this study needs redoing at the target Δz.

- [ ] **Decide whether to re-run production at a larger `L_normal`.** Now known
      to be affordable: `(Δz=50, L_fault=800, L_normal=800)` builds in 385 s,
      so a full 30-day run is ~10 min per injection model. That would make peak
      `V` less biased — but peak `V` is *already* documented as indicative-only
      because of resolution (item 2), which dominates this by ~100×, and
      re-running changes every number in `output/`. Worth doing if the runs are
      going to be regenerated anyway; not worth it on its own.
- [ ] **Push the fault-normal sweep past 800 m** to find where `V_max` settles,
      if peak `V` is ever going to be quoted quantitatively. `L_normal = 1000`
      at Δz = 50 m is ~137k DOF and may not fit.
- [ ] Note for anyone reading the older `L_fault` sweep: its 0.55% `V_max`
      spread does **not** mean domain size is settled. It varied the far
      boundary while the near one sat at 400 m.

## 2. Resolution — the actual blocker

`PROGRESS.md` "Known limitations" 1 and 2. Δz = 50 m gives 1.3 cells per
process zone `L_b ≈ 64 m`.

**The target is Δz = 20 m, not the benchmark's 10 m.** `resolution_report`
calls a grid converged at `L_b/Δz ≥ 3`, i.e. Δz ≤ 21 m — so 10 m is 6.7× more
expensive than convergence actually requires. Quote 10 m only if the submission
demands the nominal spec.

Costs from `PERFORMANCE.md` §4 (extrapolated ±2×):

| Δz | DOF | `A`+`HP_DSAT` | `K` build | converged |
|---|---|---|---|---|
| 50 m (shipped) | 245 k | 0.34 GB | 0.5 core-h | no (1.3) |
| **20 m** | 3.6 M | ~5.5 GB | **182 core-h** | **yes (3.2)** |
| 10 m (spec) | 28.2 M | ~42 GB | ~17,700 core-h | yes (6.4) |

`PROGRESS.md` limitation 2's "Δz = 50 m is the ceiling" is **stale** — that was
the removed direct solver's fill-in. Memory is no longer what binds.

- [ ] **Multi-node parallelism — the cluster blocker, and now item 2's top
      priority.** `fault_stiffness` uses `Threads.@spawn` only, so it cannot
      leave one node, and within a node the sparse mat-vec is
      memory-bandwidth-bound (2.13× on 16 threads, per PROGRESS). The `2·N_Ωf`
      columns are independent RHSs against a shared `A` — the ideal shape for
      distribution: one copy of `A` per node, a slice of the columns each,
      near-linear because each node has its own bandwidth. Needs
      `Distributed`/MPI; neither is present. Δz = 10 m is unreachable without
      it and comfortable with it.
- [x] **Test whether `K` is near-block-Toeplitz.** **Yes, and decisively.**
      `scripts/k_toeplitz_structure.jl` + `k_toeplitz_validate.jl`. A `K` rebuilt
      from **5 sources (10 CG solves)** reproduces `V_max(t)` to **0.41%** over
      30 days at the converged domain, against the full 578-solve build. Full
      write-up in `PERFORMANCE.md` §4b, including three results that each cost a
      wrong turn: averaging sources is ~500× worse than prioritising them,
      matrix-norm error is amplified 2-3 orders of magnitude into `V_max` (and
      can move in the opposite direction), and validating through the injection
      phase alone hides a 97% error that only appears after shut-in.
- [x] **Implement the Toeplitz `K` build.** `fault_stiffness_toeplitz`
      (`src/FaultResponse.jl`), opt-in via `build_model(; stiffness=:toeplitz)`
      with `:exact` still the default. Regression tests in
      `test/fault_response_test.jl` pin that the centre column is exact, that
      self-stiffness stays negative everywhere, and that it uses exactly 2
      solves — the last guards against "improving" it by adding sources, which
      measured ~500× worse.
- [x] **Validate across domain × duration.** All four combinations run at
      Δz = 50 m. The shipped 10-solve priority build gives **0.41% worst-case in
      `V_max(t)` over 30 days at the converged domain** — the only configuration
      that will actually be run. Table in `PERFORMANCE.md` §4b.
      This is what forced the design from 2 sources to 5: centre-only scores
      0.004% at 100 h and **97%** over 30 days. Validating through the injection
      phase alone would have shipped that.
- [ ] **Validate at a finer Δz, then consider making `:toeplitz` the default.**
      The last gap. The centre source's ~44% coverage gap is resolution-
      independent in *fraction*, but the far-field-decay argument behind it is
      not proven to be. Δz = 25 m at the small domain is ~2.4 h for the
      reference build; that is the cheapest meaningful check. Until it lands,
      `:exact` stays the default.
- [ ] **Try symmetrising the Toeplitz `K`.** Reciprocity says the true kernel is
      even, `kernel(−d) = kernel(d)`, so the exact `K` is symmetric (measured
      ~0.2%). The reconstruction takes the kernel from one column and does *not*
      enforce that, so `K_toep` is only as even as the sampled column. Replacing
      it with `(K + Kᵀ)/2` costs nothing and may cut the error — untested, so
      not done. Test it end-to-end, not in Frobenius norm: §4b's whole lesson is
      that matrix-norm improvements and `V_max` improvements are not the same
      thing, and can point in opposite directions.
- [ ] **Preconditioner.** Attacks the iteration count directly (236 at
      Δz = 50 m, ~450 extrapolated at Δz = 20 m). Deliberately not shipped
      unvalidated: a diagonal preconditioner preserves the `range(A)` invariant
      CG relies on here only if it commutes with `P`, i.e. if its entries agree
      within each merged fault pair. Check that before assuming it is safe —
      `CGSolver`'s docstring has the argument.
- [ ] **Decide between preconditioned/multigrid CG on this discretization and
      the boundary-integral route** most SEAS codes take (whole-space
      fault-to-fault kernel is a convolution, `O(N log N)` with FFTs, no volume
      unknowns — and no `K` build, which is the part that scales worst here).
      This is a design decision, not an increment.
- [ ] Keep `scripts/split_node_spd.jl`'s guardrails in mind when touching any
      of this: CG's internal recursive residual is only valid for symmetric
      `A`, and CG run on `A[keep,keep]` (neither reduction) converges happily
      to the wrong answer.

## 3. Smaller items

- [ ] **Order 6.** Needs Mattsson's order-6 `D1`/`D2` coefficients added to
      Diffinitive's `standard_diagonal.toml` — an upstream contribution, and a
      nontrivial derivation to get right. Blocks any higher-order convergence
      study.
- [ ] **Variable coefficients.** The notebook's operators take λ, μ as grid
      functions; this package is constant-coefficient only. Fine for BP8's
      homogeneous whole space, and it is what lets the factorization be reused
      across every RHS evaluation — but a foreclosed capability.
- [ ] **Peaceman σ̄ < 0** (`PROGRESS.md` limitation 3): pressure at the well
      cell exceeds σ, the `σ̄_min` floor binds, and eq 3's no-opening condition
      stops applying there. Not a numerical artefact — worsens as Δz shrinks.
- [ ] **Peaceman well-cell pressure ~10% below eq 25** (limitation 4). The
      `r_e = 0.198Δz` calibration is for steady radial flow; this is transient.
- [ ] **CRESCENT DET upload** (§5): files are written in the §4 formats but
      nothing has been validated against the server's parser. Gated on item 2
      anyway — the current runs are not submission-ready.

# TODO

Open work, in dependency order. Background for the symmetry items is in
`SYMMETRIC_SAT.md`; the rest is from `PROGRESS.md`'s "Known limitations".

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
- [x] **Diagnose "order 6 is unusable".** Broader than recorded: order 6 in
      `standard_diagonal.toml` has only `H`, `e`, `d1` and `D2.positivity` —
      no `D1` at all and no `D2` stencils. Upstream Diffinitive data gap, not
      fixable here. Recorded as `PROGRESS.md` limitation 5.

## 1. Act on the domain-convergence result — `L_normal = 400` is not innocent

The sweep has been run; all three fault-normal rows fit (`L_normal = 800` is
111k DOF, and its factorization completing is partly Cholesky paying off).
Against `L_normal = 800`, the shipped `L_normal = 400` is off by 0.02% in
`K_self`, 1.75% in slip and **42% in `V_max`**, with exactly the predicted
signs. It is also not converged at 800 m — 600→800 still moves `V_max` 11.6%.
Full table in `PROGRESS.md` "Results".

So the answer to the original question is *no*: the halving was not free for
peak `V`. It is free for slip and stiffness.

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
process zone `L_b ≈ 64 m` against the benchmark's 10 m / ~6 cells.

Everything cheap has now been taken and the accounting is not close:

| | effect on Δz |
|---|---|
| Cholesky (done) | 50 m → ~45 m |
| CG (converges, 73 its) | lifts the memory ceiling, but *slower* here — break-even 6-15 RHS, `fault_stiffness` needs 578 |
| needed | 50 m → 10 m |

- [ ] Decide between an iterative/multigrid solve of the same SBP-SAT system
      and the boundary-integral route most SEAS codes take (whole-space
      fault-to-fault kernel is a convolution, `O(N log N)` with FFTs, no volume
      unknowns). This is a design decision, not an increment.
- [ ] If CG is ever adopted: thread the `K` build — its columns are
      embarrassingly parallel, which is precisely what a sparse direct solve
      cannot exploit. And keep `scripts/split_node_spd.jl`'s guardrails: CG's
      internal recursive residual is only valid for symmetric `A`, and CG run
      on `A[keep,keep]` (neither reduction) converges happily to the wrong
      answer.

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

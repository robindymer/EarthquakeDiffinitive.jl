# TODO

Open work, in dependency order. Context for items 1-4 is in
`SYMMETRIC_SAT.md`; the rest is from `PROGRESS.md`'s "Known limitations".

## Done (this session)

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

## 0. Run the test suite — the `src/` edit is UNVERIFIED

`Pkg.test()` was started and cancelled before producing output, so the edited
`traction_blocks` and the new SBP test have **not** been exercised.

- [ ] `julia --project=. -e 'using Pkg; Pkg.test()'` — expect it to be slow;
      `elasticity_split_node_test.jl` and `bp8_test.jl` build and factorize 3D
      systems.
- [ ] Quicker gate if you want a signal first — just the two files that matter:
      `julia --project=. -e 'using Pkg; Pkg.activate("."); include("test/elasticity_test.jl")'`
      for the new SBP property test, then `test/elasticity_split_node_test.jl`
      for the interface conditions.

The supporting evidence that *should* make these pass, from before the edit:
`scripts/verify_notebook_traction.jl` showed every interface condition holding
at machine precision with this traction operator (n = 11 and 15), and
`scripts/symmetry_decomposition.jl` showed `-HP(D+SAT)P` going from 1.44e-01 to
7.60e-17. Both assembled the operator inside the script, though — not via the
edited `src/`.

Two things that could plausibly break and are worth looking at first if the
suite fails:

- `test/elasticity_test.jl`'s `"traction_blocks matches manufactured traction"`
  compares against the analytic `σ_i,dim` with a 5% tolerance. The boundary
  derivative is a different (probably more accurate) approximation of `∂ₙ`, but
  the tolerance is loose and the check is at order 4, so this should hold.
- `Grids._boundary_sign` is private Diffinitive API. It exists in the current
  dev checkout (`src/Grids/tensor_grid.jl:162`) and the notebook uses it the
  same way, but it is not covered by any compat guarantee.

## 1. Verify the traction fix against production runs — BLOCKING

`traction_blocks` also feeds `FaultResponse`, so `Δτ`, `K`, peak `V` and every
number in `output/` shift. Nothing downstream should be trusted until this is
checked.

- [ ] Re-run `scripts/run_bp8.jl` for both injection models at Δz = 50 and
      100 m and diff against the table in `PROGRESS.md` "Results".
- [ ] Confirm the shift is within the documented resolution spread (peak `V`
      already moves an order of magnitude between Δz = 50 and 100 m, so that is
      the yardstick).
- [ ] Re-check `K`'s symmetry at *production* domain size. Measured 2.8-3.3% at
      `L=1, n=11..15`, slightly worse than before the fix — but that domain is
      severely truncated, and `PROGRESS.md` records 0.19% at production size.
      Expect the fix not to improve it: `K`'s reciprocity needs `χ` and `T` to
      be mutually adjoint, which is a separate condition from `A`'s symmetry.
- [ ] Re-run `scripts/bp8_validate_pressure.jl` (should be unaffected — pore
      pressure does not touch elasticity — so it doubles as a control).

## 2. Galerkin reduction + Cholesky — free 2×

`A` is now symmetric, and `SᵀAS` is SPD (0/2205 negative eigenvalues, κ = 118).

- [ ] Add the Galerkin reduction `SᵀAS` to `ElasticitySplitNode`. Required:
      `factorize_reduced`'s current `E·A·S` sums columns but only *selects*
      rows, so it is not a congruence transform and stays asymmetric even for
      symmetric `A`. The prolongation `S` is already written as
      `prolongation(rs)` in `scripts/split_node_spd.jl`. Solving `SᵀAS`
      directly agrees with the current LU path to 2.6e-15.
- [ ] Switch `factorize_reduced` from `lu` to `cholesky`. Measured factor
      nonzeros, notebook traction: 1.88-1.93× smaller than LU at n = 13/17/21
      (297M → 155M at n=21, i.e. ~2.4 GB → ~1.24 GB). Cholesky *fails* on the
      pre-fix system, which is an independent confirmation it was not SPD.
- [ ] Temper expectations: fill-in scales ≈ `n^5.6` on the measured points, so
      1.9× memory buys only ≈ 1.12× finer grid — Δz 50 m → ~45 m. Free, but
      not a path to the benchmark's 10 m.

## 3. CG — optional, only if Cholesky will not fit

- [ ] CG converges in 73 iterations (true residual 9.0e-11) on `SᵀAS` after the
      fix, versus stalling at 3.7e-02 after 5000 before it.
- [ ] Note the economics: break-even against the amortized direct solve is
      ~2 right-hand sides and `fault_stiffness` needs `2·N_Ωf` ≈ 578, so CG is
      *slower* at current sizes. Its only advantage is memory.
- [ ] `K`'s columns are embarrassingly parallel, so if CG is adopted the `K`
      build should be threaded — that is what a sparse LU cannot do.
- [ ] Keep the guardrails in `scripts/split_node_spd.jl`: CG's internal
      recursive residual is only valid for symmetric `A`, and CG run on
      `A[keep,keep]` (neither reduction) converges happily to the wrong answer.

## 4. Resolution — the actual blocker

`PROGRESS.md` "Known limitations" 1 and 2. Δz = 50 m gives 1.3 cells per
process zone `L_b ≈ 64 m` against the benchmark's 10 m / ~6 cells.

- [ ] Decide between an iterative/multigrid solve of the same SBP-SAT system
      and the boundary-integral route most SEAS codes take (whole-space
      fault-to-fault kernel is a convolution, `O(N log N)` with FFTs, no volume
      unknowns). Item 2 buys ~10%, item 3 somewhat more; neither reaches 10 m.

## 5. Domain-size validation gap

- [ ] `scripts/bp8_domain_convergence.jl` sweeps `L_normal ∈ {800, 1200}` but
      the production runs use **`L_normal = 400`** (halved for memory), so the
      closest truncation boundary in the setup sits *outside* the validated
      range. Add `(L_fault=800, L_normal=400)` and `(800, 800)` rows and
      compare. `u=0` makes the medium artificially stiff, so the bias is known:
      slip too small, `|K_self|` too large.

## 6. Smaller items

- [ ] **Reproducible environment.** The `Diffinitive` dev-link lives only in
      the gitignored `Manifest.toml`; a fresh clone resolves the registry
      version and fails to precompile
      (`invalid subtyping in definition of IsotropicElasticOperator`). Add a
      `[sources]` entry to `Project.toml`.
- [ ] **Order 6 is unusable.** `first_derivative` throws `KeyError: "D1"` for
      `order=6` in `standard_diagonal.toml` — only orders 2 and 4 have the key.
      Blocks any higher-order convergence study.
- [ ] **Variable coefficients.** The notebook's operators take λ, μ as grid
      functions; this package is constant-coefficient only. Fine for BP8's
      homogeneous whole space, but a foreclosed capability.
- [ ] **Peaceman σ̄ < 0** (`PROGRESS.md` limitation 3): pressure at the well
      cell exceeds σ, the `σ̄_min` floor binds, and eq 3's no-opening condition
      stops applying there. Not a numerical artefact — worsens as Δz shrinks.
- [ ] **Peaceman well-cell pressure ~10% below eq 25** (limitation 4). The
      `r_e = 0.198Δz` calibration is for steady radial flow; this is transient.
- [ ] **CRESCENT DET upload** (§5): files are written in the §4 formats but
      nothing has been validated against the server's parser. Gated on item 4
      anyway — the current runs are not submission-ready.

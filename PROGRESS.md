# SEAS BP8-QD-GS/-PW progress

Status of the `EarthquakeDiffinitive` implementation of the SEAS
Benchmark Problems BP8-QD-GS and BP8-QD-PW
(`context/SEAS_BP8_Benchmark_Description.pdf`): a quasi-dynamic 3D
whole-space, rate-and-state fault, driven by fluid injection modelled either
as a Gaussian source or a Peaceman well. Built on the `Diffinitive` SBP-FD
library, pinned in `Project.toml`'s `[sources]`.

**Both injection models run end to end** and write the §4 output files; see
"Results" below. The implementation is complete and validated, but the runs
are **not resolution-converged** and are not submission-ready — the reason is
a hard memory limit on the 3D elastic factorization, quantified under "Known
limitations".

The discrete operator `A = -HP(D+SAT)P` is now **symmetric positive definite**
(it was ~14% asymmetric), the reduction is the Galerkin form `SᵀAS`, and the
solve is a Cholesky factorization. Full suite green at 176/176. That closed the
project's main correctness question; it did **not** move the resolution ceiling
in any material way, which remains the one open blocker.

## Done

### Environment
- **`Diffinitive` is now pinned in `Project.toml`'s `[sources]`**, to upstream
  `Diffinitive/Diffinitive` at `92d842cf`. Previously the dev-link lived only in
  the gitignored `Manifest.toml`, so a fresh clone — or any bare
  `Pkg.instantiate()` after the Manifest was lost — resolved `Diffinitive` from
  the registry instead and failed to precompile with
  `invalid subtyping in definition of IsotropicElasticOperator` (the registry
  v0.1.8 `LazyTensor` is not the dev checkout's). Verified by instantiating a
  copy of `Project.toml` + `src/` with no Manifest: it resolves the pinned
  commit, precompiles, and assembles a symmetric operator (7.6e-17).
  `92d842cf` is exactly what the working checkout at `~/.julia/dev/Diffinitive`
  has (its `src/` is clean against that commit); note this is *not* the fork at
  `~/Kod/Diffinitive`, which is a different repo and is not what was in use.
  An explicit `Pkg.develop(path=...)` still overrides `[sources]`, so the local
  dev workflow is unaffected — the existing Manifest still points at the dev
  path.
- `Diffinitive`'s own docs now build against the dev checkout
  (`docs/Project.toml` `[sources]`), with two doctest bugs fixed. One
  remaining bug (`zero(::VolumeOperator)` missing method, breaks a docs
  example) was left for the project owner to fix.
- `.claude/settings.json` allowlists the Julia/git commands this project
  keeps needing.

### Pore pressure diffusion (`src/PorePressure.jl`)
The fault's 2D pore-pressure diffusion equation (PDF §2.1, eq. 17-25). It
feeds elasticity one way, through `σ̄ = σ - p`; nothing flows back.

- `pore_pressure_operator`: assembles the constant (homogeneous-medium)
  Neumann-SAT Laplacian on the fault plane, reusing Diffinitive's built-in
  `Laplace`/`NeumannCondition`/`sat_tensors` machinery directly. Also takes a
  grid directly, so the coupled driver can build it on the elastic solver's
  own `Ω_f` nodes and share one index between `p` and `s`.
- `gaussian_source`, `injection_rate`: the eq. 19/20 forcing.
- `well_index`, `well_cell_index`, `peaceman_cell_volume`: the Peaceman well
  model's coefficients (§2.1.2), `WI = 2πkL_fwid/(η ln(r_e/r_well))` with
  `r_e = 0.198Δz` for a centred five-point stencil.
- `darcy_operators`: `q_j = -(k/η)∂p/∂x_j` (eq. 16), for the benchmark's
  `darcy_vel_2`/`darcy_vel_3` output columns.
- `solve_pore_pressure`: implicit (`OrdinaryDiffEq`) time integration of the
  standalone problem; the operator matrix is constant, so it's built once.
- Validated in `test/pore_pressure_test.jl` against the closed-form
  analytic solution (PDF eq. 21) and via grid-refinement convergence, and
  again inside the coupled model — see "Results".

### Elastic operator building blocks (`src/Elasticity.jl`)
Homogeneous isotropic elasticity (PDF §1) relating fault slip to fault
traction.

- `elastic_operator`/`elastic_blocks`: the constant-coefficient Navier
  operator, matching `context/notebooks/elastic.jl`'s discretization choice
  term-for-term rather than just its continuous-level simplification: the λ
  (Hessian-of-divergence) terms always use the "wide" sandwich `Dᵢ∘Dⱼ`
  (including the diagonal — the notebook is explicit this avoids dispersion
  mismatch between the λ- and μ-driven parts of the operator, and that
  still applies with constant coefficients), while only μ's diagonal uses
  the compact "narrow" native `second_derivative` (a physically separate
  shear-Laplacian piece). An initial version collapsed λ and μ's diagonal
  terms together (valid only by commuting partial derivatives at the
  *continuous* level) and used narrow throughout — caught before it caused
  real damage. `IsotropicElasticOperator` also has hand-specialized,
  allocation-free 2D/3D `apply` methods (mirroring the notebook's own
  non-allocating specializations, needed because Diffinitive's generic
  D-dimensional `apply` doesn't infer well) — confirmed 0 bytes/point for a
  full in-place `E*u`. Used directly by `scripts/elastic_wave_2d.jl` (a 2D
  elastic wave simulation validating the operator qualitatively — clean
  P/S wavefront separation at the right speed ratio, correct non-circular
  S-wave radiation pattern, stable reflections).
- `traction_blocks`: extracts fault traction (σᵢ,dim, fixed `+dim`-axis
  convention, not outward-normal) from a solved displacement field. **The
  normal-direction μ terms use the boundary derivative `s·normal_derivative`,
  not `e∘first_derivative`** — they come from the *narrow* `second_derivative`,
  whose SBP identity carries a different boundary operator than the wide
  sandwiches λ and μ's tangential terms come from. Getting this wrong is what
  made `-HP(D+SAT)P` ~14% asymmetric; see `SYMMETRIC_SAT.md`. The
  `"SBP property: E and T are compatible"` test is what pins it, and is the
  test that would have caught the bug immediately had it existed.
- `inject_dirichlet!`, `dof_index`, `flatten`/`unflatten`: strong-injection
  and vector-grid-function utilities (no Diffinitive helper exists for
  strong Dirichlet row-injection).
- Validated in `test/elasticity_test.jl` (the SBP compatibility identity above,
  polynomial exactness, sparse vs. `LazyTensor` consistency, traction accuracy
  against a manufactured field).

**This module used to also provide `halfspace_system`**, a single-grid
reduction of the fault problem to x1≥0, derived from an (apparently)
reflection symmetry of the whole-space problem. **That derivation had a
real physics bug**, found while cross-validating the newer split-node
solver against it: it assumed the tangential displacement gradients
(∂2u2, ∂3u3) vanish at the fault, which is only true for *spatially
uniform* slip. Confirmed numerically: the old solver produced a
normal-stress perturbation at the fault of the same order of magnitude as
the shear tractions for a Gaussian slip profile.

**Correction (this review):** an earlier version of this note read that
nonzero σ11 as the real physics the half-space BC had dropped. It is the
other way round. The mirror symmetry x1 → -x1 (u1 even, u2/u3 odd) makes
σ11 *odd* across the fault, and eq. 6a requires it *continuous*, so
**σ11 ≡ 0 on the fault** — achieved by a nonzero ∂1u1 exactly cancelling
the λ(∂2u2+∂3u3) term. Forcing ∂1u1=0 left σ11 ≈ λ(∂2s2+∂3s3)/2, so the
spurious normal stress was the symptom of the bad BC, not the physics it
was missing. The removal was right regardless, and σ11 → 0 is now a
regression test — the one that caught the split-node SAT sign error below.

MMS testing never caught the original bug because MMS validates that the
discretization correctly solves the equations *as written*, not that those
equations match BP8's actual physics — a good reminder that
discretization-accuracy tests and physical-model-correctness are genuinely
different questions.
`halfspace_system` has been removed; `ElasticitySplitNode` (below) is now
the standard approach. The building blocks above were unaffected — the
error was specifically in the fault boundary condition, not the operators.

### Two-sided split-node elastic solver (`src/ElasticitySplitNode.jl`)
Implements `context/SEAS_benchmark.pdf`'s reference formulation exactly:
`-HP(D+SAT)P u = HP(D+SAT)χ(s)`, two separate grids (`g_minus`/`g_plus`)
glued at the fault via a genuine interface SAT, rather than the (flawed)
half-space shortcut.

- **D**: block-diagonal, reusing `Elasticity.elastic_blocks` per side
  unchanged.
- **SAT**: interface traction coupling — generalizes the scalar Neumann-SAT
  pattern (`-H⁻¹∘e'∘Hᵧ`) by swapping the scalar normal-derivative for the
  vector traction operator (`Elasticity.traction_blocks`). Verified against
  the literature this session (Duru & Virta, JCP 2014, is the foundational
  reference for exactly this construction) — standard, no elasticity-specific
  subtlety beyond building the traction operator from Diffinitive's
  SBP-compatible `first_derivative`, which `traction_blocks` already does.
  Applied half-weighted to *both* sides (the note also allows applying it
  fully to one side, "since P averages it out" — full-weight-one-side
  produced a *more* asymmetric, non-PSD system empirically than
  half-weighted-both-sides, so used the latter — though, per below,
  neither is the exact symmetric construction the reference note assumes).
  **The `+` side originally had the wrong sign** (fixed in this review):
  the Neumann/traction SAT is written in terms of the *outward-normal*
  traction — in Diffinitive's own `sat_tensors(::NeumannCondition)` the
  `-H⁻¹∘e'∘Hᵧ` prefactor is side-independent and all side-dependence lives
  in `normal_derivative`'s outward sign. `traction_blocks` is
  fixed-`+x₁`-axis, so it *is* the outward traction on `g_minus` (fault =
  its upper boundary) but its negative on `g_plus` (lower boundary). Using
  one sign for both left the shear tractions discontinuous and inflated
  the solution by ~200×.
- **P**: sparse projection — averages **all three** (u1,u2,u3) fault DOF
  pairs, zeroes far-field DOFs, identity elsewhere. Confirmed symmetric and
  idempotent. Averaging `u1` is not decoration: it *is* how the no-opening
  condition u1(0⁺)=u1(0⁻) (eq. 3) is imposed, and it is what the reference
  note prescribes ("for j = 1, 2, 3"). An earlier version averaged only
  j=2,3, leaving the two fault-normal DOFs with nothing coupling them; the
  solve then opened the fault by twice the field amplitude. Nodes on the
  ring where the fault meets the truncation boundary are excluded from the
  pairing (`fault_node_pairs`) so they keep their far-field Dirichlet —
  previously the pairing loop ran after the far-field zeroing and silently
  overwrote it for j=2,3.
- **χ(s)**: forcing vector, `±s_j/2` at the fault for j=2,3 — same
  convention the (now-removed) half-space solver used as Dirichlet data, a
  good independent consistency signal that survived the rewrite.
- `A = -HP(D+SAT)P` is **singular by construction** (P's null space —
  far-field DOFs and the antisymmetric half of each tangential pair — is
  ~40% of the system). The reference note itself flags this and suggests
  CG; plain GMRES stalled well short of convergence (large null space).
  `reduced_solve` instead eliminates the redundant/zero DOFs directly from
  P's own sparsity structure (regular DOFs kept, far-field dropped,
  tangential pairs merged into one unknown) and solves the resulting
  genuinely non-singular system directly — robust where the Krylov
  approach wasn't. (Hit a classic Julia gotcha along the way: `P[i,:] .= 0.0`
  leaves *explicit* zeros stored in the sparsity pattern rather than
  removing them, silently breaking a `nzrange`-based "is this DOF free"
  check until `dropzeros!`ed.)
  The reduction is the **Galerkin form `SᵀAS`** — the congruence transform,
  built with the exported `prolongation(rs)` — and it is **Cholesky**-
  factorized. Both of those depend on the traction fix below; see "Symmetry:
  root cause found and fixed".
- **Symmetry: root cause found and fixed.** `A` used to be ~14% asymmetric, so
  the reference note's assertion that it is SPD did not hold and CG was
  unusable. `scripts/split_node_spd.jl` remains the standing diagnostic; run it
  before revisiting any of this. Measured on cubes of side `n` (order 4), before
  and after the traction fix:

  | n | reduced DOF | `A` before | `A` after | `E·A·S` after | `SᵀAS` after | CG before | CG after |
  |---|---|---|---|---|---|---|---|
  | 9 | 2205 | 0.144 | **8.5e-17** | 0.109 | **8.5e-17** | stalls, res 3.7e-2 | **73 its, 9.1e-11** |
  | 13 | 8349 | 0.113 | **9.8e-17** | 0.087 | **9.8e-17** | stalls, res 3.2e+0 | **108 its, 8.7e-11** |
  | 17 | 20925 | 0.097 | — | — | — | — | — |
  | 21 | 42237 | 0.086 | — | — | — | — | — |

  `bulk(A)` was always at round-off (6.4e-17 at n=9, 4.8e-17 at n=21), which is
  what localized the defect to the boundary rather than to `elastic_blocks`.

  Five things follow, in order of how much they changed the plan.

  1. **Definiteness was always fine; symmetry was the whole problem.** The
     symmetric part of `SᵀAS` has **0 negative eigenvalues** out of 2205, κ ≈ 200
     before the fix and **118** after. An earlier note here claimed "~19%
     asymmetry and 3 negative eigenvalues out of 2205"; the 3 do not reproduce
     and were probably measured before the `u1`-averaging and fault-edge-ring
     fixes.
  2. **The asymmetry decayed under refinement**, 0.144 → 0.086 over n = 9 → 21,
     roughly `h^0.56` — so it was a *consistency-order* defect, not a structural
     error. That was reassuring for accuracy and useless for CG, which needs
     symmetry at the resolution actually being run. It is now at round-off at
     every `n`, which is a stronger statement than convergence.
  3. **`factorize_reduced`'s reduction was nonsymmetric by construction**, and
     this was the second, independent defect. It built `E·A·S` — columns summed
     over merge pairs, rows merely *selected*. That is not a congruence
     transform. Since `P` makes each merged pair's two rows identical, a
     *perfectly symmetric* `A` still gives an `E·A·S` with merged columns
     doubled and rows not. The measurement above is the proof: after the
     traction fix `asym(A) = 8.5e-17` while `asym(E·A·S) = 0.109`. **So making
     the SAT symmetric was not enough on its own — two changes were needed, not
     one.** `factorize_reduced` now builds the congruence transform `SᵀAS`; the
     two reductions differ only by a factor of 2 on merged rows and their
     solutions agree to 2.7e-15, an independent confirmation that the old path
     was correct all along.
  4. **Cholesky replaces LU, for free.** `SᵀAS` is SPD once symmetric, and its
     Cholesky factor has **1.77×/1.88× fewer nonzeros** than the LU factor at
     n = 9/13 (10.6M vs 19.9M at n=13). Fill-in is the binding resource
     ("Known limitations" 2), so this is the cheapest memory available. Cholesky
     *fails* on the pre-fix system, which is an independent confirmation it was
     not SPD — and is why `factorize_reduced`'s fallback to LU warns loudly
     rather than degrading silently.
  5. **CG still would not pay off**, even though it now converges (73 iterations
     at n=9, true residual 9.0e-11, agreeing with the direct solve to 7.6e-11).
     The factorization costs one setup then ~0.008-0.099 s per
     back-substitution; CG costs ~0.024-0.19 s per right-hand side with no
     amortization, so break-even is **6-15 right-hand sides** and
     `fault_stiffness` needs `2·N_Ωf` ≈ 578. CG's real attraction is memory —
     it would lift the fill-in ceiling entirely — but the `K` build then becomes
     the bottleneck. Its columns are independent, so that build threads cleanly,
     which a single sparse factorization does not.

  **ROOT CAUSE — see `SYMMETRIC_SAT.md` for the full diagnosis.**
  `scripts/symmetry_decomposition.jl` takes the operator apart factor by
  factor. The defect reproduces on a *single grid with a plain free surface*
  — no interface, no projection — so none of the split-node machinery is
  implicated. Within the elastic operator, λ-only is exactly symmetric
  (1.67e-16) while μ-only is not (3.12e-01).

  The mechanism, isolated on the scalar Laplacian with one `H` throughout:
  Diffinitive gives two different approximations of `∂u/∂n` at a boundary —
  `e∘first_derivative` and `normal_derivative` — differing by
  `‖D̂ - e·D₁‖ ≈ 160` at order 4. The **narrow** `second_derivative`'s SBP
  identity expresses its boundary term in `normal_derivative`; the **wide**
  sandwich's in `first_derivative`. `traction_blocks` uses `first_derivative`
  for everything, so it cancels the wrong operator against the narrow μ
  pieces, and the leftover is the asymmetry. λ uses only wide sandwiches,
  which is why it was already exact.

  **This project's own reference notebook already gets this right.**
  `context/notebooks/elastic_clean.jl`'s `IsotropicTractionOperator` uses
  `first_derivative` for the λ terms and **the boundary derivative
  (`s·normal_derivative`) for μ's diagonal terms**, with comments saying
  exactly why. `traction_blocks` uses `e∘first_derivative` for every term and
  dropped the distinction — despite `src/Elasticity.jl` claiming to follow the
  notebook "term-for-term" on the *volume* operator, which it does.
  (`IsotropicTractionOperator` is not used anywhere in `src/`, `test/` or
  `scripts/` — the package reimplemented it as `traction_blocks`.)

  Literature agrees: Almquist & Dunham (arXiv:2003.12811) §4 call Mattsson's
  operators — what `standard_diagonal.toml` implements — *compatible* but
  **not fully compatible** (`e₀,NᵀD₁ ≠ e₀,NᵀD̂`); eq (147) gives the resulting
  `T = n·C·(D + ΔD)` and they note that proving stability in that case
  *remains an open problem*. Their remedy (eq 67) attacks it from the other
  side — adapt `D₂` so its boundary term is expressed in `D₁`.

  Both were prototyped in the script (n=9, order 4), after verifying the
  rebuilt operator matches `elastic_blocks` bit-for-bit:

  | | before | notebook traction | eq (67) |
  |---|---|---|---|
  | what changes | — | `traction_blocks` | `second_derivative` |
  | accuracy cost | — | **none** | one order at boundary point |
  | single grid `H*(D+SAT)` | 2.49e-01 | **1.15e-16** | 1.07e-16 |
  | `A = -HP(D+SAT)P` | 1.44e-01 | **7.60e-17** | 7.87e-17 |

  **The notebook's fix is the one taken** — same symmetry, no accuracy given
  up, and it makes `traction_blocks` agree with the reference implementation.
  Eq (67) would degrade the operator at exactly one grid point per boundary,
  and the fault *is* a boundary. It is in `src/Elasticity.jl`.

  Downstream, once symmetric: Galerkin `SᵀAS` asymmetry 1.98e-01 → 8.5e-17,
  **0/2205 negative eigenvalues**, κ 1.99e+02 → 1.18e+02, Cholesky admissible
  (1.9× less fill-in than LU), and **CG converges in 73 iterations** (true
  residual 9.0e-11) where it previously stalled at 3.7e-02 after 5000.

  **Both gates have passed.** (1) `test/elasticity_split_node_test.jl`'s
  interface conditions still hold — jumps and traction continuity at machine
  precision, σ₁₁ → 0 preserved — and the full suite is green (176/176).
  (2) `traction_blocks` is also what `FaultResponse` uses for physical `Δτ`, so
  `K` and peak `V` shifted; all four production runs were repeated and the shift
  is 1-2 orders of magnitude *inside* the resolution spread, while making the
  two resolutions agree better than before. Numbers in "Results" below and in
  `SYMMETRIC_SAT.md` "Gate 2".

  Two earlier claims in this file were wrong and are retracted: that a
  symmetric SAT needs *two* term types (that applies to Almquist & Dunham's
  *displacement* conditions, §5.2 — not our case, where `P` handles continuity
  and the SAT carries traction only), and that the wide cross-terms' extra μ
  contribution is a separate defect needing its own fix.

  **Trap for anyone re-testing this.** CG's internally-updated residual
  `r ← r - αAp` is only valid for `A = Aᵀ`, so on this system it reports
  convergence it has not achieved; always recompute `‖b-Ax‖/‖b‖` from
  scratch. And CG run on `A[keep,keep]` — plain row *and* column selection,
  which is neither reduction — converges happily in ~160 iterations to a
  vector 4.5% away from the true solution. Both mistakes were made while
  writing the script above, and both look like success.
- Validated in `test/elasticity_split_node_test.jl` two ways.
  1. An algebraic self-consistency check (deliberately avoiding another
     continuum-PDE derivation after the half-space episode): pick an
     arbitrary smooth, localized two-sided field, derive the exact forcing
     that makes it a solution *directly from the assembled discrete
     operator itself*, then confirm solving recovers it — this tests D,
     SAT, P, χ, and the solve together without assuming anything about the
     true physics. (Its manufactured field must have continuous `u1`, since
     `P*U + χ = U` only holds then; that is eq. 3, not a test artifact.)
  2. **Physics-level checks against BP8's interface conditions**, added
     after the self-consistency test was found to be blind to both bugs
     above — by construction it validates D/SAT/P/χ against *each other*,
     which is exactly the class of error that killed `halfspace_system`.
     For a prescribed Gaussian slip patch: no opening (eq. 3), the jump
     really is the imposed slip (eq. 4), σ21/σ31 continuous across the
     fault (eq. 6b,c), the antisymmetric split max‖U‖ = max|s|/2, and
     σ11 → 0 under refinement. Current numbers: jumps and shear-traction
     continuity at machine precision, max‖U‖ = 0.5 exactly for unit slip
     at every resolution, and max|σ11| falling 0.264 → 0.068 for n = 9 →
     21 while max|σ21| converges to ~3.03.

### Rate-and-state friction + aging law (`src/RateStateFriction.jl`)
The fault friction law (PDF eq. 8-13) as a standalone, pointwise physics
kernel — not yet wired to the elastic solver's `Δτ` or pore pressure's
`σ̄` (that's the next item).

- `friction_coefficient`/`fault_strength`/`aging_law_rhs` implement the
  regularized friction coefficient (eq. 12), fault strength `F=σ̄f(V,θ)`
  (eq. 10), and the aging law `dθ/dt=1-Vθ/D_RS` (eq. 11) directly.
- `solve_slip_rate`: the force balance `τ=τ⁰+Δτ-ηV=F(V,θ)` (eq. 8-10)
  combines into a vector equation where both the radiation-damping and
  friction terms are parallel to `V` — so the trial stress `τ⁰+Δτ` must
  itself be parallel to `V`, meaning direction comes for free and only the
  *magnitude* needs solving: the scalar equation `T=ηV+σ̄f(V,θ)`, not
  analytically invertible (`V` appears both linearly and inside `asinh`).
  Solved via Newton's method in `x=ln V` (closed-form derivative, no
  `ForwardDiff` needed) — robust across the ~10 decades of `V` the
  benchmark spans, since `f(V,θ)` is smooth and strictly increasing in `V`.
  Steps are **clamped in log space** and non-convergence now throws.
  Undamped, the iteration diverges whenever it is seeded below the root at
  `V ≳ 1 m/s`: there `asinh(u) ≈ ln(2u)` is nearly linear in `x`, so the
  step overshoots by many decades and `exp(x)` overflows, silently
  returning `NaN` or a value wrong by ~1e18 (30 of 315 sweep points before
  the fix, 0 after). BP8-QD-GS is velocity-strengthening and should never
  reach that regime, but a warm start during acceleration always
  approaches from below, so it stayed reachable in a time loop.
  No new dependency: the existing `NonlinearSolve` stack is only a
  transitive dep of `OrdinaryDiffEq`, and this is a small enough, well-behaved
  root to hand-roll instead.
- `solve_slip_velocity`: the actual vector eq. 8-10, combining direction
  (∝ `τ0+Δτ`) with `solve_slip_rate`'s magnitude — what the future time
  integration will call directly.
- `initial_state_from_strength`: unlike the forward solve, inverting eq. 12
  for `θ` given `V` and a target strength (eq. 28-29's initial-condition
  setup) *is* analytically closed-form — no iteration needed.
- Validated in `test/rate_state_friction_test.jl` against Table 1's actual
  physical parameters (not simplified O(1) values, unlike the elasticity
  tests — rate-and-state parameters are naturally at these physical scales
  regardless): aging-law steady state, strength monotonicity in `V`,
  forward round-trips (with and without radiation damping, swept across
  ~10 decades of `V`), `solve_slip_velocity`'s direction/magnitude, and the
  eq. 28-29 initial condition's self-consistency, plus regression tests for
  the seeded-from-below Newton divergence, the non-convergence error, and
  `solve_slip_velocity` returning `V=0` (not `NaN`) at zero trial stress.
  Sanity check at Table 1 values: θ₀ = 4.02e11 s, far above
  θ_ss = D_RS/V_init = 5e8 s, consistent with τ_init = 14.6 MPa sitting
  above τ_ss = 12.93 MPa — the fault starts over-healed, as an
  injection-triggered benchmark requires.

### Slip → traction map (`src/FaultResponse.jl`)
Turns the split-node solver into the operator the friction law needs.

- `boundary_selection`: fault node ↔ global DOF map. Deliberately *not*
  `boundary_indices`, whose iteration order matches neither the boundary grid
  nor `traction_blocks`' row order (checked: it doesn't). Read straight off
  `boundary_restriction`, which is what the traction blocks are built from.
- `FaultElasticity`: assembles, factorizes once (`factorize_reduced`), and
  identifies the `Ω_f` nodes, ordered x2-fastest to match the pore-pressure
  grid so slip, state and pressure share one index.
- `fault_stiffness`: the dense `K : [s2;s3] ↦ [Δτ2;Δτ3]`, built with `2·N_Ωf`
  back-substitutions. This is what makes the 30-day run tractable — an
  adaptive integrator calls the RHS ~10⁴ times, and one sparse back-solve of a
  3D system per call would dominate everything. Slip being confined to `Ω_f`
  (eq. 13) is what keeps the column count affordable.
- **Sign convention, verified numerically**: `Δτ_j = Δσ_j1` with no flip. A
  positive slip patch gives `Δσ21 = -2.89` at its centre for unit slip, i.e.
  slip relieves the stress driving it. `diag(K) < 0` is a regression test;
  getting it backwards turns the coupled system into a runaway.
- `K` comes out symmetric to **0.20%** at production size (0.2003% at Δz = 50,
  L = (800, 400); 0.1827% at Δz = 100, L = (800, 800)), which it should be by
  reciprocity. This is essentially unchanged by the traction fix — it was 0.19%
  before — and that is the expected outcome, not a loose end: reciprocity
  requires the *forcing* operator `χ` and the *extraction* operator `T` to be
  mutually adjoint, which is a **separate condition** from `A = Aᵀ`. Changing
  `T` alone shifts that balance rather than closing it. What remains of the
  0.20% is dominated by domain truncation, which is why it tracks domain size
  (2.8-3.3% at the deliberately tiny `L=1, n=11..15` test domain).

### Coupled benchmark driver (`src/BP8.jl`)
The full BP8-QD-GS/-PW problem: `ds/dt = V`, `dθ/dt = 1 - Vθ/D_RS`,
`dp/dt = α∇²p + source`, closed at each node by the algebraic force balance
`τ⁰ + Δτ(s) - ηV = σ̄f(V,θ)V/V`.

- `BP8Params`: Table 1 verbatim, with `λ`, `η`, `q0` derived. Checked in tests
  that `ν=0.25 ⇒ λ=μ`, `√(μ/ρ)=c_s`, and `k/(φβη)=α=0.05 m²/s`.
- State is integrated as **ϕ = ln θ**: θ spans ~1e8–1e12 s while slip is
  ~1e-6 m, so no single scalar tolerance serves both. Turns the aging law into
  `dϕ/dt = e^{-ϕ} - V/D_RS`. Per-block absolute tolerances on top of that.
- Initial conditions (eq. 26-29) reproduce `V_init = 1e-12` and
  `V_zero = 1e-20` to 8 digits, with `‖τ‖ = τ_init` exactly. The strength is
  balanced against `τ_init - η‖V‖`, not `τ_init` — worth ~5e-6 Pa, but it
  makes t=0 exactly self-consistent.
- The outer ring of `Ω_f` is held at `V=0`. Eq. 13 locks everything outside
  `Ω_f`, and a finite slip jump at that edge would be a stress singularity the
  elastic solve cannot represent.
- **Peaceman well** (eq. 22-23) alongside the Gaussian source, adding `p_well`
  as one extra unknown. `well_index`, `well_cell_index` and
  `peaceman_cell_volume` live in `PorePressure`.
- `analytic_pressure_gaussian`/`analytic_pressure_point` (eq. 21/25) with a
  hand-rolled `expint_e1` (series below x=1, Lentz continued fraction above),
  checked against `SpecialFunctions.expint` to 1e-12. §6 asks specifically
  that the well model be checked against the point-source solution.
- Full §4 output: nine station time series, `global.dat`, ten profile files.
  The profile layout comes from the worked example on p.13-14 — an
  `(N_t+1)×(N_coord+2)` matrix whose first row is `0 0 <coords>`, with a
  four-line field list and `author`/`code_version` header keys, all of which
  differ from the time-series files' conventions.
- `effective_stress_report` and `resolution_report` surface the two places
  this model can quietly leave its range of validity (below).

### Results

`scripts/run_bp8.jl` (30-day runs, both injection models, two resolutions),
`scripts/bp8_domain_convergence.jl` (§6), `scripts/bp8_validate_pressure.jl`.

Pore pressure against the analytic solutions at t = 100 h, Δz = 50 m:

| r (m) | GS numeric | eq. 21 | PW numeric | eq. 25 |
|---|---|---|---|---|
| 0 / r_e | 13.17 | 13.06 | 25.85 | 28.76 |
| 100 | 7.356 | 7.357 | 7.378 | 7.310 |
| 200 | 2.576 | 2.563 | 2.368 | 2.376 |
| 300 | 0.842 | 0.795 | 0.725 | 0.699 |

Away from source and no-flux edges both agree to well under a percent — a
strong check on the diffusion solver. The GS near-source error is 0.84% at
Δz = 50 m but 15% at Δz = 100 m, purely because `L_gauss = 50 m` is
unresolved. Agreement degrades past r ≈ 300 m because the analytic solutions
assume an unbounded fault while `Ω_f` has no-flux edges.

Domain-size convergence (§6, Δz = 100 m, t = 100 h) is essentially already
achieved at the smallest domain — over `L_fault` 800→1600 m and `L_normal`
800→1200 m, centre slip moves 0.25%, `V_max` 0.55%, self-stiffness 0.03%.
Pore pressure is bit-identical across those rows, as it must be.
**That sweep did not cover the production configuration**, which uses
`L_normal = 400` (halved for memory), so the nearest truncation boundary sat
outside the validated range. It could not: the sweep runs at Δz = 100 m, where
`L_normal = 400` gives only 5 points across the fault-normal direction against
SBP order 4's minimum of 9. `L_normal = 400` is legal **only at Δz = 50 m** —
which is also the production resolution.
`scripts/bp8_domain_convergence.jl` therefore now runs a second sweep at
Δz = 50 m over `L_normal ∈ {400, 600, 800}` with `L_fault` fixed at 800 m,
isolating the fault-normal truncation at the resolution that ships. All three
fit — `L_normal = 800` is 111k elastic DOF, ~2× production, and its factorization
completing at all is partly the Cholesky change paying for itself.

| `L_normal` | elastic DOF | `K_self` | slip(0,0) | `V_max` |
|---|---|---|---|---|
| 400 (production) | 58,806 | −4.38404e8 | 3.99595e-2 | 1.66456e-7 |
| 600 | 84,942 | −4.38324e8 | 4.05356e-2 | 2.54541e-7 |
| 800 | 111,078 | −4.38307e8 | 4.06708e-2 | 2.87795e-7 |

Relative to `L_normal = 800`, the production row is off by **0.02% in `K_self`,
1.75% in slip, and 42% in `V_max`** — and the signs are exactly the predicted
ones: `u=0` stiffens the medium, so slip comes out too small and `|K_self|` too
large. Pressure is identical to 5 decimals across the three, as it must be.

**This is a real gap, and it is bigger than the existing `L_fault` sweep
implied.** That sweep reported `V_max` moving 0.55% over `L_fault` 800→1600 m,
which invited the conclusion that domain size was settled; it was varying the
*far* boundary while the near one sat at 400 m. It is also **not converged at
800 m** — 600→800 still moves `V_max` 11.6%.

Proportion, though: `V_max` moves 167× between Δz = 50 and 100 m, so this 1.42×
is a second-order error sitting underneath the dominant resolution one, and it
does not disturb the quantities that are well behaved (slip, `K_self`). The
practical reading is that **peak `V` in the shipped Δz = 50 m runs is biased
low**, on top of already being flagged as indicative-only. Slip and stiffness
are unaffected at the percent level. The traction-fix comparison in "Results" is
unaffected either way, since both sides of it used `L_normal = 400`.

30-day runs, `L_fault = 800 m`. These are **post-traction-fix**; the
pre-fix column is kept because it is the evidence for `SYMMETRIC_SAT.md`'s
gate 2, and because the size of the shift is the honest measure of how much
the boundary operator matters here.

| variant | Δz | peak V (m/s) | at (days) | final slip (m) | peak p (MPa) | peak V pre-fix | slip pre-fix |
|---|---|---|---|---|---|---|---|
| GS | 50 | 1.10e-6 | 2.22 | 0.0401 | 13.17 | 2.81e-6 | 0.0422 |
| GS | 100 | 1.83e-4 | 2.18 | 0.0550 | 15.05 | 1.14e-3 | 0.0609 |
| PW | 50 | 3.44e-6 | 0.14 | 0.0586 | 25.85 | 8.59e-6 | 0.0633 |
| PW | 100 | 3.62e-4 | 0.40 | 0.0660 | 19.16 | 2.53e-3 | 0.0739 |

Reading the shift: peak `V` moves 2.5-7.0×, against a Δz = 50 → 100 m
resolution spread of 105-406× — so it is 1-2 orders of magnitude inside the
spread that limitation 1 already documents. Final slip, which integrates over
the run, moves 5-11%. Peak pressure is **bit-identical**, the intended control.

Two things the fix improved that were not asked of it. The resolution spread in
peak `V` narrows (GS 406× → 167×, PW 295× → 105×), and the GS peak *time*
becomes resolution-consistent — it was 2.26 d at Δz = 50 m against 0.62 d at
Δz = 100 m, and is now 2.22 d against 2.18 d. Nothing was tuned for either.

All four wrote their 20 §4 files to `output/BP8-QD-<GS|PW>_dz…/` (gitignored,
~120 MB total), with 9.3k-41k time-series rows — inside §4.1's requested
10⁴-10⁵ — and 721 hourly profile rows, inside §4.3's ~10³. Everything is
aseismic, as a velocity-strengthening fault (a-b = +0.006) should be: peak
slip rates are ~10⁻⁶-10⁻⁴ m/s, not the ~1 m/s of a seismic rupture.

### Figures (`scripts/plot_bp8.jl`)

Reads only the `.dat` files, so it works on any run and doubles as a check
that they parse. Writes `global.png`, `stations.png`, `slip_evolution.png`
and `spacetime.png` beside them.

Two things the figures make visible that the summary numbers did not:

- **The slipping patch is elongated along strike** — at t = 30 d, slip at
  r = 250 m is 0.0180 m along x2 against 0.0145 m along x3, a 24% difference.
  This is real, not a bug. For slip in x2, variation along x2 is mode II and
  variation along x3 is mode III, and mode II is stiffer by 1/(1-ν); measured
  directly on `K`, the ratio is 1.22 at ν = 0.25, 1.33 at ν = 1/3 and exactly
  1.00 at ν = 0, tracking 1/(1-ν) with the ~10% shortfall expected from a
  truncated domain at this resolution. A patch elongated *along* x2 is
  controlled by its narrow x3 dimension — mode III, the softer one — so
  elongating along strike is the compliant direction. The station plot shows
  the same thing: the (±200, 0) stations slip more than the (0, ±200) ones.
- **The under-resolution is plainly visible.** `V_max(t)` is a sawtooth and
  the space-time contours are a staircase, because the slip front advances
  one grid node at a time across only ~15 active nodes. This is the same
  limitation as item 1 below, seen directly rather than inferred.

## Known limitations

These are properties of the current approach, not loose ends to tidy.

1. **The runs are not resolution-converged, and cannot be on this machine.**
   The rate-and-state process zone is `L_b = μD_RS/(bσ̄) ≈ 64 m` at
   σ̄ = 25 MPa. The benchmark's Δz = 10 m gives ~6 cells per `L_b`; the
   Δz = 50 m used here gives 1.3, and Δz = 100 m gives 0.6. Because slip rate
   depends *exponentially* on σ̄ (`V ~ exp(τ/(aσ̄))`), a sub-percent pressure
   error becomes an order-of-magnitude error in peak `V` — which is exactly
   what the table above shows between the two resolutions. Final slip, which
   integrates over the whole run, is much better behaved (4.0 vs 5.5 cm).
   `resolution_report` reports this; the runs show the right physics with
   indicative, not quantitative, peak rates. The traction fix narrowed this
   spread (406× → 167× for GS) but nowhere near closed it — it is a resolution
   problem, not a discretization-quality one.
   A second, smaller contribution: `Ω_f = (-l_f, l_f)²` is an *open* interval,
   so the nodes exactly on `±l_f` are locked. That is the correct discrete
   reading of eq. 13 and converges as Δz → 0, but at these spacings it means
   the slipping patch is 600 m across at Δz = 100 m and 700 m at Δz = 50 m.
2. **Why Δz = 50 m is the ceiling.** Fill-in in the sparse factorization of the
   3D elastic system, not the grid itself. Measured on a cube with LU: 20M
   nonzeros at n=13, 93M at n=17, 297M (2.4 GB) at n=21, and OOM by n=25. The
   production configuration (58,806 DOF) takes 92 s to factorize and 125 s to
   build the 578-column stiffness; the 30-day integration itself is then 3 s.
   At the benchmark's Δz = 10 m over a domain several km across, the elastic
   system is 10⁷–10⁸ DOF — out of reach of a direct factorization by orders of
   magnitude, and the reason `L_normal = 400 m` had to be halved relative to
   `L_fault` in the production runs.
   **Cholesky (now the default) buys ≈1.9× on this, and that is nearly
   nothing.** Fill-in scales ≈ `n^5.6` on the measured points, so 1.9× less
   memory supports only a ≈1.12× finer grid — Δz 50 m → ~45 m. Free and worth
   taking; not a path to 10 m. CG would lift the ceiling entirely (it converges
   now, 73 iterations) but is *slower* here: break-even is 6-15 right-hand
   sides and `fault_stiffness` needs 578.
   Reaching spec resolution means changing the elastic solver, not tuning it:
   either an iterative/multigrid solve of the same SBP-SAT system, or the
   boundary-integral route most SEAS codes take for quasi-dynamic whole-space
   problems, where the whole-space fault-to-fault kernel is a convolution and
   costs O(N log N) per step with FFTs and no volume unknowns at all. **This is
   the one open blocker**; everything else in this file is now either done or a
   documented, bounded caveat.
3. **The Peaceman variant drives σ̄ negative at the well cell.** At Δz = 50 m
   pressure there reaches 25.85 MPa against σ = 25 MPa, so `σ̄ = σ - p` goes
   to −0.85 MPa and the `σ̄_min` floor binds. This is not a numerical
   artefact: the point-source solution is logarithmically singular, so the
   well-cell pressure *rises* as Δz shrinks (eq. 25 at `r_e = 0.198Δz` gives
   ~29 MPa at Δz = 50 m and ~44 MPa at the specified Δz = 10 m). Physically
   the fault fully unclamps and eq. 3's no-opening condition stops applying.
   The solution stays bounded because slip relieves the shear stress as fast
   as the strength drops (τ₂ → 600 Pa at that node), but those nodes are
   outside the model's stated range of validity. `effective_stress_report`
   flags it; the GS variant never gets there.
4. **Peaceman well-cell pressure is ~10% below eq. 25** at the equivalent
   radius, while matching to under 1% for r ≥ 50 m. The `r_e = 0.198Δz`
   calibration is derived for steady radial flow; this is a transient. Worth
   revisiting against the benchmark's suggested alternatives (implicit or
   Gaussian-eliminated well pressure) if the well-cell value matters.
5. **Order 6 is unusable, and the cause is upstream.** `standard_diagonal.toml`
   provides, per stencil set: orders 2 and 4 have `H`, `e`, `d1`, `D1`, `D2`,
   `D2variable`; **order 6 has only `H`, `e`, `d1` and `D2.positivity`** — no
   `D1` at all, and `D2` carries the positivity constants but no inner or
   closure stencils. So `first_derivative` throws `KeyError: "D1"` and
   `second_derivative` is missing too; this is broader than a single absent key.
   Nothing in this package can fix it — it needs Mattsson's order-6
   coefficients added to Diffinitive's operator file. Blocks any higher-order
   convergence study.
6. **Constant coefficients only.** The reference notebook's operators take λ, μ
   as grid functions; `isotropic_lambda_mu` takes scalars. Fine for BP8's
   homogeneous whole space — and it is what makes factorizing once and reusing
   across all ~10⁴ RHS evaluations possible — but a foreclosed capability.

## Not started

- **CRESCENT DET upload** (§5): the output files are written in the §4
  formats, but nothing has been uploaded or validated against the server's
  parser. Given limitation 1, the current results are not submission-ready
  anyway.

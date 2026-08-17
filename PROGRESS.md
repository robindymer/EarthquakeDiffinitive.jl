# SEAS BP8-QD-GS/-PW progress

Status of the `EarthquakeDiffinitive` implementation of the SEAS
Benchmark Problems BP8-QD-GS and BP8-QD-PW
(`context/SEAS_BP8_Benchmark_Description.pdf`): a quasi-dynamic 3D
whole-space, rate-and-state fault, driven by fluid injection modelled either
as a Gaussian source or a Peaceman well. Built on the `Diffinitive` SBP-FD
library (dev-linked from `~/.julia/dev/Diffinitive`).

**Both injection models run end to end** and write the §4 output files; see
"Results" below. The implementation is complete and validated, but the runs
are **not resolution-converged** and are not submission-ready — the reason is
a hard memory limit on the 3D elastic factorization, quantified under "Known
limitations".

## Done

### Environment
- `EarthquakeDiffinitive` dev-depends on the local `Diffinitive` checkout;
  `Pkg.instantiate()`/`Pkg.test()` work.
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
  convention, not outward-normal) from a solved displacement field.
- `inject_dirichlet!`, `dof_index`, `flatten`/`unflatten`: strong-injection
  and vector-grid-function utilities (no Diffinitive helper exists for
  strong Dirichlet row-injection).
- Validated in `test/elasticity_test.jl` (polynomial exactness, sparse vs.
  `LazyTensor` consistency, traction accuracy against a manufactured field).

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
- **The reduced system is still not quite symmetric/PSD as the reference
  note assumes**, though much closer than it looked: most of the original
  gap was the `+`-side SAT sign error above. At a small test grid the
  numbers went from ~54% relative asymmetry and 195 negative eigenvalues
  out of 2318, to **~19% and 3 out of 2205** once the sign, the `u1`
  averaging and the fault-edge ring were fixed. CG still isn't usable, but
  the residual is now a small structural gap rather than a broken operator.
  Dug into why: the far-field/bulk part of the assembled system *is*
  exactly symmetric (~1e-14) once far-field DOFs are projected out, so
  `elastic_blocks` itself is fine — the gap is entirely in the fault SAT.
  Literature (Almquist & Dunham, arXiv:2003.12811) confirms a symmetric
  interface SAT needs *two* term types (`E'·traction` **and**
  `T'·displacement-jump`), not just the one (`E'·traction`) implemented
  here. Building the second term properly surfaced a deeper mismatch:
  `traction_blocks` — accurate as a *physical* traction operator — is not
  the operator that actually appears in `elastic_blocks`'s own discrete
  SBP identity. Two confirmed discrepancies: (1) the narrow (μ) shear
  pieces need `normal_derivative` (outward-signed), not the fixed-axis
  `e∘∂/∂xₙ` `traction_blocks` uses — confirmed numerically these are
  genuinely different operators, not just a sign flip; (2) the wide
  (λ+μ) cross terms in `elastic_blocks` carry an extra μ contribution
  (from expanding `(λ+μ)∇(∇·u)`) with no counterpart in the physical
  traction formula at all. A provably symmetric SAT would need a second,
  separately-derived "SBP-consistent flux operator" distinct from
  `traction_blocks` — real work, and low-value right now since
  `reduced_solve`'s direct solve already gives verified-correct results;
  CG would only be a performance optimization. Deferred until it's
  actually needed at production scale.
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
- `K` comes out symmetric to 0.19%, which it should be by reciprocity — an
  independent confirmation that the residual interface-SAT asymmetry is the
  only thing left (see above).

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

30-day runs, `L_fault = 800 m`:

| variant | Δz | peak V (m/s) | at (days) | final slip (m) | peak p (MPa) |
|---|---|---|---|---|---|
| GS | 50 | 2.81e-6 | 2.26 | 0.0422 | 13.17 |
| GS | 100 | 1.14e-3 | 0.62 | 0.0609 | 15.05 |
| PW | 50 | 8.59e-6 | 0.14 | 0.0633 | 25.85 |
| PW | 100 | 2.53e-3 | 0.39 | 0.0739 | 19.16 |

All four wrote their 20 §4 files to `output/BP8-QD-<GS|PW>_dz…/` (gitignored,
~120 MB total), with 9.6k-45k time-series rows — inside §4.1's requested
10⁴-10⁵ — and 721 hourly profile rows, inside §4.3's ~10³. Everything is
aseismic, as a velocity-strengthening fault (a-b = +0.006) should be: peak
slip rates are ~10⁻⁶-10⁻³ m/s, not the ~1 m/s of a seismic rupture.

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
   integrates over the whole run, is much better behaved (4.2 vs 6.1 cm).
   `resolution_report` reports this; the runs show the right physics with
   indicative, not quantitative, peak rates.
   A second, smaller contribution: `Ω_f = (-l_f, l_f)²` is an *open* interval,
   so the nodes exactly on `±l_f` are locked. That is the correct discrete
   reading of eq. 13 and converges as Δz → 0, but at these spacings it means
   the slipping patch is 600 m across at Δz = 100 m and 700 m at Δz = 50 m.
2. **Why Δz = 50 m is the ceiling.** Fill-in in the sparse LU of the 3D
   elastic system, not the grid itself. Measured on a cube: 20M nonzeros at
   n=13, 93M at n=17, 297M (2.4 GB) at n=21, and OOM by n=25. The production
   configuration (58,806 DOF) takes 92 s to factorize and 125 s to build the
   578-column stiffness; the 30-day integration itself is then 3 s. At the
   benchmark's Δz = 10 m over a domain several km across, the elastic system
   is 10⁷–10⁸ DOF — out of reach of a direct factorization by orders of
   magnitude, and the reason `L_normal = 400 m` had to be halved relative to
   `L_fault` in the production runs.
   Reaching spec resolution means changing the elastic solver, not tuning it:
   either an iterative/multigrid solve of the same SBP-SAT system, or the
   boundary-integral route most SEAS codes take for quasi-dynamic whole-space
   problems, where the whole-space fault-to-fault kernel is a convolution and
   costs O(N log N) per step with FFTs and no volume unknowns at all.
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

## Not started

- **CRESCENT DET upload** (§5): the output files are written in the §4
  formats, but nothing has been uploaded or validated against the server's
  parser. Given limitation 1, the current results are not submission-ready
  anyway.

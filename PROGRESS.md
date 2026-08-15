# SEAS BP8-QD-GS progress

Status of the `EarthquakeDiffinitive` implementation of the SEAS
Benchmark Problem BP8-QD-GS (`context/SEAS_BP8_Benchmark_Description.pdf`):
a quasi-dynamic 3D whole-space, rate-and-state fault, driven by a
Gaussian-source fluid injection. Built on the `Diffinitive` SBP-FD library
(dev-linked from `~/.julia/dev/Diffinitive`).

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
Gaussian-source variant of the fault's 2D pore-pressure diffusion equation
(PDF §2.1.1, eq. 17/19-21), fully decoupled from elasticity.

- `pore_pressure_operator`: assembles the constant (homogeneous-medium)
  Neumann-SAT Laplacian on the fault plane, reusing Diffinitive's built-in
  `Laplace`/`NeumannCondition`/`sat_tensors` machinery directly.
- `gaussian_source`, `injection_rate`: the eq. 19/20 forcing.
- `solve_pore_pressure`: implicit (`OrdinaryDiffEq`) time integration; the
  operator matrix is constant, so it's built once.
- Validated in `test/pore_pressure_test.jl` against the closed-form
  analytic solution (PDF eq. 21) and via grid-refinement convergence.

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
uniform* slip — for general slip there's a genuine jump in normal stress
(σ11) proportional to the slip gradient, which `∂u1/∂x1=0` silently
dropped. Confirmed numerically: the old solver produced a normal-stress
perturbation at the fault of the same order of magnitude as the shear
tractions for a Gaussian slip profile — not the ~0 the derivation assumed.
MMS testing never caught this because MMS validates that the discretization
correctly solves the equations *as written*, not that those equations
match BP8's actual physics — a good reminder that discretization-accuracy
tests and physical-model-correctness are genuinely different questions.
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
- **P**: sparse projection — averages tangential (u2,u3) fault DOF pairs,
  zeroes far-field DOFs, identity elsewhere. Confirmed symmetric and
  idempotent.
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
- **The reduced system is *not* symmetric/PSD as the reference note
  assumes** (checked numerically: ~54% relative asymmetry, ~200 negative
  eigenvalues out of ~2300 at a small test grid), so CG genuinely isn't
  usable yet — this is a structural gap, not a bug in the delivered
  answer (the self-consistency test below still passes via direct solve).
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
- Validated in `test/elasticity_split_node_test.jl` via an algebraic
  self-consistency check (deliberately avoiding another continuum-PDE
  derivation after the half-space episode): pick an arbitrary smooth,
  localized two-sided field, derive the exact forcing that makes it a
  solution *directly from the assembled discrete operator itself*, then
  confirm solving recovers it — this tests D, SAT, P, χ, and the solve
  together without assuming anything about the true physics.

## Not started yet

Roughly in the order they'd naturally come next:

1. **Rate-and-state friction + aging law** (PDF eq. 9-12): the ODE for slip
   rate `V` and state `θ`, coupled to elasticity via traction and the
   radiation-damping term `-ηV` (eq. 8).
2. **Wire physical slip into the elastic solver**: `build_chi` currently
   takes a manufactured test slip function; needs to instead take
   `s2(x2,x3,t), s3(x2,x3,t)` from the friction ODE state, and the friction
   law needs traction computed from `reconstruct_U`'s output, not the raw
   solve variable.
3. **Factorize-once performance**: since λ,μ are constant, `split_node_system`
   rebuilds `A` from scratch every call today. Production use should
   factorize once (`reduced_solve`'s reduced, non-singular system is a
   natural factorization target) and reuse across timesteps/RK stages,
   updating only `χ(s)` (and hence the RHS) as slip evolves — the original
   architectural motivation for going constant-coefficient in the first
   place.
4. **Full coupling**: pore pressure → effective normal stress → friction
   law ⇄ elasticity (slip ⇄ traction), integrated together in time
   (likely `OrdinaryDiffEq`, matching the stiff aging-law dynamics).
5. **Peaceman well (PW) variant**: deferred from the pore-pressure
   increment; only the Gaussian-source (GS) injection model exists so far.
6. **Domain-size convergence** (PDF §6): how large `Lx,Ly,Lz` need to be
   before results stop changing — meaningful only once real slip data
   drives the solve, not with manufactured test solutions.
7. **Physical parameters**: plug in Table 1's actual values (currently the
   elasticity tests use simple O(1) λ,μ for clean MMS numerics, not the
   benchmark's GPa-scale values) and run the full 30-day simulation.
8. **Benchmark output files** (PDF §4): the time-series, `global.dat`, and
   slip/stress/pressure evolution file formats required for the CRESCENT
   DET uploader — not attempted yet.

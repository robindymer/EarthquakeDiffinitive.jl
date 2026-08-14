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

### Elastic solver (`src/Elasticity.jl`)
Homogeneous isotropic elasticity (PDF §1) relating fault slip to fault
traction.

- **Key finding**: no need for a two-block/interface-SAT scheme (Diffinitive
  has no multiblock support anyway). Because BP8's medium is identical on
  both sides of the fault and slip is the only forcing, the whole-space
  problem is provably odd in tangential displacement / even in normal
  displacement about the fault — reducing the whole-space problem to a
  **single half-space grid** (x1≥0) with a mixed boundary condition at
  x1=0: `u2=s2/2, u3=s3/2` (Dirichlet/injection) and `∂u1/∂x1=0`
  (Neumann/SAT). Far-field truncation is plain Dirichlet/injection (`u=0`).
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
  real damage, since `elastic_blocks`/`halfspace_system` were still young.
  `IsotropicElasticOperator` also has a hand-specialized, allocation-free 3D
  `apply` (mirroring the notebook's own non-allocating 2D/3D specializations,
  needed because Diffinitive's generic D-dimensional `apply` doesn't infer
  well) — confirmed 0 bytes/point for a full in-place `E*u`.
- `halfspace_system`: assembles the `3N×3N` sparse block matrix with the
  fault's Neumann SAT baked in.
- `traction_blocks`: extracts fault traction (σ21, σ31 — what the friction
  law needs) from a solved displacement field.
- `inject_dirichlet!`, `dof_index`, `flatten`/`unflatten`: strong-injection
  and vector-grid-function utilities (no Diffinitive helper exists for
  strong Dirichlet row-injection).
- Validated in `test/elasticity_test.jl`: exact polynomial differentiation,
  a manufactured-solution full-BVP solve (converges under refinement), and
  traction accuracy. **Bug found and fixed along the way**: Diffinitive's
  `normal_derivative`/`NeumannCondition` use the *outward*-normal sign
  convention, not the fixed `+x1`-axis convention assumed at first — caused
  an order-1 solution error that didn't shrink under refinement (the
  giveaway it was a real bug, not discretization error).

## Not started yet

Roughly in the order they'd naturally come next:

1. **Rate-and-state friction + aging law** (PDF eq. 9-12): the ODE for slip
   rate `V` and state `θ`, coupled to elasticity via traction and the
   radiation-damping term `-ηV` (eq. 8).
2. **Wire physical slip into the elastic solver**: `halfspace_system`
   currently gets Dirichlet/Neumann data from a manufactured test solution;
   needs to instead take `s2(x2,x3,t)/2, s3(x2,x3,t)/2` from the friction
   ODE state.
3. **Factorize-once performance**: since λ,μ are constant, `halfspace_system`
   builds the same matrix every call today (tests just call `A \ rhs`
   fresh). Production use should factorize once and reuse across
   timesteps/RK stages, updating only the RHS's Dirichlet rows as slip
   evolves — this was the original architectural motivation for going
   constant-coefficient.
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

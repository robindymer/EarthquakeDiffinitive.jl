# EarthquakeDiffinitive

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://robindymer.github.io/EarthquakeDiffinitive.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://robindymer.github.io/EarthquakeDiffinitive.jl/dev/)
[![Build Status](https://github.com/robindymer/EarthquakeDiffinitive.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/robindymer/EarthquakeDiffinitive.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/robindymer/EarthquakeDiffinitive.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/robindymer/EarthquakeDiffinitive.jl)

An implementation of the SEAS benchmark problems **BP8-QD-GS** and
**BP8-QD-PW** (`context/SEAS_BP8_Benchmark_Description.pdf`) on top of the
[Diffinitive](https://github.com/Diffinitive/Diffinitive.jl) SBP finite
difference library: a quasi-dynamic 3D whole-space with a planar
rate-and-state fault, driven to slip by fluid injection and along-fault pore
pressure diffusion.

## Solution procedure

Elasticity is quasi-static, so it collapses into a constant matrix `K`
computed before the time loop starts. Building `K` is essentially the whole
cost of a run; the 30-day integration afterwards takes seconds.

**Setup — build `K` (once per configuration; 48 h on one GPU at Δz = 10 m)**

1. Build two SBP grids meeting at the fault plane; far field `u=0`.
2. Form the split-node elastic system `-HP(D+SAT)Pu = HP(D+SAT)χ(s)`. By
   default `A` is never assembled — it is applied matrix-free as Kronecker
   products of the 1D operators. See "Solving the elastic system".
3. Solve `A u = HP(D+SAT)χ(s)` by CG straight onto the singular `A`.
4. Repeat per unit slip DOF → dense fault stiffness
   `K : [s₂;s₃] ↦ [Δτ₂;Δτ₃]`. `D4` symmetry cuts the `2·N_Ωf` solves this
   would take by 6.5-7.8×.
5. Save `K` to the on-disk cache, so no configuration is ever built twice.

**Setup — the cheap rest (seconds)**

6. Assemble the fault-plane diffusion operator `A_p` and the injection source
   on the *same* nodes as `Ω_f`.
7. Integrate pore pressure over the whole 30 days, on its own,
   **implicitly** (`Rodas5P`, analytic Jacobian) — ~180 steps, independent of
   resolution. See "Pore pressure is solved separately".

**Time loop** (ODE state `[s₂; s₃; lnθ]`, 30 days, ~3 s)

8. `Δτ = K·s` — one dense mat-vec, no PDE solve.
9. `p(t)` by interpolation from step 7; `σ̄ = σ₀ - p` per node.
10. Solve `‖τ⁰+Δτ‖ - η|V| = σ̄f(|V|,θ)` for `|V|` (Newton in `ln V`);
    direction inherited from `τ⁰+Δτ`.
11. RHS: `ṡ = V`, `ϕ̇ = e^{-ϕ} - |V|/D_RS`.
12. Advance with per-block tolerances: explicit `Tsit5` for the Gaussian
    source, implicit `QNDF` for the Peaceman well (see below). Repeat from 8.

**Output**

13. Write the §4 files (nine stations, `global.dat`, ten profiles).

## Pore pressure is solved separately

The pressure subsystem's right-hand side reads only `p`, `p_well` and `t` — the
coupling to elasticity is one-way, through `σ̄ = σ - p`. So it is integrated on
its own before the time loop (`solve_pressure_history`), with a stiff solver and
the analytic Jacobian the linear system makes available, and the elastic
integration interpolates the result. `build_model` does this by default; a
`run_bp8` over any sub-interval reuses it.

Two reasons it is worth the separation:

- **It removes `p` from the explicitly integrated state**, along with the
  loosened per-block `abstol` that used to be needed to stop an explicit method
  from tripping over a parabolic operator.
- **The Peaceman well coupling is genuinely stiff** — its eigenvalue
  `-WI(1/S_well + 1/S_e)` is ~10x diffusion's at Δz = 100 m — and an explicit
  treatment pays for it. Solved implicitly, the step count is
  resolution-independent (~180 Gaussian / ~280 Peaceman, for the full 30 days,
  at any Δz measured).

The dense output is kept in memory rather than resampled onto a fixed grid:
57-79 MB at Δz = 10 m, and its interpolant is accurate to 4.4 Pa against a
tight reference, where linear interpolation between hourly levels is 10.8 kPa.
That distinction matters because `V ~ exp(τ/(aσ̄))` turns a sub-percent pressure
error into a several-percent slip-rate error.

## The Peaceman well is integrated implicitly

Once injection drives the well cell's effective normal stress to the `σ̄_min`
floor, that node's slip rate `V ~ exp(τ/(aσ̄))` responds to its own traction on
a sub-second time scale and an explicit integrator's step count grows as
`Δz⁻⁴` — 488,788 steps at Δz = 25 m for 100 h, extrapolating to weeks at the
benchmark's 10 m. The Gaussian variant never reaches the floor and takes ~400.

`run_bp8` therefore defaults BP8-PW to `QNDF`, fed an analytic **block-diagonal
Jacobian**: each node's own 3×3 `(s2, s3, ϕ)` block from `diag(K)` and the
implicit-function derivatives of the force balance, with the elastic coupling
between nodes deliberately dropped. That block carries the stiff eigenvalue to
four digits (the mode is local), and the method's Newton iteration absorbs the
missing off-diagonals — so the converged step is the true implicit solution,
at the cost of `nf` independent 3×3 solves rather than a dense factorisation.
Measured: 1,105 steps / 20 s at Δz = 25 m, resolution-independent, agreeing
with the explicit reference to the requested tolerance. The `σ̄_min` floor
itself is a numerical guard this code adds (the benchmark does not specify
one) and stays at 1 kPa.

## Solving the elastic system

**`A` is never assembled.** `FaultElasticity` defaults to
`representation=:kronecker`, which applies `A = -HP(D+SAT)P` matrix-free: on an
equidistant grid Diffinitive's 3D derivatives are exactly Kronecker products of
the 1D operators, so one side of the system is six small `n×n` matrices plus the
1D quadrature weights, and nothing of size `nnz(A)` ever exists. At Δz = 10 m on
the (1600, 1600) domain that is ~9 GB of working set against ~85 GB for the
assembled `A`, and it removes ~73 h of host assembly. Results agree with the
assembled form to 3e-16 with identical CG iteration counts.
`representation=:assembled` still builds the explicit matrices, for validation
and for `precond=:jacobi`, which needs `diag(A)`.

**The solve is CG straight onto the singular `A`** — no factorization, no
reduction of `P`'s null space. That needs no special machinery: `b ⊥ null(A)`
exactly, and CG started from `x₀ = 0` never leaves `range(A)`, so the
singularity is simply never excited. `ElasticitySplitNode.CGSolver`'s docstring
gives the argument and the measurements. Memory is a handful of vectors, so it
survives grid refinement rather than capping it — the sparse factorization this
replaced hit `n^5.6` fill-in.

**The `2·N_Ωf` right-hand sides are independent**, which is what makes the `K`
build scale three ways:

- `D4` **symmetry** (`fault_stiffness(fe; symmetry=true)`, on by default from
  `build_model`): `Ω_f` and the elastic grid are square and centred, so the
  discretization is invariant under the 8 symmetries of the square and one
  solve pair fixes up to 8 column pairs. An exact discrete identity, verified
  to 1e-16. 6.5-7.8× fewer solves.
- **GPU** (`fault_stiffness_gpu`, needs `using CUDA`): the bandwidth-bound
  mat-vec runs on the card with the whole system resident. This is the
  production path — the Δz = 10 m, (1600, 1600) build measured 48 h of L40S
  time, split over 8 shards of 6 h each.
- **Threads and shards** on CPU: `julia -t auto`, and
  `build_stiffness_cache.jl [shard] [nshards]` across independent jobs.

`build_model`'s `stiffness` keyword selects `:exact` (all solves) or
`:toeplitz` (10 solves, reconstructing `K` from the near-block-Toeplitz
whole-space kernel, ~0.41% in `V_max`). Every submission run uses `:exact`;
`:toeplitz` predates the symmetry and GPU work and is now a cheap preview.
See `PERFORMANCE.md` §4b.

## Running

```julia
julia --project=. scripts/run_bp8.jl gs 50 800 400   # Gaussian source
julia --project=. scripts/run_bp8.jl pw 50 800 400   # Peaceman well
```

Arguments are the injection model, node spacing `Δz` (m), the half-width of
the elastic domain along the fault, its extent normal to the fault, and
optionally the `K` build mode (`exact`, the default, or `toeplitz`).
Output goes to `output/BP8-QD-<GS|PW>_dz.../` in the §4 benchmark formats:
nine station time series, `global.dat`, and ten slip/stress/pressure profiles.

### Reusing `K`

Set `EQD_STIFFNESS_CACHE` to a scratch directory and every run reads and
writes `K` there, keyed on everything that reaches it. A cache hit skips the
elastic system entirely, so a second run at the same configuration — a
different injection model, friction parameters, tolerances — starts in
seconds:

```sh
export EQD_STIFFNESS_CACHE=/path/to/scratch/eqd-stiffness
julia --project=. scripts/build_stiffness_cache.jl 10 1600 1600 exact   # or
julia --project=. scripts/build_stiffness_cache_gpu.jl 10 1600 1600     # GPU
julia --project=. scripts/merge_stiffness_cache.jl 10 1600 1600         # if sharded
```

At production resolution `K` is a cluster job. **`CLUSTER_RUNBOOK.md` is the
operational guide** — one `scripts/submit_bp8*.sh` invocation per resolution,
plus the CUDA setup a GPU node needs.

### Supporting studies

```julia
julia --project=. scripts/bp8_validate_pressure.jl      # against eq. 21 and 25
julia --project=. scripts/bp8_domain_convergence.jl     # §6 domain-size study
julia --project=. scripts/bp8_resolution_convergence.jl # §6 resolution study
julia --project=scripts scripts/bp8_compare_runs.jl     # diff finished runs
```

`scripts/extra/` holds the investigations that settled questions now answered
in the code and in `PERFORMANCE.md` / `SYMMETRIC_SAT.md` / `MATRIX_FREE_PLAN.md`
— the Toeplitz structure study, the symmetry diagnosis, the matrix-free
prototypes. Kept for reproducibility, not part of any workflow.

## Plotting

```julia
julia --project=scripts scripts/plot_bp8.jl output/BP8-QD-GS_dz50_Lf800_Ln400
```

Reads only the `.dat` files, so it works on any run, and writes four figures
beside them:

| Figure | Shows |
|---|---|
| `global.png` | max slip rate and moment rate vs time |
| `stations.png` | slip, slip rate, shear stress, pore pressure at the 9 stations |
| `slip_evolution.png` | slip profiles every 6 h along strike and depth |
| `spacetime.png` | space-time contours of slip, shear stress and pressure |

The CRESCENT DET uploader (§5.3) draws its own graphs and space-time contours
from the same files once they are uploaded.

## Layout

Modules, in `include` order — each composes the ones before it:

| Module | What it does |
|---|---|
| `PorePressure` | Fault-plane diffusion, Gaussian source and Peaceman well |
| `Elasticity` | Constant-coefficient isotropic Navier operator and traction operator |
| `ElasticitySplitNode` | Two-sided split-node fault system `-HP(D+SAT)Pu = HP(D+SAT)χ(s)`, assembled and matrix-free, plus the CG solver |
| `RateStateFriction` | Regularized rate-and-state friction and the aging law |
| `FaultResponse` | Slip → shear-traction map; the dense fault stiffness `K` and its four build paths |
| `StiffnessCache` | On-disk reuse of `K`, and the shard format a multi-job build writes |
| `BP8` | Parameters, initial conditions, coupled time integration, §4 output |

`READING_GUIDE.md` is the entry point for reading `src/` — it explains which
test file to read alongside which module, and in what order.

## Status

The implementation is complete and validated. Pore pressure matches the
analytic solutions to well under a percent away from the source, the elastic
solver satisfies BP8's interface conditions to machine precision, the discrete
operator `-HP(D+SAT)P` is symmetric positive definite, and both injection
models produce physically sensible aseismic slip over 30 days.

**The benchmark's nominal Δz = 10 m is now reachable.** The matrix-free
operator removed the memory ceiling that capped resolution — nothing of size
`nnz(A)` is formed, so the working set is vectors, not a sparse operator — and
`D4` symmetry plus the GPU build brought the `K` build down to 48 h of
single-card time at Δz = 10 m on the (1600, 1600) domain. BP8-PW's separate
blocker, the well-cell stress singularity driving step count as `Δz⁻⁴`, is
resolved by integrating it implicitly with the block-diagonal Jacobian.

What remains is production: running the full submission set at the converged
domain and resolution, and packaging it. `TODO.md` has the open items and
`PROGRESS.md` the measurements behind all of the above, including the
limitations that stand (the localized BP8-PW unclamping near the well, and the
domain-truncation bias at small `L_normal`).

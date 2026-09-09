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

Elasticity is quasi-static, so it collapses into a constant matrix computed
before the time loop starts — which is why setup costs minutes and the
30-day integration costs seconds.

**Setup (once, ~4 min)**

1. Build two SBP grids meeting at the fault plane; far field `u=0`.
2. Assemble the split-node elastic system `-HP(D+SAT)Pu = HP(D+SAT)χ(s)`.
3. Solve `A u = HP(D+SAT)χ(s)` — either directly (eliminate the redundant DOFs
   via the Galerkin reduction `SᵀAS`, then Cholesky) or iteratively (CG straight
   onto the singular `A`). See "Choosing a solver".
4. Repeat `2·N_Ωf` times → dense fault stiffness `K : [s₂;s₃] ↦ [Δτ₂;Δτ₃]`.
5. Assemble the fault-plane diffusion operator `A_p` and the injection source
   on the *same* nodes.
6. Integrate pore pressure over the whole 30 days, on its own, **implicitly**
   (`Rodas5P`, analytic Jacobian) — ~180 steps and a few seconds, independent
   of resolution. Pressure is autonomous (nothing feeds back into eq. 17), so
   it need not be in the coupled state at all; the time loop interpolates this
   solution's dense output. See "Pore pressure is solved separately".

**Time loop** (ODE state `[s₂; s₃; lnθ]`, 30 days, ~3 s)

7. `Δτ = K·s` — one dense mat-vec, no PDE solve.
8. `p(t)` by interpolation from step 6; `σ̄ = σ₀ - p` per node.
9. Solve `‖τ⁰+Δτ‖ - η|V| = σ̄f(|V|,θ)` for `|V|` (Newton in `ln V`); direction
   inherited from `τ⁰+Δτ`.
10. RHS: `ṡ = V`, `ϕ̇ = e^{-ϕ} - |V|/D_RS`.
11. Advance with `Tsit5`, per-block tolerances. Repeat from 7.

**Output**

12. Write the §4 files (nine stations, `global.dat`, ten profiles).

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

## Running

```julia
julia --project=. scripts/run_bp8.jl gs 50 800 400   # Gaussian source
julia --project=. scripts/run_bp8.jl pw 50 800 400   # Peaceman well
```

Arguments are the injection model, node spacing `Δz` (m), the half-width of
the elastic domain along the fault, and its extent normal to the fault.
Output goes to `output/BP8-QD-<GS|PW>_dz.../` in the §4 benchmark formats:
nine station time series, `global.dat`, and ten slip/stress/pressure profiles.

Two supporting studies:

```julia
julia --project=. scripts/bp8_domain_convergence.jl  # §6 domain-size study
julia --project=. scripts/bp8_validate_pressure.jl   # against eq. 21 and 25
```

## Solving the elastic system

The elastic system is solved by CG directly on the (singular) assembled
operator `A` — no factorization, no reduction of `P`'s null space. That
needs no special machinery: `b ⊥ null(A)` exactly, and CG started from
`x₀ = 0` never leaves `range(A)`, so the singularity is simply never excited.
`ElasticitySplitNode.CGSolver`'s docstring gives the argument and the
measurements behind it.

Memory scales ≈ linearly in DOF count rather than with the `n^5.6` fill-in
growth a sparse factorization would hit, so it survives grid refinement rather
than capping it. Its `2·N_Ωf` right-hand sides in `fault_stiffness` are also
independent, so the build threads across `Threads.nthreads()`:

```julia
julia -t auto --project=. scripts/run_bp8.jl gs 50 800 400
```

Threading scales sublinearly — 5.1× on 8 threads on a small case, but only 2.1×
on 16 at production size, because the sparse mat-vec is memory-bandwidth bound
once the working set leaves cache. Expect the memory system, not the core count,
to be the limit.

`build_model`'s `stiffness` keyword (`:toeplitz` by default) reconstructs the
dense fault stiffness `K` from 10 CG solves instead of the full `2·N_Ωf` — the
fault-to-fault kernel is near-block-Toeplitz, so a handful of columns near the
diagonal reproduce the rest to <1% in `V_max(t)`. `:exact` remains available as
the reference build. This is what turned the `K` build from a multi-node job
(~17 days at Δz = 20 m) into a single-node one (~2.5-3 h) — see
`PERFORMANCE.md` §4b for the validation.

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

| Module | What it does |
|---|---|
| `Elasticity` | Constant-coefficient isotropic Navier operator and traction operator |
| `ElasticitySplitNode` | Two-sided split-node fault system `-HP(D+SAT)Pu = HP(D+SAT)χ(s)` |
| `FaultResponse` | Slip → shear-traction map; dense fault stiffness, factorized once |
| `PorePressure` | Fault-plane diffusion, Gaussian source and Peaceman well |
| `RateStateFriction` | Regularized rate-and-state friction and the aging law |
| `BP8` | Parameters, initial conditions, coupled time integration, §4 output |

## Status

The implementation is complete and validated — pore pressure matches the
analytic solutions to well under a percent away from the source, the elastic
solver satisfies BP8's interface conditions to machine precision, the discrete
operator `-HP(D+SAT)P` is symmetric positive definite, and 30-day runs of both
injection models produce physically sensible aseismic slip.

**The runs are not resolution-converged on a single workstation, but both
injection models are now viable on a cluster node.** CG removes the sparse
factorization entirely, so there is no fill-in ceiling; what binds instead is
memory for the assembled operator itself — ~15 GB at Δz = 20 m (the resolution
`resolution_report` calls converged, `L_b/Δz ≥ 3`, well short of the
benchmark's nominal 10 m), against a 15 GB development workstation. The `K`
build, previously the reason a multi-node job was needed, no longer is one:
`:toeplitz` (see "Solving the elastic system") cuts it from ~17 days to
~2.5-3 h on a single node. BP8-PW was separately blocked by the well-cell
stress singularity making the coupled time integration scale ≈ Δz⁻⁴ in step
count (weeks-to-months extrapolated at Δz = 10 m); regularizing the
effective-normal-stress floor (`σ̄_min`) buys two orders of magnitude there for
well under 1% change in the reported peak slip rate. See "Known limitations"
in [PROGRESS.md](PROGRESS.md) for the measurements behind all of this and what
remains before a submission-ready run.

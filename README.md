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
solver satisfies BP8's interface conditions to machine precision, and 30-day
runs of both injection models produce physically sensible aseismic slip.

**The runs are not resolution-converged**, and cannot be on a single
workstation: fill-in in the sparse LU of the 3D elastic system caps the node
spacing at ~50 m against the benchmark's specified 10 m. See the "Known
limitations" section of [PROGRESS.md](PROGRESS.md) for the measurements
behind that and what reaching spec resolution would take.

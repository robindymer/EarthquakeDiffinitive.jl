# EarthquakeDiffinitive.jl

Julia package for SBP finite-difference earthquake-rupture simulation, built on
[Diffinitive.jl](https://github.com/Diffinitive/Diffinitive.jl).

## Context

`context/` holds background material worth checking before diving into a task:
- `lore.md` — accumulated theory/derivation notes (SAT conventions, SBP tricks, discrete-vs-continuous subtleties)
- `notes_robin.md` — workflow notes (env setup, running scripts/docs/Pluto) and a running TODO list
- `demo/`, `notebooks/` — example and exploratory scripts
- PDFs — SEAS benchmark descriptions this project targets

Root also has `PROGRESS.md`, `TODO.md`, `PERFORMANCE.md`, `SYMMETRIC_SAT.md` — check these for
standing context on where things are and why.

**`READING_GUIDE.md` is the primary onboarding doc** for the codebase itself — read it before
exploring `src/` cold. It explains the module dependency chain (each module composes the ones
before it, `src/EarthquakeDiffinitive.jl` include order):

```
PorePressure → Elasticity → ElasticitySplitNode → RateStateFriction → FaultResponse → BP8
```

...and that each module's test file should be read alongside its source, not after — the tests
carry "why this assertion, why this tolerance" comments that explain design decisions the source
doesn't.

## Dev workflow

- **Do not commit. Robin commits all code himself.** Leave finished work staged
  or in the working tree and say what changed; do not run `git commit`, `git
  push`, or anything that rewrites history. Writing files, running tests and
  `git add` are fine.

- Run the test suite: `julia --project=. -e 'using Pkg; Pkg.test()'` (or
  `julia --project=. test/runtests.jl` after `Pkg.instantiate()`).
- **Set `JULIA_NUM_THREADS=4`+ when running tests locally.** The threaded `fault_stiffness` build
  path is only exercised — and its data races only detectable — with more than one thread; CI
  runs with `JULIA_NUM_THREADS=4` for this reason.
- `scripts/` and `context/demo/`, `context/notebooks/` are separate Julia environments (own
  `Project.toml`, pull in `EarthquakeDiffinitive` as a dep rather than living inside it) — run
  their scripts with `julia --project=scripts scripts/foo.jl` etc., not `--project=.`.
- Minimum supported Julia version is 1.10 (see `Project.toml` `[compat]`); CI also tests 1.11 and
  `pre`.

## Diffinitive.jl source

This package depends on Diffinitive.jl as a git dependency (not dev'd locally). When a
question touches SBP operators, stencils, or anything Diffinitive provides
(`normal_derivative`, `D1`, `D2`, boundary operators, etc.), it's often worth reading its
source directly rather than guessing at behavior from usage here:

```
~/.julia/packages/Diffinitive/O7HXv
```

(That hash-suffixed folder name is tied to Diffinitive's pinned revision in `Manifest.toml` —
if it's missing, `ls ~/.julia/packages/Diffinitive/` to find the current one.)

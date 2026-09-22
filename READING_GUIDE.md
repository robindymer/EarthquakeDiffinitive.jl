# Reading guide

How to work through this codebase so the pieces tie together, rather than
reading files in isolation.

## 1. Anchor on the physics first

Skim `context/SEAS_benchmark.pdf` and `BP8.jl`'s own docstrings before diving
into any one module. Everything downstream — χ, `P`, the SAT signs, the
interface conditions in `elasticity_split_node_test.jl` — implements specific
numbered equations from that benchmark, and the code comments assume you can
map back to them (e.g. "BP8 eq. 3", "eq. 6b,c").

## 2. Follow the dependency chain bottom-up

This is the `include` order in `src/EarthquakeDiffinitive.jl`:

```
PorePressure → Elasticity → ElasticitySplitNode → RateStateFriction
            → FaultResponse → StiffnessCache → BP8
```

Each module composes the ones before it (`FaultResponse` wraps
`ElasticitySplitNode`, `BP8` wraps `FaultResponse` and `PorePressure`, etc.),
so reading in dependency order means you never meet a symbol before you've
seen where it comes from.

## 3. Read a module's header comment before its functions, and its test file alongside it

The top-of-file comment block states the module's intent more directly than
any individual docstring. Then read the matching test file side by side, not
after — the tests carry the "why this assertion, why this tolerance" notes
that explain design decisions the source does not.
`test/elasticity_split_node_test.jl`'s comment above `solve_gaussian_slip`
(why self-consistency alone doesn't catch a wrong model) is a good example.

## 4. Treat the markdown files as reference material, not front-to-back reading

Jump into `SYMMETRIC_SAT.md`, `PERFORMANCE.md`, `MATRIX_FREE_PLAN.md` and
`PROGRESS.md` only when a code comment cites them (`traction_blocks`'s
docstring does this). That is where the derivations and measurements behind a
one-line code comment live — reading them cold is mostly noise.
`CLUSTER_RUNBOOK.md` is the exception: read it when you are about to run
something on a cluster, front to back.

## Suggested order

1. `PorePressure.jl` + `test/pore_pressure_test.jl`
2. `Elasticity.jl` + `test/elasticity_test.jl`
3. `ElasticitySplitNode.jl` + `test/elasticity_split_node_test.jl` — the
   assembled `split_node_system` first (it is what the derivations refer to),
   then `SplitNodeOperator`, the matrix-free form production uses
   (`MATRIX_FREE_PLAN.md` for the why)
4. `RateStateFriction.jl` + `test/rate_state_friction_test.jl`
5. `FaultResponse.jl` + `test/fault_response_test.jl` — `fault_stiffness` and
   its `symmetry=true` path, then `fault_stiffness_gpu` with
   `ext/EarthquakeDiffinitiveCUDAExt.jl` and
   `test/fault_response_gpu_test.jl`
6. `StiffnessCache.jl` + `test/stiffness_cache_test.jl`
7. `BP8.jl` + `test/bp8_test.jl`

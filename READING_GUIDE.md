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
PorePressure → Elasticity → ElasticitySplitNode → RateStateFriction → FaultResponse → BP8
```

Each module composes the ones before it (`FaultResponse` wraps
`ElasticitySplitNode`, `BP8` wraps `FaultResponse` and `PorePressure`, etc.),
so reading in dependency order means you never meet a symbol before you've
seen where it comes from.

## 3. Read a module's header comment before its functions, and its test file alongside it

The top-of-file comment block in each module is usually denser with intent
than any individual docstring. Then read the matching test file side by
side, not after — the tests here aren't just usage examples, they carry
"why this assertion, why this tolerance" comments that often explain the
*design* better than the source does. `test/elasticity_split_node_test.jl`'s
comment above `solve_gaussian_slip` (why self-consistency alone doesn't catch
a wrong model) is a good example of this pattern.

## 4. Treat `SYMMETRIC_SAT.md`, `PERFORMANCE.md` and `PROGRESS.md` as reference material, not front-to-back reading

Jump into them only when a code comment cites them (`traction_blocks`'s
docstring does this, for example). That's where the actual derivations and
measurements behind a one-line code comment live — reading them cold, without
a comment pointing you there, is mostly noise.

## Suggested order

1. `PorePressure.jl` + `test/pore_pressure_test.jl`
2. `Elasticity.jl` + `test/elasticity_test.jl`
3. `ElasticitySplitNode.jl` + `test/elasticity_split_node_test.jl`
4. `RateStateFriction.jl` + `test/rate_state_friction_test.jl`
5. `FaultResponse.jl` + `test/fault_response_test.jl`
6. `BP8.jl` + `test/bp8_test.jl`

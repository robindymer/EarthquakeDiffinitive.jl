# TODO

Open work, in dependency order. Section numbers `§1`/`§2`/`§3` are cited from
code comments and are kept stable. Everything *done* lives in `PROGRESS.md`
(measurements and narrative), `PERFORMANCE.md` (cost) and
`MATRIX_FREE_PLAN.md` (the matrix-free operator); this file is only what is
still open.

## Open now

- [ ] **Merge and run the Δz = 10 m (1600, 1600) `K`.** The 8 L40S shards are
      built (48.1 h of card time, `logs/Kgpu_dz10_Lf1600_Ln1600_6889974_*`) but
      there is no merge log, so `merge_stiffness_cache.jl 10 1600 1600` still
      needs running, then `run_bp8.jl gs/pw 10 1600 1600 exact` against the
      cache entry. This answers §1 and §2 by measurement rather than
      extrapolation.

- [ ] **Update `submit_bp8_gpu.sh`'s Δz = 10 m walltime anchor.** It predicts
      38.8 h for the point that measured 48.1 h — 24% low, the wrong direction
      for a walltime request. Two Δz = 10 m measurements ((1150, 1150) at
      11.89 h, (1600, 1600) at 48.1 h) imply a DOF exponent of **~1.42**, not
      the 1.2 in the script. The comment at the `TIME` derivation also still
      says the estimates "have not been checked on an L40S yet", which is no
      longer true.

- [ ] **Decide `build_model`'s `stiffness` default.** It is `:toeplitz`, while
      `run_bp8.jl` and all three `submit_bp8*.sh` pass `exact`, and §2's own
      note below says `:exact` "stays the default". The three disagree.
      `:toeplitz` was right when `:exact` meant ~17 CPU node-days; D4 symmetry
      and the GPU build removed that gap, so the library default probably wants
      to follow the scripts.

- [ ] **Publish the branch.** Matrix-free, GPU, the stiffness cache, implicit
      pore pressure and D4 sharding are all on `GPU`, 22 commits ahead of
      `main`, which has been untouched since 2026-08-21. Also: `dev` is 6
      behind `GPU`, `direct_solver` is dead, and four dependabot branches widen
      the stdlib `[compat]` entries that currently pin
      `Dates`/`Printf`/`LinearAlgebra`/`SparseArrays` at `^1.11.0` while
      `[compat] julia = "1.10"` — those cannot both hold.

- [ ] **`CUDA.CUSPARSE` is deprecated.** `ext/EarthquakeDiffinitiveCUDAExt.jl`,
      `scripts/gpu_smoke_test.jl` and `scripts/extra/matrix_free_prototype.jl`
      all use it and warn under the resolved CUDA 6.3.1. There is no
      `CUDA.cuSPARSE` to switch to — the fix is a direct `cuSPARSE` dependency,
      which means dropping `CUDA = "5"` from `[compat]`.

- [ ] **The docs site renders nothing.** `docs/src/index.md` is the
      PkgTemplates skeleton with `@autodocs Modules = [EarthquakeDiffinitive]`.
      Every docstring lives in a submodule, which that never picks up, so ~100
      docstrings publish as an empty page. Either list the submodules or
      re-export from the top module — the latter would also make the redundant
      `using EarthquakeDiffinitive; using EarthquakeDiffinitive.BP8` pair in
      every script mean something.

## 1. The converged domain

**Required domain: `L_fault` ≥ 4·`l_f` = 1600 m, `L_normal` ≥ 3·`l_f` = 1200 m**
for ~1% in `V_max` (sweeps at Δz = 50 m in both directions; table in
`PROGRESS.md` "Results"). The old shipped configuration (2·`l_f`, 1·`l_f`) is
**~53% low** in peak slip rate, and `build_model`'s defaults (3·`l_f`, 2·`l_f`)
are also short. Convergence is slow for a physical reason: the elastostatic
kernel decays as 1/r³, so doubling the domain cuts the error only ~8×. Slip and
`K_self` were converged throughout (≤0.3%) — this is entirely a peak-`V_max`
problem, and `V_max(t)` is one of the two global source parameters the
benchmark requires.

- [ ] **Re-run production at the converged domain.** Every number in a
      pre-(1600, 1600) `output/` directory carries that bias. Gated on the
      merge above so the regeneration happens once, at the target domain *and*
      resolution.
- [ ] **Do not assume the domain requirement transfers to finer Δz.** The same
      `L_normal` 800→1200 step moves `V_max` +0.324% at Δz = 100 m but +5.17%
      at Δz = 50 m. That comparison is confounded (different `L_fault`, and
      Δz = 100 m is badly under-resolved) so it proves nothing either way — but
      it removes the grounds for assuming transferability. A matched
      two-resolution sweep settles it; ~12 h at Δz = 25 m.
- [ ] Note for anyone reading the *older* `L_fault` sweep: its 0.55% `V_max`
      spread does **not** mean domain size is settled. It varied the far
      boundary while the near one sat at 400 m.

## 2. Resolution

`PROGRESS.md` "Known limitations" 1 and 2. `resolution_report` calls a grid
converged at `L_b/Δz ≥ 3`, i.e. Δz ≤ 21 m, so the benchmark's nominal 10 m is
~6.7× more expensive than convergence requires. Quote 10 m only if the
submission demands the nominal spec.

**This is no longer a blocker.** The matrix-free operator removed the memory
ceiling (`MATRIX_FREE_PLAN.md`), and D4 symmetry plus the GPU build brought the
Δz = 10 m (1600, 1600) `K` to 48 h of single-card time. `PROGRESS.md`
limitation 2's "Δz = 50 m is the ceiling" refers to the removed direct solver's
fill-in and is stale.

- [ ] **Validate `:toeplitz` at a finer Δz** if it is ever to be relied on
      rather than used as a preview. The centre source's ~44% coverage gap is
      resolution-independent as a *fraction*, but the far-field-decay argument
      behind it is not proven to be. Δz = 25 m at the small domain is ~2.4 h
      for the reference build — the cheapest meaningful check.
      `scripts/extra/k_toeplitz_resolution.jl` is the harness.
- [ ] **Decide between iterative CG on this discretization and the
      boundary-integral route** most SEAS codes take (the whole-space
      fault-to-fault kernel is a convolution: `O(N log N)` with FFTs, no volume
      unknowns, and no `K` build at all). A design decision, not an increment,
      and much less pressing now that the `K` build fits one card.
      Preconditioning is *not* the alternative — measured at ~1×,
      `PERFORMANCE.md` §6.
- [ ] Keep `scripts/extra/split_node_spd.jl`'s guardrails in mind when touching
      any of this: CG's internal recursive residual is only valid for symmetric
      `A`, and CG run on `A[keep,keep]` converges happily to the wrong answer.

## 3. Smaller items

- [ ] **Order 6.** Needs Mattsson's order-6 `D1`/`D2` coefficients added to
      Diffinitive's `standard_diagonal.toml` — an upstream contribution, and a
      nontrivial derivation to get right. Blocks any higher-order convergence
      study. `StiffnessCache`'s `stencil_digest` already anticipates it: the
      coefficients are in the cache key, so new stencils cannot silently reuse
      an order-4 `K`.
- [ ] **Variable coefficients.** The notebook's operators take λ, μ as grid
      functions; this package is constant-coefficient only. Fine for BP8's
      homogeneous whole space, and it is what lets `K` be reused across every
      RHS evaluation — but a foreclosed capability.
- [ ] **Peaceman σ̄ < 0** (`PROGRESS.md` limitation 3): pressure at the well
      cell exceeds σ, the `σ̄_min` floor binds, and eq. 3's no-opening condition
      stops applying there. Not a numerical artefact, and the well-cell `σ̄`
      falls as Δz shrinks — but the affected disc is ~15 m across regardless of
      Δz, so it stays localized. `effective_stress_report` reports the extent.
- [ ] **CRESCENT DET upload** (§5): files are written in the §4 formats but
      nothing has been validated against the server's parser. Gated on the
      production runs above.
- [ ] **Dead exports.** `Elasticity.unflatten` and
      `Elasticity.inject_dirichlet!` are exported and used nowhere, as are
      `BP8.pressure_length`, `BP8.set_pressure_history!` and
      `RateStateFriction.friction_coefficient`. Keep or drop deliberately.
- [ ] **Housekeeping:** `diffinitive_registry` is still in
      `~/.julia/registries`.

## Resolved, for the record

Items previously tracked here, and where the evidence now lives:

| was | outcome |
|---|---|
| Multi-node `K` parallelism | Independent processes plus a shared directory, never `Distributed`/MPI: `build_stiffness_cache.jl [shard] [nshards]` + `merge_stiffness_cache.jl`, composing with D4 via `fault_stiffness_d4_shard`. |
| Is `K` near-block-Toeplitz? | Yes — 10 solves reproduce `V_max(t)` to 0.41% at the converged domain. `PERFORMANCE.md` §4b. |
| Preconditioned CG | ~1×, and AMG is not mesh-independent here. Not a route. `PERFORMANCE.md` §6. |
| Symmetrising the Toeplitz `K` | Tested — `scripts/extra/k_toeplitz_symmetrise.jl`. |
| Validate `fault_stiffness_gpu` on real hardware | Done at the production point: 48.1 h of L40S for Δz = 10 m on (1600, 1600), mean 1599 CG iterations, 0 unconverged. |
| `A` too large to assemble at Δz = 10 m | `A` is no longer assembled. `MATRIX_FREE_PLAN.md`. |
| BP8-PW's `Δz⁻⁴` step count | Implicit `QNDF` with the block-diagonal Jacobian; step count is resolution-independent. `PROGRESS.md` (2026-09-12). |
| Peaceman well-cell pressure ~10% below eq. 25 | Not a model error — it was eq. 25 evaluated at a five-point stencil's radius. `PorePressure.SBP4_RE_FACTOR` = 0.268, measured. |
| Report the computational domain in output headers | `domain_line` in `src/BP8.jl`, in both header families. |
| Regenerate `output/` | Deliberately deferred until the target domain *and* resolution — §1. |

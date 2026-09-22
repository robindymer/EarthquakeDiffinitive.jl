# scripts/extra

One-off investigations that settled a question. Their conclusions are now in
`src/` and in the project's markdown files; nothing here is part of a workflow,
and nothing in `src/` or `ext/` loads any of it.

**Several no longer run against the current API.** `split_node_spd.jl`,
`symmetry_decomposition.jl` and `verify_notebook_traction.jl` call
`factorize_reduced` / `reduced_solve`, the Cholesky-and-Galerkin path that was
removed when the solve became CG straight onto the singular `A`. They are kept
as the record of *how* each conclusion was reached, not as runnable tools — port
them to `CGSolver` if a result ever needs re-measuring.

| script | question it answered | where the answer lives |
|---|---|---|
| `split_node_spd.jl` | Is `A = -HP(D+SAT)P` SPD, and can CG solve it? | `ElasticitySplitNode.CGSolver`, `PROGRESS.md` |
| `symmetry_decomposition.jl` | Where does `A`'s ~14% asymmetry come from? | `SYMMETRIC_SAT.md` |
| `verify_notebook_traction.jl` | Does the notebook's traction operator preserve the interface conditions? | `Elasticity.traction_blocks`, `SYMMETRIC_SAT.md` |
| `k_toeplitz_structure.jl` | Is `K` near-block-Toeplitz? | `PERFORMANCE.md` §4b |
| `k_toeplitz_validate.jl` | Does the reconstruction hold end-to-end, not just in norm? | `PERFORMANCE.md` §4b |
| `k_toeplitz_symmetrise.jl` | Does enforcing reciprocity on the Toeplitz `K` help? | `PERFORMANCE.md` §4b |
| `k_toeplitz_resolution.jl` | Does the Toeplitz error grow with resolution? | open — `TODO.md` §2 |
| `matrix_free_prototype.jl`, `_fused.jl` | Can a Kronecker 1D-operator apply beat the assembled `A`? | `MATRIX_FREE_PLAN.md`; shipped as `ElasticitySplitNode.SplitNodeOperator` |
| `bp8_stiffness_spectrum.jl` | Where does BP8-PW's stiffness live? | `PROGRESS.md` "what the stiff eigenvalue is" |
| `elastic_wave_2d.jl` | Qualitative check of the 2D Navier operator | — |

Run them from the `scripts` environment:

```sh
julia --project=scripts scripts/extra/k_toeplitz_structure.jl
```

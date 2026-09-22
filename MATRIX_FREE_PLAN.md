# Matrix-free split-node operator: design and status

Goal: build `K` for **Δz = 10 m on the (1600, 1600) domain** — 100 M DOF — in
about a day on **one L40S**, without the assembled `A`.

## Status: done, and it is the default

The matrix-free operator is **production code**, not a prototype:
`representation=:kronecker` is `FaultElasticity`'s and `build_model`'s default,
and it is what every GPU `K` build runs on. Steps 1-8 below are implemented and
tested (355/355 with `JULIA_NUM_THREADS=4`; GPU tests with `EQD_TEST_GPU=1`).

The goal above was **met and measured**: Δz = 10 m on (1600, 1600), 99.5 M DOF,
8 shards of 6.01 h on one L40S each = 48.1 h of card time, mean 1599 CG
iterations, none unconverged
(`logs/Kgpu_dz10_Lf1600_Ln1600_6889974_*.out`). That is 24% above what
`scripts/submit_bp8_gpu.sh`'s `EST_H` predicts for the same point (38.8 h), so
its Δz = 10 m anchor wants updating — two Δz = 10 m measurements now imply a
DOF exponent of ~1.42 rather than the 1.2 in the script.

Note `scripts/extra/matrix_free_prototype.jl` is **not** this code. It is the
standalone feasibility study that preceded it, with its own `Ops1D`,
`axpass!` and `navier!`; nothing in `src/` or `ext/` loads it. Kept for
reproducibility only.

What remains is the production runs themselves (step 9), not the operator.

| where | what |
|---|---|
| `src/ElasticitySplitNode.jl` | `SplitNodeOperator`, `split_node_operator`, `Rows1D`, `SideOps`, host `axpass!`/`scale_H!`, `apply_P!`, `hp_dsat!`, `duplicate_operator`, `AssembledSplitNode`; `sat_matrix`, `projection_parts`, `projection_matrix` factored out of `split_node_system` (which now builds `P` from triplets) |
| `src/FaultResponse.jl` | `FaultElasticity` holds `op` (default `representation=:kronecker`; `:assembled` kept), `shear_traction!` uses `solver.A`'s own scratch |
| `src/BP8.jl` | `representation` keyword threaded through `build_model` → `stiffness_matrix` → `build_fault_elasticity`; not part of the cache key |
| `ext/EarthquakeDiffinitiveCUDAExt.jl` | `to_device`, device `axpass!`/`scale_H!` kernels, `fault_stiffness_gpu` on the operator interface with RHS and `P u + χ` on the device, `shard`/`nshards` |
| `scripts/build_stiffness_cache_gpu.jl` | vector-based VRAM estimate, no assembly phase, optional `shard nshards` |
| `scripts/submit_bp8_gpu.sh` | L40S default at every Δz, `MEM=32G`, (1600, 1600) default, optional `nshards` → array + merge |
| `scripts/submit_bp8.sh` | sizing for a shard that holds no `A` |
| tests | operator vs assembled to 1e-14, `P` triplets vs the old row loops, `K` parity across representations and build paths, sharded GPU vs whole |

## Why the assembled path cannot reach 1600² at Δz = 10 m

Measured from the five GPU builds in `logs/Kgpu_*.out` and extrapolated:

| quantity | Δz=10 (1150,1150), measured | Δz=10 (1600,1600), extrapolated |
|---|---|---|
| DOF | 37.1 M | 99.5 M |
| nnz(A) | 1.76e9 (Int32) | 4.7e9 (**Int64**, > 2³¹) |
| device memory | 22.8 GB | ~85 GB (H100 NVL only, marginally) |
| host assembly (`split-node system ready`) | 17.5 h | **~73 h** (scales as DOF^1.45, 5 points) |
| CG solves, 1681 D4 representatives | 23 h on L40S | ~4 days H100 / ~8 days L40S |
| total | 40 h | **~1 week, on the wrong side of the 47 h walltime** |

Where the assembly time goes (`scripts/extra/matrix_free_prototype.jl`, (1600,1200)):

| step | Δz=50, 634 k DOF | Δz=40, 1.2 M DOF | scaling |
|---|---|---|---|
| `_to_sparse_matrix(elastic_blocks(...))` — sparsifying the lazy `D` | 136 s (69%) | 251 s (60%) | ~linear |
| `P[r, :] .= 0.0` row loops on a CSC | 52 s (27%) | 158 s (37%) | **DOF^1.7** |
| SAT (`traction_blocks`, prolongation, triplets) | 4.7 s | 6.7 s | sub-linear |
| H, `H*P*DSAT`, `-HP_DSAT*P` | 1.5 s | 2.4 s | — |

Both dominant steps produce `D` and `P`, which are exactly the two pieces the
matrix-free operator never materialises. SAT, `T2`, `T3` stay as the small
boundary-local sparse matrices they already are.

## What the prototype established

`scripts/extra/matrix_free_prototype.jl` (+ `_fused.jl`), run on the laptop RTX 2000 Ada:

1. **Diffinitive's 3D `D1`/`D2` on an equidistant `TensorGrid` are exact
   Kronecker products of the 1D operators** (`max|diff| = 0.0`). The whole
   Navier block operator is therefore six `n×n` 1D sparse matrices per side
   (`d1[1:3]`, `d2[1:3]`, n ≤ 321) plus the Kronecker inner-product weights.
2. **`A_free v = −H·P·(D_kron + SAT)·P·v` matches the assembled `A v` to
   3e-16**, on CPU and GPU; CG converges in the identical number of iterations
   to the identical solution (1e-14). Same operator, same `K`, same cache key.
3. **GPU cost per mat-vec** (cuSPARSE CSR runs at this card's 222 GB/s
   roofline, so the ratio is a bytes-moved ratio and transfers to the L40S/H100):

   | DOF | CSR SpMV | axis-pass matrix-free | speedup | device memory | assembly it replaces |
   |---|---|---|---|---|---|
   | 634 k | 1.52 ms | 0.81 ms | 1.9× | 0.33 → 0.03 GB | 197 s |
   | 1.2 M | 3.00 ms | 1.33 ms | 2.25× | 0.64 → 0.06 GB | 422 s |
   | 4.9 M | 13.4 ms | 5.9 ms | 2.27× | 2.69 → 0.22 GB | 3149 s (59% the `P` loops) |

   Settling near 2.3× on this card (the ~3× traffic ratio, ≈530 vs ≈1590
   B/point, less the CG-independent kernel overheads); a full CG solve runs
   1.85× faster because Krylov's vector ops are common to both. A naive **fused** single-kernel stencil is
   *slower* (0.56× the axis-pass): Ada GPUs — this one and the L40S — do FP64
   at 1/64 rate, so it is FLOP-bound. Use the axis-pass design; a fused kernel
   is a later H100-only optimisation, not part of this plan.

The Navier block, written to minimise passes (`w_k = D1_k u_k`):

    out_j = μ·Σ_i D2_i u_j  +  μ·D2_j u_j  +  D1_j[(λ+μ)·div − μ·w_j],   div = Σ_k w_k

18 axis passes + 4 element-wise ops per side per mat-vec. Each pass is one
kernel: `out[i,j,k] += α·Σ_p M[r_d, p]·u[…p along axis d…]` with the 1D
operator's rows in CSR — no interior/closure special-casing, the 1D matrix
carries both. Reads are coalesced along axis 1 for every `d`.

## Projected cost at Δz = 10 m, (1600, 1600), one L40S

- Device memory: CG vectors (4 × 0.8 GB) + `pv`, `dsat` (2 × 0.8 GB) + per-side
  scratch `w[3]`, `div`, `tmp` (1.3 GB) + `P` index/mask (~0.4 GB) + SAT
  (~1 GB) + `T2`/`T3` ≈ **9 GB**. Fits an L40S with 4× headroom; fits the
  laptop at Δz = 10 (1150,1150).
- Per CG iteration: operator ≈ 16 GB traffic → ~19 ms at 864 GB/s; Krylov's
  vector ops ≈ 8 GB → ~9 ms; **≈ 28 ms**.
- Iterations: `n23^0.93` fit gives ~1,620 per solve at n23 = 321; 1681
  representatives → 2.7 M iterations → **~21 h**. Same 1681 solves as today,
  none of the 73 h assembly, none of the 155 GB host RAM.
- On an H100: ~7 ms/iteration → ~5 h. On the CPU shard path the same operator
  cuts memory from ~155 GB to < 1 GB per shard and traffic 3×, so the
  180 node-day estimate becomes ~60 — a fallback, not the route.

## Design

One new operator type, used by both CPU and GPU paths; everything downstream
(`CGSolver`, Krylov `cg!`, `fault_stiffness*`, D4 orbits, sharding, merge,
cache key) is unchanged because they only ever call `mul!` / `*`.

```julia
# src/ElasticitySplitNode.jl (as implemented)
struct Rows1D{VI,VV}                 # one 1D operator by rows (CSR): rowptr, colind, val
struct SideOps{R<:Rows1D,VV}         # one elastic half-space
    n::NTuple{3,Int}
    d1::NTuple{3,R}; d2::NTuple{3,R} # 1D first/second derivatives
    h::NTuple{3,VV}                  # 1D inner-product weights; H = h1[i]·h2[j]·h3[k]
end
struct SplitNodeOperator{S,VV,VI,TS}
    minus::S; plus::S
    λ::Float64; μ::Float64
    mask::VV; pm::VI; pp::VI         # P: far-field mask + averaged fault-pair DOFs
    SAT::TS                          # boundary-local sparse, Ntot × Ntot
    Ntot::Int
    pv::VV; dsat::VV; pavg::VV; w::NTuple{3,VV}; div::VV; tmp::VV   # scratch
end
mul!(y, op, v)          # y = -H P (D+SAT) P v
apply_P!(y, op, v)      # P v
hp_dsat!(y, op, χ)      # H P (D+SAT) χ — the RHS, and `shear_traction!`
duplicate_operator(op)  # shares everything but scratch (threaded builds)
AssembledSplitNode(A, HP_DSAT, P)   # the old matrices behind the same interface
```

Fields are flat vectors; every field access is `u[off + i + (j-1)n1 + (k-1)n1n2]`
so the identical `mul!` runs on `Vector` and `CuVector` — the CUDA extension
adds only the two hot loops (`axpass!`, `scale_H!`) as kernels and `to_device`.

`P` as mask + pair list reproduces `split_node_system`'s `P` exactly: far-field
rows zero (`boundary_indices` of every non-fault boundary, all 3 components),
fault pairs from `fault_node_pairs` (which already excludes the fault ∩
far-field ring) averaged for all 3 components. `P` is symmetric, so
`P` and `Pᵀ` need no separate code.

The 1D operators come from `sparse(first_derivative(g.grids[d], set))` etc. on
each `TensorGrid`'s 1D factor grids — 6 tiny matrices per side, milliseconds.

## Steps

**0. Prototype in the repo (done).** `scripts/extra/matrix_free_prototype.jl`
reproduces every number above: `julia --project=scripts
scripts/extra/matrix_free_prototype.jl 50 1600 1200 gpu`.

**1. `src/ElasticitySplitNode.jl` — the operator, CPU first.**
- `SideOps`, `SplitNodeOperator`, `split_node_operator(g_minus, g_plus, λ, μ, set)`.
- CPU `mul!` with the three axis-pass loops from the prototype, `Threads.@threads`
  over the outermost index; scratch preallocated in the struct.
- `apply_P!`, `hp_dsat!`.
- Build `mask`/pairs with the same `boundary_indices` / `fault_node_pairs`
  calls `split_node_system` uses. While there, **rebuild the assembled `P` from
  triplets too** (`sparse(I,J,V)` from the same mask + pairs) — it removes the
  DOF^1.7 term from the assembled path for free and keeps the two `P`s
  provably identical.
- `split_node_system` stays as is for tests and the diagnostic scripts
  (`split_node_spd.jl`, `symmetry_decomposition.jl`, `bp8_stiffness_spectrum.jl`).

**2. `CGSolver` — nothing to change** (`A::TA` is already generic; `:none` is the
only preconditioner in use and `jacobi_preconditioner` errors cleanly on a
non-matrix if someone asks).

**3. `src/FaultResponse.jl` — `FaultElasticity` holds the operator.**
- Replace fields `P`, `HP_DSAT`, `rs.A` by one `op::SplitNodeOperator` (keep
  `T2`, `T3`, `chi_rows*`, `omega`, …). Add keyword
  `representation = :kronecker | :assembled` to the constructor, default
  `:kronecker`; `:assembled` wraps the old matrices in the same interface
  (`apply_P!` → `P*v`, `hp_dsat!` → `HP_DSAT*χ`) so tests can diff them.
- `shear_traction!` (line ~203): `U = apply_P!(…, split_node_solve(solver,
  hp_dsat!(…, χ))) .+ χ`.
- `build_fault_elasticity` (BP8.jl:228) passes `representation` through; the
  `@info "split-node system ready"` line will now report seconds, not hours.

**4. `ext/EarthquakeDiffinitiveCUDAExt.jl` — GPU operator.**
- `to_device` for `SideOps` / `SplitNodeOperator` (CSR arrays of the
  1D ops → `CuVector{Int32}`/`CuVector{Float64}`, `SAT` → `CuSparseMatrixCSR`
  via the existing `to_csr`, scratch → `CuVector`), giving a device-resident
  operator with the same `mul!` interface. The prototype's `axpass_kernel!`
  and `Op1Dgpu` are the implementation; `apply_P!` is one gather kernel.
- `fault_stiffness_gpu`: `A_gpu = adapt(CuArray, fe.op)`; RHS
  `hp_dsat!` **on the device** (today `fe.HP_DSAT * χ` is a 1.7e9-nnz CPU
  SpMV per representative); `U = P u + χ` via `apply_P!` on the device;
  `T2`/`T3` as now. Loop, D4 orbit fill, stats, warnings unchanged.
- Drop the `precond === :none` guard's dependence on `rs.A`.

**5. Sharded GPU build** (so 1600² can also be split across several L40S, and
so a timed-out job loses at most one shard).
- `fault_stiffness_gpu(fe; shard, nshards)` = the existing loop over the slice
  of representatives `fault_stiffness_d4_shard` computes, writing the same
  shard file format; `merge_stiffness_cache.jl` then needs no change.
- `submit_bp8_gpu.sh`: optional `nshards` argument → `--array`, then the
  existing merge job, then the run. Without it: one job, as now.

**6. Scripts / submit sizing.**
- `build_stiffness_cache_gpu.jl`: replace the `nnz(A)`-based VRAM estimate and
  the `csr_bytes(fe.rs.A)` line with the vector-based one above; the pre-flight
  "needs ~23 GB" message becomes "~9 GB".
- `submit_bp8_gpu.sh`: Δz=10 default GPU `l40s` (drop the "needs h100" guard for
  L_fault > 1200), host `MEM` 32G at every Δz (no `A`), `TIME` 47:00:00 for a
  single 1600² job or 12:00:00 per shard with `nshards=4`.

**7. Tests** (add to `test/elasticity_split_node_test.jl`,
`test/fault_response_test.jl`, `test/fault_response_gpu_test.jl`; small grids,
`n1=9, n23=13` like the existing ones, seconds each):
- `mul!(SplitNodeOperator)` vs `split_node_system`'s `A` to 1e-14 on random `v`;
  `apply_P!` vs `P*v` exact; `hp_dsat!` vs `HP_DSAT*χ`.
- The rebuilt triplet `P` `==` the old `P` (structurally, after `dropzeros!`).
- `fault_stiffness` with `representation=:kronecker` vs `:assembled`: `K` equal
  to 1e-12 (CG-tolerance level); same with `fault_stiffness_d4` and the GPU
  path when CUDA is functional.
- A `JULIA_NUM_THREADS=4` run: the threaded CPU `mul!` shares no scratch across
  `duplicate`d solvers (give each `CGSolver` its own operator scratch, or make
  scratch per-call) — the same race class `fault_stiffness` already fixed once.

**8. Validation at scale — DONE locally (2026-09-14).** On the laptop RTX 2000
Ada, `scripts/build_stiffness_cache_gpu.jl 20 1200 1200` rebuilt the cluster's
own cache entry `K_exact_o4_dz20_lf400_Lf1200_Ln1200_1d794b2be6168695`
matrix-free:

| | cluster, assembled (L40S) | here, matrix-free (laptop RTX 2000 Ada) |
|---|---|---|
| assembly | 3811 s (1.06 h) | **23 s** (operator build) |
| solves | 441 reps, mean 665 CG iterations, 0 unconverged | 441 reps, **mean 665**, 0 unconverged |
| device memory | 3.2 GB | **0.5 GB** |
| total | 1.49 h on an L40S | 0.80 h on a laptop GPU with 1/4 the bandwidth |

Identical iteration counts across different hardware and representations is the
sharp check: CG's iteration count is a property of the operator's spectrum, so
matching it to the iteration means the two operators are the same one.

**End to end**: `run_bp8.jl gs 20 1200 1200 exact` against that matrix-free `K`
reproduces the cluster's `output/BP8-QD-GS_dz20_Lf1200_Ln1200_exact_gpu/`
**exactly** — worst relative `|ΔV_max|` over all 12,961 output rows is `0.0`,
as are slip and moment rate, and the summary line matches digit for digit
(`peak slip rate 3.3050E-07 m/s at t = 1.387 days`, `final slip 3.8469E-02 m`).
The §4 files are written at 6-7 significant figures, so this says the two `K`
agree to well beyond output precision, not that they are bitwise equal.

Smaller checks, all in the test suite: `A v` to 3e-16, `K` across
representations and build paths to 1e-9, GPU shards vs whole to 1e-8, and one
`K` built three ways (one GPU job, two GPU shards + merge, threaded CPU)
agreeing to 1e-16.

**Done on the cluster**: Δz = 10 m (1150, 1150) matrix-free on an L40S took
11.89 h against 40.4 h assembled, and that measurement is `submit_bp8_gpu.sh`'s
Δz = 10 m `EST_H` anchor. The (1600, 1600) run then came in at 48.1 h against
the 38.8 h that anchor predicts — see "Status" above.

**9. Production — what remains.** The (1600, 1600) Δz = 10 m `K` has been
built as 8 L40S shards; merge them (`merge_stiffness_cache.jl 10 1600 1600`)
and run `run_bp8.jl gs/pw 10 1600 1600 exact` against the cache entry. On
measured timings a single job wants ~48 h of L40S, over the 47 h cap, so either
shard it or use an h100 (`… 10 1600 1600 h100`), where the script's /3
bandwidth heuristic puts it near 16 h.

## Risks and what bounds them

- **Kernel speed on the L40S vs the laptop.** Both are Ada; the L40S has 3.9×
  the bandwidth. If the axis passes reach only half the roofline there, the
  1600² build is ~2 days on one card — still inside a 47 h walltime with
  `nshards=2`. Step 8's (1150,1150) rerun measures this before 1600² is queued.
- **CG vector ops are now a third of the per-iteration cost.** Krylov's `cg!`
  on `CuVector` is fine; do not add a preconditioner "while there" — Jacobi is
  measured to regress (PERFORMANCE.md §6).
- **Round-off parity.** The Kronecker apply sums the same terms in a different
  order than the assembled SpMV — 3e-16 measured, CG-tolerance on `K`. Not a
  concern, but it is why step 8's diff is against `rtol`, not `==`.
- **`P` on the fault ∩ far-field ring.** Reusing `fault_node_pairs` and the
  same `boundary_indices` loops as `split_node_system` is the guarantee; the
  step-7 `P == P` test is the check.
- **Toeplitz** (`fault_stiffness_toeplitz`, 10 solves, 0.41–0.87 % `V_max`
  error) becomes minutes with this operator but is not needed once exact costs
  a day; keep it as the fallback if step 8 shows the kernels well below roofline.

Effort: steps 1–4 and 7 are ~2 days; 5–6 half a day; 8 is machine time.

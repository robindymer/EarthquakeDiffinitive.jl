# Running BP8-QD-GS on UPPMAX

Step-by-step for producing a SEAS BP8 submission on a cluster. Written to be
read over SSH.

Background for the numbers here: `PERFORMANCE.md` §4 (cost model),
`PROGRESS.md` "Domain requirement relaxes with resolution", and §4/§5/§6 of
`context/SEAS_BP8_Benchmark_Description.pdf`.

---

## TL;DR — the whole thing is one command per resolution

Edit the config block at the top of `scripts/submit_bp8.sh` **once** — project
id, cache directory, partition names, mail address. It refuses to run while the
`CHANGE-ME` placeholders are still there. Then:

```bash
./scripts/submit_bp8.sh 20      # Δz = 20 m
./scripts/submit_bp8.sh 10      # Δz = 10 m
```

Each call submits three chained SLURM jobs and returns immediately:

1. an **array job** building `K`'s columns in independent shards
2. a **merge** job stitching them into one cache entry
3. the **run**, which reads that `K` and writes the §4 output files

You submit once and collect results when the last one finishes. Both
resolutions can be in flight at the same time — they use different cache keys
and different output directories, so they never collide. §6 of the PDF allows
**at most two resolutions and two domain sizes**, so exactly these two is the
plan.

Watch: `squeue -u $USER`. Outputs: `output/BP8-QD-GS_dz<N>_Lf<N>_Ln<N>_exact/`.
Logs: `logs/`.

Defaults the script picks:

| Δz | domain | shards | RAM/node | total `K` build |
|---|---|---|---|---|
| 20 m | (1600, 1200) | 32 | ~15 GB | ~12.8 node-days |
| 10 m | (1200, 1200) | 440 | ~65 GB | ~415 node-days |

Override any of them: `./scripts/submit_bp8.sh 10 1600 1200 600`
(`<Δz> <L_fault> <L_normal> <nshards>`).

---

## Before the first submission

The config block at the top of `scripts/submit_bp8.sh` is already filled in from
your working Pelle script (account `uppmax2026-1-45`, partition `pelle`, mail
`robin.dymer@it.uu.se`). One line in it is marked **CHECK** — the
storage path. Then run the one-time setup below.

### Use Julia 1.11, not 1.10

Pelle has both, and 1.11 is the default:

```
Julia/1.10.9-LTS-linux-x86_64    Julia/1.11.3-linux-x86_64 (D)
```

The config block is set to `Julia/1.11.3-linux-x86_64`. **Do not switch it to
1.10.** `Project.toml` pins the exact Diffinitive revision in a `[sources]`
block, 1.10 ignores `[sources]` entirely, and `Manifest.toml` is gitignored so
there is no other pin to fall back on — a 1.10 instantiate resolves Diffinitive
from the registry and fails to compile with `invalid subtyping in definition of
IsotropicElasticOperator`. All 440 array tasks would fail identically.

### Add the Diffinitive registry first

`Diffinitive` and `Tokens` are **not in the General registry** — they come from
`https://github.com/Diffinitive/diffinitive_registry`. Your laptop has it
added already, so this is invisible locally, but a fresh depot on Pelle does
not, and `instantiate` fails with:

```
ERROR: expected package `Tokens [040c2ec2]` to be registered
```

Registries are per-depot, so adding it once covers every project on Pelle.
(`context/notes_robin.md`'s "registry up" assumes it is already there.)

The `scripts/` environment additionally needs `EarthquakeDiffinitive` itself,
which is in no registry at all, **and** the Diffinitive revision pinned rather
than the registry's tagged `0.1.8` — those are different git trees, and the
tagged one does not compile. Both are now pinned in a `[sources]` block in
`scripts/Project.toml`, so a plain `instantiate` is all you need. If you ever
bump the Diffinitive revision in the root `Project.toml`, change it there too —
the root's `[sources]` does not propagate to `scripts/`.

### One-time setup

```bash
module load Julia/1.11.3-linux-x86_64
cd /path/to/EarthquakeDiffinitive.jl

# 1. the custom registry — HTTPS, not the SSH URL in its own Registry.toml
julia -e 'using Pkg
          Pkg.Registry.add(RegistrySpec(
              url="https://github.com/Diffinitive/diffinitive_registry"))'

# 2. then the environments
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'

# 3. both must print ok before you submit anything
julia --project=. -e 'using EarthquakeDiffinitive; println("ok")'
julia --project=scripts -e 'using EarthquakeDiffinitive; println("ok")'
```

If step 1 says the registry already exists, `Pkg.Registry.update()` instead.

### Partitions and memory

The script requests memory explicitly (`--mem`), sized from the table below. If
Pelle rejects `--mem` because it ties memory to core count, drop it and raise
`CORES` in the script until cores x GB-per-core clears the requirement:

```bash
sinfo -o "%P %D %c %m %f"        # partitions, cores, MB per node
```

## Does it fit in memory on UPPMAX?

**Sharding parallelises time, not memory.** Every shard assembles its own copy
of `A`+`HP_DSAT`, so more nodes never reduces what each one needs. This is the
first thing to check.

| run | RAM/node | on a 128 GB node |
|---|---|---|
| **Δz = 20 m, (1600, 1200)** | **~15 GB** | trivial, any standard node |
| Δz = 10 m, (1200, 1200) | ~65 GB | fits with real headroom |
| Δz = 10 m, (1600, 1200) | ~116 GB | **too tight** — needs a fat node |

So Δz = 20 m is comfortable anywhere. Δz = 10 m fits standard nodes **only** with
the relaxed (1200, 1200) domain — which was measured sufficient at Δz = 25 m and
is an **extrapolation** at 10 m (`PROGRESS.md` says so; (1200, 800) was never
tested). The conservative domain at 10 m needs fat nodes and a `-C mem256GB`-style
constraint; pass it explicitly if you want it.

Per-shard extras are negligible — a shard's slice of `K` is a few MB — and the
merge needs only ~2x the final `K` (~2.8 GB at Δz = 10 m), which is why it runs
on 4 cores rather than a whole node.

The script asks for **8 cores / 24 GB** per shard at Δz = 20 m and **16 cores /
96 GB** at Δz = 10 m. Cores barely matter — `PERFORMANCE.md` §4 measured
threading efficiency at ~15% because the sparse mat-vec is bandwidth-bound, so
the core count is really there to secure the memory and to keep each shard
inside its 24 h walltime.

**Walltime, not total compute, sets the shard count.** At Δz = 10 m a column
costs ~2730 s, so a 24 h job fits ~30 columns and 13,122 columns need ~440
tasks. At Δz = 20 m a column is ~328 s, so 32 tasks is already generous.

---

## Running it on a GPU instead

**One card replaces the whole job array.** `fault_stiffness_gpu`
(`PERFORMANCE.md` §5 item 0c) holds `A` resident on one device and runs the
same D4 orbit representatives against it, so there are no shards and no merge —
two chained jobs instead of three, and the build writes the finished cache
entry itself.

### One-time setup, on top of the CPU setup above

```bash
julia --project=scripts -e 'using Pkg; Pkg.instantiate()'
```

`CUDA` is a weakdep of the root project, so the environment that wants the
extension has to depend on it directly; it is now in `scripts/Project.toml`.

**Then provision CUDA.jl's runtime from inside a GPU allocation.** This is not
optional on Pelle, and skipping it is the single most likely way for the GPU
chain to fail:

```bash
interactive -A uppmax2026-1-45 -p gpu --gpus=l40s:1 -c 4 -t 01:00:00
module load Julia/1.11.3-linux-x86_64
cd /proj/efficient_elastic/efficient_elastic/nobackup/EarthquakeDiffinitive.jl
nvidia-smi                       # confirm a card and a driver are actually here

# 1. Recompile the CUDA JLLs *on this node*, where the driver is visible, so
#    CUDA.jl re-runs its artifact selection against a real driver. All of
#    them, not just the runtime — see below.
julia --project=scripts -e '
    using Pkg
    for (uuid, e) in Pkg.Types.Context().env.manifest
        startswith(e.name, "CUDA_") && endswith(e.name, "_jll") || continue
        @info "recompiling $(e.name)"
        Base.compilecache(Base.PkgId(uuid, e.name))
    end'

# 2. Fresh process — the JLLs are read at load time. This is also the CUDA.jl
#    precompile; do it here, not in a batch job you are paying for.
julia --project=scripts scripts/gpu_smoke_test.jl

# 3. Read the runtime version the smoke test printed and pin it, so the next
#    login-node `Pkg.instantiate()` cannot undo any of this.
julia --project=scripts -e 'using CUDA; CUDA.set_runtime_version!(v"X.Y")'

# 4. Re-run the smoke test *before releasing the allocation*. Step 3 changes a
#    Preferences entry, which invalidates every CUDA precompile cache that
#    depends on it — so the next `using CUDA` recompiles. Spend that here, in
#    an allocation you are already holding, rather than inside a batch job on
#    GPU walltime. It also confirms the pinned version actually resolves.
julia --project=scripts scripts/gpu_smoke_test.jl
```

After step 4 passes, **everything else happens from the login node.**
`submit_bp8_gpu.sh` only calls `sbatch`; it needs no GPU, and running it from
inside an interactive allocation just burns the allocation waiting on a queue.
Exit the interactive session first.

**Recompile every `CUDA_*_jll`, not only `CUDA_Runtime_jll`.** The CUDA.jl
error message names only the runtime, and following it literally gets you a
node that reports `runtime 13.3.0, artifact installation` and then dies with

```
ERROR: UndefVarError: `ptxas` not defined in `CUDA_Compiler_jll`
```

The runtime and the *compiler* are separate artifacts resolved by separate
JLLs, and both were precompiled without a driver, so both are stale. The same
missing `CUDA_Compiler_jll` is why CUDA.jl warns

```
CUDA runtime library `libnvJitLink.so.13` was loaded from a system path,
`/usr/local/cuda/targets/x86_64-linux/lib/libnvJitLink.so.13`
```

— nvJitLink ships in that JLL, so when it is unavailable CUDA.jl falls back to
Pelle's `/usr/local/cuda`. That warning is a *symptom* of the stale compiler
JLL, not the separate "do not `module load CUDA`" problem below, and it should
disappear once step 1 covers every `CUDA_*_jll`. If it survives step 1, then it
is a genuine `LD_LIBRARY_PATH` leak and worth chasing.

**Why recompile rather than pin a version straight away.** The instantiate
above ran on a login node, which has no NVIDIA driver. CUDA.jl chooses its CUDA
toolkit *artifact* by asking the driver what it supports **at precompile
time**, so that instantiate recorded "no runtime found" — and Julia will not
invalidate the cache when you later land on a GPU node, because nothing in the
*environment* changed, only the hardware. `compilecache` re-runs that selection
here, where the driver is real, and so needs no version from you. Step 3 then
freezes whatever it picked into `scripts/LocalPreferences.toml`, turning the
choice into a stated preference rather than a driver query, which is what makes
it survive the next login-node precompile.

Do not try to read the version out of `nvidia-smi` and pin *that*. Pelle's
header reads `CUDA UMD Version: 13.3`, not the `CUDA Version: 13.3` that every
scripted extraction expects, so the obvious `sed` silently yields an empty
string; and the number is in any case a *ceiling* — the newest runtime the
driver can support — rather than one CUDA.jl is guaranteed to ship an artifact
for. Let step 1 choose and pin only what it chose.

If step 2 reports no runtime at all, CUDA.jl could not match the driver — pin
the newest toolkit it ships instead (`v"12.9"`, then `v"12.6"`); a CUDA 13
driver runs a CUDA 12 runtime fine, the compatibility only fails the other way.

**Observed on Pelle, 2026-09-10:** driver 610.57.04 / CUDA 13.3 on an L40S
(46068 MiB, i.e. ~45 GiB usable, not the 48 GB the sizing tables round to);
step 1 selected `runtime 13.3.0, artifact installation`.

### Run them in this order

```bash
# 0. smoke test, in an interactive allocation — this is where CUDA.jl
#    precompiles, better here than inside a batch job. It compiles a kernel
#    and runs a cuSPARSE spmv, which `CUDA.versioninfo()` does not.
julia --project=scripts scripts/gpu_smoke_test.jl

./scripts/submit_bp8_gpu.sh 20              # ~1 h on an L40S — do this first
./scripts/submit_bp8_gpu.sh 10              # relaxed (1200,1200), H100
./scripts/submit_bp8_gpu.sh 10 1600 1200    # converged, H100
```

Δz = 20 m first is not caution for its own sake: it exercises the entire path
at a size any card handles, its `K` can be **diffed against the CPU build** of
the same configuration, and it prints `assembly X.XX h`, which is the single
most uncertain number in the estimates below.

Fourth argument overrides the card (`l40s` / `h100`).

### Sizing

| | Δz = 20 m | Δz = 10 m relaxed (1200,1200) | Δz = 10 m converged (1600,1200) |
|---|---|---|---|
| DOF | 9.5 M | 42.2 M | 74.8 M |
| `nnz(A)` | 0.44e9 | 1.99e9 (Int32, 8% margin) | **3.55e9 (forces Int64)** |
| **VRAM** | ~7 GB | **~28 GB** | **~65 GB** |
| host RAM, **peak** | ~29 GB | ~125 GB | **~222 GB** |
| card | L40S | H100 (L40S: ~28 h) | **H100 only** |
| assembly (est.) | ~2 h | ~16 h | ~36 h |
| solve (est.) | ~1 h | ~7 h | ~20 h |

`sinfo` on Pelle gives ten 48 GB L40S and only **two** H100, both on 386 GB
nodes — so card choice is partly a queue-time decision, which is why Δz = 20 m
defaults to an L40S.

**Host RAM is the peak during assembly, ~2x the final matrix, not 1x.**
Measured at four sizes (92 k – 2.77 M DOF), `peak = 2.03 * final + 1.6 GB` fits
to within 0.3 GB everywhere. Sizing the request from the *final* `A`+`HP_DSAT`
figure under-requests by half and OOMs during assembly, hours before the GPU is
touched. `submit_bp8_gpu.sh` sizes from the peak.

**The device figure is not the host figure, and §5 item 0c's fit table
conflated them.** That table rules the converged domain out at "~116 GB", but
that is the *host* CSC/Int64 footprint of `A` **plus** `HP_DSAT`. `HP_DSAT`
never goes to the device — the right-hand side is formed on the host — so the
device holds only `A`, `P`, `T2`, `T3`. The converged domain is a one-GPU job.

### Assembly, not the solve, is now the bottleneck

Measured, single build, warm JIT, **single-threaded** (100% of one core — extra
cores do nothing for this phase):

| DOF | assemble | local exponent |
|---|---|---|
| 92 k | 24.9 s | — |
| 360 k | 99.8 s | 1.02 |
| 692 k | 216.4 s | 1.19 |
| 2.77 M | 1435 s | **1.37** |

Superlinear with a **rising** exponent. This contradicts `PERFORMANCE.md`
§4c's "minutes-long sparse assembly at Δz = 20 m" — that is an extrapolation
in the doc from a Δz = 100 m measurement, and it is off by roughly two orders
of magnitude.

The extrapolations in the sizing table are therefore **lower bounds**, and they
are 27x beyond the largest measured point on different hardware. If they hold
on Pelle, the converged Δz = 10 m job is ~36 h assembly + ~20 h solve = **over
the 2-day GPU limit**. Do not submit it until Δz = 20 m has reported its real
`assembly` figure.

**This also bears on the CPU path.** Every shard re-assembles `A`, so an
880-shard Δz = 10 m array spends the overwhelming majority of its allocation on
redundant assembly rather than on solving — roughly 90% at these rates. Far
fewer shards (~120) still fit a 24 h walltime and are several times cheaper.

### Nothing gets overwritten

`run_bp8.jl` appends `$BP8_OUTPUT_SUFFIX` to the output directory, and
`submit_bp8_gpu.sh` sets it to `gpu`:

```
CPU chain   output/BP8-QD-GS_dz20_Lf1600_Ln1200_exact/
GPU chain   output/BP8-QD-GS_dz20_Lf1600_Ln1200_exact_gpu/
```

The **`K` cache is deliberately still shared** — same key, both paths. They
produce the same matrix, so re-keying it would force a redundant multi-hour
rebuild of something already on disk. If the CPU chain finishes a configuration
first, the GPU build sees the entry and exits in seconds.

Note the GPU outputs carry **12,961 rows against an older CPU run's 8,641**,
because `saveat` was corrected from 300 s to 200 s (§4.1 requires 1e4–1e5;
saveat=300 gives 8,641, which violates it). That is the fix, not a discrepancy.

### Two things that had to be fixed to make the converged domain fit

1. **The upload used to peak at ~122 GB** on a 94 GB card.
   `CuSparseMatrixCSR(::SparseMatrixCSC)` expands to
   `CuSparseMatrixCSR(CuSparseMatrixCSC(M))` — it uploads CSC, then converts
   to CSR *on the device*, so both plus a cuSPARSE scratch buffer are resident
   at once. `to_csr` now skips the conversion for a matrix that is its own
   transpose (`A` is, to 6.4e-17; `P` exactly), since CSC arrays read as CSR
   are already the transpose. Checked at runtime, values as well as pattern —
   `P` is structurally symmetric, so a pattern-only test would wave through a
   silent `Pᵀ`.
2. **`nnz` crosses `typemax(Int32)`.** The converged `A` has ~3.55e9
   nonzeros; a 32-bit CSR row pointer cannot hold that and would overflow
   silently. `to_csr` picks the width from `nnz`. The relaxed domain is under
   the limit by only ~8%, so this is not a margin to spend.

### How the solve estimate was made, and how much to trust it

A **bandwidth model, not a measurement** — `TODO.md`'s open item is exactly
this, and achieved cuSPARSE bandwidth on an H100 is the unverified input.

CG iterations were measured at Δz = 50/40/25 m (293, 358, 551) and fit
`~n23^0.93`, close to the `O(h⁻¹)` theory for unpreconditioned CG, giving
~1,280 iterations at Δz = 10 m relaxed and ~1,670 converged. Cost per iteration
is `bytes(A)/bandwidth` at 70% of 3.9 TB/s.

The cross-check: running the model *backwards* through the two documented CPU
per-solve times (2730 s relaxed, 6150 s converged) implies effective CPU
bandwidth of **15.0 and 15.6 GB/s** — two independent configurations agreeing
to 4%, and consistent with a laptop measurement of 13.2 GB/s and with §4's
"threading efficiency ~15%".

The implied speedup is ~150-180x, far above the 8.4x measured on an RTX 2060 at
56k DOF. Mechanically consistent — at 56k DOF the CPU works out of cache, at
production it does not — but a large extrapolation.

### Known inefficiency, not yet fixed

`rhs = fe.HP_DSAT * χ` traverses the entire ~59 GB matrix while `χ` has exactly
**two** nonzeros. On CPU that is 0.06% of a 6150 s solve and invisible; on GPU
it is ~4 s against a ~39 s solve, i.e. ~10% of solve time, and it is a few
lines to fix by slicing the two columns.

---

## Note on §4.3 profile spacing

`write_profiles` emits the profile at the **computational grid** spacing: at
Δz = 10 m the coordinate row is at 10 m, at Δz = 20 m it is at 20 m. §4.3 asks
for "a spacing of 10 m (exactly)", but §6 allows results from two different
spatial resolutions, and the profiles are reported on the grid each run actually
used. **Decided (Robin, 2026-09-09): submit as-is, no interpolation onto a
separate reporting grid.**

§4.1 station files and §4.2 `global.dat` are resolution-independent and
compliant either way.

---

## If something fails

**CUDA.jl was precompiled without a driver.** The job dies immediately with

```
CUDA.jl could not find an appropriate CUDA runtime to use.
CUDA.jl's JLLs were precompiled without an NVIDIA driver present.
...
ERROR: LoadError: no functional CUDA device
```

This is **not** a missing `--gpus`, even though the older error text said so —
`submit_bp8_gpu.sh` always passes `--gpus`, and the build script now probes
`nvidia-smi` to tell the two apart. It is the login-node precompile described
under "One-time setup" above: run `CUDA.set_runtime_version!` from a GPU
allocation and resubmit. The submit script also carries an in-job fallback that
recompiles `CUDA_Runtime_jll` on the allocated node, but that can cost ~45 min
of a walltime you paid for, so pin the version instead of relying on it.

**A shard times out or dies.** Resubmit just the array — a shard whose file
already exists exits immediately, so only the missing ones rebuild:

Simplest fix: just run `./scripts/submit_bp8.sh 20` again. The shards that
already wrote their files exit immediately, so only the missing ones rebuild,
and you get a fresh merge and run chained behind them.

**The merge reports missing columns.** That is the check working. It validates
the union of columns actually on disk, not a claimed shard count, and refuses to
write a partial `K` — which matters because a partial `K` is the right shape and
full of plausible numbers, so it would produce a wrong submission rather than an
error. Rebuild the named shards and merge again.

**Running the steps by hand**, if you'd rather not use the chain:

```bash
export EQD_STIFFNESS_CACHE=/proj/.../eqd-stiffness
export JULIA_NUM_THREADS=8
julia --project=scripts scripts/build_stiffness_cache.jl 20 1600 1200 exact $i 32
julia --project=scripts scripts/merge_stiffness_cache.jl 20 1600 1200
julia --project=. scripts/run_bp8.jl gs 20 1600 1200 exact
```

Delete the `.shard*` files once merged; the `.eqdk` is all `run_bp8.jl` needs.

---

## Packaging and upload (§5)

```bash
cd output/BP8-QD-GS_dz20_Lf1600_Ln1200_exact
zip ../RobinDymer_v1.zip *.dat
```

Log in to the CRESCENT code verification server, choose benchmark ID
**`BP8-QD-GS`**, and upload a single zip named `nameOfModeler_version.zip`. The
platform sorts by **filename**, so do not rename anything — wrong filenames mean
nothing is visualised.

`BP8_MODELER` (default "Robin Dymér") is what lands in every file header.

---

## Submission checklist (§4)

20 files, all written by `write_outputs`:

- [ ] **§4.1** — 9 station files `fltst_strk±NNNdp±NNN.dat`, 11 fields each
      (`t`, `slip_2/3`, `slip_rate_2/3` as log10, `shear_stress_2/3`,
      `pore_pressure`, `darcy_vel_2/3`, `state` as log10).
- [ ] Row count 1e4–1e5. **`run_bp8.jl` uses `saveat=300.0`, giving 8,641 rows
      over 30 days — just under.** `saveat=200.0` gives 12,961; change it in
      `scripts/run_bp8.jl` if you want to be safely inside the range.
- [ ] **§4.2** — `global.dat`: `t`, max slip rate (log10 m/s), moment rate.
- [ ] **§4.3** — 10 profile files (`slip_2`, `slip_3`, `shear_stress_2`,
      `shear_stress_3`, `pore_pressure` × `strike`, `depth`), hourly. Coordinate
      row is at the run's own grid spacing — see the note above.
- [ ] Headers carry modeler, date, `element_size`, and the domain line.

---

## Known gaps

1. **BP8-PW is not ready.** It is stiffness-bound (`K_ww/D` at the floored well
   cell) and the `σ̄_min` = 100 kPa regularisation is measured but **not adopted**
   (`par.σ̄_min` is still 1 kPa). `TODO.md` item 2 also flags it as a deliberate
   physics perturbation worth raising with the organisers. GS is unaffected — it
   is not stiffness-bound.
2. **The (1200, 1200) domain at Δz = 10 m is an extrapolation**, not a
   measurement.

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

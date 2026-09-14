# Build `K` for one configuration on a single GPU and put it in the stiffness
# cache — the GPU counterpart of `build_stiffness_cache.jl`.
#
# WHY A SEPARATE ENTRY POINT (rather than a flag on the CPU script). The two
# have different job shapes. The CPU build is sharded across many nodes; the
# GPU build is one job on one device — or, with `[shard] [nshards]`, a few —
# and writes a complete cache entry directly via `save_stiffness` when
# unsharded, so `run_bp8.jl` reads it through the normal `EQD_STIFFNESS_CACHE`
# lookup with no further step. Same cache key as the CPU path, so the two are
# interchangeable: whichever finishes first satisfies the run, and the second
# one sees the file and exits.
#
# Run:
#   export EQD_STIFFNESS_CACHE=/path/to/scratch/eqd-stiffness
#   julia --project=scripts scripts/build_stiffness_cache_gpu.jl [Δz] [L_fault] [L_normal] [shard nshards]
#
# With no arguments it builds the Δz = 20 m submission configuration. Only
# `:exact` is supported — `:toeplitz` is 10 solves and does not need a GPU.
# With `shard nshards` it writes a `.shard<k>` file of the D4 representatives
# `shard:nshards:end` (same format as the CPU shards) for
# `merge_stiffness_cache.jl` to assemble — to spread one build over several
# cards, or to keep one inside a walltime.
#
# MEMORY. The elastic system is applied **matrix-free** (`SplitNodeOperator`,
# MATRIX_FREE_PLAN.md): nothing of size `nnz(A)` is ever formed, on the host
# or the device. What the device holds is ~9 vectors of system length (the
# CG workspace, rhs, solution, the operator's scratch) plus the small
# boundary-local SAT block and `T2`/`T3`:
#
#     Δz = 20 m (1600, 1600), 12.6 M DOF   ~1 GB
#     Δz = 10 m (1150, 1150), 37 M DOF     ~3 GB   (assembled: 23 GB measured)
#     Δz = 10 m (1600, 1600), 100 M DOF    ~9 GB   (assembled: ~85 GB, Int64 CSR)
#
# so any BP8 configuration fits an L40S, and host RAM is the dense `K`
# (1.3 GB at Δz = 10 m) plus a few vectors. There is no assembly step to wait
# through either: the operator is built in seconds, where the assembled `A`
# took 17.5 h at Δz = 10 m (1150, 1150) and would take ~73 h on (1600, 1600).
#
# The script prints the estimate and the device's actual free VRAM before it
# starts, and refuses to begin a build that cannot fit.
using EarthquakeDiffinitive
using EarthquakeDiffinitive.BP8
using EarthquakeDiffinitive.FaultResponse: fault_stiffness_gpu, fault_grid_axes,
                                           frictional_node_count, elastic_solver_report
using EarthquakeDiffinitive.StiffnessCache
using Diffinitive.SbpOperators
using SparseArrays
using Printf
using CUDA

include(joinpath(@__DIR__, "gpu_vram.jl"))

const DEFAULT_Δz = 20.0
const DEFAULT_L_FAULT = 1600.0
const DEFAULT_L_NORMAL = 1200.0

# TWO DIFFERENT FAULTS LOOK IDENTICAL HERE, and saying only "no GPU" sent one
# debugging session down the wrong path entirely. `CUDA.functional()` is false
# both when the allocation has no device and when it has one that CUDA.jl
# cannot use. `nvidia-smi` separates them, because the *driver* is the half
# that comes from the node while the *runtime* is the half that was chosen at
# precompile time — deliberately a shell probe rather than a Julia one, since
# the Julia side is exactly what is broken in the second case.
if !CUDA.functional()
    driver_present = try
        success(pipeline(`nvidia-smi -L`; stdout=devnull, stderr=devnull))
    catch
        false
    end
    if !driver_present
        error("""
            no functional CUDA device and no NVIDIA driver on this node — this
            script is the GPU build path. On SLURM that usually means the job
            was submitted without `--gpus`; use `scripts/submit_bp8_gpu.sh`, or
            run the CPU path (`scripts/build_stiffness_cache.jl`) instead.""")
    else
        error("""
            this node has a GPU driver but CUDA.jl has no usable CUDA runtime,
            so it cannot touch the card. This is a *precompilation* fault, not
            a SLURM one: CUDA.jl picks its toolkit artifact by asking the
            driver at precompile time, so a `Pkg.instantiate()` run on a login
            node bakes in "no runtime found", and Julia will not invalidate
            that cache merely because the hardware changed.

            Fix it once, from a node that has a driver — re-run the artifact
            selection *there*, then freeze what it picked:

                julia --project=scripts -e 'pkg = Base.PkgId(Base.UUID(
                    "76a88914-d11a-5bdc-97e0-2f5a05c973a2"), "CUDA_Runtime_jll")
                    Base.compilecache(pkg)'
                julia --project=scripts -e 'using CUDA; CUDA.versioninfo()'
                julia --project=scripts -e 'using CUDA; CUDA.set_runtime_version!(v"X.Y")'

            with `X.Y` the `CUDA runtime` line the second command prints — not
            the version in `nvidia-smi`'s header, which is the newest the
            driver *could* support and may have no CUDA.jl artifact. The last
            command writes `scripts/LocalPreferences.toml`, so later precompiles
            no longer need a driver present. See CLUSTER_RUNBOOK.md "CUDA.jl was
            precompiled without a driver".""")
    end
end

dir = stiffness_cache_dir()
dir === nothing && error("""
    EQD_STIFFNESS_CACHE is not set, so there is nowhere to put the result.
    Set it to a directory with room for the file — 2·N_Ωf square, Float64, so
    ~86 MB at Δz = 20 m and ~1.3 GB at Δz = 10 m.""")

Δz = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : DEFAULT_Δz
L_fault = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : DEFAULT_L_FAULT
L_normal = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : DEFAULT_L_NORMAL
shard = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : nothing
nshards = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : nothing
(shard === nothing) == (nshards === nothing) ||
    error("pass both [shard] and [nshards], or neither")
shard === nothing || 1 <= shard <= nshards ||
    error("shard must be in 1:nshards, got shard=$shard nshards=$nshards")

par = benchmark_parameters()
order = 4
set = read_stencil_set(SbpOperators.sbp_operators_path() * "standard_diagonal.toml"; order)
key = stiffness_cache_key(; λ=EarthquakeDiffinitive.BP8.lame_lambda(par), μ=par.μ,
                          l_f=par.l_f, Δz, L_fault, L_normal, order, stencil=set,
                          stiffness=:exact)
path = shard === nothing ? stiffness_cache_path(dir, key) :
                           joinpath(dir, key.name * ".shard$shard")

nf = (round(Int, 2par.l_f / Δz) + 1)^2
n1, n23 = fault_grid_sizes(par, Δz, L_fault, L_normal, order)
Ntot = 3 * 2 * n1 * n23 * n23

@printf("""
target      Δz = %g m, L_fault = %g m, L_normal = %g m, exact (GPU, matrix-free)
grid        %d x %d x %d per side, %d DOF
Ω_f nodes   %d  (K is %d x %d, %.1f MB on disk)
shard       %s
device      %s, %.1f GB total, %.1f GB free
output      %s
""", Δz, L_fault, L_normal, n1, n23, n23, Ntot, nf, 2nf, 2nf, (2nf)^2 * 8 / 2^20,
     shard === nothing ? "none (whole K in one job)" : "$shard of $nshards (D4 representatives $shard:$nshards:end)",
     CUDA.name(CUDA.device()), vram_total() / 2^30,
     vram_free() / 2^30, path)

if isfile(path)
    println("\nalready written — nothing to do (delete the file to rebuild it)")
    exit(0)
end

# ---- VRAM check --------------------------------------------------------------
# Matrix-free, the device footprint is arithmetic on `Ntot`: Krylov's CG
# workspace (4 vectors), rhs, solution, χ and the operator's own scratch
# (`pv`, `dsat`, five field-length vectors ≈ 1.7 system-length ones), plus the
# SAT block (boundary-local, ~1% of an assembled `A`) and `T2`/`T3`.
need = (7 + 2 + 5 / 3) * Ntot * 8 * 1.10
free0 = vram_free()
@printf("VRAM        ~%.1f GB needed (vectors + SAT), %.1f GB free\n", need / 2^30, free0 / 2^30)
need < free0 || error("""
    this configuration needs ~$(round(need / 2^30, digits=1)) GB of VRAM but only \
    $(round(free0 / 2^30, digits=1)) GB is free on $(CUDA.name(CUDA.device())).""")

t0 = time()
fe = build_fault_elasticity(; par, Δz, L_fault, L_normal, n1, n23, set, verbose=true)
@printf("operator    built in %.1f s (matrix-free; no assembly)\n", time() - t0)

x2, x3 = collect.(fault_grid_axes(fe))
if shard === nothing
    K = fault_stiffness_gpu(fe; verbose=true)
    save_stiffness(path, key, K, x2, x3)
else
    cols, Kc = fault_stiffness_gpu(fe; verbose=true, shard, nshards)
    save_stiffness_shard(path, key, cols, Kc, x2, x3)
end

rep = elastic_solver_report(fe)
@printf("\ndone in %.2f h → %s (%.1f MB)\nsolves %d, mean CG iterations %.0f, unconverged %d\n",
        (time() - t0) / 3600, path, filesize(path) / 2^20,
        rep.solves, rep.iterations / max(rep.solves, 1), rep.unconverged)
rep.unconverged == 0 || error("$(rep.unconverged) solve(s) did not converge — K is not trustworthy")

# Build `K` for one configuration on a single GPU and put it in the stiffness
# cache — the GPU counterpart of `build_stiffness_cache.jl`.
#
# WHY A SEPARATE ENTRY POINT (rather than a flag on the CPU script). The two
# have opposite job shapes. The CPU build is sharded across many nodes and
# needs a merge step afterwards; the GPU build is *one* job on *one* device
# holding the whole `A` in VRAM, with no shards and nothing to merge. It
# therefore writes a complete cache entry directly via `save_stiffness`, and
# `run_bp8.jl` reads it through the normal `EQD_STIFFNESS_CACHE` lookup with
# no further step. Same cache key as the CPU path, so the two are
# interchangeable: whichever finishes first satisfies the run, and the second
# one sees the file and exits.
#
# Run:
#   export EQD_STIFFNESS_CACHE=/path/to/scratch/eqd-stiffness
#   julia --project=scripts scripts/build_stiffness_cache_gpu.jl [Δz] [L_fault] [L_normal]
#
# With no arguments it builds the Δz = 20 m submission configuration. Only
# `:exact` is supported — `:toeplitz` is 10 solves and does not need a GPU.
#
# MEMORY. Two separate requirements, and the host one is the larger:
#
#   * host RAM  — `build_fault_elasticity` assembles `A` and `HP_DSAT` as
#     Float64/Int64 CSC. That is ~15 GB at Δz = 20 m, ~65 GB at Δz = 10 m on
#     the relaxed (1200, 1200) domain and ~116 GB on the converged
#     (1600, 1200) one (PERFORMANCE.md §4).
#   * VRAM — only `A`, `P`, `T2`, `T3` go to the device, as CSR, and
#     `HP_DSAT` stays on the host because the right-hand side is formed
#     there. The index width follows `nnz`: Int32 (12 bytes per nonzero)
#     while `nnz` fits in one, Int64 (16, same as the host CSC) once it does
#     not — which the converged Δz = 10 m `A` does not, at ~3.55e9. So the
#     device needs a bit over *half* the host figure: ~28 GB at Δz = 10 m
#     relaxed and ~65 GB converged, CG vectors included. Only the 94 GB H100
#     NVL holds the converged case; the relaxed one also fits a 48 GB L40S.
#
# The script prints both estimates and the device's actual free VRAM before it
# starts, and refuses to begin a build that cannot fit — an OOM 40 minutes into
# a 10-hour allocation is worth one second of arithmetic up front.
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

par = benchmark_parameters()
order = 4
set = read_stencil_set(SbpOperators.sbp_operators_path() * "standard_diagonal.toml"; order)
key = stiffness_cache_key(; λ=EarthquakeDiffinitive.BP8.lame_lambda(par), μ=par.μ,
                          l_f=par.l_f, Δz, L_fault, L_normal, order, stencil=set,
                          stiffness=:exact)
path = stiffness_cache_path(dir, key)

nf = (round(Int, 2par.l_f / Δz) + 1)^2
n1, n23 = fault_grid_sizes(par, Δz, L_fault, L_normal, order)
Ntot = 3 * 2 * n1 * n23 * n23

@printf("""
target      Δz = %g m, L_fault = %g m, L_normal = %g m, exact (GPU)
grid        %d x %d x %d per side, %d DOF
Ω_f nodes   %d  (K is %d x %d, %.1f MB on disk)
device      %s, %.1f GB total, %.1f GB free
cache       %s
""", Δz, L_fault, L_normal, n1, n23, n23, Ntot, nf, 2nf, 2nf, (2nf)^2 * 8 / 2^20,
     CUDA.name(CUDA.device()), vram_total() / 2^30,
     vram_free() / 2^30, path)

if isfile(path)
    println("\nalready cached — nothing to do (delete the file to rebuild it)")
    exit(0)
end

# ---- a-priori VRAM check, BEFORE the hours of host assembly -----------------
# `nnz(A)` is not known until `A` exists, but it is very predictable: measured
# at Δz = 100/80/50/40/25 m, nonzeros per row fit `48.52 - 349/n23` to within
# 0.25% at every point. That is more than accurate enough to reject a job that
# is off by a factor of two, which is the case that matters — landing on a
# 48 GB L40S when the run needs 65 GB. The exact check still runs after
# assembly; this one exists so that failure costs seconds, not hours.
est_nnz_row(n) = 48.52 - 349.0 / n
est_nnzA = Ntot * est_nnz_row(n23)
est_Ti = est_nnzA <= typemax(Int32) - 1 ? 4 : 8
est_need = est_nnzA * (8 + est_Ti) * 1.03 + 9 * Ntot * 8   # +3% for P, T2, T3
free0 = vram_free()
@printf("estimate    nnz(A) ~%.2fe9 (%s), VRAM ~%.0f GB needed, %.0f GB free\n",
        est_nnzA / 1e9, est_Ti == 4 ? "Int32" : "Int64", est_need / 2^30, free0 / 2^30)
est_need < free0 || error("""
    this configuration needs ~$(round(est_need / 2^30, digits=0)) GB of VRAM but only \
    $(round(free0 / 2^30, digits=0)) GB is free on $(CUDA.name(CUDA.device())).
    Refusing before the (multi-hour) host assembly rather than after it.
    Use a larger GPU (--gpus=h100:1), or the relaxed (1200, 1200) domain.""")

t0 = time()
fe = build_fault_elasticity(; par, Δz, L_fault, L_normal, n1, n23, set, verbose=true)
@printf("assembly    %.2f h\n", (time() - t0) / 3600)

# Now that `A` exists, the VRAM requirement is known exactly rather than
# estimated — check it before uploading anything, while the failure is still
# a clear message instead of a CUDA OOM.
# Mirrors `to_csr`'s index-width rule exactly: assuming Int32 here would
# under-report by ~14 GB on the converged Δz = 10 m `A`, which is precisely
# the configuration this check exists to catch.
csr_bytes(M) = (w = nnz(M) <= typemax(Int32) - 1 ? 4 : 8;
                nnz(M) * (8 + w) + (size(M, 1) + 1) * w)
resident = csr_bytes(fe.rs.A) + csr_bytes(fe.P) + csr_bytes(fe.T2) + csr_bytes(fe.T3)
# Krylov's CG workspace plus the rhs/solution vectors this build keeps live.
vectors = 9 * fe.Ntot * 8
need = resident + vectors
free = vram_free()
@printf("""
A           %d nonzeros (%.2f per row)
VRAM        %.1f GB matrices + %.1f GB vectors = %.1f GB needed, %.1f GB free
""", nnz(fe.rs.A), nnz(fe.rs.A) / fe.Ntot, resident / 2^30, vectors / 2^30,
     need / 2^30, free / 2^30)

need < free || error("""
    this build needs ~$(round(need / 2^30, digits=1)) GB of VRAM but only \
    $(round(free / 2^30, digits=1)) GB is free on $(CUDA.name(CUDA.device())).
    Use a larger GPU, or drop to the relaxed (1200, 1200) domain.""")

K = fault_stiffness_gpu(fe; verbose=true)
x2, x3 = collect.(fault_grid_axes(fe))
save_stiffness(path, key, K, x2, x3)

rep = elastic_solver_report(fe)
@printf("\ndone in %.2f h → %s (%.1f MB)\nsolves %d, mean CG iterations %.0f, unconverged %d\n",
        (time() - t0) / 3600, path, filesize(path) / 2^20,
        rep.solves, rep.iterations / max(rep.solves, 1), rep.unconverged)
rep.unconverged == 0 || error("$(rep.unconverged) solve(s) did not converge — K is not trustworthy")

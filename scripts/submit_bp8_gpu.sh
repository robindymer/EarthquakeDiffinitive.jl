#!/bin/bash -l
#
# One command to produce a BP8-QD-GS submission using a single GPU for the `K`
# build — the GPU counterpart of `submit_bp8.sh`.
#
#   scripts/submit_bp8_gpu.sh <Δz> [L_fault] [L_normal] [gpu_type] [nshards] [walltime]
#
# Submits two chained jobs and returns immediately:
#
#   1. the `K` build, one GPU, one job — or, with `nshards`, an array of
#      `nshards` one-GPU jobs each solving a slice of the D4 representatives,
#      followed by the merge job from `submit_bp8.sh`
#   2. the BP8-QD-GS run, which reads that `K` and writes the §4 files
#
# The build is MATRIX-FREE (`SplitNodeOperator`, MATRIX_FREE_PLAN.md): the
# elastic system is applied from six 1D operators per side, nothing of size
# `nnz(A)` is assembled on the host or uploaded to the device, and the whole
# build is CG solves from the first minute. Same cache key as the CPU path, so
# the two are interchangeable and can even race: whichever writes the entry
# first satisfies the run.
#
# See CLUSTER_RUNBOOK.md "Running it on a GPU instead" for the sizing behind
# the defaults below.
set -euo pipefail

# ============================ EDIT THIS BLOCK ================================
SLURM_ACCOUNT="uppmax2026-1-45"
SLURM_PARTITION_GPU="gpu"            # confirmed present: l40s:10 and h100:2
SLURM_PARTITION_SMALL="pelle"        # the run: a few cores, minutes

# CHECK: same storage as submit_bp8.sh. NOT $HOME.
EQD_STIFFNESS_CACHE="/proj/efficient_elastic/efficient_elastic/nobackup/GPU/EarthquakeDiffinitive.jl/eqd-stiffness"

JULIA_MODULE="Julia/1.11.3-linux-x86_64"
BP8_MODELER="Robin Dymér"
# =============================================================================
#
# DELIBERATELY NO `module load CUDA/...` HERE. CUDA.jl ships its own CUDA
# toolkit as a Julia artifact and picks the right one for the driver it finds.
# Loading the system CUDA module puts /sw/.../cuda/lib64 on LD_LIBRARY_PATH,
# CUDA.jl then loads *those* libraries instead of its artifacts, and warns
# ("CUDA runtime library `libcusparse.so.12` was loaded from a system path") —
# with a real chance of a version mismatch against what CUDA.jl was built for.
# The GPU *driver* comes from the node, not the module, so nothing is missing.
# This is the one place the Julia GPU workflow differs from the CUDA/C++ one in
# submit_diocotron.sh.

# Appended to the output directory name by run_bp8.jl, so a GPU run never
# overwrites a CPU run of the same configuration — the directory name is
# otherwise built only from Δz, the domain and the stiffness mode. Not part of
# the `K` cache key: both paths produce the same `K` and are meant to share it.
BP8_OUTPUT_SUFFIX="gpu"

export EQD_STIFFNESS_CACHE BP8_MODELER BP8_OUTPUT_SUFFIX

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DZ="${1:?usage: submit_bp8_gpu.sh <dz> [L_fault] [L_normal] [gpu_type] [nshards] [walltime]}"

# Defaults per resolution.
#
# DOMAIN. Δz = 10 m defaults to the (1600, 1600) domain — the domain sweep at
# Δz = 20 m (`scripts/bp8_compare_runs.jl`, 2026-09-14) shows the post-shut-in
# `V_max` still moving 2-5 % between 1200² and 1600², so (1200, 1200) is not
# domain-converged and there is no longer a memory reason to prefer it.
#
# GPU TYPE. `sinfo` on Pelle:
#     gpu  gpu:l40s:10(S:0-1)  386000
#     gpu  gpu:h100:2(S:1)     386000
# Matrix-free, every configuration fits either card with a wide margin, so the
# choice is purely queue length vs bandwidth (L40S 864 GB/s, H100 NVL 3.9 TB/s):
#
#   run                          VRAM    L40S              H100
#   Δz = 20 m (1600, 1600)      ~1 GB   ~1 h              ~15 min
#   Δz = 10 m (1150, 1150)      ~3 GB   ~7 h              ~2 h        (assembled path: 40 h measured)
#   Δz = 10 m (1600, 1600)      ~9 GB   ~1 day            ~5 h
#
# Estimated from the laptop measurements in MATRIX_FREE_PLAN.md scaled by
# bandwidth; the first Δz = 10 m run on the cluster calibrates them. Ten L40S
# against two H100 makes the L40S the default everywhere — and since an L40S
# job is short enough to backfill, it will usually *start* sooner too.
#
# CORES: the host does the per-solve right-hand side bookkeeping and the D4
# orbit fill, nothing heavier; 8 is plenty.
#
# EST_H is the estimated *solve* time in hours for the whole build on one L40S
# — there is no assembly phase any more, so this is the whole job. It is what
# the walltime is derived from below, and the only thing to update once the
# cluster has measured a real figure.
case "$DZ" in
  10) DEF_LF=1600; DEF_LN=1600; CORES=8; DEF_GPU=l40s ;;
  *)  DEF_LF=1600; DEF_LN=1600; CORES=8; DEF_GPU=l40s ;;
esac
L_FAULT="${2:-$DEF_LF}"
L_NORMAL="${3:-$DEF_LN}"
GPU_TYPE="${4:-$DEF_GPU}"
NSHARDS="${5:-1}"

if [[ "${DZ%.*}" == "10" ]]; then
    if [[ "${L_FAULT%.*}" -gt 1200 ]]; then EST_H=21; else EST_H=7; fi
else
    EST_H=1
fi
# An H100 NVL has 4.5x an L40S's bandwidth and the build is bandwidth-bound, but
# only 3x is claimed here: the estimates themselves are unmeasured, and there
# are only two H100s, so the walltime that gets the job *started* matters more
# than shaving the last hour off the request.
[[ "$GPU_TYPE" == "h100" ]] && EST_H=$(( (EST_H + 2) / 3 ))

# WALLTIME IS DERIVED, NOT FIXED PER RESOLUTION, for two reasons that both cost
# queue time when got wrong:
#
#  1. **A 47 h request queues far worse than a 12 h one.** SLURM backfills short
#     jobs into gaps ahead of long ones, so asking for the partition maximum
#     "to be safe" can cost more waiting than the job takes to run. A Δz = 10 m
#     (1150, 1150) build is ~7 h — asking 47 h for it is pure queue penalty.
#  2. **A shard does 1/nshards of the work**, so with `nshards` the per-job
#     walltime must shrink too. Handing every array task the whole build's
#     walltime is the same mistake, multiplied.
#
# So: 2x the estimate (the estimates are scaled from laptop measurements and
# have not been checked on an L40S yet), divided across the shards, floored at
# 2 h so a small build still has room to precompile, capped at the partition's
# 47 h. Override with the 6th argument when the estimate is wrong — the one
# number to trust more than this arithmetic is a previous log's `done in X h`.
if [[ -n "${6:-}" ]]; then
    TIME="$6"
else
    TIME_H=$(( (2 * EST_H + NSHARDS - 1) / NSHARDS ))
    [[ "$TIME_H" -lt 2 ]] && TIME_H=2
    [[ "$TIME_H" -gt 47 ]] && TIME_H=47
    TIME=$(printf '%02d:00:00' "$TIME_H")
fi

# HOST MEMORY: the dense `K` (86 MB at Δz = 20 m, 1.3 GB at 10 m), a few
# system-length vectors and the small SAT block. Nothing of size `nnz(A)`.
MEM=32G

mkdir -p "$REPO/logs"
mkdir -p "$EQD_STIFFNESS_CACHE" || {
    echo "error: cannot create EQD_STIFFNESS_CACHE=$EQD_STIFFNESS_CACHE" >&2
    echo "       edit the config block at the top of $0" >&2
    exit 1
}

TAG="dz${DZ%.*}_Lf${L_FAULT%.*}_Ln${L_NORMAL%.*}"

cat <<EOF
BP8-QD-GS submission chain (GPU K build)
  repo         $REPO
  dz           $DZ m
  domain       L_fault = $L_FAULT m, L_normal = $L_NORMAL m
  K build      $NSHARDS x $GPU_TYPE, $CORES cores, $MEM host RAM, walltime $TIME each
               (estimate ${EST_H} h total on one L40S; override walltime with the 6th argument)
  K cache      $EQD_STIFFNESS_CACHE  (shared with the CPU path)
  account      $SLURM_ACCOUNT   partition $SLURM_PARTITION_GPU
EOF

PREAMBLE="module load $JULIA_MODULE
cd $REPO
export EQD_STIFFNESS_CACHE=$EQD_STIFFNESS_CACHE
export JULIA_NUM_THREADS=\$SLURM_CPUS_PER_TASK"

# --- CUDA.jl provisioning preflight, run *on the allocated node* -------------
#
# WHY THIS EXISTS. `Pkg.instantiate()` is normally run on the login node, which
# has no NVIDIA driver. CUDA.jl selects its CUDA toolkit *artifact* by asking
# the driver which version it supports **at precompile time**, so a login-node
# precompile bakes "no runtime found" into `CUDA_Runtime_jll`'s cache. Landing
# on a GPU node afterwards does not fix it: nothing in the *environment*
# changed, only the hardware, so Julia reuses the stale cache and CUDA.jl
# reports
#
#     CUDA.jl could not find an appropriate CUDA runtime to use.
#     CUDA.jl's JLLs were precompiled without an NVIDIA driver present.
#
# and `CUDA.functional()` is false — on a node that has a perfectly good card.
# The durable fix is to pin the toolkit version so the choice no longer depends
# on a driver being visible (CLUSTER_RUNBOOK.md "CUDA.jl was precompiled
# without a driver"); this block is the in-job safety net for when that has not
# been done, and it distinguishes the two failure modes that produce the same
# symptom.
GPU_PREFLIGHT=$(cat <<'PRE'
if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "error: no NVIDIA driver visible on $(hostname)." >&2
    echo "       This allocation has no GPU — check that --gpus survived sbatch." >&2
    exit 1
fi
nvidia-smi -L

# One cheap load to see whether the JLLs are usable here. If they are not, they
# were precompiled somewhere without a driver: recompile them on this node and
# let a fresh process pick them up. ALL of `CUDA_*_jll` and not just
# `CUDA_Runtime_jll`, because the runtime and the compiler are separate
# artifacts that go stale independently — fixing only the runtime gets you as
# far as `UndefVarError: ptxas not defined in CUDA_Compiler_jll`, and a missing
# `libnvJitLink` from the same JLL is what makes CUDA.jl fall back to
# /usr/local/cuda and warn about a system path.
#
# This can take ~45 min if it cascades into a full CUDA.jl precompile, which is
# why the runbook asks you to do it once in an interactive allocation instead.
if ! julia --project=scripts -e 'using CUDA; exit(CUDA.functional() ? 0 : 1)' >/dev/null 2>&1; then
    echo "CUDA.jl not functional on first load — recompiling CUDA JLLs on $(hostname)"
    julia --project=scripts -e '
        using Pkg
        for (uuid, e) in Pkg.Types.Context().env.manifest
            startswith(e.name, "CUDA_") && endswith(e.name, "_jll") || continue
            @info "recompiling $(e.name)"
            Base.compilecache(Base.PkgId(uuid, e.name))
        end'
    julia --project=scripts scripts/gpu_smoke_test.jl
fi
PRE
)

# --- 1. build K on the GPU ----------------------------------------------------
if [[ "$NSHARDS" -le 1 ]]; then
JID_K=$(sbatch --parsable \
  -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION_GPU" -c "$CORES" --mem="$MEM" \
  --gpus="$GPU_TYPE:1" -t "$TIME" -J "bp8Kgpu_$TAG" \
  -o "$REPO/logs/Kgpu_${TAG}_%j.out" <<EOF
#!/bin/bash -l
$PREAMBLE
$GPU_PREFLIGHT
julia --project=scripts scripts/build_stiffness_cache_gpu.jl $DZ $L_FAULT $L_NORMAL
EOF
)
echo "  [1] K build (GPU) job $JID_K"
JID_DEP="$JID_K"
else
# Sharded: an array of one-GPU jobs, then the same merge job the CPU chain
# uses (`afterany`, so a timed-out shard still gets its missing columns named
# by the merge's coverage check; a shard whose file exists exits at once, so
# resubmitting the array re-runs only what is missing).
JID_K=$(sbatch --parsable \
  -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION_GPU" -c "$CORES" --mem="$MEM" \
  --gpus="$GPU_TYPE:1" -t "$TIME" -J "bp8Kgpu_$TAG" --array="1-$NSHARDS" \
  -o "$REPO/logs/Kgpu_${TAG}_%A_%a.out" <<EOF
#!/bin/bash -l
$PREAMBLE
$GPU_PREFLIGHT
julia --project=scripts scripts/build_stiffness_cache_gpu.jl $DZ $L_FAULT $L_NORMAL \$SLURM_ARRAY_TASK_ID $NSHARDS
EOF
)
echo "  [1] K shards (GPU) job $JID_K  (array 1-$NSHARDS)"
JID_M=$(sbatch --parsable \
  -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION_SMALL" -c 4 --mem=16G -t 02:00:00 \
  -J "bp8M_$TAG" --dependency="afterany:$JID_K" --kill-on-invalid-dep=yes \
  -o "$REPO/logs/merge_${TAG}_%j.out" <<EOF
#!/bin/bash -l
$PREAMBLE
julia --project=scripts scripts/merge_stiffness_cache.jl $DZ $L_FAULT $L_NORMAL
EOF
)
echo "  [1b] merge        job $JID_M  (after $JID_K)"
JID_DEP="$JID_M"
fi

# --- 2. run -------------------------------------------------------------------
# `afterok`, not `afterany`: there is no merge step to diagnose a partial
# build, and build_stiffness_cache_gpu.jl writes the cache entry atomically, so
# a failed build leaves no file and the run would only fail again, slower.
JID_R=$(sbatch --parsable \
  -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION_SMALL" -c 4 --mem=16G -t 04:00:00 \
  -J "bp8R_$TAG" --dependency="afterok:$JID_DEP" --kill-on-invalid-dep=yes \
  -o "$REPO/logs/run_${TAG}_%j.out" <<EOF
#!/bin/bash -l
$PREAMBLE
export BP8_MODELER="$BP8_MODELER"
export BP8_OUTPUT_SUFFIX="$BP8_OUTPUT_SUFFIX"
julia --project=. scripts/run_bp8.jl gs $DZ $L_FAULT $L_NORMAL exact
EOF
)
echo "  [2] run + outputs job $JID_R  (after $JID_DEP)"

cat <<EOF

Submitted. Watch with:  squeue -u \$USER
Outputs will appear in: $REPO/output/BP8-QD-GS_${TAG}_exact_${BP8_OUTPUT_SUFFIX}/
Logs in:                $REPO/logs/
EOF

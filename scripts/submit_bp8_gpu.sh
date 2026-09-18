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

error_exit() { echo "error: $*" >&2; exit 1; }

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
#   run                          VRAM    L40S                H100
#   Δz = 20 m (1600, 1600)      ~1 GB   ~1.5 h (extrap.)   ~30 min
#   Δz = 10 m (1150, 1150)      ~3 GB   **11.89 h measured** ~4 h (extrap.)  (assembled path: 40.4 h measured)
#   Δz = 10 m (1600, 1600)      ~9 GB   ~32 h (extrap.)     ~11 h (extrap.)
#
# Calibrated 2026-09-16 from the one real cluster figure above
# (Kgpu_dz10_Lf1150_Ln1150_6839867.out, L40S, matrix-free): grid DOF scales
# almost exactly as (L_fault/Δz)^3 (measured ratio 1.1354 for 1150->1200 m at
# Δz=10, predicted (1200/1150)^3=1.1362), representative count as 1/Δz^2 (Ω_f
# is the physical fault patch, independent of the padding domain — 1681 at
# Δz=10 vs 441 at Δz=20 m is 3.81x, matching (20/10)^2=4x), and mean CG
# iterations move only mildly with Δz (1194 at Δz=10 vs 951 at Δz=20 m,
# 1150-1800 m domains). Multiplying those factors through calibrates every
# other cell in the table to within the same laptop-scaled assumption used
# before, now anchored to one cluster measurement instead of zero. **Still
# extrapolated for every cell but the one in bold** — the (1600, 1600) row is
# the production target and has not itself been measured; treat 32 h as
# central, not a ceiling, and shard it (see EST_H below).
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
# `nshards` accepts `N` or `N%C`: N independent shards, at most C of them
# running at once (SLURM's own `--array=1-N%C` throttle). The two are separate
# concerns and conflating them is what makes sharding look antisocial — N sets
# how small each task is (and so how easily it backfills), C sets how much of
# the partition you occupy. Pelle has ten L40S, so `8%2` is eight ~5 h tasks
# that never hold more than two cards: friendlier than one 47 h job, and it
# starts sooner.
NSHARDS_SPEC="${5:-1}"
NSHARDS="${NSHARDS_SPEC%%\%*}"          # count, for sizing the walltime
NSHARDS_CONC="${NSHARDS_SPEC#*%}"       # concurrency cap, or == NSHARDS_SPEC if absent
[[ "$NSHARDS" =~ ^[0-9]+$ ]] ||
    error_exit "nshards must be N or N%C with integer N, got '$NSHARDS_SPEC'"
if [[ "$NSHARDS_CONC" != "$NSHARDS_SPEC" ]]; then
    [[ "$NSHARDS_CONC" =~ ^[0-9]+$ ]] ||
        error_exit "nshards concurrency must be an integer, got '$NSHARDS_SPEC'"
fi

# EST_H scales from a MEASURED anchor per resolution, by DOF — not a per-domain
# constant. A single constant for every `L_fault > 1200` under-sizes the larger
# domains badly: it gave 2000^2 the same 32 h as 1600^2, so a 4-way shard asked
# 16 h per task for ~23 h of work and would have been killed with nothing
# written (job 6861315, cancelled).
#
# Anchors, both measured on an L40S on this matrix-free path:
#   Δz = 10 m  11.89 h at 37.1 M DOF  (1150, 1150)  Kgpu_dz10_Lf1150_Ln1150_6839867.out
#   Δz = 20 m   1.79 h at 24.5 M DOF  (2000, 2000)  Kgpu_dz20_Lf2000_Ln2000_6857776.out
# Separate anchors per Δz because the *number* of solves is set by Δz alone
# (441 at 20 m, 1681 at 10 m — Ω_f is a fixed 400 m patch), while DOF is set by
# the domain; one global law cannot carry both.
#
# Exponent 1.2, not 1.0: cost is DOF x iterations, and mean CG iterations climb
# with domain size too — 665 -> 858 from (1200,1200) to (1600,1600) at Δz = 20 m,
# and 730 -> 1042 across the new 7.1 M -> 24.5 M sweep. Checked back against the
# four Δz = 20 m measurements it predicts 0.41/0.53/0.98/1.79 h against 0.33/
# 0.43/0.80/1.79 measured — 10-25% high, which is the safe direction for a
# walltime request.
case "${DZ%.*}" in
  10) DOF_REF=37139256; T_REF=11.89 ;;
  *)  DOF_REF=24483006; T_REF=1.79  ;;
esac
DZI="${DZ%.*}"; LFI="${L_FAULT%.*}"; LNI="${L_NORMAL%.*}"
N1=$(( LNI / DZI + 1 )); N23=$(( 2 * LFI / DZI + 1 ))
DOF=$(( 6 * N1 * N23 * N23 ))
EST_H=$(LC_ALL=C awk -v t="$T_REF" -v d="$DOF" -v r="$DOF_REF" 'BEGIN{printf "%.1f", t*(d/r)^1.2}')

# An H100 NVL has 4.5x an L40S's bandwidth and the build is bandwidth-bound, but
# only 3x is claimed here: the estimates themselves are unmeasured, and there
# are only two H100s, so the walltime that gets the job *started* matters more
# than shaving the last hour off the request.
[[ "$GPU_TYPE" == "h100" ]] && EST_H=$(LC_ALL=C awk -v e="$EST_H" 'BEGIN{printf "%.1f", e/3}')

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
    TIME_H=$(LC_ALL=C awk -v e="$EST_H" -v n="$NSHARDS" \
        'BEGIN{h=2*e/n; h=int(h)+(h>int(h)); if(h<2)h=2; if(h>47)h=47; printf "%d", h}')
    TIME=$(printf '%02d:00:00' "$TIME_H")
    # A request pinned at the 47 h cap means the estimate exceeds what one task
    # can finish — more shards, not a longer request.
    [[ "$TIME_H" -eq 47 ]] && echo "  warning: estimate ${EST_H}h/task exceeds the 47h cap — raise nshards" >&2
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
  K build      $NSHARDS x $GPU_TYPE$([[ "$NSHARDS_CONC" != "$NSHARDS_SPEC" ]] && echo " (max $NSHARDS_CONC at once)"), $CORES cores, $MEM host RAM, walltime $TIME each
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
  --gpus="$GPU_TYPE:1" -t "$TIME" -J "bp8Kgpu_$TAG" --array="1-$NSHARDS_SPEC" \
  -o "$REPO/logs/Kgpu_${TAG}_%A_%a.out" <<EOF
#!/bin/bash -l
$PREAMBLE
$GPU_PREFLIGHT
julia --project=scripts scripts/build_stiffness_cache_gpu.jl $DZ $L_FAULT $L_NORMAL \$SLURM_ARRAY_TASK_ID $NSHARDS
EOF
)
echo "  [1] K shards (GPU) job $JID_K  (array 1-$NSHARDS_SPEC)"
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

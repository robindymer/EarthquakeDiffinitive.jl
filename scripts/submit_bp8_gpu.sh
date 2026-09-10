#!/bin/bash -l
#
# One command to produce a BP8-QD-GS submission using a single GPU for the `K`
# build — the GPU counterpart of `submit_bp8.sh`.
#
#   scripts/submit_bp8_gpu.sh <Δz> [L_fault] [L_normal] [gpu_type]
#
# Submits two chained jobs and returns immediately:
#
#   1. the `K` build, one GPU, one job, no shards and no merge
#   2. the BP8-QD-GS run, which reads that `K` and writes the §4 files
#
# WHY TWO JOBS AND NOT THREE. `submit_bp8.sh`'s middle job merges shard files.
# The GPU build is a single process holding all of `A` on one device, so there
# is nothing to shard and nothing to merge — it writes the finished cache entry
# itself. Same cache key as the CPU path, so the two are interchangeable and
# can even race: whichever writes the entry first satisfies the run.
#
# See CLUSTER_RUNBOOK.md "Running it on a GPU instead" for the sizing behind
# the defaults below.
set -euo pipefail

# ============================ EDIT THIS BLOCK ================================
SLURM_ACCOUNT="uppmax2026-1-45"
SLURM_PARTITION_GPU="gpu"            # confirmed present: l40s:10 and h100:2
SLURM_PARTITION_SMALL="pelle"        # the run: a few cores, minutes

# CHECK: same storage as submit_bp8.sh. NOT $HOME.
EQD_STIFFNESS_CACHE="/proj/efficient_elastic/efficient_elastic/nobackup/EarthquakeDiffinitive.jl/eqd-stiffness"

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

DZ="${1:?usage: submit_bp8_gpu.sh <dz> [L_fault] [L_normal] [gpu_type]}"

# Defaults per resolution.
#
# DOMAIN. Δz = 10 m defaults to the relaxed (1200, 1200) domain, matching
# submit_bp8.sh — pass 1600 1200 for the converged one. Both now fit a single
# GPU, which was not true of the host-memory figures: only `A`, `P`, `T2` and
# `T3` go to the device, while `HP_DSAT` stays on the host, so the device needs
# ~28 GB relaxed and ~65 GB converged against ~61/~110 GB of host RAM.
# PERFORMANCE.md §5 item 0c's fit table sized the device from the host number
# and wrongly concluded the converged domain fits nothing.
#
# GPU TYPE. `sinfo` on Pelle:
#     gpu  gpu:l40s:10(S:0-1)  386000
#     gpu  gpu:h100:2(S:1)     386000
# so there are ten 48 GB L40S and only *two* H100. Sizing and speed:
#
#   run                     VRAM    L40S (864 GB/s)   H100 (3.9 TB/s)
#   Δz = 20 m               ~7 GB   ~4 h              ~1 h
#   Δz = 10 m relaxed      ~28 GB   ~28 h             ~7 h
#   Δz = 10 m converged    ~65 GB   does not fit      ~20 h
#
# Δz = 20 m therefore defaults to an L40S — it fits easily, and with ten cards
# the queue is far shorter than for the two H100s. Δz = 10 m defaults to an
# H100 because 28 h on an L40S is uncomfortably close to the 2-day GPU limit
# once assembly is added, and the converged domain does not fit an L40S at all.
# Override as the 4th argument if the H100 queue is long.
#
# CORES: 16, and not more, because **assembly is single-threaded** (measured:
# 100% of one core, not 400% of four). Extra cores do nothing for the phase
# that turns out to dominate; they are here only for the per-solve host
# right-hand side. Do not raise this expecting assembly to speed up.
#
# TIME: the GPU limit is 2 days, and the budget is assembly + solve. Assembly
# is NOT the "minutes" PERFORMANCE.md §4c suggests — measured at four sizes it
# is superlinear with a *rising* exponent (1.02, 1.19, 1.37 across the range),
# extrapolating to ~2 h at Δz = 20 m and much worse at 10 m. That extrapolation
# is the least certain number in this script: it is 27x beyond the largest
# measured point, on different hardware, and the rising exponent means it is a
# lower bound. **Run Δz = 20 m first** — the build script prints `assembly
# X.XX h`, which replaces this guess with a measurement before you commit a
# 47 h H100 allocation.
case "$DZ" in
  10) DEF_LF=1200; DEF_LN=1200; CORES=16; DEF_GPU=h100; TIME="47:00:00" ;;
  20) DEF_LF=1600; DEF_LN=1200; CORES=16; DEF_GPU=l40s; TIME="12:00:00" ;;
  *)  DEF_LF=1600; DEF_LN=1200; CORES=16; DEF_GPU=l40s; TIME="12:00:00" ;;
esac
L_FAULT="${2:-$DEF_LF}"
L_NORMAL="${3:-$DEF_LN}"
GPU_TYPE="${4:-$DEF_GPU}"

# HOST MEMORY is set from the *domain*, not just Δz, because the converged
# domain needs roughly twice the relaxed one and the difference straddles what
# a default request would cover.
#
# Peak RSS during assembly is **~2x the final `A`+`HP_DSAT`**, not ~1x —
# measured at four sizes (92 k to 2.77 M DOF), fitting
# `peak = 2.03 * final + 1.6 GB` to within 0.3 GB at every point. So:
#
#   Δz = 20 m converged   final  13 GB -> peak  ~29 GB
#   Δz = 10 m relaxed     final  61 GB -> peak ~125 GB
#   Δz = 10 m converged   final 108 GB -> peak ~222 GB
#
# GPU nodes have 386 GB, so even the largest fits with room; the point of the
# table is that sizing from the *final* matrix size would under-request by 2x
# and OOM during assembly, hours before the GPU is ever touched.
if [[ "${DZ%.*}" == "10" ]]; then
    if [[ "${L_FAULT%.*}" -gt 1200 ]]; then MEM=320G; else MEM=200G; fi
else
    MEM=64G
fi

# The converged domain at Δz = 10 m needs ~65 GB and an L40S has 48 GB. The
# build script checks this too, but only after assembling `A` — catching it
# here saves hours of host work before a certain failure.
if [[ "$GPU_TYPE" == "l40s" && "${DZ%.*}" == "10" && "${L_FAULT%.*}" -gt 1200 ]]; then
    echo "error: Δz = 10 m on the converged (${L_FAULT%.*}, ${L_NORMAL%.*}) domain needs ~65 GB of VRAM;" >&2
    echo "       an L40S has 48 GB. Use h100, or the relaxed (1200, 1200) domain." >&2
    exit 1
fi

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
  K build      1x $GPU_TYPE, $CORES cores, $MEM host RAM, walltime $TIME
  K cache      $EQD_STIFFNESS_CACHE  (shared with the CPU path)
  account      $SLURM_ACCOUNT   partition $SLURM_PARTITION_GPU
EOF

PREAMBLE="module load $JULIA_MODULE
cd $REPO
export EQD_STIFFNESS_CACHE=$EQD_STIFFNESS_CACHE
export JULIA_NUM_THREADS=\$SLURM_CPUS_PER_TASK"

# --- 1. build K on the GPU ----------------------------------------------------
# Threads still matter even though the solves are on the device: the host
# assembles A and HP_DSAT, and forms every right-hand side.
JID_K=$(sbatch --parsable \
  -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION_GPU" -c "$CORES" --mem="$MEM" \
  --gpus="$GPU_TYPE:1" -t "$TIME" -J "bp8Kgpu_$TAG" \
  -o "$REPO/logs/Kgpu_${TAG}_%j.out" <<EOF
#!/bin/bash -l
$PREAMBLE
julia --project=scripts scripts/build_stiffness_cache_gpu.jl $DZ $L_FAULT $L_NORMAL
EOF
)
echo "  [1] K build (GPU) job $JID_K"

# --- 2. run -------------------------------------------------------------------
# `afterok`, not `afterany`: there is no merge step to diagnose a partial
# build, and build_stiffness_cache_gpu.jl writes the cache entry atomically, so
# a failed build leaves no file and the run would only fail again, slower.
JID_R=$(sbatch --parsable \
  -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION_SMALL" -c 4 --mem=16G -t 04:00:00 \
  -J "bp8R_$TAG" --dependency="afterok:$JID_K" --kill-on-invalid-dep=yes \
  -o "$REPO/logs/run_${TAG}_%j.out" <<EOF
#!/bin/bash -l
$PREAMBLE
export BP8_MODELER="$BP8_MODELER"
export BP8_OUTPUT_SUFFIX="$BP8_OUTPUT_SUFFIX"
julia --project=. scripts/run_bp8.jl gs $DZ $L_FAULT $L_NORMAL exact
EOF
)
echo "  [2] run + outputs job $JID_R  (after $JID_K)"

cat <<EOF

Submitted. Watch with:  squeue -u \$USER
Outputs will appear in: $REPO/output/BP8-QD-GS_${TAG}_exact_${BP8_OUTPUT_SUFFIX}/
Logs in:                $REPO/logs/
EOF

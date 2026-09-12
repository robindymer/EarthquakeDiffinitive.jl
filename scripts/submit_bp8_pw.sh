#!/bin/bash -l
#
# Submit a BP8-QD-PW (Peaceman well) run against an ALREADY CACHED `K`.
#
#   scripts/submit_bp8_pw.sh <Δz> [L_fault] [L_normal] [walltime]
#
# One job, no `K` build and no dependency: the stiffness cache key does not
# include the injection model (`StiffnessCache.stiffness_cache_key`), so a `K`
# built for the GS run of the same (Δz, L_fault, L_normal) is exactly the `K`
# the PW run reads. If no entry exists, `run_bp8.jl` will try to build `:exact`
# inline in this small allocation and fail or crawl — build it first with
# `submit_bp8.sh` / `submit_bp8_gpu.sh`.
#
# WHY A SEPARATE SCRIPT AND NOT A FLAG ON `submit_bp8_gpu.sh`. That script's
# job is the `K` build; this one only runs. PW used to be stiffness-bound
# (PROGRESS.md "Known limitations" 3) — with the explicit integrator it would
# have needed tens of hours at Δz = 20 m and was not going to finish at 10 m.
# `run_bp8` now integrates PW implicitly (`QNDF` + analytic block-diagonal
# Jacobian, PROGRESS.md "BP8-PW stiffness: RESOLVED"), ~1,000-2,000 steps for
# the benchmark at any Δz measured, so the expected cost is minutes to an
# hour — the same order as GS. The default walltime is padded because the
# Δz = 10 m number has not been measured yet; watch the progress bar in the
# log and record the actual figure in TODO.md item 2.
#
# The code on the cluster must include the implicit integrator (commit after
# 2026-09-12), and both environments need re-resolving once, because
# `OrdinaryDiffEqBDF` became a direct dependency and the Manifests are not
# tracked:
#     julia --project=.       -e 'using Pkg; Pkg.instantiate()'
#     julia --project=scripts -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
set -euo pipefail

# ============================ EDIT THIS BLOCK ================================
SLURM_ACCOUNT="uppmax2026-1-45"
SLURM_PARTITION="pelle"
# CHECK: same storage as submit_bp8.sh / submit_bp8_gpu.sh. NOT $HOME.
EQD_STIFFNESS_CACHE="/proj/efficient_elastic/efficient_elastic/nobackup/temp/EarthquakeDiffinitive.jl/eqd-stiffness"
JULIA_MODULE="Julia/1.11.3-linux-x86_64"
BP8_MODELER="Robin Dymér"
# Appended to the output directory name so this run never overwrites another
# PW run of the same configuration. Change it per experiment.
BP8_OUTPUT_SUFFIX="${BP8_OUTPUT_SUFFIX:-cluster}"
# =============================================================================

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DZ="${1:?usage: submit_bp8_pw.sh <dz> [L_fault] [L_normal] [walltime]}"

case "$DZ" in
  10) DEF_LF=1200; DEF_LN=1200 ;;
  *)  DEF_LF=1600; DEF_LN=1200 ;;
esac
L_FAULT="${2:-$DEF_LF}"
L_NORMAL="${3:-$DEF_LN}"
TIME="${4:-08:00:00}"

# CORES: the per-step cost is a dense `K` mat-vec (BLAS-threaded) plus `nf`
# independent scalar Newton solves and a sparse block-diagonal factorisation;
# 16 cores is plenty and cheap to queue.
# MEMORY: dense `K` is 90 MB at Δz = 20 m and 1.4 GB at Δz = 10 m, plus the
# pressure history (~80 MB at 10 m). 32 GB leaves room for the output pass.
CORES=16
MEM=32G

TAG="dz${DZ%.*}_Lf${L_FAULT%.*}_Ln${L_NORMAL%.*}"
mkdir -p "$REPO/logs"

cat <<EOT
BP8-QD-PW run (cached K)
  repo      $REPO
  dz        $DZ m
  domain    L_fault = $L_FAULT m, L_normal = $L_NORMAL m
  job       $CORES cores, $MEM, walltime $TIME on $SLURM_PARTITION
  K cache   $EQD_STIFFNESS_CACHE
  suffix    $BP8_OUTPUT_SUFFIX
EOT

JID=$(sbatch --parsable \
  -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION" -c "$CORES" --mem="$MEM" -t "$TIME" \
  -J "bp8PW_$TAG" -o "$REPO/logs/runPW_${TAG}_%j.out" <<EOT
#!/bin/bash -l
module load $JULIA_MODULE
cd $REPO
export EQD_STIFFNESS_CACHE=$EQD_STIFFNESS_CACHE
export JULIA_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export BP8_MODELER="$BP8_MODELER"
export BP8_OUTPUT_SUFFIX="$BP8_OUTPUT_SUFFIX"
julia --project=. scripts/run_bp8.jl pw $DZ $L_FAULT $L_NORMAL exact
EOT
)
echo "  job $JID"

cat <<EOT

Submitted. Watch with:  squeue -u \$USER   /   tail -f logs/runPW_${TAG}_${JID}.out
Outputs will appear in: $REPO/output/BP8-QD-PW_${TAG}_exact_${BP8_OUTPUT_SUFFIX}/
EOT

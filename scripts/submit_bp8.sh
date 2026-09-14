#!/bin/bash -l
#
# One command to produce a BP8-QD-GS submission on a SLURM cluster.
#
#   scripts/submit_bp8.sh <Δz> [L_fault] [L_normal] [nshards]
#
# Submits three chained jobs and returns immediately:
#
#   1. an array job that builds the `K` columns in `nshards` independent pieces
#   2. a merge job that stitches them into one cache entry
#   3. the actual BP8-QD-GS run, which reads that `K` and writes the §4 files
#
# Each waits on the one before it, so you submit once and collect results when
# the last one finishes. See CLUSTER_RUNBOOK.md for what to do with them.
#
set -euo pipefail

# ============================ EDIT THIS BLOCK ================================
# Filled in from Robin's working UPPMAX/Pelle submission script. Check the one
# line marked CHECK before the first run.

SLURM_ACCOUNT="uppmax2026-1-45"
SLURM_PARTITION="pelle"              # shard array (needs the memory, see below)
SLURM_PARTITION_SMALL="pelle"        # merge and run (a few cores, minutes)

# CHECK: this is the storage for the `efficient_elastic` project. If
# uppmax2026-1-45 is a different allocation, point this at that project's
# /proj/<name>/nobackup instead. NOT $HOME — K is ~86 MB at Δz = 20 m but
# ~1.3 GB at 10 m, and the shard files add the same again until the merge.
EQD_STIFFNESS_CACHE="/proj/efficient_elastic/efficient_elastic/nobackup/EarthquakeDiffinitive.jl/eqd-stiffness"

SLURM_MAIL="robin.dymer@it.uu.se"    # END/FAIL notifications; "" to disable

# Must be >= 1.11. The 1.10 module on Pelle silently ignores the `[sources]`
# pin in Project.toml, resolves Diffinitive to whatever master is now, and
# fails to compile. 1.11.3 is Pelle's default module anyway.
JULIA_MODULE="Julia/1.11.3-linux-x86_64"

BP8_MODELER="Robin Dymér"            # goes into every output file header
# =============================================================================

export EQD_STIFFNESS_CACHE BP8_MODELER

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DZ="${1:?usage: submit_bp8.sh <dz> [L_fault] [L_normal] [nshards]}"

# Defaults per resolution.
#
# The elastic system is applied MATRIX-FREE (`SplitNodeOperator`,
# MATRIX_FREE_PLAN.md), so a shard no longer holds a copy of `A`+`HP_DSAT`:
# memory is the dense `K` slice plus a few system-length vectors, and the
# hours of assembly per shard are gone. Shard counts are still set by
# *walltime*: each solve moves ~3x fewer bytes than the CSR mat-vec did, so
# the per-column figures CLUSTER_RUNBOOK.md quotes (~2730 s at Δz = 10 m) are
# an upper bound — measure the first shard and resize.
#
# Prefer `submit_bp8_gpu.sh`: one L40S does what ~100 of these shards do.
case "$DZ" in
  10) DEF_LF=1600; DEF_LN=1600; DEF_SHARDS=200; CORES=16; MEM=32G ;;
  20) DEF_LF=1600; DEF_LN=1600; DEF_SHARDS=16;  CORES=8;  MEM=16G ;;
  *)  DEF_LF=1600; DEF_LN=1600; DEF_SHARDS=16;  CORES=8;  MEM=16G ;;
esac
L_FAULT="${2:-$DEF_LF}"
L_NORMAL="${3:-$DEF_LN}"
NSHARDS="${4:-$DEF_SHARDS}"

MAILOPT=()
[[ -n "$SLURM_MAIL" ]] && MAILOPT=(--mail-type=END,FAIL --mail-user="$SLURM_MAIL")

mkdir -p "$REPO/logs"
mkdir -p "$EQD_STIFFNESS_CACHE" || {
    echo "error: cannot create EQD_STIFFNESS_CACHE=$EQD_STIFFNESS_CACHE" >&2
    echo "       edit the config block at the top of $0 to point at storage you can write" >&2
    exit 1
}

TAG="dz${DZ%.*}_Lf${L_FAULT%.*}_Ln${L_NORMAL%.*}"

cat <<EOF
BP8-QD-GS submission chain
  repo         $REPO
  dz           $DZ m
  domain       L_fault = $L_FAULT m, L_normal = $L_NORMAL m
  shards       $NSHARDS  ($CORES cores, $MEM each)
  K cache      $EQD_STIFFNESS_CACHE
  account      $SLURM_ACCOUNT   partition $SLURM_PARTITION
EOF

# `JULIA_NUM_THREADS` from the allocation, not `-t auto`: on a shared node
# `auto` sees every core on the machine, not the ones we were actually given.
PREAMBLE="module load $JULIA_MODULE
cd $REPO
export EQD_STIFFNESS_CACHE=$EQD_STIFFNESS_CACHE
export JULIA_NUM_THREADS=\$SLURM_CPUS_PER_TASK"

# --- 1. build K in shards -----------------------------------------------------
# A shard whose file already exists exits immediately, so resubmitting this
# array after a partial failure re-runs only what is missing.
JID_K=$(sbatch --parsable "${MAILOPT[@]}" \
  -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION" -c "$CORES" --mem="$MEM" -t 24:00:00 \
  -J "bp8K_$TAG" --array="1-$NSHARDS" \
  -o "$REPO/logs/K_${TAG}_%A_%a.out" <<EOF
#!/bin/bash -l
$PREAMBLE
julia --project=scripts scripts/build_stiffness_cache.jl $DZ $L_FAULT $L_NORMAL exact \$SLURM_ARRAY_TASK_ID $NSHARDS
EOF
)
echo "  [1] K shards      job $JID_K  (array 1-$NSHARDS)"

# --- 2. merge -----------------------------------------------------------------
# `afterany`, not `afterok`: if a shard times out we still want the merge to run,
# because its coverage check names the exact missing columns. It refuses to
# write a partial K, so nothing downstream can consume a half-built matrix.
JID_M=$(sbatch --parsable "${MAILOPT[@]}" \
  -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION_SMALL" -c 4 --mem=16G -t 02:00:00 \
  -J "bp8M_$TAG" --dependency="afterany:$JID_K" --kill-on-invalid-dep=yes \
  -o "$REPO/logs/merge_${TAG}_%j.out" <<EOF
#!/bin/bash -l
$PREAMBLE
julia --project=scripts scripts/merge_stiffness_cache.jl $DZ $L_FAULT $L_NORMAL
EOF
)
echo "  [2] merge         job $JID_M  (after $JID_K)"

# --- 3. run -------------------------------------------------------------------
# Cheap: a cache hit skips FaultElasticity entirely, so this needs room for K
# (~86 MB at Δz = 20 m, ~1.3 GB at 10 m) and minutes of CPU, not a whole node.
JID_R=$(sbatch --parsable "${MAILOPT[@]}" \
  -A "$SLURM_ACCOUNT" -p "$SLURM_PARTITION_SMALL" -c 4 --mem=16G -t 04:00:00 \
  -J "bp8R_$TAG" --dependency="afterok:$JID_M" --kill-on-invalid-dep=yes \
  -o "$REPO/logs/run_${TAG}_%j.out" <<EOF
#!/bin/bash -l
$PREAMBLE
export BP8_MODELER="$BP8_MODELER"
julia --project=. scripts/run_bp8.jl gs $DZ $L_FAULT $L_NORMAL exact
EOF
)
echo "  [3] run + outputs job $JID_R  (after $JID_M)"

cat <<EOF

Submitted. Watch with:  squeue -u \$USER
Outputs will appear in: $REPO/output/BP8-QD-GS_${TAG}_exact/
Logs in:                $REPO/logs/
EOF

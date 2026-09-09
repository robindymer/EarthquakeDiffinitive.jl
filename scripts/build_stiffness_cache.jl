# Build `K` for one configuration and put it in the stiffness cache — nothing
# else. No pore pressure, no time integration, no outputs.
#
# WHY A SEPARATE ENTRY POINT. The `:exact` build at a production configuration
# is hours to days on one node (PERFORMANCE.md §4) while everything downstream
# of it is seconds. Those belong in different jobs: this one asks for many cores
# and a long wall-clock limit and produces a file; the runs that use the file
# ask for neither. It also means an interrupted or requeued run costs nothing —
# the cache entry is already on disk.
#
# Run:
#   export EQD_STIFFNESS_CACHE=/path/to/scratch/eqd-stiffness
#   julia --project=scripts -t auto scripts/build_stiffness_cache.jl [Δz] [L_fault] [L_normal] [stiffness]
#
# With no arguments it builds the BP8-QD-GS submission configuration below.
# `stiffness` is `exact` (default here — the submission route) or `toeplitz`.
#
#   julia --project=scripts scripts/build_stiffness_cache.jl --list
#
# lists what is already cached, with the full key of each entry.
#
# SHARDING ACROSS NODES. At a resolution where even :exact's embarrassingly
# parallel column build does not fit in one node's wall-clock budget (Δz = 10 m
# is ~934 node-days, PERFORMANCE.md §4 — no single node finishes that), add two
# more arguments:
#
#   julia --project=scripts -t auto scripts/build_stiffness_cache.jl [Δz] [L_fault] [L_normal] [stiffness] [shard] [nshards]
#
# `shard` is 1-based, `nshards` the total shard count for this configuration.
# Each shard rebuilds `FaultElasticity` (minutes, not the bottleneck) and
# computes only its slice of the 2·N_Ωf columns, writing a *shard* file rather
# than a complete cache entry — no communication between shards, and no
# framework beyond independent processes and a shared directory. Submit
# `nshards` of these as a job array (one node each), then run
# `merge_stiffness_cache.jl` once to assemble the final entry that
# `run_bp8.jl` reads. A failed or requeued shard is just rerun with the same
# `shard` number; the merge checks actual column coverage, not a shard count.
using EarthquakeDiffinitive
using EarthquakeDiffinitive.BP8
using EarthquakeDiffinitive.FaultResponse: fault_stiffness, fault_grid_axes
using EarthquakeDiffinitive.StiffnessCache
using Diffinitive.SbpOperators
using Printf

# The converged domain from the PROGRESS.md domain study: L_fault ≥ 4·l_f,
# L_normal ≥ 3·l_f. Δz = 20 m is the coarsest grid that resolves L_b
# (PERFORMANCE.md §4), so it is the target rather than the nominal 10 m.
const DEFAULT_Δz = 20.0
const DEFAULT_L_FAULT = 1600.0
const DEFAULT_L_NORMAL = 1200.0

dir = stiffness_cache_dir()
dir === nothing && error("""
    EQD_STIFFNESS_CACHE is not set, so there is nowhere to put the result.
    Set it to a directory with room for the file — 2·N_Ωf square, Float64, so
    ~86 MB at Δz = 20 m and ~1.3 GB at Δz = 10 m.""")

if !isempty(ARGS) && ARGS[1] == "--list"
    entries = stiffness_cache_entries(dir)
    @printf("%s: %d entr%s, %.2f GB\n", dir, length(entries),
            length(entries) == 1 ? "y" : "ies",
            sum(e -> e.bytes, entries; init=0) / 2^30)
    for e in entries
        @printf("\n%s  (%.1f MB)\n    %s\n", e.name, e.bytes / 2^20, e.key)
    end
    exit(0)
end

Δz = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : DEFAULT_Δz
L_fault = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : DEFAULT_L_FAULT
L_normal = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : DEFAULT_L_NORMAL
stiffness = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :exact
shard = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : nothing
nshards = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : nothing
(shard === nothing) == (nshards === nothing) ||
    error("pass both [shard] and [nshards], or neither")
shard === nothing || 1 <= shard <= nshards ||
    error("shard must be in 1:nshards, got shard=$shard nshards=$nshards")
shard === nothing || stiffness === :exact ||
    error("sharding only makes sense for stiffness=:exact — :toeplitz is already " *
          "just 10 solves, not worth splitting across nodes")

par = benchmark_parameters()
order = 4
set = read_stencil_set(SbpOperators.sbp_operators_path() * "standard_diagonal.toml"; order)
key = stiffness_cache_key(; λ=EarthquakeDiffinitive.BP8.lame_lambda(par), μ=par.μ,
                          l_f=par.l_f, Δz, L_fault, L_normal, order, stencil=set, stiffness)
path = stiffness_cache_path(dir, key)

nf = (round(Int, 2par.l_f / Δz) + 1)^2
ncols = 2nf

if shard !== nothing
    cols = shard:nshards:ncols
    shard_path = joinpath(dir, key.name * ".shard$shard")
    @printf("""
    target      Δz = %g m, L_fault = %g m, L_normal = %g m, %s
    shard       %d of %d  (%d of %d columns)
    Ω_f nodes   %d
    threads     %d
    shard file  %s
    """, Δz, L_fault, L_normal, stiffness, shard, nshards, length(cols), ncols,
         nf, Threads.nthreads(), shard_path)

    if isfile(shard_path)
        println("\nshard already written — nothing to do (delete the file to rebuild it)")
        exit(0)
    end

    n1, n23 = fault_grid_sizes(par, Δz, L_fault, L_normal, order)

    t0 = time()
    fe = build_fault_elasticity(; par, Δz, L_fault, L_normal, n1, n23, set, verbose=true)
    Kc = fault_stiffness(fe; cols, verbose=true)
    x2, x3 = collect.(fault_grid_axes(fe))
    save_stiffness_shard(shard_path, key, collect(cols), Kc, x2, x3)
    @printf("\ndone in %.1f h → %s (%.1f MB)\n", (time() - t0) / 3600, shard_path,
            filesize(shard_path) / 2^20)
    exit(0)
end

@printf("""
target      Δz = %g m, L_fault = %g m, L_normal = %g m, %s
Ω_f nodes   %d  (K is %d×%d, %.1f MB on disk)
solves      %d
threads     %d
cache       %s
""", Δz, L_fault, L_normal, stiffness, nf, 2nf, 2nf, (2nf)^2 * 8 / 2^20,
     stiffness === :exact ? 2nf : 10, Threads.nthreads(), path)

if isfile(path)
    println("\nalready cached — nothing to do (delete the file, or pass cache=:refresh, to rebuild)")
    exit(0)
end

t0 = time()
# `pressure=false`: this script exists to produce `K` and nothing else, and the
# pore-pressure history `build_model` would otherwise solve is never used here.
m = build_model(; par, Δz, L_fault, L_normal, stiffness, order, cache=:auto,
                cache_dir=dir, pressure=false, verbose=true)
@printf("\ndone in %.1f h → %s (%.1f MB)\n", (time() - t0) / 3600, path,
        filesize(path) / 2^20)

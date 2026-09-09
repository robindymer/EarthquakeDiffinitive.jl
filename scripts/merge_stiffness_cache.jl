# Assembles the shard files `build_stiffness_cache.jl [Δz] [L_fault] [L_normal]
# exact [shard] [nshards]` writes into one cache entry that `build_model`
# (and hence `run_bp8.jl`) reads via the normal `EQD_STIFFNESS_CACHE` lookup.
#
# Run once all shards for a configuration are done:
#
#   export EQD_STIFFNESS_CACHE=/path/to/scratch/eqd-stiffness
#   julia --project=scripts scripts/merge_stiffness_cache.jl [Δz] [L_fault] [L_normal]
#
# `stiffness` is always `exact` here — sharding a `:toeplitz` build was never
# offered by `build_stiffness_cache.jl` (10 solves total, not worth splitting).
#
# Coverage is checked against the *columns the shards actually carry*, not
# against a claimed `nshards`: rerun a failed shard under the same `shard`
# number and rerun this script, and it just works, because
# `merge_stiffness_shards` errors on any gap or duplicate rather than
# assembling a partial or double-counted `K` — see `StiffnessCache.jl`.
using EarthquakeDiffinitive
using EarthquakeDiffinitive.BP8
using EarthquakeDiffinitive.StiffnessCache
using Diffinitive.SbpOperators
using Printf

const DEFAULT_Δz = 20.0
const DEFAULT_L_FAULT = 1600.0
const DEFAULT_L_NORMAL = 1200.0

dir = stiffness_cache_dir()
dir === nothing && error("EQD_STIFFNESS_CACHE is not set")

Δz = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : DEFAULT_Δz
L_fault = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : DEFAULT_L_FAULT
L_normal = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : DEFAULT_L_NORMAL

par = benchmark_parameters()
order = 4
set = read_stencil_set(SbpOperators.sbp_operators_path() * "standard_diagonal.toml"; order)
key = stiffness_cache_key(; λ=EarthquakeDiffinitive.BP8.lame_lambda(par), μ=par.μ,
                          l_f=par.l_f, Δz, L_fault, L_normal, order, stencil=set, stiffness=:exact)
path = stiffness_cache_path(dir, key)

if isfile(path)
    println("already merged — nothing to do: $path")
    exit(0)
end

shard_paths = filter(p -> startswith(basename(p), key.name * ".shard"), readdir(dir; join=true))
@printf("target      Δz = %g m, L_fault = %g m, L_normal = %g m, exact\nshards      %d file(s) found\n",
        Δz, L_fault, L_normal, length(shard_paths))
isempty(shard_paths) && error("no shard files for $(key.name) under $dir — nothing to merge")

t0 = time()
K, x2, x3 = merge_stiffness_shards(dir, key)
save_stiffness(path, key, K, x2, x3)
@printf("\nmerged %d shard(s) in %.1f s → %s (%.1f MB)\n", length(shard_paths),
        time() - t0, path, filesize(path) / 2^20)
println("shard files are no longer needed and can be deleted:")
foreach(println, shard_paths)

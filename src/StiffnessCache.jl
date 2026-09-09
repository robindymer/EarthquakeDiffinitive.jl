module StiffnessCache

using ..ElasticitySplitNode: CG_DEFAULTS

export stiffness_cache_dir, stiffness_cache_key, stiffness_cache_path,
       save_stiffness, load_stiffness, stiffness_cache_entries, stencil_digest,
       save_stiffness_shard, load_stiffness_shard, merge_stiffness_shards

# ==============================================================================
# On-disk reuse of the fault stiffness `K`.
#
# `K` is ~98% of the cost of `build_model` (PERFORMANCE.md §5) and depends on
# nothing that changes between runs of the same configuration: the elastic
# constants, the geometry, the grid, the SBP order, the build mode
# (`:exact`/`:toeplitz`) and the CG settings. Everything after it — the
# pore-pressure operators, the time integration, the outputs — is cheap. So a
# configuration that has been built once should never be built again.
#
# WHAT IS IN THE KEY is the whole correctness question. A cache that hits when
# it should have missed silently runs the wrong physics, and `K` carries no
# self-evident signature you could check it against afterwards. So the key
# spells out every input that reaches `K`:
#
#   λ, μ            elastic constants (from BP8Params μ and ν)
#   l_f             Ω_f half-width — sets which nodes K is indexed over
#   Δz              grid spacing
#   L_fault, L_normal   truncation distances — K is NOT truncation-independent,
#                       that is exactly what the domain study measures
#   order           SBP order
#   stencil         digest of the SBP coefficients themselves — `order` names
#                   them, but Diffinitive is pinned by git revision and could
#                   change them under a fixed `order`
#   stiffness       :exact or :toeplitz — different K for the same physics,
#                   which is why it is in the FILENAME and not just the key
#   rtol, atol, itmax, precond   CG settings; rtol especially, since it sets
#                   how close each column is to the exact solve
#
# The key string is stored in the file and re-compared on load, so the 64-bit
# filename hash only has to make names unique-in-practice, not collision-proof:
# a collision produces a miss, not a wrong `K`.
#
# The file also carries the Ω_f axes `x2`, `x3`. That is what lets a cache hit
# skip `FaultElasticity` — the several-GB sparse assembly — entirely, rather
# than only skipping the solves. Nothing else in `BP8Model` needs the elastic
# system once `K` exists.
# ==============================================================================

const MAGIC = "EQDKSTF1"           # bump the trailing digit on any layout change

"""
    stiffness_cache_dir() -> String or nothing

Where cached `K` files live: `\$EQD_STIFFNESS_CACHE` if it is set and non-empty,
otherwise `nothing`, which disables caching.

Deliberately opt-in per machine rather than defaulting to some path under
`\$HOME`: these files run from tens of MB to a few GB, and where they are put
(scratch, project storage, node-local disk) is a decision only the person
running knows the answer to. Set it once —

    export EQD_STIFFNESS_CACHE=/path/to/scratch/eqd-stiffness

— and every `build_model` call in every script picks it up with no code change.
"""
function stiffness_cache_dir()
    d = get(ENV, "EQD_STIFFNESS_CACHE", "")
    return isempty(d) ? nothing : d
end

# FNV-1a. Hand-rolled rather than `hash`, because `Base.hash` is only promised
# to be stable within a Julia version — a Julia upgrade would silently rename
# every cache file and rebuild everything. This is stable forever.
function fnv1a(s::AbstractString)
    h = 0xcbf29ce484222325
    for b in codeunits(s)
        h = (h ⊻ b) * 0x100000001b3
    end
    return h
end

# --- canonical serialisation of the SBP operator coefficients ------------------
#
# `order` alone does NOT pin the operators: the coefficients live in
# Diffinitive's `standard_diagonal.toml`, and Diffinitive is a git dependency
# pinned by revision. Bump that revision — or add the order-6 stencils that
# `TODO.md` wants — and `K` changes while every other part of the key stays put.
# That is precisely the silent wrong-hit this cache is built to prevent, so the
# coefficients go in the key too.
#
# Recursion is structural rather than by field name, so it does not depend on
# `StencilSet`'s internals. Dictionary keys are SORTED because the table is a
# `Dict{String,Any}`, whose iteration order follows `Base.hash` and is therefore
# only stable within a Julia version — unsorted, a Julia upgrade would rename
# every cache file and silently rebuild a multi-day `K`.
canonical(io, x::AbstractDict) =
    for k in sort!(collect(keys(x)); by=string)
        print(io, k, "=")
        canonical(io, x[k])
        print(io, ";")
    end
canonical(io, x::Union{AbstractVector,Tuple}) =
    for v in x
        canonical(io, v)
        print(io, ",")
    end
canonical(io, x::Union{Real,AbstractString,Symbol,Nothing}) = print(io, repr(x))
canonical(io, x) =                                  # any other struct, incl. StencilSet
    for f in fieldnames(typeof(x))
        canonical(io, getfield(x, f))
        print(io, "|")
    end

"""
    stencil_digest(stencil_set) -> String

A 16-hex digest of the SBP operator coefficients, for the cache key. Stable
across processes and Julia versions; changes if any coefficient changes.
"""
stencil_digest(set) = string(fnv1a(sprint(canonical, set)); base=16, pad=16)

# `repr` on a Float64 round-trips exactly, so the key distinguishes values that
# print the same but are not equal.
num(x::Float64) = repr(x)
num(x::Integer) = string(x)

# Short, readable filename fragment: 25.0 → "25", 12.5 → "12.5".
short(x::Real) = isinteger(x) ? string(Int(round(x))) : string(float(x))

"""
    stiffness_cache_key(; λ, μ, l_f, Δz, L_fault, L_normal, order, stiffness,
                          solver_kwargs...) -> NamedTuple

The full identity of a cached `K`: `(; text, name)`, where `text` is the
canonical string stored in and verified against the file, and `name` is the
filename it goes under.

`stencil` is the `StencilSet` the operators are built from (or a digest string
from [`stencil_digest`](@ref)). It is separate from `order` on purpose — see the
comment above `canonical`.

`solver_kwargs` are the `CGSolver` keywords `build_model` forwards. They are
normalised against [`CG_DEFAULTS`](@ref) so that passing a value explicitly and
letting it default produce the *same* key, and an unrecognised keyword is an
error rather than a silent omission from the key — the failure mode being a
cache hit for settings that were never the ones cached.
"""
function stiffness_cache_key(; λ, μ, l_f, Δz, L_fault, L_normal, order, stiffness,
                             stencil, solver_kwargs...)
    unknown = setdiff(keys(solver_kwargs), keys(CG_DEFAULTS))
    isempty(unknown) ||
        error("solver keyword(s) $(join(unknown, ", ")) are not part of the stiffness " *
              "cache key; add them to CG_DEFAULTS (and to the key) or the cache " *
              "could hand back a K built with different settings")
    cg = merge(CG_DEFAULTS, values(solver_kwargs))

    text = join(["magic=" * MAGIC,
                 "lambda=" * num(Float64(λ)),
                 "mu=" * num(Float64(μ)),
                 "l_f=" * num(Float64(l_f)),
                 "dz=" * num(Float64(Δz)),
                 "L_fault=" * num(Float64(L_fault)),
                 "L_normal=" * num(Float64(L_normal)),
                 "order=" * num(Int(order)),
                 "stencil=" * (stencil isa AbstractString ? stencil : stencil_digest(stencil)),
                 "stiffness=" * string(stiffness),
                 "rtol=" * num(Float64(cg.rtol)),
                 "atol=" * num(Float64(cg.atol)),
                 "itmax=" * num(Int(cg.itmax)),
                 "precond=" * string(cg.precond)], " ")

    name = string("K_", stiffness, "_o", Int(order),
                  "_dz", short(Δz), "_lf", short(l_f),
                  "_Lf", short(L_fault), "_Ln", short(L_normal),
                  "_", string(fnv1a(text); base=16, pad=16), ".eqdk")
    return (; text, name)
end

"""
    stiffness_cache_path(dir, key) -> String

Full path of the cache file for `key` under `dir`.
"""
stiffness_cache_path(dir, key) = joinpath(dir, key.name)

# --- file layout -------------------------------------------------------------
#
#   magic     8 bytes, MAGIC
#   keylen    Int64            length of the key text, padded to a multiple of 8
#   key       keylen bytes     UTF-8, NUL-padded
#   n2, n3    Int64            Ω_f grid shape
#   x2, x3    n2+n3 Float64    Ω_f node coordinates
#   nrow,ncol Int64            = (2nf, 2nf)
#   K         nrow*ncol Float64, column-major
#
# Every field is 8 bytes or a multiple of 8 (hence the padded key), so the
# Float64 payloads stay 8-aligned and the file could be mmapped later without a
# layout change.

pad8(n) = (8 - n % 8) % 8

"""
    save_stiffness(path, key, K, x2, x3)

Write `K` and the `Ω_f` axes to `path`, tagged with `key`.

Written to a temporary name in the same directory and `mv`d into place, so a
run that dies mid-write — or two runs racing on the same configuration, which
is exactly what a shared cache directory invites — cannot leave a truncated
file that a later run would read as valid.
"""
function save_stiffness(path, key, K::AbstractMatrix{Float64}, x2, x3)
    mkpath(dirname(path))
    tmp = string(path, ".tmp.", getpid(), ".", rand(UInt32))
    try
        open(tmp, "w") do io
            kb = codeunits(key.text)
            npad = pad8(length(kb))
            write(io, MAGIC)
            write(io, Int64(length(kb) + npad))
            write(io, kb)
            npad > 0 && write(io, zeros(UInt8, npad))
            write(io, Int64(length(x2)), Int64(length(x3)))
            write(io, Vector{Float64}(x2), Vector{Float64}(x3))
            write(io, Int64(size(K, 1)), Int64(size(K, 2)))
            write(io, K)
        end
        mv(tmp, path; force=true)
    catch
        isfile(tmp) && rm(tmp; force=true)
        rethrow()
    end
    return path
end

"""
    load_stiffness(path, key) -> (K, x2, x3) or nothing

Read back a `K` saved under `key`. Returns `nothing` — never throws — if the
file is absent, truncated, written by a different layout version, or carries a
different key: all of those mean "no usable cache entry", and the caller's
response to every one of them is the same, to build `K`.

A key *mismatch* on an existing file is a hash collision, and is worth a warning
rather than a silent rebuild: it would otherwise present as a cache that never
hits, for no visible reason.
"""
function load_stiffness(path, key)
    isfile(path) || return nothing
    try
        return open(path, "r") do io
            String(read(io, ncodeunits(MAGIC))) == MAGIC || return nothing
            keylen = read(io, Int64)
            0 < keylen < 1 << 20 || return nothing
            text = rstrip(String(read(io, keylen)), '\0')
            if text != key.text
                @warn "stiffness cache: filename hash collision, rebuilding" path
                return nothing
            end
            n2 = read(io, Int64)
            n3 = read(io, Int64)
            x2 = read!(io, Vector{Float64}(undef, n2))
            x3 = read!(io, Vector{Float64}(undef, n3))
            nrow = read(io, Int64)
            ncol = read(io, Int64)
            nrow == ncol == 2n2 * n3 ||
                error("cached K is $(nrow)×$(ncol) but the axes give nf = $(n2*n3)")
            K = read!(io, Matrix{Float64}(undef, nrow, ncol))
            eof(io) || error("trailing bytes after K")
            return (K, x2, x3)
        end
    catch err
        @warn "stiffness cache: unreadable entry, rebuilding" path err
        return nothing
    end
end

"""
    stiffness_cache_entries(dir=stiffness_cache_dir()) -> Vector{NamedTuple}

What is in the cache: `(; name, path, bytes, key)` per entry, newest last. For
looking at a cache directory without opening the files by hand — the key text
is the readable record of what each `K` actually is.
"""
function stiffness_cache_entries(dir=stiffness_cache_dir())
    (dir === nothing || !isdir(dir)) && return NamedTuple[]
    out = NamedTuple[]
    for name in sort(readdir(dir))
        endswith(name, ".eqdk") || continue
        path = joinpath(dir, name)
        key = try
            open(path, "r") do io
                String(read(io, ncodeunits(MAGIC))) == MAGIC || return ""
                keylen = read(io, Int64)
                rstrip(String(read(io, keylen)), '\0')
            end
        catch
            ""
        end
        push!(out, (; name, path, bytes=filesize(path), key))
    end
    return out
end

# ==============================================================================
# Sharding an `:exact` build across independent processes (e.g. cluster nodes).
#
# `fault_stiffness`'s `2·N_Ωf` columns are independent right-hand sides against
# the same assembled `A`: no communication is needed between them, only a
# private copy of `A` per shard (`FaultElasticity` assembly is minutes,
# PERFORMANCE.md §4c) and a slice of the columns. That is enough to spread an
# otherwise multi-node-days build across many single-node jobs with nothing
# fancier than the filesystem as the coordination point — see
# `build_stiffness_cache.jl [shard] [nshards]` and `merge_stiffness_cache.jl`.
#
# A shard file is smaller than the final cache entry (only its columns, not
# the full `2N_Ωf × 2N_Ωf` matrix) and carries the *global* column indices it
# covers, so merging does not trust a claimed `nshards` — it trusts the union
# of `cols` actually found on disk. That is what lets a failed or re-run shard
# job be dropped in without renumbering anything else.
# ==============================================================================

const SHARD_MAGIC = "EQDKSHD1"

"""
    save_stiffness_shard(path, key, cols, Kshard, x2, x3)

Write one shard of a `K` build: the global column indices `cols` (into
`1:2N_Ωf`) and the corresponding `2N_Ωf × length(cols)` slice `Kshard`, tagged
with the same `key` the final assembled cache entry will carry. Same
write-to-temp-then-`mv` safety as [`save_stiffness`](@ref).
"""
function save_stiffness_shard(path, key, cols::AbstractVector{<:Integer},
                              Kshard::AbstractMatrix{Float64}, x2, x3)
    size(Kshard, 2) == length(cols) ||
        error("Kshard has $(size(Kshard, 2)) columns but cols has $(length(cols)) entries")
    mkpath(dirname(path))
    tmp = string(path, ".tmp.", getpid(), ".", rand(UInt32))
    try
        open(tmp, "w") do io
            kb = codeunits(key.text)
            npad = pad8(length(kb))
            write(io, SHARD_MAGIC)
            write(io, Int64(length(kb) + npad))
            write(io, kb)
            npad > 0 && write(io, zeros(UInt8, npad))
            write(io, Int64(length(x2)), Int64(length(x3)))
            write(io, Vector{Float64}(x2), Vector{Float64}(x3))
            write(io, Int64(size(Kshard, 1)), Int64(length(cols)))
            write(io, Vector{Int64}(cols))
            write(io, Kshard)
        end
        mv(tmp, path; force=true)
    catch
        isfile(tmp) && rm(tmp; force=true)
        rethrow()
    end
    return path
end

"""
    load_stiffness_shard(path, key) -> (; cols, K, x2, x3) or nothing

Read back one shard written by [`save_stiffness_shard`](@ref). Returns
`nothing` — never throws — for anything that means "not a usable shard for
this `key`": absent, truncated, wrong magic, or a key mismatch (warned, since
that would otherwise look like a shard that silently never merges).
"""
function load_stiffness_shard(path, key)
    isfile(path) || return nothing
    try
        return open(path, "r") do io
            String(read(io, ncodeunits(SHARD_MAGIC))) == SHARD_MAGIC || return nothing
            keylen = read(io, Int64)
            0 < keylen < 1 << 20 || return nothing
            text = rstrip(String(read(io, keylen)), '\0')
            if text != key.text
                @warn "stiffness shard: key mismatch, ignoring" path
                return nothing
            end
            n2 = read(io, Int64)
            n3 = read(io, Int64)
            x2 = read!(io, Vector{Float64}(undef, n2))
            x3 = read!(io, Vector{Float64}(undef, n3))
            nrow = read(io, Int64)
            ncol = read(io, Int64)
            nrow == 2n2 * n3 ||
                error("shard K has $nrow rows but the axes give nf = $(n2*n3)")
            cols = read!(io, Vector{Int64}(undef, ncol))
            K = read!(io, Matrix{Float64}(undef, nrow, ncol))
            eof(io) || error("trailing bytes after shard K")
            return (; cols, K, x2, x3)
        end
    catch err
        @warn "stiffness shard: unreadable entry, skipping" path err
        return nothing
    end
end

"""
    merge_stiffness_shards(dir, key) -> (K, x2, x3)

Assemble the full `K` for `key` from shard files `<key.name>.shard*` under
`dir`. Errors — rather than silently returning a partial matrix — if the
union of columns found across shards is not exactly `1:2N_Ωf` with no gaps and
no duplicates, or if no shards are found at all.
"""
function merge_stiffness_shards(dir, key)
    paths = filter(p -> startswith(basename(p), key.name * ".shard"),
                   isdir(dir) ? readdir(dir; join=true) : String[])
    isempty(paths) && error("no shard files found for $(key.name) under $dir")

    shards = filter(!isnothing, [load_stiffness_shard(p, key) for p in paths])
    isempty(shards) && error("found $(length(paths)) file(s) matching $(key.name).shard*, " *
                             "but none had a matching key — see warnings above")

    x2, x3 = shards[1].x2, shards[1].x3
    n2, n3 = length(x2), length(x3)
    nf = n2 * n3
    ncols = 2nf
    K = fill(NaN, 2nf, ncols)

    seen = falses(ncols)
    for s in shards
        s.x2 == x2 && s.x3 == x3 ||
            error("shard axes disagree with another shard for the same key — corrupt cache dir?")
        for (pos, col) in enumerate(s.cols)
            1 <= col <= ncols || error("shard column index $col out of range 1:$ncols")
            seen[col] && error("column $col is covered by more than one shard")
            seen[col] = true
            K[:, col] .= @view s.K[:, pos]
        end
    end
    all(seen) ||
        error("shards cover $(count(seen))/$ncols columns — missing: ", findall(!, seen))

    return K, x2, x3
end

end # module StiffnessCache

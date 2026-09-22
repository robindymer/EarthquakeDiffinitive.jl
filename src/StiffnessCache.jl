module StiffnessCache

using ..ElasticitySplitNode: CG_DEFAULTS

export stiffness_cache_dir, stiffness_cache_key, stiffness_cache_path,
       save_stiffness, load_stiffness, stiffness_cache_entries, stencil_digest,
       save_stiffness_shard, load_stiffness_shard, merge_stiffness_shards

# On-disk reuse of the fault stiffness `K`.
#
# `K` is ~98% of `build_model`'s cost (PERFORMANCE.md §5) and depends on nothing
# that changes between runs of the same configuration, so a configuration built
# once should never be built again.
#
# WHAT IS IN THE KEY is the whole correctness question: a cache that hits when
# it should have missed runs the wrong physics silently, and `K` carries no
# signature to check afterwards. So the key spells out every input:
#
#   λ, μ                       elastic constants
#   l_f                        Ω_f half-width — sets which nodes K spans
#   Δz                         grid spacing
#   L_fault, L_normal          truncation distances; K is not independent of them
#   order                      SBP order
#   stencil                    digest of the SBP coefficients — `order` names
#                              them, but Diffinitive is pinned by git revision
#   stiffness                  :exact or :toeplitz; also in the FILENAME
#   rtol, atol, itmax, precond CG settings
#
# The key text is stored in the file and re-compared on load, so the 64-bit
# filename hash only needs to be unique in practice: a collision is a miss, not
# a wrong `K`.
#
# The file also carries the Ω_f axes, which is what lets a hit skip
# `FaultElasticity` entirely rather than only the solves.

const MAGIC = "EQDKSTF1"           # bump the trailing digit on any layout change

"""
    stiffness_cache_dir() -> String or nothing

Where cached `K` files live: `\$EQD_STIFFNESS_CACHE` if set and non-empty,
otherwise `nothing`, which disables caching.

Opt-in per machine rather than defaulting under `\$HOME`, because these files
run from tens of MB to a few GB and only the person running knows whether that
belongs on scratch, project storage or node-local disk. Set it once —

    export EQD_STIFFNESS_CACHE=/path/to/scratch/eqd-stiffness

— and every `build_model` call picks it up with no code change.
"""
function stiffness_cache_dir()
    d = get(ENV, "EQD_STIFFNESS_CACHE", "")
    return isempty(d) ? nothing : d
end

# FNV-1a, not `Base.hash`: the latter is only stable within a Julia version, so
# an upgrade would rename every cache file and rebuild everything.
function fnv1a(s::AbstractString)
    h = 0xcbf29ce484222325
    for b in codeunits(s)
        h = (h ⊻ b) * 0x100000001b3
    end
    return h
end

# --- canonical serialisation of the SBP operator coefficients ----------------
#
# `order` alone does not pin the operators: the coefficients live in
# Diffinitive's `standard_diagonal.toml` and Diffinitive is pinned by git
# revision, so bumping it changes `K` while the rest of the key stays put. That
# is the silent wrong-hit this cache exists to prevent, so they go in the key.
#
# Recursion is structural, not by field name, so it does not depend on
# `StencilSet`'s internals. Dict keys are sorted because iteration order follows
# `Base.hash` and is only stable within a Julia version.
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

`stencil` is the `StencilSet` (or a [`stencil_digest`](@ref) string). Separate
from `order` on purpose — see the comment above `canonical`.

`solver_kwargs` are the `CGSolver` keywords `build_model` forwards, normalised
against [`CG_DEFAULTS`](@ref) so an explicit value and a defaulted one give the
same key. An unrecognised keyword errors rather than being silently left out of
the key, which would mean a hit for settings that were never cached.
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
# Every field is a multiple of 8 bytes (hence the padded key), so the Float64
# payloads stay 8-aligned and the file could be mmapped without a layout change.

pad8(n) = (8 - n % 8) % 8

"""
    save_stiffness(path, key, K, x2, x3)

Write `K` and the `Ω_f` axes to `path`, tagged with `key`.

Written to a temp name in the same directory and `mv`d into place, so a run
that dies mid-write — or two runs racing on one configuration, which a shared
cache directory invites — cannot leave a truncated file that reads as valid.
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

Read back a `K` saved under `key`. Returns `nothing`, never throws, for
absent, truncated, wrong-layout or wrong-key files — all of which mean "build
`K`".

A key mismatch on an existing file is a hash collision and is warned about,
since it would otherwise look like a cache that never hits for no reason.
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

What is in the cache: `(; name, path, bytes, key)` per entry. For inspecting a
cache directory without opening files by hand — the key text is the readable
record of what each `K` is.
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

# Sharding an `:exact` build across independent processes.
#
# `fault_stiffness`'s columns are independent right-hand sides against one `A`,
# so a shard needs only its own `fe` (minutes to build) and a slice of the
# columns. The filesystem is the whole coordination mechanism — see
# `build_stiffness_cache.jl [shard] [nshards]` and `merge_stiffness_cache.jl`.
#
# A shard file holds only its columns and carries their *global* indices, so
# merging trusts the union of `cols` found on disk rather than a claimed
# `nshards`. A failed or re-run shard can therefore be dropped in without
# renumbering anything.
#
# The format does not care how a shard picked its columns, which is what lets
# `FaultResponse.fault_stiffness_d4_shard` reuse it: it splits D4 orbit
# representatives, so each solve fixes up to 8 columns and the covered set is
# only known afterwards. The coverage check below only needs the union across
# shards to be exactly `1:2N_Ωf`, which holds either way.

const SHARD_MAGIC = "EQDKSHD1"

"""
    save_stiffness_shard(path, key, cols, Kshard, x2, x3)

Write one shard: the global column indices `cols` (into `1:2N_Ωf`) and the
matching `2N_Ωf × length(cols)` slice, tagged with the same `key` the final
cache entry will carry. Same temp-then-`mv` safety as
[`save_stiffness`](@ref).
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

Read back one shard. Returns `nothing`, never throws, for absent, truncated,
wrong-magic or wrong-key files. A key mismatch is warned about, since it would
otherwise look like a shard that silently never merges.
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

Assemble the full `K` for `key` from `<key.name>.shard*` files under `dir`.
Errors rather than returning a partial matrix if no shards are found or if
their columns are not exactly `1:2N_Ωf` with no gaps or duplicates.
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

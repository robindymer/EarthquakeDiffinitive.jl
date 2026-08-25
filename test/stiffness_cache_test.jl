using EarthquakeDiffinitive
using EarthquakeDiffinitive.BP8
using EarthquakeDiffinitive.StiffnessCache
using Diffinitive.SbpOperators
using Test

# The cache's only real hazard is a hit that should have been a miss: `K` looks
# perfectly plausible whatever configuration produced it, so a wrong hit is a
# silent physics error, not a crash. Hence most of what follows tests the KEY
# rather than the file — that every input reaching `K` changes the key, and that
# nothing else does.

@testset "StiffnessCache" begin
    # Shared baseline key. Values are arbitrary but fixed; only differences matter.
    ops(order) = read_stencil_set(SbpOperators.sbp_operators_path() *
                                  "standard_diagonal.toml"; order)
    base = (; λ=3.204e10, μ=3.204e10, l_f=400.0, Δz=100.0,
            L_fault=800.0, L_normal=800.0, order=4, stencil=ops(4), stiffness=:exact)
    k0 = stiffness_cache_key(; base...)

    @testset "the key separates everything K depends on" begin
        # Each of these changes K, so each must change the filename. λ and μ
        # scale it, the geometry and grid change its shape and entries, `order`
        # changes the operators, `stiffness` changes the build, and the CG
        # settings change how converged each column is.
        for (field, value) in (:λ => 2.0e10, :μ => 1.0e10, :l_f => 200.0,
                               :Δz => 50.0, :L_fault => 1600.0, :L_normal => 1200.0,
                               :order => 2, :stiffness => :toeplitz)
            k = stiffness_cache_key(; merge(base, NamedTuple{(field,)}((value,)))...)
            @test k.name != k0.name
            @test k.text != k0.text
        end
        for kw in (:rtol => 1e-8, :atol => 1e-12, :itmax => 500, :precond => :jacobi)
            k = stiffness_cache_key(; base..., (kw,)...)
            @test k.text != k0.text
            @test k.name != k0.name
        end
    end

    @testset "the SBP coefficients are in the key, separately from `order`" begin
        # `order` names the operators but does not pin them: Diffinitive is a git
        # dependency pinned by revision, so its coefficients can change while
        # `order` stays 4. Holding `order` fixed and swapping only the stencil
        # set must still change the key, or bumping that revision would silently
        # reuse a stale K.
        k = stiffness_cache_key(; merge(base, (; stencil=ops(2)))...)
        @test k.name != k0.name
        @test occursin("order=4", k.text) && occursin("order=4", k0.text)

        # The digest must be stable across processes and Julia versions, which
        # means it cannot depend on `Dict` iteration order.
        @test stencil_digest(ops(4)) == stencil_digest(ops(4))
        @test stencil_digest(ops(4)) != stencil_digest(ops(2))
        @test length(stencil_digest(ops(4))) == 16
    end

    @testset "defaults normalise: explicit == implicit" begin
        # `build_model(...)` and `build_model(...; rtol=1e-10)` are the same
        # build. If they keyed differently the cache would simply never hit for
        # one of them, which is the kind of miss nobody notices.
        @test stiffness_cache_key(; base..., rtol=1e-10, precond=:none) == k0
    end

    @testset "an unknown solver keyword is an error, not a silent omission" begin
        # The dangerous version of this is a new CGSolver keyword that changes K
        # but is dropped from the key: every configuration would then collide on
        # one filename. Better to fail loudly at the call.
        @test_throws ErrorException stiffness_cache_key(; base..., bogus=1)
    end

    @testset "filename records the readable configuration" begin
        # The hash makes it unique; the prefix makes a cache directory something
        # you can read. `stiffness` in particular has to be visible — :exact and
        # :toeplitz K for the same physics are different matrices.
        @test startswith(k0.name, "K_exact_o4_dz100_lf400_Lf800_Ln800_")
        @test endswith(k0.name, ".eqdk")
        @test occursin("stiffness=toeplitz",
                       stiffness_cache_key(; merge(base, (; stiffness=:toeplitz))...).text)
    end

    @testset "round trip" begin
        dir = mktempdir()
        K = reshape(collect(1.0:72.0), 8, 9)   # deliberately not square/symmetric
        x2, x3 = [-1.0, 0.0, 1.0, 2.0], [-1.0, 0.0]   # 4×2 = 8 nodes ⇒ 2nf = 16 ≠ 8
        path = stiffness_cache_path(dir, k0)

        # The shape check must catch a K that does not match the axes, since
        # that is what would corrupt the slip ↔ traction indexing downstream.
        save_stiffness(path, k0, K, x2, x3)
        @test load_stiffness(path, k0) === nothing

        Kgood = reshape(collect(1.0:256.0), 16, 16)
        save_stiffness(path, k0, Kgood, x2, x3)
        got = load_stiffness(path, k0)
        @test got !== nothing
        @test got[1] == Kgood          # bit-exact: no tolerance, it is the same bytes
        @test got[2] == x2
        @test got[3] == x3

        # A different key against the same file is a hash collision. Return
        # nothing rather than the wrong K.
        @test load_stiffness(path, stiffness_cache_key(; merge(base, (; Δz=50.0))...)) === nothing

        # Absent and damaged files are both just "no entry".
        @test load_stiffness(joinpath(dir, "nope.eqdk"), k0) === nothing
        open(path, "a") do io
            write(io, 0x00)
        end
        @test load_stiffness(path, k0) === nothing

        # No temporary files survive a successful save.
        @test all(f -> !occursin(".tmp.", f), readdir(dir))
    end

    @testset "the cache directory is opt-in" begin
        # Unset means off, so the test suite and CI never write cache files and
        # never read a stale one.
        withenv("EQD_STIFFNESS_CACHE" => nothing) do
            @test stiffness_cache_dir() === nothing
        end
        withenv("EQD_STIFFNESS_CACHE" => "") do
            @test stiffness_cache_dir() === nothing
        end
        withenv("EQD_STIFFNESS_CACHE" => "/some/where") do
            @test stiffness_cache_dir() == "/some/where"
        end
    end

    @testset "build_model: a hit reproduces the built model exactly" begin
        # :toeplitz keeps this to 10 solves. What is being tested is the cache
        # path, which is identical for :exact.
        dir = mktempdir()
        cfg = (; Δz=100.0, L_fault=800.0, L_normal=800.0, stiffness=:toeplitz)

        cold = build_model(; cfg..., cache=:auto, cache_dir=dir)
        @test length(stiffness_cache_entries(dir)) == 1

        warm = build_model(; cfg..., cache=:auto, cache_dir=dir)
        # Bit-exact, not approximate. A hit returns the same bytes, so anything
        # less would mean the file is not faithfully round-tripping K.
        @test warm.K == cold.K
        @test warm.x2 == cold.x2
        @test warm.x3 == cold.x3
        @test warm.nf == cold.nf
        @test warm.active == cold.active
        @test warm.weights == cold.weights

        # :off ignores an entry that is sitting right there...
        off = build_model(; cfg..., cache=:off, cache_dir=dir)
        @test off.K ≈ cold.K
        # ...and writes nothing.
        @test length(stiffness_cache_entries(dir)) == 1

        # :read hits but does not create an entry for a configuration it missed.
        other = build_model(; cfg..., L_normal=900.0, cache=:read, cache_dir=dir)
        @test length(stiffness_cache_entries(dir)) == 1
        @test size(other.K) == size(cold.K)

        # :refresh rebuilds over an existing entry rather than reading it.
        mtime_before = mtime(stiffness_cache_entries(dir)[1].path)
        refreshed = build_model(; cfg..., cache=:refresh, cache_dir=dir)
        @test refreshed.K == cold.K
        @test length(stiffness_cache_entries(dir)) == 1
        @test mtime(stiffness_cache_entries(dir)[1].path) >= mtime_before

        @test_throws ErrorException build_model(; cfg..., cache=:sometimes, cache_dir=dir)

        # `cache_dir=nothing` (what an unset EQD_STIFFNESS_CACHE gives) must be
        # a plain build, not an error.
        plain = build_model(; cfg..., cache=:auto, cache_dir=nothing)
        @test plain.K ≈ cold.K
    end
end

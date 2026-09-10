# Prove that CUDA.jl on this node can do everything the GPU `K` build needs.
#
# WHY THIS EXISTS RATHER THAN `CUDA.versioninfo()`. The runbook used to smoke
# test with `julia --project=scripts -e 'using CUDA; CUDA.versioninfo()'`. That
# is wrong in both directions at once:
#
#   * it reaches *too far* for a status print — `versioninfo` asks for the
#     compiler version, so a stale `CUDA_Compiler_jll` surfaces as
#     `UndefVarError: ptxas not defined` from inside a print routine, which
#     reads like a CUDA.jl bug rather than what it is (a precompilation cache
#     written where no driver was visible);
#   * it never compiles a kernel and never calls cuSPARSE, which are precisely
#     the two things `fault_stiffness_gpu` is made of. A node can print a clean
#     `versioninfo` and still be unable to run the build.
#
# So this checks the capabilities, in the order they fail, and says what to do
# about each. Run it inside a GPU allocation:
#
#   julia --project=scripts scripts/gpu_smoke_test.jl
using CUDA
using CUDA.CUSPARSE
using SparseArrays
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "gpu_vram.jl"))

const FIX_STALE_JLL = """
    This is a stale precompilation cache, not a broken node. CUDA.jl resolves
    its CUDA artifacts by asking the driver at *precompile* time, so anything
    precompiled on a login node records "nothing available" — and Julia will
    not invalidate that when you later land on a GPU, because the environment
    did not change, only the hardware. Recompile the CUDA JLLs here:

        julia --project=scripts -e '
            using Pkg
            for (uuid, e) in Pkg.Types.Context().env.manifest
                startswith(e.name, "CUDA_") && endswith(e.name, "_jll") || continue
                @info "recompiling \$(e.name)"
                Base.compilecache(Base.PkgId(uuid, e.name))
            end'

    then re-run this script. See CLUSTER_RUNBOOK.md "CUDA.jl was precompiled
    without a driver"."""

failures = String[]

# `f` FIRST, because every call site below uses `do` block syntax and Julia
# passes the block as the *first* argument — `check(name, fix) do ... end` is
# `check(block, name, fix)`. Getting this backwards makes every check fail
# identically with "objects of type String are not callable", which looks like
# a CUDA fault and is not one.
function check(f, name, fix)
    print(rpad(name, 34))
    try
        result = f()
        println(result === nothing ? "ok" : result)
        return true
    catch err
        println("FAILED")
        println()
        showerror(stdout, err)
        println("\n")
        push!(failures, "$name\n\n$fix")
        return false
    end
end

println("CUDA.jl smoke test on ", gethostname())
println()

# The driver half — this is what the *node* provides, and it is either there or
# the allocation has no GPU. Checked first because every later failure is
# ambiguous without it.
check("driver present", """
    No CUDA driver on this node. The allocation has no GPU: check that
    `--gpus` survived sbatch, or that your `interactive` request included it.
    This is a SLURM problem, not a Julia one.""") do
    CUDA.driver_version()
end

check("runtime available", FIX_STALE_JLL) do
    CUDA.runtime_version()
end

# The compiler is a *separate* artifact from the runtime (`CUDA_Compiler_jll`
# ships ptxas and libnvJitLink), so it goes stale independently — and it is the
# one `versioninfo` trips over. A missing libnvJitLink here is also why CUDA.jl
# falls back to `/usr/local/cuda`'s copy and warns about a system path: the
# warning is a *symptom* of this, not an independent problem.
check("compiler (ptxas) available", FIX_STALE_JLL) do
    isdefined(CUDA, :compiler_version) ? CUDA.compiler_version() : "skipped (old CUDA.jl)"
end

check("CUDA.functional()", FIX_STALE_JLL) do
    CUDA.functional() || error("CUDA.functional() is false")
    "ok"
end

if !isempty(failures)
    println("\n", "="^72)
    for f in failures
        println("\n", f)
    end
    exit(1)
end

dev = CUDA.device()
free, total = vram_free(), vram_total()
@printf("\ndevice   %s\n", CUDA.name(dev))
@printf("VRAM     %.1f GB free of %.1f GB\n", free / 2^30, total / 2^30)
@printf("runtime  %s   driver %s\n\n", CUDA.runtime_version(), CUDA.driver_version())

# Everything above is metadata. These two actually run.
check("kernel compile + launch", """
    The toolchain reports itself healthy but cannot compile a kernel. If the
    error mentions ptxas or nvJitLink being loaded from `/usr/local/cuda`,
    strip that from LD_LIBRARY_PATH and re-run — CUDA.jl must use its own
    artifacts, and mixing them with a system toolkit is exactly the version
    mismatch CLUSTER_RUNBOOK.md warns about.""") do
    # Broadcast plus a reduction: both go through the GPU compiler, so this is
    # the real ptxas test rather than a version string.
    a, b = CUDA.rand(4096), CUDA.rand(4096)
    got = sum(a .* b)
    ref = dot(Array(a), Array(b))
    isapprox(got, ref; rtol=1e-4) || error("dot product wrong: $got vs $ref")
    @sprintf("ok  (dot rel.err %.1e)", abs(got - ref) / abs(ref))
end

check("cuSPARSE CSR spmv", """
    cuSPARSE is the whole GPU build path — `fault_stiffness_gpu` is CG against
    a CSR `A`. If this fails while kernels compile, suspect a library version
    mismatch from a system CUDA on LD_LIBRARY_PATH.""") do
    n = 5000
    A = sprand(n, n, 0.002) + 10I
    x = rand(n)
    got = Array(CuSparseMatrixCSR(A) * CuArray(x))
    ref = A * x
    err = norm(got - ref) / norm(ref)
    err < 1e-10 || error("spmv wrong: relative error $err")
    @sprintf("ok  (rel.err %.1e)", err)
end

if isempty(failures)
    println("\nAll checks passed — this node can run scripts/build_stiffness_cache_gpu.jl.")
else
    println("\n", "="^72)
    for f in failures
        println("\n", f)
    end
    exit(1)
end

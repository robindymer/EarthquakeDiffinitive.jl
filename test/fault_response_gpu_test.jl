# `fault_stiffness_gpu` (EarthquakeDiffinitiveCUDAExt, PERFORMANCE.md §5 item
# 0c) needs a real GPU, so this file is entirely opt-in: skipped by default,
# including in CI, so nobody without a GPU pays for loading CUDA.jl (a heavy
# precompile) just to run the suite.
#
# Enable with:
#   EQD_TEST_GPU=1 julia --project=. -e 'using Pkg; Pkg.test()'
#
# This only checks CORRECTNESS (agreement with the CPU D4 build) at small
# scale on whatever GPU is present. It is not a speed benchmark, and passing
# here says nothing about the speedup at production scale or on a specific
# datacenter GPU — see `fault_stiffness_gpu`'s docstring.
if get(ENV, "EQD_TEST_GPU", "") == "1"
    using CUDA
    if !CUDA.functional()
        @info "EQD_TEST_GPU=1 but CUDA.functional() is false — skipping GPU tests"
    else
        @testset "FaultResponse GPU (EQD_TEST_GPU=1)" begin
            set = fr_stencil_set()
            gm, gp = fr_grids(13; L=1.2)
            fe_cpu = FaultElasticity(gm, gp, λ_fr, μ_fr, set; l_f=0.6)
            fe_gpu = FaultElasticity(gm, gp, λ_fr, μ_fr, set; l_f=0.6)

            K_cpu = fault_stiffness(fe_cpu; symmetry=true, threaded=false)
            K_gpu = fault_stiffness_gpu(fe_gpu)

            @test size(K_gpu) == size(K_cpu)
            @test K_gpu ≈ K_cpu rtol = 1e-8
            @test solver_report(fe_gpu.rs).solves == solver_report(fe_cpu.rs).solves
            @test all(<(0), diag(K_gpu))

            # precond=:jacobi is explicitly unsupported on the GPU path (untested
            # there) — must error, not silently ignore the keyword.
            set2 = fr_stencil_set()
            gmj, gpj = fr_grids(13; L=1.2)
            fe_jacobi = FaultElasticity(gmj, gpj, λ_fr, μ_fr, set2; l_f=0.6, precond=:jacobi)
            @test_throws ErrorException fault_stiffness_gpu(fe_jacobi)
        end
    end
else
    @info "skipping GPU tests (set EQD_TEST_GPU=1 to enable, requires a functional CUDA GPU)"
end

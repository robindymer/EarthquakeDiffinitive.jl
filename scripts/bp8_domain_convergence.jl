# SEAS BP8-QD-GS §6: how large does the computational domain have to be before
# results stop changing?
#
# The whole-space problem is truncated with u=0 Dirichlet boundaries at
# |x1| = L_normal and |x2|,|x3| = L_fault. Those boundaries make the medium
# artificially stiff, so slip comes out too small; the question is how far away
# they have to be before that stops mattering.
#
# Run with:  julia --project=. scripts/bp8_domain_convergence.jl
using EarthquakeDiffinitive
using EarthquakeDiffinitive.BP8
using Printf

const Δz = 100.0                      # coarse: this study varies domain, not resolution
const T_END = 100 * 3600.0            # through the injection period

# Sorted by cost. The sparse LU's fill-in — not the grid itself — is what
# binds: it grows far faster than the DOF count, and an over-large
# configuration gets the process OOM-killed rather than throwing, which is why
# each row is printed as soon as it finishes.
configs = [(L_fault=800.0, L_normal=800.0),
           (L_fault=1200.0, L_normal=800.0),
           (L_fault=1200.0, L_normal=1200.0),
           (L_fault=1600.0, L_normal=800.0),
           (L_fault=1600.0, L_normal=1200.0)]

results = NamedTuple[]
for cfg in configs
    @info "building" cfg...
    flush(stdout)
    local m
    try
        t0 = time()
        m = build_model(; Δz, cfg.L_fault, cfg.L_normal)
        t_build = time() - t0
        sol = run_bp8(m; tspan=(0.0, T_END), saveat=3600.0)
        c = evaluate!(m, sol.u[end], sol.t[end])
        nf = m.nf
        ic = argmin(abs.(m.x2)) + (argmin(abs.(m.x3)) - 1) * length(m.x2)
        push!(results, (; cfg.L_fault, cfg.L_normal,
                        dofs=m.grid_info.elastic_dofs,
                        t_build,
                        K_self=m.K[ic, ic],
                        slip_centre=sol.u[end][ic],
                        Vmax=maximum(c.Vmag),
                        moment_rate=m.par.μ * sum(m.weights .* c.Vmag),
                        p_centre=sol.u[end][3nf+ic]))
        r = results[end]
        @printf("DONE L_fault=%.0f L_normal=%.0f dofs=%d build=%.1fs K_self=%.5E slip=%.5E Vmax=%.5E p=%.4f MPa\n",
                r.L_fault, r.L_normal, r.dofs, r.t_build, r.K_self, r.slip_centre,
                r.Vmax, r.p_centre / 1e6)
        flush(stdout)
    catch e
        @warn "configuration failed" cfg... exception = e
        flush(stdout)
    end
end

println("\n", "="^100)
println("BP8-QD-GS domain-size convergence,  Δz = $Δz m,  t = $(T_END/3600) h")
println("="^100)
@printf("%9s %9s %10s %8s %13s %13s %13s %13s\n",
        "L_fault", "L_norm", "elast.DOF", "build s", "K_self", "slip(0,0)", "Vmax", "p(0,0) MPa")
for r in results
    @printf("%9.0f %9.0f %10d %8.1f %13.5E %13.5E %13.5E %13.5f\n",
            r.L_fault, r.L_normal, r.dofs, r.t_build, r.K_self, r.slip_centre,
            r.Vmax, r.p_centre / 1e6)
end

if length(results) > 1
    println("\nrelative change against the largest domain run:")
    ref = results[end]
    @printf("%9s %9s %13s %13s %13s\n", "L_fault", "L_norm", "d(K_self)", "d(slip)", "d(Vmax)")
    for r in results
        @printf("%9.0f %9.0f %12.2f%% %12.2f%% %12.2f%%\n", r.L_fault, r.L_normal,
                100abs(r.K_self - ref.K_self) / abs(ref.K_self),
                100abs(r.slip_centre - ref.slip_centre) / abs(ref.slip_centre),
                100abs(r.Vmax - ref.Vmax) / abs(ref.Vmax))
    end
end
println("\nPore pressure is independent of the elastic domain (it lives on Ω_f only),")
println("so p(0,0) should be identical across rows — a check that nothing else drifted.")

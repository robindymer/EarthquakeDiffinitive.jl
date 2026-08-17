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

# Sorted by cost. The sparse factorization's fill-in — not the grid itself — is
# what binds: it grows far faster than the DOF count, and an over-large
# configuration gets the process OOM-killed rather than throwing, which is why
# each row is printed as soon as it finishes.
configs = [(L_fault=800.0, L_normal=800.0),
           (L_fault=1200.0, L_normal=800.0),
           (L_fault=1200.0, L_normal=1200.0),
           (L_fault=1600.0, L_normal=800.0),
           (L_fault=1600.0, L_normal=1200.0)]

# The sweep above cannot reach the production configuration. `run_bp8.jl`
# defaults to `L_normal = 400` — halved relative to `L_fault` to fit the
# factorization in memory — and at Δz = 100 m that is only 5 points across the
# fault-normal direction, below SBP order 4's minimum of 9. `L_normal = 400` is
# legal only at Δz = 50 m, which is also the production resolution.
#
# So the fault-normal truncation is studied separately, at Δz = 50 m with
# `L_fault` held at 800 m. That is the comparison that actually bears on the
# shipped runs: `u=0` at |x1| = L_normal makes the medium artificially stiff, so
# the expected bias is slip too small and |K_self| too large, and this varies
# exactly the distance responsible while changing nothing else. Sorted by cost,
# and the large rows may simply not fit — an OOM here is a result, not a bug.
const NORMAL_STUDY_Δz = 50.0
normal_configs = [(L_fault=800.0, L_normal=400.0),   # production
                  (L_fault=800.0, L_normal=600.0),
                  (L_fault=800.0, L_normal=800.0)]

"""
    sweep(configs, Δz) -> Vector{NamedTuple}

Runs each configuration to `T_END` and collects the quantities the truncation
boundary would bias. A configuration that fails — too coarse for the SBP
closure, or out of memory — is warned about and skipped rather than aborting
the rest, since the expensive rows are the ones most likely to fail.
"""
function sweep(configs, Δz)
    results = NamedTuple[]
    for cfg in configs
        @info "building" Δz cfg...
        flush(stdout)
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
        catch e
            @warn "configuration failed" Δz cfg... exception = e
        end
        flush(stdout)
        flush(stderr)
    end
    return results
end

"""
    report(results, title, Δz; ref_label)

Prints the table and the relative changes against `results[end]`, which is the
largest (least truncated) configuration that actually ran.
"""
function report(results, title, Δz; ref_label="the largest domain run")
    println("\n", "="^100)
    println("$title,  Δz = $Δz m,  t = $(T_END/3600) h")
    println("="^100)
    isempty(results) && (println("no configuration completed"); return)
    @printf("%9s %9s %10s %8s %13s %13s %13s %13s\n",
            "L_fault", "L_norm", "elast.DOF", "build s", "K_self", "slip(0,0)", "Vmax", "p(0,0) MPa")
    for r in results
        @printf("%9.0f %9.0f %10d %8.1f %13.5E %13.5E %13.5E %13.5f\n",
                r.L_fault, r.L_normal, r.dofs, r.t_build, r.K_self, r.slip_centre,
                r.Vmax, r.p_centre / 1e6)
    end
    if length(results) > 1
        println("\nrelative change against $ref_label:")
        ref = results[end]
        @printf("%9s %9s %13s %13s %13s\n", "L_fault", "L_norm", "d(K_self)", "d(slip)", "d(Vmax)")
        for r in results
            @printf("%9.0f %9.0f %12.2f%% %12.2f%% %12.2f%%\n", r.L_fault, r.L_normal,
                    100abs(r.K_self - ref.K_self) / abs(ref.K_self),
                    100abs(r.slip_centre - ref.slip_centre) / abs(ref.slip_centre),
                    100abs(r.Vmax - ref.Vmax) / abs(ref.Vmax))
        end
    end
end

results = sweep(configs, Δz)
report(results, "BP8-QD-GS domain-size convergence", Δz)

normal_results = sweep(normal_configs, NORMAL_STUDY_Δz)
report(normal_results, "BP8-QD-GS fault-normal truncation at PRODUCTION resolution",
       NORMAL_STUDY_Δz; ref_label="the largest L_normal that fitted")
println("""
The first row of that second table is what `run_bp8.jl` ships. If d(slip) and
d(K_self) against the largest L_normal are small, `L_normal = 400` is vindicated
and the halving done for memory costs nothing. `u=0` makes the medium
artificially stiff, so the expected sign is slip too small, |K_self| too large.
If the largest rows are missing they did not fit in memory — which is itself the
reason `L_normal` was halved.""")
println("\nPore pressure is independent of the elastic domain (it lives on Ω_f only),")
println("so p(0,0) should be identical WITHIN each table — a check that nothing else")
println("drifted. It differs BETWEEN the tables only because they use different Δz.")

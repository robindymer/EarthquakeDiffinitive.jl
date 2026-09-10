# Runs SEAS BP8-QD and writes the §4 benchmark output files.
#
#   julia --project=. scripts/run_bp8.jl [gs|pw] [Δz] [L_fault] [L_normal] [stiffness]
#
# `$BP8_OUTPUT_SUFFIX`, if set, is appended to the output directory name. The
# GPU chain (`submit_bp8_gpu.sh`) sets it to `gpu` so its results land beside
# a CPU run of the same configuration instead of silently overwriting it —
# the directory name is otherwise built only from Δz, the domain and the
# stiffness mode, none of which distinguish the two.
#
# Deliberately NOT part of the `K` cache key: the GPU and CPU builds produce
# the same `K` for the same configuration (that is the point of them sharing a
# key), so keying the cache on this would force a redundant multi-hour rebuild
# to obtain a matrix that is already on disk. It affects where results are
# written and nothing else.
#
# `stiffness` is `exact` (default here — the submission route, matching
# `build_stiffness_cache.jl`) or `toeplitz`. Pass `exact` only once `K` for
# this configuration is already sitting in `$EQD_STIFFNESS_CACHE` (built
# separately with `build_stiffness_cache.jl`, which is sized for a standalone
# cluster job) — otherwise this run pays the `:exact` build cost inline.
#
# Δz = 20 m, L_fault = 1600 m, L_normal = 1200 m is the coarsest grid that
# resolves the process zone (`L_b/Δz ≥ 3`) at the domain size the §6 study
# found necessary; see PERFORMANCE.md §4 and PROGRESS.md "Domain requirement
# relaxes with resolution". Those are also this script's defaults below.
using EarthquakeDiffinitive
using EarthquakeDiffinitive.BP8
using Printf

const MODELER = get(ENV, "BP8_MODELER", "Robin Dymér")

injection = length(ARGS) >= 1 ? Symbol(lowercase(ARGS[1]) == "pw" ? :peaceman : :gaussian) : :gaussian
Δz = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 20.0
L_fault = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 1600.0
L_normal = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1200.0
stiffness = length(ARGS) >= 5 ? Symbol(ARGS[5]) : :exact
stiffness ∈ (:exact, :toeplitz) ||
    error("stiffness must be exact or toeplitz, got $stiffness")

tag = injection === :gaussian ? "GS" : "PW"
suffix = get(ENV, "BP8_OUTPUT_SUFFIX", "")
outdir = joinpath(@__DIR__, "..", "output",
                  "BP8-QD-$(tag)_dz$(Int(Δz))_Lf$(Int(L_fault))_Ln$(Int(L_normal))_$(stiffness)" *
                  (isempty(suffix) ? "" : "_$(suffix)"))

@info "BP8-QD-$tag" Δz L_fault L_normal stiffness outdir
flush(stdout)

t0 = time()
m = build_model(; Δz, L_fault, L_normal, injection, stiffness, verbose=true)
@info "model built" seconds = round(time() - t0, digits=1) frictional_nodes = m.nf
flush(stdout)

t0 = time()
# §4.1 asks for 1e4-1e5 rows in the time series; §4.3 for ~1e3 in the
# profiles.
#
# `saveat` fixes the output grid exactly, so the row count is arithmetic, not
# an estimate: over `t_f` = 30 d = 2,592,000 s, saveat=300 gives 8,641 rows —
# **below** the 1e4 floor, not "in range" as this comment used to claim (the
# solver's adaptive steps do not add rows once saveat is set). saveat=200
# gives 12,961, comfortably inside. The cost is output size only; it does not
# touch `K`, the integration itself, or accuracy.
sol = run_bp8(m; saveat=200.0, verbose=true, progress=true)
@info "integrated" seconds = round(time() - t0, digits=1) steps = length(sol.t) retcode = sol.retcode
flush(stdout)

function summarize(m, sol)
    nf = m.nf
    ic = argmin(abs.(m.x2)) + (argmin(abs.(m.x3)) - 1) * length(m.x2)
    Vpeak = 0.0
    tpeak = 0.0
    ppeak = 0.0
    for (j, t) in enumerate(sol.t)
        c = evaluate!(m, sol.u[j], t)
        v = maximum(c.Vmag)
        v > Vpeak && (Vpeak = v; tpeak = t)
        ppeak = max(ppeak, pressure_at!(m, t)[ic])
    end
    uend = sol.u[end]
    @printf("\npeak slip rate      %.4E m/s at t = %.3f days\n", Vpeak, tpeak / 86400)
    @printf("final slip (0,0)    %.4E m\n", uend[ic])
    @printf("final max slip      %.4E m\n", maximum(abs, uend[1:nf]))
    @printf("peak pressure (0,0) %.4f MPa   (min σ̄ = %.4f MPa)\n",
            ppeak / 1e6, (m.par.σ0 - ppeak) / 1e6)

    es = effective_stress_report(m)
    if es.bound
        @printf("\nWARNING: effective normal stress went to %.4f MPa (below the %.0f Pa floor)\n",
                es.σ̄_lowest / 1e6, es.floor)
        println("         in $(es.floor_hits) evaluations. Fluid pressure has fully unclamped")
        println("         the fault there, so BP8's no-opening condition (eq. 3) no longer")
        println("         holds and the model is outside its range of validity at those nodes.")
    end

    rr = resolution_report(m)
    @printf("\nresolution: Δz = %.0f m, process zone L_b = %.1f m -> %.2f cells per L_b\n",
            rr.Δz, rr.L_b, rr.cells_per_Lb)
    @printf("            Gaussian source width L_gauss = %.0f m -> %.2f cells\n",
            m.par.L_gauss, rr.cells_per_L_gauss)
    if !rr.converged
        println("            NOT resolution-converged: slip rate depends exponentially on σ̄,")
        println("            so peak V here is indicative, not quantitative. See PROGRESS.md.")
    end
    return nothing
end
summarize(m, sol)

write_outputs(m, sol, outdir; modeler=MODELER)
@info "outputs written" outdir files = length(readdir(outdir))

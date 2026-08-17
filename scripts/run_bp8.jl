# Runs SEAS BP8-QD and writes the §4 benchmark output files.
#
#   julia --project=. scripts/run_bp8.jl [gs|pw] [Δz] [L_fault] [L_normal]
#
# Defaults to the largest configuration that fits in memory on a 13 GB
# machine. The binding constraint is fill-in in the sparse LU of the 3D
# elastic system, not the grid size itself — see PROGRESS.md.
using EarthquakeDiffinitive
using EarthquakeDiffinitive.BP8
using Printf

const MODELER = get(ENV, "BP8_MODELER", "Robin Dymér")

injection = length(ARGS) >= 1 ? Symbol(lowercase(ARGS[1]) == "pw" ? :peaceman : :gaussian) : :gaussian
Δz = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 50.0
L_fault = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 800.0
L_normal = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 400.0

tag = injection === :gaussian ? "GS" : "PW"
outdir = joinpath(@__DIR__, "..", "output", "BP8-QD-$(tag)_dz$(Int(Δz))_Lf$(Int(L_fault))_Ln$(Int(L_normal))")

@info "BP8-QD-$tag" Δz L_fault L_normal outdir
flush(stdout)

t0 = time()
m = build_model(; Δz, L_fault, L_normal, injection, verbose=true)
@info "model built" seconds = round(time() - t0, digits=1) frictional_nodes = m.nf
flush(stdout)

t0 = time()
# §4.1 asks for 1e4-1e5 rows in the time series; §4.3 for ~1e3 in the
# profiles. 5-minute saves plus the solver's own adaptive steps lands in range.
sol = run_bp8(m; saveat=300.0, verbose=true)
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
        ppeak = max(ppeak, sol.u[j][3nf+ic])
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

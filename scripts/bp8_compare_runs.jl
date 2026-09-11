# Compare finished BP8 output directories against each other.
#
#   julia --project=scripts scripts/bp8_compare_runs.jl [dir ...]
#
# With no arguments it picks up every `output/BP8-QD-*` directory. The
# **reference is the last directory listed** (with no arguments: the one whose
# name sorts last, which for a domain sweep is the largest); pass them
# explicitly to choose.
#
# WHY THIS EXISTS RATHER THAN `bp8_domain_convergence.jl`. That script and
# `bp8_resolution_convergence.jl` *run the models themselves*, at Δz = 100 m
# and through t = 100 h. Once runs are produced on a cluster the models are
# already integrated and the `.dat` files are on disk; re-running them locally
# is neither possible at production size nor necessary. This reads only the §4
# output, so it works on any run from any machine, including ones built by the
# GPU path and ones whose `K` came from different builds.
#
# ---------------------------------------------------------------------------
# WHAT TO COMPARE, AND WHEN — the two traps this script exists to avoid
#
# 1. **`V_max` is the metric, and it amplifies.** `PERFORMANCE.md` §4b measured
#    that an error in `K` shows up in `V_max` 2-3 orders of magnitude larger
#    than in the matrix norm, because `V ~ exp(τ/aσ̄)`. Slip and moment rate are
#    *not* amplified (slip error stayed ≤0.02% where `V_max` moved percent).
#    So `V_max` is the quantity that decides convergence, and slip is reported
#    alongside it as a control: a run where `V_max` differs but slip does not is
#    showing amplification of a small stress difference, not a different
#    solution.
#
# 2. **100 h is not 30 days, and the difference is not small.** Injection stops
#    at `t_off` = 100 h, `V_max` drops an order of magnitude, and the relaxation
#    phase is far more sensitive to the long-range kernel than the driven phase.
#    §4b measured a configuration at 0.004% through 100 h and **97%** over 30
#    days. Every existing convergence study in this repo stops at 100 h, so this
#    script reports the two windows *separately* and the full run — if the
#    post-injection column is much larger than the injection one, the 100 h
#    figure was hiding the answer.
#
# Runs may differ in `saveat` (it was corrected 300 -> 200 s partway through
# this project), so everything is linearly interpolated onto the reference's
# own time grid, restricted to the overlap. Comparing row-by-row would silently
# compare different times.
#
# §4.3 profiles are deliberately NOT compared: they are written at each run's
# own grid spacing, so a Δz = 20 m and a Δz = 10 m run do not share abscissae
# and the comparison would need spatial interpolation too. §4.1 stations and
# §4.2 `global.dat` are resolution-independent and are what these tables use.
using Printf

const DAY = 86400.0
const T_OFF = 100 * 3600.0        # injection shut-in (PDF §2.1)

datarows(path) = [r for r in (tryparse.(Float64, split(l)) for l in eachline(path))
                  if !isempty(r) && !any(isnothing, r)]

"""
    series(dir) -> NamedTuple

`t`, `Vmax` (m/s, undoing `global.dat`'s log10) and `moment_rate` from
`global.dat`, plus centre-station `slip_2` and `shear_stress_2`.
"""
function series(dir)
    g = joinpath(dir, "global.dat")
    s = joinpath(dir, "fltst_strk+000dp+000.dat")
    isfile(g) || error("no global.dat in $dir")
    G = datarows(g)
    isempty(G) && error("global.dat in $dir has no data rows")
    t = [r[1] for r in G]
    out = (; dir, t, Vmax = [10.0^r[2] for r in G], moment = [r[3] for r in G],
           slip = Float64[], stress = Float64[])
    if isfile(s)
        S = datarows(s)
        ts = [r[1] for r in S]
        ts == t || @warn "station and global time grids differ in $dir; station metrics skipped"
        if ts == t
            return (; out..., slip = [r[2] for r in S], stress = [r[6] for r in S])
        end
    end
    return out
end

"Linear interpolation of `(xs, ys)` at `x`, clamped to the data range."
function interp(xs, ys, x)
    x <= xs[1] && return ys[1]
    x >= xs[end] && return ys[end]
    i = searchsortedlast(xs, x)
    w = (x - xs[i]) / (xs[i+1] - xs[i])
    return ys[i] * (1 - w) + ys[i+1] * w
end

"""
    worstrel(ref, cmp, field, window) -> (worst, t_at_worst)

Worst |a-b|/|b| over the times of `ref` inside `window`, with `cmp`
interpolated onto them. Zero/absent reference values are skipped rather than
producing an Inf that would swamp the maximum.
"""
function worstrel(ref, cmp, field, window)
    rv, cv = getfield(ref, field), getfield(cmp, field)
    (isempty(rv) || isempty(cv)) && return (NaN, NaN)
    lo, hi = window
    tmin, tmax = max(ref.t[1], cmp.t[1]), min(ref.t[end], cmp.t[end])
    worst, at = 0.0, NaN
    for (k, t) in enumerate(ref.t)
        (lo <= t <= hi && tmin <= t <= tmax) || continue
        b = rv[k]
        abs(b) > 0 || continue
        d = abs(interp(cmp.t, cv, t) - b) / abs(b)
        d > worst && (worst = d; at = t)
    end
    return (100worst, at)
end

dirs = isempty(ARGS) ?
       sort(filter(isdir, [joinpath("output", d) for d in
                           (isdir("output") ? readdir("output") : String[])
                           if startswith(d, "BP8-QD-")])) : ARGS
length(dirs) >= 2 || error("""
    need at least two output directories to compare, found $(length(dirs)).
    Pass them explicitly, or copy the cluster's `output/` down first.""")

runs = series.(dirs)
ref = runs[end]

println("reference (last listed): ", basename(ref.dir))
@printf("            %d rows, t = 0 .. %.2f d\n\n", length(ref.t), ref.t[end] / DAY)

windows = [("injection (t<=100h)", (0.0, T_OFF)),
           ("post-shutin (t>100h)", (T_OFF, Inf)),
           ("FULL RUN", (0.0, Inf))]

for (field, label) in ((:Vmax, "V_max  [the deciding metric, amplified ~1e2-1e3x]"),
                       (:slip, "slip_2 [control: NOT amplified]"),
                       (:moment, "moment_rate [control: NOT amplified]"))
    println(label)
    @printf("  %-42s %14s %14s %14s\n", "run", "injection", "post-shutin", "FULL RUN")
    for r in runs[1:end-1]
        vals = [worstrel(ref, r, field, w)[1] for (_, w) in windows]
        @printf("  %-42s %13s %13s %13s\n", basename(r.dir),
                map(v -> isnan(v) ? "n/a" : @sprintf("%.3f%%", v), vals)...)
    end
    println()
end

# The trap from the header, made explicit: a run that looks converged through
# injection and is not converged over the full 30 days is the exact failure
# §4b documents, and it is invisible in any table that stops at 100 h.
println("-"^80)
for r in runs[1:end-1]
    inj, _ = worstrel(ref, r, :Vmax, (0.0, T_OFF))
    full, at = worstrel(ref, r, :Vmax, (0.0, Inf))
    (isnan(inj) || isnan(full) || inj <= 0) && continue
    if full > 3inj
        @printf("WARNING  %s: V_max agrees to %.3f%% through injection but %.3f%% over the \
                 full run (worst at t = %.2f d).\n         The injection-only figure \
                 understates the difference %.0fx — see PERFORMANCE.md §4b.\n",
                basename(r.dir), inj, full, at / DAY, full / inj)
    end
end
@printf("\nAll figures are worst relative difference against %s over the stated window.\n",
        basename(ref.dir))
println("Convergence means the numbers SHRINK as runs approach the reference, in the")
println("FULL RUN column. A single small number proves nothing on its own.")

# ---------------------------------------------------------------------------
# Overlay plot. The table above is what you make the decision on; this is what
# shows you *where* and *why* the runs part company — the tables collapse the
# whole history into one worst-case number, which cannot distinguish "differs
# everywhere by a little" from "tracks perfectly until shut-in, then diverges".
# Both appear in this problem and they mean different things.
if get(ENV, "BP8_COMPARE_PLOT", "1") != "0"
    using CairoMakie
    fig = Figure(size=(1100, 820))

    ax1 = Axis(fig[1, 1]; xlabel="t (days)", ylabel="log10 V_max (m/s)",
               title="V_max — overlay")
    ax2 = Axis(fig[1, 2]; xlabel="t (days)", ylabel="|ΔV_max| / V_max (%)",
               yscale=log10, title="relative difference vs $(basename(ref.dir))")
    ax3 = Axis(fig[2, 1]; xlabel="t (days)", ylabel="slip_2 (m)",
               title="centre-station slip — overlay (control)")
    ax4 = Axis(fig[2, 2]; xlabel="t (days)", ylabel="moment rate (N m/s)",
               yscale=log10, title="moment rate — overlay")

    for r in runs
        isref = r.dir == ref.dir
        lab = basename(r.dir)
        sty = (; label=lab, linewidth = isref ? 2.5 : 1.5,
               linestyle = isref ? :solid : :dash)
        lines!(ax1, r.t ./ DAY, log10.(max.(r.Vmax, 1e-30)); sty...)
        isempty(r.slip) || lines!(ax3, r.t ./ DAY, r.slip; sty...)
        lines!(ax4, r.t ./ DAY, max.(r.moment, 1e-30); sty...)
        if !isref
            d = [(b = ref.Vmax[k]; abs(b) > 0 ?
                  max(100abs(interp(r.t, r.Vmax, t) - b) / abs(b), 1e-6) : NaN)
                 for (k, t) in enumerate(ref.t)]
            lines!(ax2, ref.t ./ DAY, d; label=lab, linewidth=1.5)
        end
    end
    # Shut-in is where the tables split their windows, so mark it on every axis.
    for ax in (ax1, ax2, ax3, ax4)
        vlines!(ax, [T_OFF / DAY]; color=:gray, linestyle=:dot)
    end
    Legend(fig[3, 1:2], ax1; orientation=:horizontal, nbanks=2, framevisible=false)

    out = joinpath(dirname(ref.dir), "comparison.png")
    save(out, fig)
    println("\noverlay plot -> $out   (set BP8_COMPARE_PLOT=0 to skip)")
end

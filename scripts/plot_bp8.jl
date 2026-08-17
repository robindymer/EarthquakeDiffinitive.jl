# Draws the standard SEAS figure set from a BP8 output directory.
#
#   julia --project=scripts scripts/plot_bp8.jl [outdir]
#
# Reads only the §4 .dat files, so it works on any run (and doubles as a check
# that those files are actually parseable). Figures are written next to them.
#
# Four figures:
#   global.png          max slip rate and moment rate vs time
#   stations.png        slip, slip rate, shear stress and pressure at the 9 stations
#   slip_evolution.png  slip profiles along strike and depth, one curve per interval
#   spacetime.png       space-time contours of slip, stress and pressure
using CairoMakie
using Printf

outdir = length(ARGS) >= 1 ? ARGS[1] :
         joinpath(@__DIR__, "..", "output", "BP8-QD-GS_dz50_Lf800_Ln400")
isdir(outdir) || error("no such output directory: $outdir")
@info "plotting" outdir

const DAY = 86400.0

"""
Numeric rows of a §4 data file. Header lines start with `#`; the field list is
plain text (one line for time series, four for profiles), so a row counts only
if every token parses as a float.
"""
function datarows(path)
    rows = Vector{Float64}[]
    for line in eachline(path)
        (isempty(strip(line)) || startswith(line, "#")) && continue
        toks = split(line)
        vals = tryparse.(Float64, toks)
        any(isnothing, vals) && continue
        push!(rows, Vector{Float64}(vals))
    end
    return rows
end

timeseries(path) = reduce(hcat, datarows(path))'   # rows × columns

"Profile file: returns (coords, times, values[time, coord]) plus max slip rate."
function profile(path)
    rows = datarows(path)
    coords = rows[1][3:end]
    body = rows[2:end]
    times = [r[1] for r in body]
    vmax = [r[2] for r in body]
    vals = reduce(hcat, (r[3:end] for r in body))'  # time × coord
    return coords, times, vals, vmax
end

# Thin very long series so Makie is not asked to draw 45k points per curve.
thin(n; target=4000) = n <= target ? (1:n) : (1:cld(n, target):n)

problem = occursin("PW", basename(rstrip(outdir, '/'))) ? "BP8-QD-PW" : "BP8-QD-GS"
label = basename(rstrip(outdir, '/'))

# ---------------------------------------------------------------- global.dat
let
    d = timeseries(joinpath(outdir, "global.dat"))
    k = thin(size(d, 1))
    t = d[k, 1] ./ DAY
    fig = Figure(size=(900, 640))
    Label(fig[0, 1], "$problem — source parameters   ($label)",
          fontsize=17, font=:bold, tellwidth=false)

    ax1 = Axis(fig[1, 1], ylabel="max slip rate  (log₁₀ m/s)",
               xlabel="time (days)", title="V_max(t)")
    lines!(ax1, t, d[k, 2], color=:firebrick, linewidth=2)

    ax2 = Axis(fig[2, 1], ylabel="moment rate  (N·m/s)", xlabel="time (days)",
               yscale=log10, title="M₀̇(t)")
    lines!(ax2, t, max.(d[k, 3], eps()), color=:steelblue, linewidth=2)

    for ax in (ax1, ax2)
        vlines!(ax, [100 * 3600 / DAY], color=(:black, 0.45), linestyle=:dash)
    end
    text!(ax1, 100 * 3600 / DAY, minimum(d[k, 2]); text=" injection off",
          align=(:left, :bottom), fontsize=11, color=(:black, 0.6))

    save(joinpath(outdir, "global.png"), fig)
    @info "wrote global.png"
end

# ------------------------------------------------------------ station series
let
    stations = ["fltst_strk+000dp+000", "fltst_strk+200dp+000", "fltst_strk-200dp+000",
                "fltst_strk+000dp+200", "fltst_strk+000dp-200", "fltst_strk+200dp+200",
                "fltst_strk-200dp-200", "fltst_strk+200dp-200", "fltst_strk-200dp+200"]
    present = filter(s -> isfile(joinpath(outdir, s * ".dat")), stations)
    colors = cgrad(:viridis, max(length(present), 2), categorical=true)

    fig = Figure(size=(1100, 800))
    Label(fig[0, 1:2], "$problem — on-fault stations   ($label)",
          fontsize=17, font=:bold, tellwidth=false)

    # (column index in the file, axis label, unit scaling)
    panels = [(2, "slip  s₂ (m)", identity),
              (4, "slip rate  log₁₀|V₂| (m/s)", identity),
              (6, "shear stress  τ₂ (MPa)", identity),
              (8, "pore pressure  p (MPa)", identity)]

    axes_ = Axis[]
    for (n, (col, ylab, f)) in enumerate(panels)
        ax = Axis(fig[cld(n, 2), mod1(n, 2)], xlabel="time (days)", ylabel=ylab)
        push!(axes_, ax)
        for (i, s) in enumerate(present)
            d = timeseries(joinpath(outdir, s * ".dat"))
            k = thin(size(d, 1))
            lines!(ax, d[k, 1] ./ DAY, f.(d[k, col]), color=colors[i], linewidth=1.6,
                   label=replace(s, "fltst_" => ""))
        end
        vlines!(ax, [100 * 3600 / DAY], color=(:black, 0.4), linestyle=:dash)
    end
    Legend(fig[1:2, 3], axes_[1], "station", framevisible=false, labelsize=10)

    save(joinpath(outdir, "stations.png"), fig)
    @info "wrote stations.png"
end

# --------------------------------------------------------- slip evolution
# The classic SEAS plot: one slip profile per fixed interval, so the spacing
# between curves reads directly as slip rate.
let
    fig = Figure(size=(1000, 440))
    Label(fig[0, 1:2], "$problem — slip evolution, one profile every 6 h   ($label)",
          fontsize=17, font=:bold, tellwidth=false)

    for (n, (fname, xlab)) in enumerate([("slip_2_strike.dat", "along strike  x₂ (m)"),
                                         ("slip_2_depth.dat", "along depth  x₃ (m)")])
        path = joinpath(outdir, fname)
        isfile(path) || continue
        coords, times, vals, _ = profile(path)
        ax = Axis(fig[1, n], xlabel=xlab, ylabel="slip  s₂ (m)")
        step = max(1, round(Int, 6 * 3600 / max(times[2] - times[1], 1)))
        idx = 1:step:length(times)
        for i in idx
            lines!(ax, coords, vals[i, :],
                   color=times[i] / DAY, colorrange=(0, times[end] / DAY),
                   colormap=:plasma, linewidth=1.2)
        end
        n == 2 && Colorbar(fig[1, 3], colormap=:plasma,
                           colorrange=(0, times[end] / DAY), label="time (days)")
    end
    save(joinpath(outdir, "slip_evolution.png"), fig)
    @info "wrote slip_evolution.png"
end

# ------------------------------------------------------------- space-time
let
    wanted = [("slip_2_strike.dat", "slip s₂ (m)", :viridis),
              ("shear_stress_2_strike.dat", "shear stress τ₂ (MPa)", :RdBu),
              ("pore_pressure_strike.dat", "pore pressure p (MPa)", :inferno),
              ("slip_2_depth.dat", "slip s₂ along depth (m)", :viridis)]
    fig = Figure(size=(1100, 780))
    Label(fig[0, 1:2], "$problem — space-time evolution   ($label)",
          fontsize=17, font=:bold, tellwidth=false)

    for (n, (fname, title, cmap)) in enumerate(wanted)
        path = joinpath(outdir, fname)
        isfile(path) || continue
        coords, times, vals, _ = profile(path)
        r, c = cld(n, 2), mod1(n, 2)
        ax = Axis(fig[r, 2c-1], xlabel=occursin("depth", fname) ? "x₃ (m)" : "x₂ (m)",
                  ylabel="time (days)", title=title)
        hm = heatmap!(ax, coords, times ./ DAY, permutedims(vals), colormap=cmap)
        Colorbar(fig[r, 2c], hm)
    end
    save(joinpath(outdir, "spacetime.png"), fig)
    @info "wrote spacetime.png"
end

@info "done" figures = filter(f -> endswith(f, ".png"), readdir(outdir))

# One station's 10 time series in the SEAS code-verification platform layout
# (5 rows × 2 columns, same panel order), overlaying one or more runs.
#
#   julia --project=scripts scripts/plot_bp8_station.jl [--station=NAME] outdir1 [outdir2 ...]
#
# NAME defaults to fltst_strk+000dp+000. Written to <outdir1>/<NAME>_cvp.png.
using CairoMakie

station = "fltst_strk+000dp+000"
outdirs = String[]
for a in ARGS
    if startswith(a, "--station=")
        global station = split(a, "=", limit=2)[2]
    else
        push!(outdirs, a)
    end
end
isempty(outdirs) && push!(outdirs, joinpath(@__DIR__, "..", "output",
                                            "BP8-QD-GS_dz20_Lf2000_Ln2000_exact_gpu"))

"Numeric rows of a §4 time-series file (header and field-name lines skipped)."
function timeseries(path)
    rows = Vector{Float64}[]
    for line in eachline(path)
        (isempty(strip(line)) || startswith(line, "#")) && continue
        vals = tryparse.(Float64, split(line))
        any(isnothing, vals) && continue
        push!(rows, Vector{Float64}(vals))
    end
    return reduce(hcat, rows)'
end

# Column order of the station file == panel order on the platform.
panels = ["slip 2 (m)", "slip 3 (m)",
          "slip rate 2 (log10 m/s)", "slip rate 3 (log10 m/s)",
          "shear stress 2 (MPa)", "shear stress 3 (MPa)",
          "pore pressure (MPa)", "Darcy velocity 2 (m/s)",
          "Darcy velocity 3 (m/s)", "state (log10 s)"]

thin(n; target=4000) = n <= target ? (1:n) : (1:cld(n, target):n)

fig = Figure(size=(1200, 1500))
Label(fig[0, 1:2], station, fontsize=18, font=:bold, tellwidth=false)
axs = [Axis(fig[cld(i, 2), mod1(i, 2)], title=p, ylabel=p, xlabel="time (s)",
            titlefont=:regular) for (i, p) in enumerate(panels)]

colors = Makie.wong_colors()
peak = zeros(length(panels))
for (j, dir) in enumerate(outdirs)
    path = joinpath(dir, station * ".dat")
    isfile(path) || (@warn "missing" path; continue)
    d = timeseries(path)
    k = thin(size(d, 1))
    for (i, ax) in enumerate(axs)
        peak[i] = max(peak[i], maximum(abs, d[:, i+1]))
        lines!(ax, d[k, 1], d[k, i+1], color=colors[mod1(j, end)], linewidth=1.5,
               label=basename(rstrip(dir, '/')))
    end
end
# Roundoff-level series (e.g. Darcy velocity at a symmetry point) would otherwise
# autoscale to ~1e-20 and give unreadable ticks; show them as flat zero.
for (ax, m) in zip(axs, peak)
    m < 1e-15 && ylims!(ax, -1e-9, 1e-9)
end
Legend(fig[1, 3], axs[1], framevisible=false, labelsize=11)

out = joinpath(outdirs[1], station * "_cvp.png")
save(out, fig)
@info "wrote" out

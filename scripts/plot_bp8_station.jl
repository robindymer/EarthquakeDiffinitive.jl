# One station's 10 time series in the SEAS code-verification platform layout
# (5 rows × 2 columns, same panel order), overlaying one or more runs.
#
#   julia --project=scripts scripts/plot_bp8_station.jl [--station=NAME|all] dir1 [dir2 ...]
#   BP8_COMPARE_VARIANT=GS julia --project=scripts scripts/plot_bp8_station.jl --station=all pelle/output
#
# NAME defaults to fltst_strk+000dp+000. A `dir` without a `global.dat` is a
# parent: its `BP8-QD-*` runs are overlaid (same rule as `bp8_compare_runs.jl`),
# and the plot goes to <parent>/<NAME>_cvp.png; otherwise to <dir1>/<NAME>_cvp.png.
# Line style encodes Δz (solid = finest), colour the run.
using CairoMakie

station = "fltst_strk+000dp+000"
args = String[]
for a in ARGS
    if startswith(a, "--station=")
        global station = split(a, "=", limit=2)[2]
    else
        push!(args, a)
    end
end
isempty(args) && push!(args, joinpath(@__DIR__, "..", "output",
                                      "BP8-QD-GS_dz20_Lf2000_Ln2000_exact_gpu"))

isrun(d) = isfile(joinpath(d, "global.dat"))
discover(parent) = sort(filter(isdir, [joinpath(parent, d) for d in readdir(parent)
                                       if startswith(d, "BP8-QD-")]))
outdirs = reduce(vcat, [isrun(a) ? [a] : discover(a) for a in args])
variant(d) = (m = match(r"^BP8-QD-([A-Za-z]+)", basename(d)); m === nothing ? "?" : m.captures[1])
want = get(ENV, "BP8_COMPARE_VARIANT", "")
isempty(want) || filter!(d -> variant(d) == want, outdirs)
isempty(outdirs) && error("no BP8 run directories found in $(join(args, ", "))")
length(unique(variant.(outdirs))) > 1 &&
    @warn "overlaying different variants $(unique(variant.(outdirs))); set BP8_COMPARE_VARIANT to pick one"
savedir = isrun(args[1]) ? outdirs[1] : args[1]

"`Δz` in m from a station header, `NaN` if absent."
function gridsize(path)
    for (i, l) in enumerate(eachline(path))
        i > 40 && break
        m = match(r"^#\s*element_size\s*=\s*([0-9.eE+-]+)", l)
        m === nothing || return parse(Float64, m.captures[1])
    end
    return NaN
end

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

stations = station == "all" ?
    sort([splitext(f)[1] for f in readdir(outdirs[1]) if startswith(f, "fltst_") && endswith(f, ".dat")]) :
    [station]

colors = length(outdirs) <= 7 ? Makie.wong_colors() :
         Makie.resample_cmap(:tab10, max(length(outdirs), 10))
styles = [:solid, :dash, :dot, :dashdot]

for st in stations
    fig = Figure(size=(1300, 1500))
    Label(fig[0, 1:2], st, fontsize=18, font=:bold, tellwidth=false)
    axs = [Axis(fig[cld(i, 2), mod1(i, 2)], title=p, ylabel=p, xlabel="time (s)",
                titlefont=:regular) for (i, p) in enumerate(panels)]
    paths = [joinpath(dir, st * ".dat") for dir in outdirs]
    dzs = [isfile(p) ? gridsize(p) : NaN for p in paths]
    dzlevels = sort(unique(filter(!isnan, dzs)))
    peak = zeros(length(panels))
    for (j, (dir, path)) in enumerate(zip(outdirs, paths))
        isfile(path) || (@warn "missing" path; continue)
        d = timeseries(path)
        k = thin(size(d, 1))
        ls = isnan(dzs[j]) ? :solid : styles[mod1(findfirst(==(dzs[j]), dzlevels), end)]
        for (i, ax) in enumerate(axs)
            peak[i] = max(peak[i], maximum(abs, d[:, i+1]))
            lines!(ax, d[k, 1], d[k, i+1], color=colors[mod1(j, end)], linewidth=1.5,
                   linestyle=ls, label=basename(rstrip(dir, '/')))
        end
    end
    # Roundoff-level series (e.g. Darcy velocity at a symmetry point) would otherwise
    # autoscale to ~1e-20 and give unreadable ticks; show them as flat zero.
    for (ax, m) in zip(axs, peak)
        m < 1e-15 && ylims!(ax, -1e-9, 1e-9)
    end
    Legend(fig[1:2, 3], axs[1], framevisible=false, labelsize=11)

    out = joinpath(savedir, st * "_cvp.png")
    save(out, fig)
    @info "wrote" out
end

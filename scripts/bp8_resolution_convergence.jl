# Run with:  julia --project=. scripts/bp8_resolution_convergence.jl
#
# ONCE WE HAVE A CLUSTER-PRODUCED REFERENCE SOLUTION:
#   1. Copy its output/ directory down locally (gitignored, so this only
#      exists once you've done that), e.g. from
#      `julia --project=. scripts/run_bp8.jl gs 20 1600 1200`.
#   2. Point REFERENCE_DIR, L_FAULT and L_NORMAL below at it — all three must
#      match, or `read_reference` will warn (domain mismatch) or the sweep and
#      reference just won't be comparable.
#   3. Make sure every entry in Δzs is coarser than the reference's own Δz —
#      see the note at Δzs below for why.
#   4. Re-run the script. The "observed order of convergence" section at the
#      bottom of `report` is the actual answer to "what is the convergence
#      rate" — the relative-difference table above it is just the raw numbers
#      that rate is fit from.
using EarthquakeDiffinitive
using EarthquakeDiffinitive.BP8
using Printf

const T_END = 100 * 3600.0   # through the injection period, matches bp8_domain_convergence.jl
const L_FAULT = 1600.0
const L_NORMAL = 1200.0

# Coarsest first, so a table row prints as soon as it's done. Every entry must
# be coarser than REFERENCE_DIR's own Δz, or its "error" is partly just the
# reference's own discretization error rather than this sweep's.
const Δzs = [100.0, 50.0, 25.0]

# Produced by `julia --project=. scripts/run_bp8.jl gs 20 1600 1200` (or finer)
# on a cluster — must match L_FAULT/L_NORMAL above to be a fair comparison
# (checked at load time) and should be the finest Δz you can afford, since
# every rate computed below is only as trustworthy as this reference is
# converged. `output/` is gitignored, so this only exists locally once that
# run has been copied down.
const REFERENCE_DIR = "output/BP8-QD-GS_dz20_Lf1600_Ln1200"

"""
    datarows(file)

Parses a §4 `.dat` file's time-series rows, skipping the header block and the
one field-name line (neither starts a row with a parseable number).
"""
datarows(file) = [parse.(Float64, split(l)) for l in eachline(file)
                   if !isempty(strip(l)) &&
                      tryparse(Float64, first(split(l))) !== nothing]

"""
    read_reference(dir, t) -> NamedTuple

Reads the centre-station (`fltst_strk+000dp+000.dat`) and whole-fault
(`global.dat`) quantities nearest time `t` from a previously written BP8
output directory, converting units to match `sweep`'s (m, m/s, Pa). Warns if
the reference was written for a different domain, since that would make the
comparison meaningless rather than merely stale.
"""
function read_reference(dir, t)
    station, global_ = joinpath(dir, "fltst_strk+000dp+000.dat"), joinpath(dir, "global.dat")
    isfile(station) && isfile(global_) ||
        error("no BP8 output at $dir — run e.g. `julia --project=. scripts/run_bp8.jl gs " *
              "$(Int(Δzs[1])) $(Int(L_FAULT)) $(Int(L_NORMAL))` first")

    header = read(global_, String)
    dom = match(r"\|x1\| <= ([\d.]+) m, \|x2\|,\|x3\| <= ([\d.]+) m", header)
    if dom !== nothing && (parse(Float64, dom[1]) != L_NORMAL || parse(Float64, dom[2]) != L_FAULT)
        @warn "reference domain does not match this sweep" dir ref_L_normal = dom[1] ref_L_fault = dom[2] L_NORMAL L_FAULT
    end

    nearest(rows) = argmin(r -> abs(r[1] - t), rows)
    s = nearest(datarows(station))   # t slip_2 slip_3 slip_rate_2 slip_rate_3 τ2 τ3 p darcy_2 darcy_3 state
    g = nearest(datarows(global_))   # t max_slip_rate(log10) moment_rate
    abs(s[1] - t) > 0.01T_END && @warn "reference's nearest saved time is off target" t_ref = s[1] t
    return (; slip_centre=s[2], p_centre=s[8] * 1e6, Vmax=10.0^g[2])
end

"""
    sweep(Δzs) -> Vector{NamedTuple}

Runs BP8-QD-GS at each Δz to `T_END` and collects the quantities resolution
would bias: the fault's self-stiffness, centre slip and peak slip rate, and
centre pore pressure (which lives on Ω_f only, so it is the cleanest check
that nothing outside resolution drifted).
"""
function sweep(Δzs)
    results = NamedTuple[]
    for Δz in Δzs
        @info "building" Δz L_FAULT L_NORMAL
        flush(stdout)
        try
            t0 = time()
            m = build_model(; Δz, L_fault=L_FAULT, L_normal=L_NORMAL)
            t_build = time() - t0
            sol = run_bp8(m; tspan=(0.0, T_END), saveat=T_END)
            c = evaluate!(m, sol.u[end], sol.t[end])
            nf = m.nf
            ic = argmin(abs.(m.x2)) + (argmin(abs.(m.x3)) - 1) * length(m.x2)
            push!(results, (; Δz,
                            dofs=m.grid_info.elastic_dofs,
                            t_build,
                            K_self=m.K[ic, ic],
                            slip_centre=sol.u[end][ic],
                            Vmax=maximum(c.Vmag),
                            p_centre=sol.u[end][3nf+ic]))
            r = results[end]
            @printf("DONE Δz=%.0f dofs=%d build=%.1fs K_self=%.5E slip=%.5E Vmax=%.5E p=%.4f MPa\n",
                    r.Δz, r.dofs, r.t_build, r.K_self, r.slip_centre, r.Vmax, r.p_centre / 1e6)
        catch e
            @warn "configuration failed" Δz exception = e
        end
        flush(stdout)
        flush(stderr)
    end
    return results
end

"""
    report(results, ref)

Prints the table, each row's relative difference from `ref` (the imported
reference solution, not one of the sweep's own rows), and the observed order
of convergence `p` between each consecutive pair of Δz, fit from
`e(Δz) ∝ Δz^p` as `p = log(e_coarse/e_fine) / log(Δz_coarse/Δz_fine)`. Needs
at least two rows on either side of a pair, so the first row has no `p`.
"""
function report(results, ref)
    println("\n", "="^90)
    println("BP8-QD-GS resolution convergence,  L_fault=$L_FAULT L_normal=$L_NORMAL,  t = $(T_END/3600) h")
    println("reference: $REFERENCE_DIR")
    println("="^90)
    isempty(results) && (println("no configuration completed"); return)
    @printf("%6s %10s %8s %13s %13s %13s %13s\n",
            "Δz", "elast.DOF", "build s", "K_self", "slip(0,0)", "Vmax", "p(0,0) MPa")
    for r in results
        @printf("%6.0f %10d %8.1f %13.5E %13.5E %13.5E %13.5f\n",
                r.Δz, r.dofs, r.t_build, r.K_self, r.slip_centre, r.Vmax, r.p_centre / 1e6)
    end

    err_slip(r) = abs(r.slip_centre - ref.slip_centre) / abs(ref.slip_centre)
    err_Vmax(r) = abs(r.Vmax - ref.Vmax) / abs(ref.Vmax)

    println("\nrelative difference from the imported reference (K_self has no reference",
            " — it isn't in the §4 output):")
    @printf("%6s %13s %13s\n", "Δz", "d(slip)", "d(Vmax)")
    for r in results
        @printf("%6.0f %12.2f%% %12.2f%%\n", r.Δz, 100err_slip(r), 100err_Vmax(r))
    end

    length(results) < 2 && return
    println("\nobserved order of convergence p, fit between consecutive Δz from e(Δz) ∝ Δz^p",
            " (an SBP order-4 interior / order-2 boundary scheme should land near p=2, per",
            " test/convergence_test.jl's solution-order figures — well short of that means",
            " something other than truncation error, e.g. the reference itself, is binding):")
    @printf("%6s %13s %13s\n", "Δz pair", "p(slip)", "p(Vmax)")
    rate(err, c, f) = log(err(c) / err(f)) / log(c.Δz / f.Δz)
    for (c, f) in zip(results, results[2:end])   # coarse, fine
        @printf("%3.0f→%-3.0f %12.2f  %12.2f\n", c.Δz, f.Δz,
                rate(err_slip, c, f), rate(err_Vmax, c, f))
    end
end

ref = read_reference(REFERENCE_DIR, T_END)
results = sweep(Δzs)
report(results, ref)

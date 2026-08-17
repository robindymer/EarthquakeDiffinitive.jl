# Checks the pore-pressure part of both injection models against the analytic
# solutions the benchmark supplies — eq. 21 for the Gaussian source, eq. 25 for
# the point source the Peaceman well is meant to reproduce (§6 asks for this
# check explicitly).
#
#   julia --project=. scripts/bp8_validate_pressure.jl
using EarthquakeDiffinitive
using EarthquakeDiffinitive.BP8
using Printf

const T_CHECK = 100 * 3600.0    # t_off; well inside t ≪ l_f²/4α ≈ 220 h

function check(injection, Δz)
    # Pressure is one-way decoupled from elasticity (σ̄ feeds friction, nothing
    # feeds back into eq. 17), so the elastic block only has to be legal, not
    # converged — 8Δz is the smallest an SBP order-4 grid can be.
    m = build_model(; Δz, L_fault=800.0, L_normal=8Δz, injection)
    sol = run_bp8(m; tspan=(0.0, T_CHECK), saveat=T_CHECK)
    nf = m.nf
    p = sol.u[end][3nf+1:4nf]
    n2 = length(m.x2)
    i0, j0 = argmin(abs.(m.x2)), argmin(abs.(m.x3))
    rows = NamedTuple[]
    for i in i0:n2
        r = abs(m.x2[i])
        num = p[i+(j0-1)*n2]
        ana = injection === :gaussian ? analytic_pressure_gaussian(r, T_CHECK, m.par) :
              analytic_pressure_point(max(r, 0.198Δz), T_CHECK, m.par)
        push!(rows, (; r, num, ana))
    end
    return m, rows
end

for injection in (:gaussian, :peaceman), Δz in (100.0, 50.0)
    m, rows = check(injection, Δz)
    println("\n", "="^64)
    println("$(injection === :gaussian ? "Gaussian source (eq. 21)" : "Peaceman well vs point source (eq. 25)"),  Δz = $Δz m,  t = $(T_CHECK/3600) h")
    println("="^64)
    @printf("%8s %14s %14s %10s\n", "r (m)", "numeric MPa", "analytic MPa", "rel.err")
    for row in rows
        row.r > 400 && continue
        @printf("%8.0f %14.5f %14.5f %9.2f%%\n", row.r, row.num / 1e6, row.ana / 1e6,
                100abs(row.num - row.ana) / max(abs(row.ana), 1e-12))
    end
    if injection === :peaceman
        rep = effective_stress_report(m)
        @printf("  effective normal stress: lowest σ̄ = %.4f MPa, floor bound %s\n",
                rep.σ̄_lowest / 1e6, rep.bound ? "YES ($(rep.floor_hits) evaluations)" : "no")
    end
end

println("""

Notes
-----
* The Gaussian source is smeared over L_gauss = 50 m, so Δz = 50 m resolves it
  only marginally and Δz = 100 m not at all — the near-source error shrinks
  quickly with resolution, which is the point of the comparison.
* The point-source solution is singular at r = 0, so the Peaceman row at r = 0
  is compared at the equivalent radius r_e = 0.198Δz, which is what the well
  cell's pressure is supposed to represent.
* Both analytic solutions assume an unbounded fault; ours has no-flux edges at
  |x2|,|x3| = 400 m, so agreement degrades near the boundary and for later t.
""")

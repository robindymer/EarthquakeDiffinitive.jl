using EarthquakeDiffinitive
using EarthquakeDiffinitive.PorePressure
using Diffinitive.SbpOperators
using SpecialFunctions: expint
using Test

# BP8-QD-GS parameters (Table 1 of the benchmark description).
const β = 1e-8      # Pa⁻¹
const φ = 0.1
const k = 5e-14     # m²
const η = 1e-3      # Pa·s
const α = k / (φ * β * η)
const l_f = 400.0   # m
const L_gauss = 50.0 # m
const Q0 = 0.003    # m³/s
const L_fwid = 1.0  # m
const q0 = Q0 / L_fwid
const t_off = 100 * 3600.0 # s

function analytic_pressure(x2, x3, t)
    prefactor = q0 / (4π * α * β * φ)
    r2 = x2^2 + x3^2
    if iszero(r2)
        return prefactor * log((2L_gauss^2 + 4α * t) / (2L_gauss^2))
    else
        return prefactor * (expint(r2 / (2L_gauss^2 + 4α * t)) - expint(r2 / (2L_gauss^2)))
    end
end

# Index of the grid point at (x2,x3) into the vec()-flattened n×n solution.
function point_index(x2, x3, Δz, n)
    i = round(Int, (x2 + l_f) / Δz) + 1
    j = round(Int, (x3 + l_f) / Δz) + 1
    return i + (j - 1) * n
end

function max_relative_error(Δz; times_hours=(1, 10, 50))
    stencil_set = read_stencil_set(SbpOperators.sbp_operators_path() * "standard_diagonal.toml", order=4)
    g, A = pore_pressure_operator(l_f, Δz, α, stencil_set)
    source = gaussian_source(g, L_gauss)
    n = round(Int, 2l_f / Δz) + 1

    tspan = (0.0, maximum(times_hours) * 3600.0)
    sol = solve_pore_pressure(g, A, source; q0, t_off, β, φ, tspan)

    points = ((0.0, 0.0), (200.0, 0.0), (0.0, 200.0))

    errs = Float64[]
    for h in times_hours
        t = h * 3600.0
        p = sol(t)
        for (x2, x3) in points
            numeric = p[point_index(x2, x3, Δz, n)]
            analytic = analytic_pressure(x2, x3, t)
            push!(errs, abs(numeric - analytic) / abs(analytic))
        end
    end
    return maximum(errs)
end

@testset "PorePressure" begin
    @testset "matches analytic Gaussian-source solution" begin
        @test max_relative_error(10.0) < 0.01
    end

    @testset "error decreases under grid refinement" begin
        e20 = max_relative_error(20.0)
        e10 = max_relative_error(10.0)
        e5 = max_relative_error(5.0)
        @test e10 < e20
        @test e5 < e10
    end
end

module BP8

using Diffinitive
using Diffinitive.Grids
using Diffinitive.SbpOperators
using SparseArrays
using StaticArrays
using LinearAlgebra: mul!, norm, diag
using OrdinaryDiffEq
using Printf
using Dates: today
using Tokens

using ..RateStateFriction
using ..PorePressure
using ..FaultResponse
using ..Elasticity

export BP8Params, benchmark_parameters, BP8Model, build_model, initial_state,
       run_bp8, evaluate!, write_outputs, station_locations,
       effective_stress_report, analytic_pressure_gaussian, analytic_pressure_point,
       resolution_report, process_zone

# ==============================================================================
# SEAS BP8-QD-GS / -PW: the coupled problem.
#
#   slip      ds_j/dt = V_j                                   (eq. 5)
#   state     dθ/dt   = 1 - Vθ/D_RS                           (eq. 11)
#   pressure  dp/dt   = α∇²p + source(t)                      (eq. 17/19-23)
#
# closed at each node by the algebraic force balance
#
#   τ⁰ + Δτ(s) - ηV = σ̄ f(V,θ) V/V,   σ̄ = σ - p               (eq. 7-10)
#
# solved for V by `RateStateFriction.solve_slip_velocity`. `Δτ(s)` is the
# elastic response, a dense mat-vec against the fault stiffness precomputed by
# `FaultResponse.fault_stiffness`.
#
# The state variable is integrated as ϕ = ln θ. θ spans ~1e8-1e12 s here while
# slip is ~1e-6 m, so a single scalar tolerance cannot serve both; ln θ is
# well scaled and turns the aging law into dϕ/dt = e^{-ϕ} - V/D_RS.
# ==============================================================================

"""
    BP8Params

Table 1 of the benchmark description, plus the quantities derived from it
(`λ`, `η`, `q0`). Use [`benchmark_parameters`](@ref) for the specified values.
"""
Base.@kwdef struct BP8Params
    # elastic
    μ::Float64 = 32.04e9        # shear modulus, Pa
    ν::Float64 = 0.25           # Poisson's ratio
    ρ::Float64 = 2670.0         # density, kg/m³
    c_s::Float64 = 3464.0       # shear wave speed, m/s
    # friction
    a::Float64 = 0.016
    b::Float64 = 0.010
    D_RS::Float64 = 0.5e-3      # m
    V_star::Float64 = 1e-6      # m/s
    f_star::Float64 = 0.6
    σ0::Float64 = 25.0e6        # initial effective normal stress, Pa
    τ_init::Float64 = 14.6e6    # Pa
    V_init::Float64 = 1e-12     # m/s
    V_zero::Float64 = 1e-20     # m/s
    # fluid
    Q0::Float64 = 0.003         # m³/s
    L_gauss::Float64 = 50.0     # m
    L_fwid::Float64 = 1.0       # m
    β::Float64 = 1e-8           # 1/Pa
    φ::Float64 = 0.1
    k::Float64 = 5e-14          # m²
    viscosity::Float64 = 1e-3   # Pa·s
    α::Float64 = 0.05           # m²/s
    S_well::Float64 = 1e-7      # m³/Pa
    r_well::Float64 = 0.05      # m
    # geometry / time
    l_f::Float64 = 400.0        # m
    Δz::Float64 = 10.0          # m
    t_off::Float64 = 100 * 3600.0     # s
    t_f::Float64 = 30 * 24 * 3600.0   # s
    # numerical guard: `σ̄ = σ - p` must stay positive for the friction law to
    # make sense. BP8's peak pressure is ~13 MPa against σ = 25 MPa so this
    # should never bind; if it does, the run is outside the model's validity
    # rather than merely under-resolved.
    σ̄_min::Float64 = 1.0e3      # Pa
end

benchmark_parameters(; kwargs...) = BP8Params(; kwargs...)

"Lamé's first parameter from μ and ν."
lame_lambda(p::BP8Params) = 2p.μ * p.ν / (1 - 2p.ν)

"Radiation damping coefficient η = μ/(2c_s) (eq. 8)."
damping(p::BP8Params) = radiation_damping_coefficient(p.μ, p.c_s)

"Injection rate per unit fault thickness, q0 = Q0/L_fwid (eq. 14b)."
q0_per_thickness(p::BP8Params) = p.Q0 / p.L_fwid

friction_params(p::BP8Params) = FrictionParams(p.a, p.b, p.D_RS, p.V_star, p.f_star)

# ------------------------------------------------------------------------------

mutable struct Cache
    Δτ::Vector{Float64}
    V2::Vector{Float64}
    V3::Vector{Float64}
    Vmag::Vector{Float64}
    τ2::Vector{Float64}
    τ3::Vector{Float64}
    Vprev::Vector{Float64}
    σ̄_lowest::Float64   # smallest σ̄ = σ - p seen, BEFORE the floor is applied
    floor_hits::Int     # how many times that floor actually bound
end
Cache(nf) = Cache(zeros(2nf), zeros(nf), zeros(nf), zeros(nf), zeros(nf), zeros(nf),
                  fill(1e-12, nf), Inf, 0)

"""
    BP8Model

Everything needed to evaluate the coupled right-hand side: the precomputed
fault stiffness, the pore-pressure operator and source, the Darcy operators
for output, the `Ω_f` quadrature weights, and the locked-edge mask.
"""
struct BP8Model
    par::BP8Params
    K::Matrix{Float64}              # slip → Δτ, stacked [s2;s3] → [Δτ2;Δτ3]
    Ap::SparseMatrixCSC{Float64,Int}
    source::Vector{Float64}         # Gaussian spatial source (eq. 19)
    Q2::SparseMatrixCSC{Float64,Int}
    Q3::SparseMatrixCSC{Float64,Int}
    weights::Vector{Float64}        # Ω_f quadrature weights, m²
    active::Vector{Bool}            # false on the locked Ω_f edge ring
    x2::Vector{Float64}
    x3::Vector{Float64}
    nf::Int
    τ0::SVector{2,Float64}
    injection::Symbol               # :gaussian or :peaceman
    well_cell::Int
    WI::Float64
    S_e::Float64
    cache::Cache
    grid_info::NamedTuple
end

"""
    build_model(; par=benchmark_parameters(), Δz, L_fault, L_normal,
                  injection=:gaussian, order=4, verbose=false)

Assembles the whole coupled model. `Δz` is the node spacing (must divide
`par.l_f`), `L_fault` the half-width of the elastic grids in the two
fault-parallel directions, `L_normal` their extent in the fault-normal
direction. The far field is truncated with `u=0` at those boundaries, so both
have to be comfortably larger than `l_f`; see the §6 domain-size study.

The expensive part is `fault_stiffness`, which does `2·N_Ωf` back-substitutions
of the factorized 3D elastic system.
"""
function build_model(; par::BP8Params=benchmark_parameters(),
                     Δz=par.Δz, L_fault=3par.l_f, L_normal=2par.l_f,
                     injection=:gaussian, order=4, verbose=false,
                     solver=:cholesky, solver_kwargs...)
    injection ∈ (:gaussian, :peaceman) ||
        error("injection must be :gaussian or :peaceman, got $injection")
    isapprox(par.l_f / Δz, round(par.l_f / Δz); atol=1e-9) ||
        error("Δz=$Δz must divide l_f=$(par.l_f) so the grid has nodes on ±l_f")
    L_fault >= par.l_f || error("L_fault=$L_fault must be at least l_f=$(par.l_f)")

    set = read_stencil_set(SbpOperators.sbp_operators_path() * "standard_diagonal.toml"; order)
    n1 = round(Int, L_normal / Δz) + 1
    n23 = round(Int, 2L_fault / Δz) + 1
    # SBP closures need more than two closure widths of points per dimension.
    n_min = 2order + 1
    n1 >= n_min || error("L_normal/Δz gives only $n1 points across the fault-normal " *
                         "direction; SBP order $order needs at least $n_min. " *
                         "Increase L_normal or decrease Δz.")
    n23 >= n_min || error("2*L_fault/Δz gives only $n23 points along the fault; " *
                          "SBP order $order needs at least $n_min.")
    g_minus = equidistant_grid((-L_normal, -L_fault, -L_fault), (0.0, L_fault, L_fault), n1, n23, n23)
    g_plus = equidistant_grid((0.0, -L_fault, -L_fault), (L_normal, L_fault, L_fault), n1, n23, n23)
    verbose && @info "elastic grids" points_per_side = n1 * n23^2 dofs = 6 * n1 * n23^2

    t0 = time()
    fe = FaultElasticity(g_minus, g_plus, lame_lambda(par), par.μ, set;
                         l_f=par.l_f, solver, solver_kwargs...)
    verbose && @info "split-node system ready" seconds = round(time() - t0, digits=1) solver

    t0 = time()
    K = fault_stiffness(fe; verbose)
    verbose && @info "fault stiffness built" seconds = round(time() - t0, digits=1) size = size(K) elastic_solver_report(fe)...

    x2, x3 = fault_grid_axes(fe)
    nf = frictional_node_count(fe)
    n2f, n3f = length(x2), length(x3)

    # Pore pressure on exactly the Ω_f fault nodes, so p and s share an index.
    n2f >= n_min || error("Ω_f spans only $n2f nodes at Δz=$Δz; the pore-pressure " *
                          "operator needs at least $n_min. Use Δz ≤ $(par.l_f * 2 / (n_min - 1)).")
    g_p = equidistant_grid((-par.l_f, -par.l_f), (par.l_f, par.l_f), n2f, n3f)
    all(isapprox.([g_p[CartesianIndex(i, 1)][1] for i in 1:n2f], x2; atol=1e-9)) ||
        error("pore-pressure grid does not line up with the elastic fault nodes")
    Ap = pore_pressure_operator(g_p, par.α, set)
    source = gaussian_source(g_p, par.L_gauss)
    Q2, Q3 = darcy_operators(g_p, set; k=par.k, viscosity=par.viscosity)
    weights = diag(sparse(inner_product(g_p, set)))

    # eq. 13: locked outside Ω_f. The outer ring of Ω_f nodes is held at V=0
    # so slip is continuous into the locked region — a finite jump there would
    # be a stress singularity the elastic solve cannot represent.
    active = trues(nf)
    lin = LinearIndices((n2f, n3f))
    for j in 1:n3f, i in 1:n2f
        (i == 1 || i == n2f || j == 1 || j == n3f) && (active[lin[i, j]] = false)
    end

    # eq. 28-29: τ⁰ parallel to the initial slip velocity.
    Vvec = SVector(par.V_init, par.V_zero)
    τ0 = par.τ_init * Vvec / norm(Vvec)

    WI = injection === :peaceman ?
         well_index(; k=par.k, L_fwid=par.L_fwid, viscosity=par.viscosity,
                    Δz, r_well=par.r_well) : 0.0
    S_e = peaceman_cell_volume(Δz, par.L_fwid) * par.φ * par.β

    return BP8Model(par, K, Ap, source, Q2, Q3, weights, active, collect(x2), collect(x3),
                    nf, τ0, injection, well_cell_index(g_p), WI, S_e, Cache(nf),
                    (; Δz, L_fault, L_normal, n1, n23, order,
                       elastic_dofs=6 * n1 * n23^2, n2f, n3f))
end

# ------------------------------------------------------------------------------

state_length(m::BP8Model) = 4m.nf + (m.injection === :peaceman ? 1 : 0)

"""
    effective_stress_report(m) -> NamedTuple

Whether the effective-normal-stress floor `par.σ̄_min` ever bound, and how low
`σ̄ = σ - p` actually went. `σ̄ ≤ 0` means fluid pressure has fully unclamped
the fault, at which point BP8's no-opening condition (eq. 3) no longer holds
and the model is outside its range of validity — worth knowing about rather
than silently clamping. The Peaceman-well variant reaches this at Table 1's
parameters; the Gaussian-source variant does not.
"""
effective_stress_report(m::BP8Model) =
    (; σ̄_lowest=m.cache.σ̄_lowest, floor_hits=m.cache.floor_hits,
       floor=m.par.σ̄_min, bound=m.cache.floor_hits > 0)

"""
    process_zone(par, σ̄) = μ·D_RS/(b·σ̄)

The rate-and-state process (nucleation) zone `L_b`. Resolving it is the
binding requirement on `Δz` for this class of problem — the usual rule of
thumb is several cells per `L_b`. Note `L_b` *grows* as fluid injection
reduces `σ̄`, so the tightest constraint is at the initial `σ̄`.
"""
process_zone(par::BP8Params, σ̄=par.σ0) = par.μ * par.D_RS / (par.b * σ̄)

"""
    resolution_report(m) -> NamedTuple

How well `Δz` resolves the two length scales that matter: the rate-and-state
process zone `L_b` and the Gaussian source width `L_gauss`. Rate-and-state
slip rate depends exponentially on `σ̄` (through `V ~ exp(τ/(aσ̄))`), so an
under-resolved pressure field turns into an order-of-magnitude error in peak
slip rate, not a proportional one. Treat `cells_per_Lb < 3` as "the run shows
the right physics but the numbers are not converged".
"""
function resolution_report(m::BP8Model)
    par = m.par
    Δz = m.grid_info.Δz
    Lb0 = process_zone(par, par.σ0)
    σ̄_lo = isfinite(m.cache.σ̄_lowest) ? max(m.cache.σ̄_lowest, par.σ̄_min) : par.σ0
    return (; Δz, L_b=Lb0, cells_per_Lb=Lb0 / Δz,
            L_b_at_lowest_σ̄=process_zone(par, σ̄_lo),
            cells_per_L_gauss=par.L_gauss / Δz,
            converged=Lb0 / Δz >= 3)
end

"""
    initial_state(m) -> u

BP8 eq. 26-29: zero slip, zero pressure change, and the state variable that
makes the initial slip rate exactly `V_init` under the initial shear traction
`τ_init`. Note the strength has to balance `τ_init - η‖V‖`, not `τ_init` — the
radiation-damping term is only ~5e-6 Pa here, but including it makes the
initial condition exactly self-consistent.
"""
function initial_state(m::BP8Model)
    p = m.par
    u = zeros(state_length(m))
    Vmag = norm(SVector(p.V_init, p.V_zero))
    θ0 = initial_state_from_strength(Vmag, p.τ_init - damping(p) * Vmag, p.σ0,
                                     friction_params(p))
    u[2m.nf+1:3m.nf] .= log(θ0)
    return u
end

"""
    evaluate!(m, u, t) -> cache

Fills the model cache with the derived fields at state `u`: the elastic
traction change, the slip velocity from the force balance, and the total shear
stress. Used by both the right-hand side and the output writers, so they
cannot drift apart.
"""
function evaluate!(m::BP8Model, u, t)
    p = m.par
    nf = m.nf
    c = m.cache
    fp = friction_params(p)
    η = damping(p)

    slip = @view u[1:2nf]
    ϕ = @view u[2nf+1:3nf]
    pres = @view u[3nf+1:4nf]

    mul!(c.Δτ, m.K, slip)

    @inbounds for i in 1:nf
        if !m.active[i]
            c.V2[i] = 0.0
            c.V3[i] = 0.0
            c.Vmag[i] = 0.0
            c.τ2[i] = m.τ0[1] + c.Δτ[i]
            c.τ3[i] = m.τ0[2] + c.Δτ[nf+i]
            continue
        end
        σ̄_raw = p.σ0 - pres[i]
        σ̄_raw < c.σ̄_lowest && (c.σ̄_lowest = σ̄_raw)
        σ̄_raw < p.σ̄_min && (c.floor_hits += 1)
        σ̄ = max(σ̄_raw, p.σ̄_min)
        θ = exp(ϕ[i])
        Δτv = SVector(c.Δτ[i], c.Δτ[nf+i])
        V = solve_slip_velocity(m.τ0, Δτv, θ, σ̄, η, fp; V0=c.Vprev[i])
        c.V2[i] = V[1]
        c.V3[i] = V[2]
        c.Vmag[i] = norm(V)
        c.Vprev[i] = max(c.Vmag[i], 1e-30)
        # eq. 8: the shear stress actually acting on the fault.
        c.τ2[i] = m.τ0[1] + c.Δτ[i] - η * V[1]
        c.τ3[i] = m.τ0[2] + c.Δτ[nf+i] - η * V[2]
    end
    return c
end

"""
    rhs!(du, u, m, t)

The coupled right-hand side (eq. 5, 11, 17/19-23).
"""
function rhs!(du, u, m::BP8Model, t)
    p = m.par
    nf = m.nf
    c = evaluate!(m, u, t)

    @inbounds for i in 1:nf
        du[i] = c.V2[i]
        du[nf+i] = c.V3[i]
        # aging law in ϕ = ln θ:  dϕ/dt = e^{-ϕ} - V/D_RS
        du[2nf+i] = exp(-u[2nf+i]) - c.Vmag[i] / p.D_RS
    end

    dp = @view du[3nf+1:4nf]
    pres = @view u[3nf+1:4nf]
    mul!(dp, m.Ap, pres)

    if m.injection === :gaussian
        q = injection_rate(t; q0=q0_per_thickness(p), t_off=p.t_off)
        @inbounds for i in 1:nf
            dp[i] += q / (p.β * p.φ) * m.source[i]
        end
    else
        # Peaceman well, eq. 22-23.
        p_well = u[end]
        transfer = m.WI * (p_well - pres[m.well_cell])
        dp[m.well_cell] += transfer / m.S_e
        Qinj = t < p.t_off ? p.Q0 : 0.0
        du[end] = (Qinj - transfer) / p.S_well
    end
    return nothing
end

"""
    run_bp8(m; tspan=(0.0, m.par.t_f), alg=Tsit5(), reltol=1e-8,
              saveat=3600.0, verbose=false, kwargs...)

Integrates the coupled system. Absolute tolerances are set per block (slip,
ln θ, pressure) because they live on wildly different scales.
"""
function run_bp8(m::BP8Model; tspan=(0.0, m.par.t_f), alg=Tsit5(), reltol=1e-8,
                 saveat=3600.0, verbose=false, kwargs...)
    u0 = initial_state(m)
    nf = m.nf
    abstol = similar(u0)
    abstol[1:2nf] .= 1e-14        # slip, m
    abstol[2nf+1:3nf] .= 1e-10    # ln θ
    abstol[3nf+1:4nf] .= 1e-3     # pressure, Pa
    m.injection === :peaceman && (abstol[end] = 1e-3)

    prob = ODEProblem(rhs!, u0, tspan, m)
    t0 = time()
    sol = solve(prob, alg; reltol, abstol, saveat, save_everystep=true,
                tstops=[m.par.t_off], kwargs...)
    verbose && @info "integration finished" seconds = round(time() - t0, digits=1) steps = length(sol.t) retcode = sol.retcode
    return sol
end

# ==============================================================================
# Analytic pore-pressure solutions (eq. 21 and 25), valid for t ≪ l_f²/(4α) ≈
# 220 h — i.e. before diffusion reaches the no-flux edges of Ω_f. §6 asks
# explicitly that the Peaceman well be checked against eq. 25.
# ==============================================================================

"""
    expint_e1(x)

The exponential integral `E₁(x) = ∫_x^∞ e^{-s}/s ds` for `x > 0`: series for
`x < 1`, Lentz continued fraction otherwise. Implemented here rather than
depending on `SpecialFunctions` for this one function.
"""
function expint_e1(x::Real)
    x > 0 || throw(DomainError(x, "E₁ is defined here for x > 0"))
    if x < 1
        # E₁(x) = -γ - ln x + Σ (-1)^{k+1} xᵏ/(k·k!)
        s = -0.5772156649015329 - log(x)
        term = 1.0
        for k in 1:60
            term *= -x / k
            s -= term / k
            abs(term / k) < 1e-18 * abs(s) && break
        end
        return s
    end
    # Modified Lentz continued fraction, E₁(x) = e^{-x}/(x+1-1/(x+3-4/(x+5-…)))
    tiny = 1e-300
    b = x + 1.0
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in 1:300
        a = -i * i
        b += 2.0
        d = 1 / (a * d + b)
        c = b + a / c
        del = c * d
        h *= del
        abs(del - 1) < 1e-16 && break
    end
    return h * exp(-x)
end

"""
    analytic_pressure_gaussian(r, t, p::BP8Params)

BP8 eq. 21: pore pressure from the Gaussian source at radius `r` and time `t`
(with the closed form at `r = 0`), for `0 ≤ t < t_off`.
"""
function analytic_pressure_gaussian(r, t, p::BP8Params)
    t <= 0 && return 0.0
    q0 = q0_per_thickness(p)
    amp = q0 / (4π * p.α * p.β * p.φ)
    L2 = 2p.L_gauss^2
    return r ≈ 0 ? amp * log((L2 + 4p.α * t) / L2) :
           amp * (expint_e1(r^2 / (L2 + 4p.α * t)) - expint_e1(r^2 / L2))
end

"""
    analytic_pressure_point(r, t, p::BP8Params)

BP8 eq. 25: pore pressure from a point source, the reference the Peaceman well
model is meant to reproduce away from the well.
"""
function analytic_pressure_point(r, t, p::BP8Params)
    (t <= 0 || r <= 0) && return 0.0
    q0 = q0_per_thickness(p)
    return q0 / (4π * p.α * p.β * p.φ) * expint_e1(r^2 / (4p.α * t))
end

# ==============================================================================
# Benchmark output (§4).
# ==============================================================================

"""
    station_locations()

The nine on-fault observation points of §4.1, as
`(filename, x2, x3)` with `strk` = `x2` and `dp` = `x3`.
"""
function station_locations()
    stations = Tuple{String,Float64,Float64}[]
    fmt(v) = @sprintf("%s%03d", v < 0 ? "-" : "+", abs(round(Int, v)))
    for (x2, x3) in ((0.0, 0.0), (-200.0, 0.0), (0.0, 200.0), (200.0, 0.0), (0.0, -200.0),
                     (-200.0, -200.0), (-200.0, 200.0), (200.0, -200.0), (200.0, 200.0))
        push!(stations, ("fltst_strk$(fmt(x2))dp$(fmt(x3))", x2, x3))
    end
    return stations
end

nearest_node(axis, x) = argmin(abs.(axis .- x))

problem_name(m::BP8Model) = m.injection === :gaussian ? "BP8-QD-GS" : "BP8-QD-PW"

domain_line(m::BP8Model) =
    "# elastic_domain=|x1| <= $(m.grid_info.L_normal) m, |x2|,|x3| <= " *
    "$(m.grid_info.L_fault) m, SBP order $(m.grid_info.order), " *
    "$(m.grid_info.elastic_dofs) elastic DOF"

"Header for the §4.1/§4.2 time-series files, following that section's example."
function ts_header(m::BP8Model; modeler, location, extra=String[])
    return vcat([
            "# This is the file header:",
            "# problem=SEAS Benchmark $(problem_name(m))",
            "# code=EarthquakeDiffinitive",
            "# version=0.1.0",
            "# modeler=$modeler",
            "# date=$(replace(string(today()), "-" => "/"))",
            "# element_size=$(m.grid_info.Δz) m",
            "# location=$location",
        ], extra, [domain_line(m)])
end

const TS_FIELDS = ["t", "slip_2", "slip_3", "slip_rate_2", "slip_rate_3",
                   "shear_stress_2", "shear_stress_3", "pore_pressure",
                   "darcy_vel_2", "darcy_vel_3", "state"]

const TS_COLUMNS = ["Time (s)", "Slip_2 (m)", "Slip_3 (m)",
                    "Slip_rate_2 (log10 m/s)", "Slip_rate_3 (log10 m/s)",
                    "Shear_stress_2 (MPa)", "Shear_stress_3 (MPa)",
                    "Pore_pressure (MPa)", "Darcy_velocity_2 (m/s)",
                    "Darcy_velocity_3 (m/s)", "State (log10 s)"]

safelog10(x) = log10(max(abs(x), 1e-300))

"""
    write_outputs(m, sol, dir; modeler="", profile_dt=3600.0)

Writes the three families of §4 output files into `dir`: the nine station time
series, `global.dat`, and the ten slip/stress/pressure profiles.
"""
function write_outputs(m::BP8Model, sol, dir; modeler="", profile_dt=3600.0)
    mkpath(dir)
    times = sol.t
    nf = m.nf

    # Derived fields at every saved time, computed through the same
    # `evaluate!` the integrator used.
    ns = length(times)
    V2 = Matrix{Float64}(undef, nf, ns)
    V3 = similar(V2); τ2 = similar(V2); τ3 = similar(V2)
    q2 = similar(V2); q3 = similar(V2)
    Vmax = Vector{Float64}(undef, ns)
    moment_rate = Vector{Float64}(undef, ns)

    for (j, t) in enumerate(times)
        u = sol.u[j]
        c = evaluate!(m, u, t)
        V2[:, j] .= c.V2
        V3[:, j] .= c.V3
        τ2[:, j] .= c.τ2
        τ3[:, j] .= c.τ3
        pres = @view u[3nf+1:4nf]
        mul!(view(q2, :, j), m.Q2, pres)
        mul!(view(q3, :, j), m.Q3, pres)
        Vmax[j] = maximum(c.Vmag)
        moment_rate[j] = m.par.μ * sum(m.weights .* c.Vmag)
    end

    dts = diff(times)
    step_lines = isempty(dts) ? String[] : [
        @sprintf("# minimum_time_step=%.3E", minimum(dts)),
        @sprintf("# maximum_time_step=%.3E", maximum(dts)),
        "# num_time_steps=$(ns)"]

    write_time_series(m, sol, dir, times, V2, V3, τ2, τ3, q2, q3, step_lines, modeler)
    write_global(m, dir, times, Vmax, moment_rate, step_lines, modeler)
    write_profiles(m, sol, dir, profile_dt, modeler)
    return dir
end

function write_time_series(m, sol, dir, times, V2, V3, τ2, τ3, q2, q3, step_lines, modeler)
    nf = m.nf
    lin = LinearIndices((length(m.x2), length(m.x3)))
    for (name, sx2, sx3) in station_locations()
        i = nearest_node(m.x2, sx2)
        j = nearest_node(m.x3, sx3)
        idx = lin[i, j]
        loc = @sprintf("on fault, strike = %.4g m, depth = %.4g m", m.x2[i], m.x3[j])
        extra = String[]
        (isapprox(m.x2[i], sx2; atol=1e-6) && isapprox(m.x3[j], sx3; atol=1e-6)) ||
            push!(extra, "# note=requested station ($sx2, $sx3) m is not a grid node; nearest node used")
        open(joinpath(dir, name * ".dat"), "w") do io
            for l in ts_header(m; modeler, location=loc, extra=vcat(extra, step_lines))
                println(io, l)
            end
            for (c, d) in enumerate(TS_COLUMNS)
                println(io, "# Column #$c = $d")
            end
            println(io, "# The line below lists the names of the data fields")
            println(io, join(TS_FIELDS, " "))
            println(io, "# Here is the time-series data.")
            for (n, t) in enumerate(times)
                u = sol.u[n]
                @printf(io, "%21.13E %14.6E %14.6E %14.6E %14.6E %14.6E %14.6E %14.6E %14.6E %14.6E %14.6E\n",
                        t, u[idx], u[nf+idx],
                        safelog10(V2[idx, n]), safelog10(V3[idx, n]),
                        τ2[idx, n] / 1e6, τ3[idx, n] / 1e6,
                        u[3nf+idx] / 1e6, q2[idx, n], q3[idx, n],
                        u[2nf+idx] / log(10))
            end
        end
    end
end

function write_global(m, dir, times, Vmax, moment_rate, step_lines, modeler)
    open(joinpath(dir, "global.dat"), "w") do io
        for l in ts_header(m; modeler, location="frictional domain", extra=step_lines)
            println(io, l)
        end
        println(io, "# Column #1 = Time (s)")
        println(io, "# Column #2 = Max_slip_rate (log10 m/s)")
        println(io, "# Column #3 = Moment_rate (N.m/s)")
        println(io, "# The line below lists the names of the data fields")
        println(io, "t max_slip_rate moment_rate")
        println(io, "# Here is the time-series data.")
        for (n, t) in enumerate(times)
            @printf(io, "%21.13E %14.6E %14.6E\n", t, safelog10(Vmax[n]), moment_rate[n])
        end
    end
end

"""
Profile files (§4.3), in the exact layout of that section's worked example:
an `(N_t+1) × (N_coord+2)` matrix whose first row is `0 0 <coordinates>` and
whose remaining rows are `t  max_slip_rate  <quantity at each coordinate>`.
The two leading zeros on the coordinate row keep every row the same width.
The field list is four separate lines, and the header keys (`author`,
`code_version`) differ from the time-series files' (`modeler`, `version`) —
both follow their own section's example.
"""
function write_profiles(m, sol, dir, profile_dt, modeler)
    nf = m.nf
    n2f, n3f = length(m.x2), length(m.x3)
    lin = LinearIndices((n2f, n3f))
    ts = collect(0.0:profile_dt:sol.t[end])
    isempty(ts) && (ts = [sol.t[end]])
    last(ts) < sol.t[end] && push!(ts, sol.t[end])

    # Along strike: x3 = 0, varying x2.  Along depth: x2 = 0, varying x3.
    j0 = nearest_node(m.x3, 0.0)
    i0 = nearest_node(m.x2, 0.0)
    lines = ["strike" => (m.x2, [lin[i, j0] for i in 1:n2f]),
             "depth" => (m.x3, [lin[i0, j] for j in 1:n3f])]

    states = [sol(t) for t in ts]
    caches = [begin
                  c = evaluate!(m, u, t)
                  (; τ2=copy(c.τ2), τ3=copy(c.τ3), Vmax=maximum(c.Vmag))
              end for (u, t) in zip(states, ts)]

    quantities = ["slip_2" => ((u, c, k) -> u[k], "Horizontal slip (Slip_2) (m)"),
                  "slip_3" => ((u, c, k) -> u[nf+k], "Vertical slip (Slip_3) (m)"),
                  "shear_stress_2" => ((u, c, k) -> c.τ2[k] / 1e6,
                                       "Horizontal shear stress (Shear_stress_2) (MPa)"),
                  "shear_stress_3" => ((u, c, k) -> c.τ3[k] / 1e6,
                                       "Vertical shear stress (Shear_stress_3) (MPa)"),
                  "pore_pressure" => ((u, c, k) -> u[3nf+k] / 1e6,
                                      "Pore pressure (Pore_pressure) (MPa)")]

    lf = m.par.l_f
    for (qname, (getter, qdesc)) in quantities, (lname, (axis, idxs)) in lines
        coord = lname == "strike" ? "x2" : "x3"
        along = lname == "strike" ? "along strike" : "along depth"
        ncol = length(axis) + 2
        open(joinpath(dir, "$(qname)_$(lname).dat"), "w") do io
            println(io, "# This is the file header:")
            println(io, "# problem=SEAS Benchmark $(problem_name(m))")
            println(io, "# author=$modeler")
            println(io, "# date=$(replace(string(today()), "-" => "/"))")
            println(io, "# code=EarthquakeDiffinitive")
            println(io, "# code_version=0.1.0")
            println(io, "# element_size=$(m.grid_info.Δz) m")
            println(io, "# Row #1 = $(lname == "strike" ? "Strike" : "Depth") (m) with two zeros first")
            println(io, "# Column #1 = Time (s)")
            println(io, "# Column #2 = Max slip rate (log10 m/s)")
            println(io, "# Columns #3-$ncol = $qdesc $along")
            println(io, "# Computational domain size:  $(-lf)m < x2 < $(lf)m, $(-lf)m < x3 < $(lf)m")
            println(io, domain_line(m))
            println(io, "# The line below lists the names of the data fields")
            println(io, coord)
            println(io, "t")
            println(io, "max_slip_rate")
            println(io, qname)
            println(io, "# Here are the data")
            @printf(io, "%21.13E %14.6E", 0.0, 0.0)
            for x in axis
                @printf(io, " %14.6E", x)
            end
            println(io)
            for (n, t) in enumerate(ts)
                u = states[n]
                c = caches[n]
                @printf(io, "%21.13E %14.6E", t, safelog10(c.Vmax))
                for k in idxs
                    @printf(io, " %14.6E", getter(u, c, k))
                end
                println(io)
            end
        end
    end
end

end # module BP8

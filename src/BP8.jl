module BP8

using Diffinitive
using Diffinitive.Grids
using Diffinitive.SbpOperators
using SparseArrays
using StaticArrays
using LinearAlgebra: mul!, norm, diag
using OrdinaryDiffEq
using OrdinaryDiffEqBDF: QNDF
using SciMLBase: u_modified!
using ProgressMeter: Progress, update!, finish!
using Printf
using Dates: today
using Tokens

using ..RateStateFriction
using ..PorePressure
using ..FaultResponse
using ..Elasticity
using ..StiffnessCache

export BP8Params, benchmark_parameters, BP8Model, build_model, initial_state,
       run_bp8, default_integrator, evaluate!, state_jacobian!, state_jacobian_prototype,
       write_outputs, station_locations,
       solve_pressure_history, set_pressure_history!, pressure_at!,
       pressure_operator, pressure_length, well_pressure,
       effective_stress_report, analytic_pressure_gaussian, analytic_pressure_point,
       resolution_report, process_zone, fault_grid_sizes, build_fault_elasticity

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
# The state variable is integrated as ϕ = ln θ. θ spans ~1e8-1e12 s while slip
# is ~1e-6 m, so one scalar tolerance cannot serve both; ln θ is well scaled and
# turns the aging law into dϕ/dt = e^{-ϕ} - V/D_RS.

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
    # `σ̄ = σ - p` must stay positive for the friction law to make sense. Peak
    # pressure is ~13 MPa against σ = 25 MPa, so if this binds the run is
    # outside the model's validity rather than merely under-resolved.
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
    # Which nodes were ever unclamped. `floor_hits` counts RHS evaluations, so
    # it scales with the step count and says nothing about the affected area.
    floor_nodes::BitVector
    # Pore pressure at the current `t`, interpolated from the separately
    # integrated history. `pbuf` is the raw subsystem state (`nf`, or `nf+1` for
    # Peaceman, whose last entry is the well-bore pressure); `pres` views its
    # first `nf` entries. In-place keeps the RHS allocation-free.
    pbuf::Vector{Float64}
    pres::SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int}},true}
end
function Cache(nf, plen)
    pbuf = zeros(plen)
    return Cache(zeros(2nf), zeros(nf), zeros(nf), zeros(nf), zeros(nf), zeros(nf),
                 fill(1e-12, nf), Inf, 0, falses(nf), pbuf, view(pbuf, 1:nf))
end

"""
    PressureHistory

The separately integrated pore-pressure solution, or `nothing`. Mutable so
`run_bp8` can attach a history to an otherwise immutable `BP8Model`. `sol` is
untyped on purpose; reads go through `pressure_at!`'s function barrier.
"""
mutable struct PressureHistory
    sol::Any
    tspan::Tuple{Float64,Float64}
end
PressureHistory() = PressureHistory(nothing, (0.0, 0.0))

"""
    BP8Model

Everything needed to evaluate the coupled right-hand side: the precomputed
fault stiffness, the pore-pressure operator and source, the Darcy operators
for output, the `Ω_f` quadrature weights, and the locked-edge mask.

Pore pressure is **not** part of the integrated state — it is solved on its own
beforehand and stored in `pressure`. See [`solve_pressure_history`](@ref).
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
    pressure::PressureHistory
    grid_info::NamedTuple
end

"""
    fault_grid_sizes(par, Δz, L_fault, L_normal, order) -> (; n1, n23)

Validates a configuration and computes the elastic grid point counts
`build_model` uses. Factored out so `build_stiffness_cache.jl`'s sharding path
can reproduce exactly the `n1`, `n23` `build_model` would use — which is what
makes a sharded build match the single-process one.
"""
function fault_grid_sizes(par::BP8Params, Δz, L_fault, L_normal, order)
    isapprox(par.l_f / Δz, round(par.l_f / Δz); atol=1e-9) ||
        error("Δz=$Δz must divide l_f=$(par.l_f) so the grid has nodes on ±l_f")
    L_fault >= par.l_f || error("L_fault=$L_fault must be at least l_f=$(par.l_f)")

    n1 = round(Int, L_normal / Δz) + 1
    n23 = round(Int, 2L_fault / Δz) + 1
    # SBP closures need more than two closure widths of points per dimension.
    n_min = 2order + 1
    n1 >= n_min || error("L_normal/Δz gives only $n1 points across the fault-normal " *
                         "direction; SBP order $order needs at least $n_min. " *
                         "Increase L_normal or decrease Δz.")
    n23 >= n_min || error("2*L_fault/Δz gives only $n23 points along the fault; " *
                          "SBP order $order needs at least $n_min.")
    return (; n1, n23)
end

"""
    build_fault_elasticity(; par, Δz, L_fault, L_normal, n1, n23, set, verbose=false,
                           solver_kwargs...) -> FaultElasticity

The split-node elastic system `fault_stiffness` solves against, for grids
[`fault_grid_sizes`](@ref) has already sized and validated.

Factored out of the cache-miss path so a `K` build can be **sharded across
processes**: each shard calls this (minutes, not the bottleneck) and then
`fault_stiffness(fe; cols=...)` for its own columns. `build_model` uses the
same function on every miss, so there is one definition of "the elastic system
for this configuration" — which is what makes shards from separate processes
safe to merge (`merge_stiffness_cache.jl`).
"""
function build_fault_elasticity(; par::BP8Params, Δz, L_fault, L_normal, n1, n23, set,
                                verbose=false, representation=:kronecker, solver_kwargs...)
    g_minus = equidistant_grid((-L_normal, -L_fault, -L_fault), (0.0, L_fault, L_fault), n1, n23, n23)
    g_plus = equidistant_grid((0.0, -L_fault, -L_fault), (L_normal, L_fault, L_fault), n1, n23, n23)

    t0 = time()
    fe = FaultElasticity(g_minus, g_plus, lame_lambda(par), par.μ, set;
                         l_f=par.l_f, representation, solver_kwargs...)
    verbose && @info "split-node system ready" seconds = round(time() - t0, digits=1)
    return fe
end

# `K` for one configuration, from the cache if it is there.
#
# Split out of `build_model` so the load and build-then-save paths sit together,
# and so `FaultElasticity` is constructed in the miss branch only — a hit has no
# use for it.
function stiffness_matrix(; par, Δz, L_fault, L_normal, n1, n23, order, set,
                          stiffness, cache, cache_dir, verbose, solver_kwargs,
                          representation=:kronecker)
    cache ∈ (:auto, :read, :refresh, :off) ||
        error("cache must be :auto, :read, :refresh or :off, got $cache")

    key = stiffness_cache_key(; λ=lame_lambda(par), μ=par.μ, l_f=par.l_f,
                              Δz, L_fault, L_normal, order, stencil=set, stiffness,
                              solver_kwargs...)
    path = (cache === :off || cache_dir === nothing) ? nothing :
           stiffness_cache_path(cache_dir, key)

    if path !== nothing && cache !== :refresh
        hit = load_stiffness(path, key)
        if hit !== nothing
            K, x2, x3 = hit
            verbose && @info "fault stiffness: cache hit" path stiffness size = size(K)
            return K, x2, x3
        end
    end

    fe = build_fault_elasticity(; par, Δz, L_fault, L_normal, n1, n23, set,
                                verbose, representation, solver_kwargs...)

    t0 = time()
    # `symmetry=true` is unconditionally safe here: `build_fault_elasticity`
    # always gives both fault-parallel directions the same `L_fault`/`n23` about
    # 0, which is `fault_stiffness`'s D4 precondition. Same K, 6.5-7.8× fewer
    # solves.
    K = stiffness === :toeplitz ? fault_stiffness_toeplitz(fe; verbose) :
                                  fault_stiffness(fe; verbose, symmetry=true)
    verbose && @info "fault stiffness built" seconds = round(time() - t0, digits=1) stiffness size = size(K) elastic_solver_report(fe)...

    x2, x3 = collect.(fault_grid_axes(fe))
    if path !== nothing && cache !== :read
        # A failed save must not lose a build that just cost hours: warn and
        # hand back the K we have.
        try
            save_stiffness(path, key, K, x2, x3)
            verbose && @info "fault stiffness: cached" path bytes = filesize(path)
        catch err
            @warn "could not write the stiffness cache; continuing with the K in memory" path err
        end
    end
    return K, x2, x3
end

"""
    build_model(; par=benchmark_parameters(), Δz, L_fault, L_normal,
                  injection=:gaussian, order=4, verbose=false)

Assembles the whole coupled model. `Δz` is the node spacing (must divide
`par.l_f`), `L_fault` the half-width of the elastic grids in the two
fault-parallel directions, `L_normal` their extent in the fault-normal
direction. The far field is truncated with `u=0` at those boundaries, so both
have to be comfortably larger than `l_f`; see the §6 domain-size study.

The expensive part is `fault_stiffness`, which does `2·N_Ωf` CG solves of the
assembled 3D elastic system.

`stiffness` selects how `K` is built. `:exact` does all `2·N_Ωf` solves, cut
6.5-7.8× by D4 symmetry, and is **what every submission run uses** — the submit
scripts and `run_bp8.jl` all pass it, and `fault_stiffness_gpu` builds it in
48 h on one L40S at the Δz = 10 m production point. `:toeplitz` is the current
default here: 10 solves expanding 5 sources by the whole-space kernel's
translation invariance, an approximation worth 0.41% in `V_max` at the converged
domain (`PERFORMANCE.md` §4b). It dates from when `:exact` meant ~17 days on one
CPU node; D4 symmetry and the GPU path have since removed that gap, so it is now
a cheap preview rather than the production route.

`precond` goes to [`CGSolver`](@ref) and is an independent axis: `stiffness`
sets how *many* solves happen, `precond` how each converges. `:none` (default)
or `:jacobi`, the latter measuring 0.92× and kept only to record that.

## Reusing `K` from disk

`K` depends only on the elastic constants, the geometry, the grid, the SBP
operators, `stiffness` and the CG settings — nothing that varies between runs of
the same configuration — so it is cached. `cache_dir` defaults to
[`stiffness_cache_dir`](@ref) (the `EQD_STIFFNESS_CACHE` environment variable);
caching is off when that is unset. `cache` selects the mode:

| `cache` | on a hit | on a miss |
|---|---|---|
| `:auto` (default) | load | build, then save |
| `:read` | load | build, save nothing |
| `:refresh` | ignored | build, then overwrite |
| `:off` | — | build |

**A hit skips `FaultElasticity` as well as the solves** — the file carries the
`Ω_f` axes, and nothing else in `BP8Model` needs the elastic system once `K`
exists. So a cached `:exact` model at a production configuration builds in
seconds rather than days, which is what makes `:exact` usable for sweeps over
the cheap axes (injection, friction, `t_f`, tolerances).

The key spells out every input reaching `K` and is verified on load, so a
changed configuration misses rather than returning the wrong `K`. See
`StiffnessCache`.
"""
function build_model(; par::BP8Params=benchmark_parameters(),
                     Δz=par.Δz, L_fault=3par.l_f, L_normal=2par.l_f,
                     injection=:gaussian, order=4, verbose=false,
                     stiffness=:toeplitz, representation=:kronecker,
                     cache=:auto, cache_dir=stiffness_cache_dir(),
                     pressure=true, pressure_kwargs=(;),
                     solver_kwargs...)
    injection ∈ (:gaussian, :peaceman) ||
        error("injection must be :gaussian or :peaceman, got $injection")

    set = read_stencil_set(SbpOperators.sbp_operators_path() * "standard_diagonal.toml"; order)
    n1, n23 = fault_grid_sizes(par, Δz, L_fault, L_normal, order)
    # Also checked below for the pore-pressure grid, which `fault_grid_sizes`
    # does not know about.
    n_min = 2order + 1
    verbose && @info "elastic grids" points_per_side = n1 * n23^2 dofs = 6 * n1 * n23^2

    stiffness ∈ (:exact, :toeplitz) ||
        error("stiffness must be :exact or :toeplitz, got $stiffness")
    K, x2, x3 = stiffness_matrix(; par, Δz, L_fault, L_normal, n1, n23, order, set,
                                 stiffness, cache, cache_dir, verbose, solver_kwargs,
                                 representation)
    nf = length(x2) * length(x3)
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

    # eq. 13: locked outside Ω_f. The outer ring is held at V=0 so slip is
    # continuous into the locked region; a jump there would be a stress
    # singularity the elastic solve cannot represent.
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

    plen = nf + (injection === :peaceman ? 1 : 0)
    m = BP8Model(par, K, Ap, source, Q2, Q3, weights, active, collect(x2), collect(x3),
                 nf, τ0, injection, well_cell_index(g_p), WI, S_e, Cache(nf, plen),
                 PressureHistory(),
                 (; Δz, L_fault, L_normal, n1, n23, order,
                    elastic_dofs=6 * n1 * n23^2, n2f, n3f))

    # Up front over the full benchmark duration, so `evaluate!` works
    # immediately and any `run_bp8` sub-interval reuses it. Seconds, against
    # hours for `K`. `pressure=false` is for callers that only want `K`.
    if pressure
        tspan = (0.0, par.t_f)
        set_pressure_history!(m, solve_pressure_history(m; tspan, verbose, pressure_kwargs...),
                              tspan)
    end
    return m
end

# ------------------------------------------------------------------------------

# Slip (2·nf), ln θ (nf). Pore pressure is integrated separately and is
# deliberately absent — see `solve_pressure_history`.
state_length(m::BP8Model) = 3m.nf

"Length of the *pressure* subsystem's own state: `nf`, plus `p_well` for BP8-PW."
pressure_length(m::BP8Model) = m.nf + (m.injection === :peaceman ? 1 : 0)

"""
    effective_stress_report(m) -> NamedTuple

Whether the floor `par.σ̄_min` ever bound, how low `σ̄ = σ - p` went, and how
much of the fault was affected. `σ̄ ≤ 0` means fluid pressure has fully unclamped
the fault and BP8's no-opening condition (eq. 3) no longer holds. BP8-PW reaches
this at Table 1's parameters; BP8-GS does not.

`nodes` and `radius` bound the damage. The unclamped region is a disc whose
physical radius is set by where eq. 25's pressure crosses `σ` — about **15 m**
at Table 1's parameters — so it is independent of `Δz`: refining resolves it
rather than enlarging it (~0.3 cells across at Δz = 50 m, ~1.5 at 10 m).
Against `l_f` = 400 m that is a localized defect in BP8-PW's own point-source
specification, not a discretization problem. `PROGRESS.md` limitation 3.

`floor_hits` counts RHS evaluations, so it tracks step count, not extent — use
`nodes`.
"""
function effective_stress_report(m::BP8Model)
    idx = findall(m.cache.floor_nodes)
    n2 = length(m.x2)
    radius = isempty(idx) ? 0.0 :
        maximum(hypot(m.x2[mod1(i, n2)], m.x3[(i - 1) ÷ n2 + 1]) for i in idx)
    return (; σ̄_lowest=m.cache.σ̄_lowest, floor_hits=m.cache.floor_hits,
              floor=m.par.σ̄_min, bound=m.cache.floor_hits > 0,
              nodes=length(idx), fraction=length(idx) / m.nf, radius)
end

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

How well `Δz` resolves the process zone `L_b` and the source width
`L_gauss`. `V ~ exp(τ/(aσ̄))`, so an under-resolved pressure field gives an
order-of-magnitude error in peak slip rate, not a proportional one. Treat
`cells_per_Lb < 3` as "right physics, unconverged numbers".
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

# Pore pressure: integrated separately, implicitly, once.
#
# The pressure subsystem is autonomous — its RHS reads only `p`, `p_well` and
# `t`, never slip or state (eq. 17-23; the coupling is one-way through
# `σ̄ = σ - p`). So it does not belong in the coupled state at all: integrate it
# on its own with a method suited to it, then interpolate.
#
# `Rodas5P` with the analytic Jacobian: 180 steps (GS) / 283 (PW) over 30 days,
# ~6 s, and resolution-independent (94/88/95 at Δz = 50/25/10 m).

"""
    pressure_operator(m) -> J

The constant sparse operator of the pressure subsystem: `Ap` alone for the
Gaussian source, and [`well_coupled_operator`](@ref)'s `[p; p_well]` system for
the Peaceman well — the benchmark's **option 2**, one additional unknown in the
same linear system.
"""
pressure_operator(m::BP8Model) =
    m.injection === :peaceman ?
    well_coupled_operator(m.Ap, m.well_cell, m.WI, m.S_e, m.par.S_well) : m.Ap

"""
    solve_pressure_history(m; tspan, alg=Rodas5P(), reltol=1e-8, abstol=1e-3, verbose=false)

Integrates the autonomous pore-pressure subsystem over `tspan` and returns the
`ODESolution`, whose dense output is what `evaluate!` later interpolates.

The subsystem is linear, so its Jacobian is exactly the constant `J` from
[`pressure_operator`](@ref). Supplying it is what makes the implicit solve
cheap and avoids a finite-difference dense Jacobian (400 s and 1 GB at
Δz = 10 m).

**`save_everystep=true` is deliberate.** The solve takes only ~180-280 steps,
so the dense output is 57-79 MB at Δz = 10 m and the solver's own interpolant
is far better than linear interpolation on a fixed grid: 4.4 Pa max error
(0.003% in `V`) against 10.8 kPa (7.1%) for hourly levels. `V ~ exp(τ/(aσ̄))`
amplifies pressure error exponentially, and the error concentrates at the two
kinks in the forcing (`t = 0`, `t_off`) — where an adaptive solver puts steps
and a fixed grid does not.
"""
function solve_pressure_history(m::BP8Model; tspan=(0.0, m.par.t_f), alg=Rodas5P(),
                                reltol=1e-8, abstol=1e-3, verbose=false)
    par = m.par
    J = pressure_operator(m)
    n = size(J, 1)

    # Both variants force through the same eq. 20 on/off switch, so the time
    # dependence is one scalar times a fixed vector: the Gaussian source over
    # the fault (eq. 19), or the rate into the well bore's storage (eq. 23).
    b = zeros(n)
    if m.injection === :gaussian
        b .= (q0_per_thickness(par) / (par.β * par.φ)) .* m.source
    else
        b[end] = par.Q0 / par.S_well
    end

    function f!(du, u, _, t)
        mul!(du, J, u)
        du .+= injection_rate(t; q0=1.0, t_off=par.t_off) .* b
        return nothing
    end
    jac!(Jout, u, _, t) = copyto!(Jout, J)

    F = ODEFunction(f!; jac=jac!, jac_prototype=J)
    prob = ODEProblem(F, zeros(n), tspan)   # eq. 27: zero pressure change at t = 0
    t0 = time()
    sol = solve(prob, alg; reltol, abstol, tstops=[par.t_off], save_everystep=true)
    verbose && @info "pressure history integrated" seconds = round(time() - t0, digits=1) steps = length(sol.t) retcode = sol.retcode
    return sol
end

"""
    set_pressure_history!(m, sol, tspan) -> m

Attaches a pressure solution to the model so `evaluate!` can interpolate it.
"""
function set_pressure_history!(m::BP8Model, sol, tspan)
    m.pressure.sol = sol
    m.pressure.tspan = (Float64(tspan[1]), Float64(tspan[2]))
    return m
end

# A function barrier, because `PressureHistory.sol` is untyped: the body
# specializes on the concrete solution type, so the cost is one dynamic dispatch
# rather than a type-unstable inner loop.
_interp_pressure!(dest, sol, t) = (sol(dest, t); dest)

"""
    well_pressure(m, t)

The Peaceman well-bore pressure `p_well` at time `t` (BP8 eq. 23), the extra
unknown [`well_coupled_operator`](@ref) carries. Errors for the Gaussian
variant.

Not a §4 output, but the injected-volume balance and the eq. 25 point-source
check need it. After the initial transient it sits at `p[well_cell] + Q0/WI`,
matching `Q0/WI` to three figures at `t_off`.
"""
function well_pressure(m::BP8Model, t)
    m.injection === :peaceman ||
        error("well_pressure is only defined for the Peaceman variant (injection=:peaceman)")
    pressure_at!(m, t)          # fills the cache buffer, bounds-checks `t`
    return m.cache.pbuf[end]
end

"""
    pressure_at!(m, t) -> p

Pore pressure on the fault at time `t`, interpolated from the stored history
into the model cache without allocating. Errors rather than extrapolating past
the history's window, which would be a quiet source of wrong answers.
"""
function pressure_at!(m::BP8Model, t)
    ph = m.pressure
    ph.sol === nothing &&
        error("this model has no pressure history; call `solve_pressure_history` and " *
              "`set_pressure_history!` first (`run_bp8` does both)")
    t0, t1 = ph.tspan
    tol = 1e-6 * max(one(t1), abs(t1))
    (t0 - tol <= t <= t1 + tol) ||
        error("t=$t is outside the pressure history's tspan $((t0, t1)); " *
              "re-solve the pressure over the interval you mean to integrate")
    _interp_pressure!(m.cache.pbuf, ph.sol, t)
    return m.cache.pres
end

"""
    initial_state(m) -> u

BP8 eq. 26-29: zero slip, zero pressure change, and the state variable making
the initial slip rate exactly `V_init` under `τ_init`. The strength balances
`τ_init - η‖V‖`, not `τ_init`: the damping term is only ~5e-6 Pa, but including
it makes the initial condition exactly self-consistent.
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
    evaluate!(m, u, t; onfail=:error) -> cache

Fills the model cache with the derived fields at state `u`: elastic traction
change, slip velocity from the force balance, total shear stress. Used by both
the right-hand side and the output writers, so they cannot drift apart.

`onfail` goes to [`solve_slip_rate`](@ref). The RHS and Jacobian pass `:nan`,
since an implicit integrator evaluates them at trial states it is about to
discard and a `NaN` is the signal to reject the step. The writers keep
`:error`: they only see accepted states, and a silent `NaN` in a benchmark file
would be worse than an exception.
"""
function evaluate!(m::BP8Model, u, t; onfail::Symbol=:error)
    p = m.par
    nf = m.nf
    c = m.cache
    fp = friction_params(p)
    η = damping(p)

    slip = @view u[1:2nf]
    ϕ = @view u[2nf+1:3nf]
    # A non-finite `t` means the integrator's step has already gone bad;
    # `pressure_at!` would throw, and under `:nan` the contract is to hand the
    # NaN back so the step is rejected.
    if !isfinite(t) && onfail === :nan
        for v in (c.Δτ, c.V2, c.V3, c.Vmag, c.τ2, c.τ3)
            fill!(v, NaN)
        end
        return c
    end
    pres = pressure_at!(m, t)

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
        if σ̄_raw < p.σ̄_min
            c.floor_hits += 1
            c.floor_nodes[i] = true
        end
        σ̄ = max(σ̄_raw, p.σ̄_min)
        θ = exp(ϕ[i])
        Δτv = SVector(c.Δτ[i], c.Δτ[nf+i])
        V = solve_slip_velocity(m.τ0, Δτv, θ, σ̄, η, fp; V0=c.Vprev[i], onfail)
        c.V2[i] = V[1]
        c.V3[i] = V[2]
        c.Vmag[i] = norm(V)
        # The warm start must survive a failed evaluation: `max(NaN, x)` is
        # `NaN`, which would poison every later solve at this node.
        isfinite(c.Vmag[i]) && (c.Vprev[i] = max(c.Vmag[i], 1e-30))
        # eq. 8: the shear stress actually acting on the fault.
        c.τ2[i] = m.τ0[1] + c.Δτ[i] - η * V[1]
        c.τ3[i] = m.τ0[2] + c.Δτ[nf+i] - η * V[2]
    end
    return c
end

"""
    rhs!(du, u, m, t)

Slip and state (eq. 5, 11). Pore pressure is **not** here: it is autonomous, so
it is integrated separately by [`solve_pressure_history`](@ref) and enters only
through `evaluate!`'s `σ̄ = σ - p`.
"""
function rhs!(du, u, m::BP8Model, t)
    p = m.par
    nf = m.nf
    c = evaluate!(m, u, t; onfail=:nan)

    @inbounds for i in 1:nf
        du[i] = c.V2[i]
        du[nf+i] = c.V3[i]
        # aging law in ϕ = ln θ:  dϕ/dt = e^{-ϕ} - V/D_RS
        du[2nf+i] = exp(-u[2nf+i]) - c.Vmag[i] / p.D_RS
    end
    return nothing
end

# The block-diagonal Jacobian that makes an implicit integrator affordable.
#
# BP8-PW is stiff: once pressure has floored σ̄ at the well cell, that node's
# `V ~ exp(τ/(aσ̄))` reacts to its own traction on a time scale `D/K_ww` with
# `D ∝ σ̄`, and an explicit step count grows like `Δz⁻⁴` (weeks-to-months
# extrapolated to Δz = 10 m). Splitting pressure out of the state did not touch
# it — the mode is on the friction side.
#
# The cure is implicit integration of that mode, made cheap by the mode being
# LOCAL: each node's own 3×3 block `(s2, s3, ϕ)` — only `diag(K)`, no elastic
# coupling — reproduces the stiff eigenvalue to a ratio of 1.0000
# (`scripts/extra/bp8_stiffness_spectrum.jl`). So the integrator gets this
# block-diagonal `J` and its Newton iteration solves `nf` independent 3×3
# systems per stage instead of factorising a dense `3nf × 3nf`.
#
# WHY AN INEXACT JACOBIAN IS ENOUGH. The true `J` differs by the off-diagonal
# `(∂V_i/∂T)·K_ij`, large only in the rows of floored nodes. Newton's error
# propagation `(I - γhJ_blk)⁻¹·γh·(J - J_blk)` therefore has O(1) entries in a
# few rows only, and a matrix whose large entries sit in one row has eigenvalues
# equal to that row's diagonal, which is zero. Newton still contracts, just with
# a couple more iterations, and the converged stage solution is unchanged — this
# approximates how the answer is found, not the answer. A full finite-difference
# `J` would be ~10 min *per Jacobian* at Δz = 10 m.
#
# Derivatives are closed-form from the converged force balance
# (`RateStateFriction.slip_rate_derivatives`); `test/bp8_test.jl` checks the
# block against finite differences of `rhs!`.

"""
    state_jacobian_prototype(m) -> SparseMatrixCSC

[`state_jacobian!`](@ref)'s sparsity: `nf` independent 3×3 blocks coupling each
node's `(s2, s3, ϕ)`, which in the stacked `[s2; s3; ϕ]` ordering sit at rows
and columns `(i, nf+i, 2nf+i)`. Passed as `jac_prototype` so the integrator's
linear solver sees a sparse `W` with no fill.
"""
function state_jacobian_prototype(m::BP8Model)
    nf = m.nf
    rows = Vector{Int}(undef, 9nf)
    cols = Vector{Int}(undef, 9nf)
    k = 0
    for i in 1:nf
        idx = (i, nf + i, 2nf + i)
        for col in idx, row in idx
            k += 1
            rows[k] = row
            cols[k] = col
        end
    end
    return sparse(rows, cols, zeros(9nf), 3nf, 3nf)
end

"""
    state_jacobian!(J, u, m, t)

The per-node block-diagonal approximation to `∂rhs!/∂u` described above,
written into the entries of a matrix with [`state_jacobian_prototype`](@ref)'s
pattern. For node `i` with trial stress `T = τ⁰ + Δτ_i`, unit direction `t̂`,
slip-rate magnitude `V` and the derivatives `∂V/∂T`, `∂V/∂ϕ`:

    ∂V⃗/∂s_i = [ (∂V/∂T)·t̂t̂ᵀ + (V/|T|)·(I - t̂t̂ᵀ) ]·K_ii     (2×2)
    ∂V⃗/∂ϕ   = (∂V/∂ϕ)·t̂
    ∂ϕ̇/∂s_i = -(∂V/∂T)·t̂ᵀ·K_ii / D_RS
    ∂ϕ̇/∂ϕ   = -e^{-ϕ} - (∂V/∂ϕ) / D_RS

where `K_ii` is the node's own 2×2 block of `K`. The first line differentiates
`V⃗ = V(|T|)·T/|T|`: the magnitude responds along `t̂`, the direction rotates at
`V/|T|` perpendicular to it. Locked nodes, zero trial stress and failed force
balances get only the `-e^{-ϕ}` diagonal — exact for the first two, and safe for
the third since the step is being rejected anyway.
"""
function state_jacobian!(J, u, m::BP8Model, t)
    p = m.par
    nf = m.nf
    fp = friction_params(p)
    η = damping(p)
    K = m.K
    c = evaluate!(m, u, t; onfail=:nan)
    pres = c.pres

    @inbounds for i in 1:nf
        i2, i3, iϕ = i, nf + i, 2nf + i
        ϕ = u[iϕ]
        J[i2, i2] = J[i2, i3] = J[i2, iϕ] = 0.0
        J[i3, i2] = J[i3, i3] = J[i3, iϕ] = 0.0
        J[iϕ, i2] = J[iϕ, i3] = 0.0
        J[iϕ, iϕ] = -exp(-ϕ)

        m.active[i] || continue
        T = SVector(m.τ0[1] + c.Δτ[i2], m.τ0[2] + c.Δτ[i3])
        Tn = norm(T)
        V = c.Vmag[i]
        (iszero(Tn) || !isfinite(V)) && continue

        σ̄ = max(p.σ0 - pres[i], p.σ̄_min)
        d = slip_rate_derivatives(V, exp(ϕ), σ̄, η, fp)
        that = T / Tn
        P = that * that'
        dVvec_dT = d.dV_dT * P + (V / Tn) * (SMatrix{2,2}(1.0, 0.0, 0.0, 1.0) - P)
        Kii = SMatrix{2,2}(K[i2, i2], K[i3, i2], K[i2, i3], K[i3, i3])
        dVvec_ds = dVvec_dT * Kii
        dVmag_ds = d.dV_dT * (that' * Kii)

        J[i2, i2] = dVvec_ds[1, 1]; J[i2, i3] = dVvec_ds[1, 2]; J[i2, iϕ] = d.dV_dϕ * that[1]
        J[i3, i2] = dVvec_ds[2, 1]; J[i3, i3] = dVvec_ds[2, 2]; J[i3, iϕ] = d.dV_dϕ * that[2]
        J[iϕ, i2] = -dVmag_ds[1] / p.D_RS
        J[iϕ, i3] = -dVmag_ds[2] / p.D_RS
        J[iϕ, iϕ] -= d.dV_dϕ / p.D_RS
    end
    return nothing
end

"""
    default_integrator(m) -> alg

`Tsit5()` for the Gaussian source, `QNDF()` for the Peaceman well; see
[`run_bp8`](@ref) for the measurements behind the split.
"""
default_integrator(m::BP8Model) = m.injection === :peaceman ? QNDF() : Tsit5()

"""
    run_bp8(m; tspan=(0.0, m.par.t_f), alg=default_integrator(m), reltol=1e-8,
              saveat=3600.0, verbose=false, pressure_kwargs=(;), kwargs...)

Integrates slip and state. Absolute tolerances are per block (slip, ln θ),
which live on very different scales.

**The integrator defaults per injection model** ([`default_integrator`](@ref)):
explicit `Tsit5` for the Gaussian source, whose ~400 steps leave nothing to
gain, and implicit `QNDF` with the block-diagonal [`state_jacobian!`](@ref) for
the Peaceman well. Over 100 h at Δz = 25 m: Tsit5 488,788 steps / 4,901 s
against QNDF 1,105 steps / 20 s, and the implicit count does not grow under
refinement where the explicit one grew as `Δz⁻⁴`. Both reach the same solution
to tolerance (`test/bp8_test.jl`). Pass `alg` to override; the Jacobian is
always attached, but Rosenbrock methods do not tolerate its dropped off-diagonal
blocks — they have no Newton loop to absorb them.

**Pore pressure is solved first, separately and implicitly**
([`solve_pressure_history`](@ref)) and attached to `m`; the slip integration
interpolates it. An already-attached history covering `tspan` is reused, and
`build_model` attaches one over the full `par.t_f`, so sub-interval runs and
sweeps over elastic settings pay for it once. `pressure_kwargs` overrides that
solve.

`land_on_saveat` (default: on for the Peaceman well) adds the `saveat` grid to
`tstops`, so every saved row is a step solution rather than the interpolant. At
a `σ̄_min`-floored node V is ill-conditioned in (τ, θ), and interpolated rows
showed V spikes >1 decade above the true slip rate; see `PEACEMAN_SPIKES.md`.

`progress` (defaults to `verbose`) shows a bar tracking `t/tspan[2]`.
"""
function run_bp8(m::BP8Model; tspan=(0.0, m.par.t_f), alg=default_integrator(m), reltol=1e-8,
                 saveat=3600.0, land_on_saveat=m.injection === :peaceman, verbose=false, progress=verbose,
                 pressure_kwargs=(;), kwargs...)
    covers = m.pressure.sol !== nothing &&
             m.pressure.tspan[1] <= tspan[1] && tspan[2] <= m.pressure.tspan[2]
    covers || set_pressure_history!(m,
                  solve_pressure_history(m; tspan, verbose, pressure_kwargs...), tspan)

    u0 = initial_state(m)
    nf = m.nf
    abstol = similar(u0)
    abstol[1:2nf] .= 1e-14        # slip, m
    abstol[2nf+1:3nf] .= 1e-10    # ln θ

    # Attached unconditionally: explicit methods ignore it and the prototype is
    # only 9nf entries. See the comment block above `state_jacobian!`.
    f = ODEFunction(rhs!; jac=state_jacobian!, jac_prototype=state_jacobian_prototype(m))
    prob = ODEProblem(f, u0, tspan, m)
    t0 = time()
    tstops = [m.par.t_off]
    if land_on_saveat
        grid = saveat isa Number ? collect(tspan[1]:saveat:tspan[2]) : collect(saveat)
        tstops = sort!(unique!(vcat(tstops, grid)))
    end
    solve_kwargs = (; reltol, abstol, saveat, save_everystep=false, tstops, kwargs...)
    if progress
        prog = Progress(100; desc="bp8: ")
        # `u_modified!(integrator, false)` is NOT optional. A `DiscreteCallback`
        # leaves `integrator.u_modified == true` unless told otherwise, which
        # makes the integrator assume the callback changed `u` and reinitialise
        # after EVERY step — refactorising `W` and discarding the multistep
        # history. `Tsit5` is explicit and single-step, so it barely notices
        # (BP8-GS: 664 steps either way); `QNDF` is an implicit BDF method and
        # is destroyed by it. Measured at Δz = 10 m on (1600, 1600), t = 2 h:
        # 241 steps / 16.7 s without the callback against >40x slower with it,
        # which is why every BP8-PW run through `run_bp8.jl` sat at 0% forever
        # while the same model solved fine from the REPL.
        cb = DiscreteCallback((u, t, integrator) -> true,
            function (integrator)
                update!(prog, floor(Int, 100 * (integrator.t - tspan[1]) / (tspan[2] - tspan[1])))
                u_modified!(integrator, false)
                return nothing
            end;
            save_positions=(false, false))
        solve_kwargs = (; solve_kwargs..., callback=cb)
    end
    # `save_everystep=false`: `saveat` already defines the output grid, and
    # storing every accepted step on top of it is what made BP8-PW blow up —
    # 284 MB against 1.1 MB at Δz = 50 m over 100 h, and a Δz = 25 m run reached
    # 7.9 GB without finishing.
    #
    # `sol(t)` still works and is bit-identical on the `saveat` grid, which is
    # all the shipped path asks for (`write_profiles`' default `profile_dt`
    # equals the default `saveat`). A `profile_dt` that is not a multiple of
    # `saveat` interpolates across hour-wide gaps: 1.3e-5 relative, not zero.
    sol = solve(prob, alg; solve_kwargs...)
    progress && finish!(prog)
    verbose && @info "integration finished" seconds = round(time() - t0, digits=1) saved = length(sol.t) steps = sol.stats.naccept retcode = sol.retcode
    return sol
end

# Analytic pore-pressure solutions (eq. 21 and 25), valid for t ≪ l_f²/(4α) ≈
# 220 h, i.e. before diffusion reaches Ω_f's no-flux edges. §6 asks explicitly
# that the Peaceman well be checked against eq. 25.

"""
    expint_e1(x)

The exponential integral `E₁(x) = ∫_x^∞ e^{-s}/s ds` for `x > 0`: series for
`x < 1`, Lentz continued fraction otherwise. Here rather than a
`SpecialFunctions` dependency for one function.
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

# Benchmark output (§4).

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

    # Derived fields at every saved time, through the same `evaluate!` the
    # integrator used.
    ns = length(times)
    V2 = Matrix{Float64}(undef, nf, ns)
    V3 = similar(V2); τ2 = similar(V2); τ3 = similar(V2)
    q2 = similar(V2); q3 = similar(V2); pr = similar(V2)
    Vmax = Vector{Float64}(undef, ns)
    moment_rate = Vector{Float64}(undef, ns)

    for (j, t) in enumerate(times)
        u = sol.u[j]
        c = evaluate!(m, u, t)
        V2[:, j] .= c.V2
        V3[:, j] .= c.V3
        τ2[:, j] .= c.τ2
        τ3[:, j] .= c.τ3
        # `evaluate!` has just interpolated the pressure history at `t`.
        pres = c.pres
        pr[:, j] .= pres
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

    write_time_series(m, sol, dir, times, V2, V3, τ2, τ3, q2, q3, pr, step_lines, modeler)
    write_global(m, dir, times, Vmax, moment_rate, step_lines, modeler)
    write_profiles(m, sol, dir, profile_dt, modeler)
    return dir
end

function write_time_series(m, sol, dir, times, V2, V3, τ2, τ3, q2, q3, pr, step_lines, modeler)
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
                        pr[idx, n] / 1e6, q2[idx, n], q3[idx, n],
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
        println(io, "# Column #3 = Moment_density_rate (N.m/s)")
        println(io, "# The line below lists the names of the data fields")
        # SEAS uploader expects `moment_density_rate`, not the description PDF's `moment_rate`.
        println(io, "t max_slip_rate moment_density_rate")
        println(io, "# Here is the time-series data.")
        for (n, t) in enumerate(times)
            @printf(io, "%21.13E %14.6E %14.6E\n", t, safelog10(Vmax[n]), moment_rate[n])
        end
    end
end

"""
Profile files (§4.3), in the layout of that section's worked example: an
`(N_t+1) × (N_coord+2)` matrix whose first row is `0 0 <coordinates>` and whose
rest are `t  max_slip_rate  <quantity at each coordinate>`. The two leading
zeros keep every row the same width. The field list is four separate lines, and
the header keys (`author`, `code_version`) differ from the time-series files'
(`modeler`, `version`) — each follows its own section's example.
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
                  (; τ2=copy(c.τ2), τ3=copy(c.τ3), p=copy(c.pres), Vmax=maximum(c.Vmag))
              end for (u, t) in zip(states, ts)]

    quantities = ["slip_2" => ((u, c, k) -> u[k], "Horizontal slip (Slip_2) (m)"),
                  "slip_3" => ((u, c, k) -> u[nf+k], "Vertical slip (Slip_3) (m)"),
                  "shear_stress_2" => ((u, c, k) -> c.τ2[k] / 1e6,
                                       "Horizontal shear stress (Shear_stress_2) (MPa)"),
                  "shear_stress_3" => ((u, c, k) -> c.τ3[k] / 1e6,
                                       "Vertical shear stress (Shear_stress_3) (MPa)"),
                  "pore_pressure" => ((u, c, k) -> c.p[k] / 1e6,
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

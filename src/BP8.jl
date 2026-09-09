module BP8

using Diffinitive
using Diffinitive.Grids
using Diffinitive.SbpOperators
using SparseArrays
using StaticArrays
using LinearAlgebra: mul!, norm, diag
using OrdinaryDiffEq
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
       run_bp8, evaluate!, write_outputs, station_locations,
       solve_pressure_history, set_pressure_history!, pressure_at!,
       pressure_operator, pressure_length, well_pressure,
       effective_stress_report, analytic_pressure_gaussian, analytic_pressure_point,
       resolution_report, process_zone, fault_grid_sizes, build_fault_elasticity

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
    # WHICH nodes were ever unclamped, not just how many evaluations hit the
    # floor. `floor_hits` counts RHS evaluations, so it scales with the
    # integrator's step count and says nothing about how much of the fault is
    # affected — the question a reader of the limitation actually has.
    floor_nodes::BitVector
    # Pore pressure at the current `t`, interpolated out of the separately
    # integrated pressure history (`solve_pressure_history`). `pbuf` is the
    # raw subsystem state — length `nf`, or `nf+1` for the Peaceman variant,
    # whose last entry is the well-bore pressure — and `pres` views its first
    # `nf` entries, the fault field itself. Interpolating in place keeps the
    # right-hand side allocation-free.
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

Holds the separately integrated pore-pressure solution, or `nothing` before
one has been computed. Mutable so `run_bp8` can attach a history to an
otherwise immutable `BP8Model`; `sol` is deliberately untyped, and every read
of it goes through the `pressure_at!` function barrier.
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
`build_model` uses. Factored out so external tooling — currently
`build_stiffness_cache.jl`'s sharding path — can reproduce exactly the `n1`,
`n23` `build_model` would use for the same configuration, which a sharded
`:exact` build depends on to actually match the single-process one.
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

Assembles the split-node elastic system `fault_stiffness` solves against, for
the grids `build_model` has already sized and validated via
[`fault_grid_sizes`](@ref). Factored out of the cache-miss path below so a `K`
build can be **sharded across independent processes**: each shard calls this
— minutes, not the bottleneck, PERFORMANCE.md §4c — and then
`fault_stiffness(fe; cols=..., ...)` for its own slice of the `2·N_Ωf`
columns, with no communication needed between shards (the columns are
independent right-hand sides against the same `A`). `build_model` itself goes
through this same function on every cache miss, so there is exactly one
definition of "the elastic system for this configuration" — what makes it
safe to assemble a `K` from shards built in separate processes and merge them
into one cache entry (`merge_stiffness_cache.jl`).
"""
function build_fault_elasticity(; par::BP8Params, Δz, L_fault, L_normal, n1, n23, set,
                                verbose=false, solver_kwargs...)
    g_minus = equidistant_grid((-L_normal, -L_fault, -L_fault), (0.0, L_fault, L_fault), n1, n23, n23)
    g_plus = equidistant_grid((0.0, -L_fault, -L_fault), (L_normal, L_fault, L_fault), n1, n23, n23)

    t0 = time()
    fe = FaultElasticity(g_minus, g_plus, lame_lambda(par), par.μ, set;
                         l_f=par.l_f, solver_kwargs...)
    verbose && @info "split-node system ready" seconds = round(time() - t0, digits=1)
    return fe
end

# `K` for one configuration, from the cache if it is there.
#
# Split out of `build_model` so the two paths that produce a `K` — load, and
# build-then-maybe-save — sit next to each other, and so `FaultElasticity` is
# constructed inside the miss branch only. That placement is the point of the
# whole cache: assembling the split-node system is ~15 GB and minutes at the
# Δz = 20 m target (PERFORMANCE.md §4), and a hit has no use for it.
function stiffness_matrix(; par, Δz, L_fault, L_normal, n1, n23, order, set,
                          stiffness, cache, cache_dir, verbose, solver_kwargs)
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
                                verbose, solver_kwargs...)

    t0 = time()
    K = stiffness === :toeplitz ? fault_stiffness_toeplitz(fe; verbose) :
                                  fault_stiffness(fe; verbose)
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

`stiffness` selects how `K` is built. **`:toeplitz` is the default**: it does
**10** solves, expanding 5 sources by the whole-space kernel's translation
invariance, instead of `2·N_Ωf`. `:exact` does all `2·N_Ωf` and remains available
as the reference — use it whenever you are measuring the approximation itself.

It is an approximation, and the case for defaulting to it is that its error is
bounded and shrinks along **both** axes production moves along
(`PERFORMANCE.md` §4b): 5.78% at the small domain and Δz = 50 m, **0.41%** at the
converged domain, **0.87%** at Δz = 25 m. Every configuration that will actually
be run is more favourable than the ones measured, and all of them sit far below
the resolution error. The cost difference at the Δz = 20 m target is ~17 days
against ~3 h on one node, which is the difference between the run happening on a
single machine and needing a cluster.

`precond` is forwarded to [`CGSolver`](@ref) and selects the preconditioner for
each solve. It is an **independent** axis from `stiffness`: `stiffness` sets how
*many* solves are done, `precond` how each one converges, and every combination
is valid. `:none` (default) or `:jacobi` — the latter measures 0.92×, i.e. worse
than none, and exists so that stays visible rather than being rediscovered.

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

**A hit skips `FaultElasticity` as well as the solves**, which is the bulk of
the remaining cost: the cache file carries the `Ω_f` axes, and nothing else in
`BP8Model` needs the elastic system once `K` exists. So a cached `:exact` model
at a production configuration builds in seconds rather than days — which is what
makes `:exact` usable for a sweep over the *cheap* axes (injection, friction,
`t_f`, tolerances) where `K` does not change at all.

The cache key spells out every input that reaches `K` and is verified against
the file on load, so a changed configuration misses rather than silently
returning the wrong `K`; see `StiffnessCache`.
"""
function build_model(; par::BP8Params=benchmark_parameters(),
                     Δz=par.Δz, L_fault=3par.l_f, L_normal=2par.l_f,
                     injection=:gaussian, order=4, verbose=false,
                     stiffness=:toeplitz,
                     cache=:auto, cache_dir=stiffness_cache_dir(),
                     pressure=true, pressure_kwargs=(;),
                     solver_kwargs...)
    injection ∈ (:gaussian, :peaceman) ||
        error("injection must be :gaussian or :peaceman, got $injection")

    set = read_stencil_set(SbpOperators.sbp_operators_path() * "standard_diagonal.toml"; order)
    n1, n23 = fault_grid_sizes(par, Δz, L_fault, L_normal, order)
    # SBP closures need more than two closure widths of points per dimension —
    # also used below for the pore-pressure grid, which fault_grid_sizes does
    # not know about.
    n_min = 2order + 1
    verbose && @info "elastic grids" points_per_side = n1 * n23^2 dofs = 6 * n1 * n23^2

    stiffness ∈ (:exact, :toeplitz) ||
        error("stiffness must be :exact or :toeplitz, got $stiffness")
    K, x2, x3 = stiffness_matrix(; par, Δz, L_fault, L_normal, n1, n23, order, set,
                                 stiffness, cache, cache_dir, verbose, solver_kwargs)
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

    plen = nf + (injection === :peaceman ? 1 : 0)
    m = BP8Model(par, K, Ap, source, Q2, Q3, weights, active, collect(x2), collect(x3),
                 nf, τ0, injection, well_cell_index(g_p), WI, S_e, Cache(nf, plen),
                 PressureHistory(),
                 (; Δz, L_fault, L_normal, n1, n23, order,
                    elastic_dofs=6 * n1 * n23^2, n2f, n3f))

    # Solve the pressure history up front, over the full benchmark duration, so
    # the model is complete: `evaluate!` works immediately and any `run_bp8`
    # sub-interval reuses it. Seconds, against hours for `K`. `pressure=false`
    # skips it for callers that only want `K` (the cache builder, say).
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

Whether the effective-normal-stress floor `par.σ̄_min` ever bound, how low
`σ̄ = σ - p` went, and — the part that decides whether it matters — **how much
of the fault was affected**. `σ̄ ≤ 0` means fluid pressure has fully unclamped
the fault, at which point BP8's no-opening condition (eq. 3) no longer holds and
the model is outside its range of validity. The Peaceman-well variant reaches
this at Table 1's parameters; the Gaussian-source variant does not.

`nodes` and `radius` are what bound the damage. The unclamped region is a disc
whose *physical* radius is set by where eq. 25's pressure crosses `σ` — about
**15 m** at Table 1's parameters — and is therefore **independent of Δz**.
Refining the grid does not enlarge it, it only resolves it: ~0.3 cells across at
Δz = 50 m, ~1.5 at Δz = 10 m. So the well-cell `σ̄` falls steeply with resolution
(−0.87 MPa at 50 m, −16.3 MPa at 10 m) while the affected *area* does not grow.
Against `l_f` = 400 m that is a localized defect in BP8-PW's own point-source
specification, not a discretization problem and not one that contaminates the
fault at large. See `PROGRESS.md` "Known limitations" 3.

`floor_hits` counts RHS **evaluations**, so it scales with the integrator's step
count and is not a measure of extent; use `nodes` for that.
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

# ==============================================================================
# Pore pressure: integrated separately, implicitly, once.
#
# The pressure subsystem is *autonomous* — its right-hand side reads only `p`,
# `p_well` and `t`, never slip or state (eq. 17-23; the coupling to elasticity
# is one-way, through `σ̄ = σ - p`). So it does not belong in the coupled state
# vector at all: it can be integrated on its own, with a method suited to it,
# and the elastic integration then interpolates the result.
#
# Measured at Δz = 10 m over the full 30 days, `Rodas5P` with the analytic
# Jacobian below: 180 steps (Gaussian) / 283 steps (Peaceman), ~6 s, and the
# step count is **resolution-independent** (94/88/95 at Δz = 50/25/10 m).
# Carrying `p` explicitly in the coupled system instead cost `nf` of `4nf+1`
# state entries and needed the loosened per-block `abstol` that used to sit in
# `run_bp8`.
# ==============================================================================

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

The subsystem is *linear*, so its Jacobian is exactly the constant `J` from
[`pressure_operator`](@ref); supplying it explicitly is what makes the implicit
solve cheap (and avoids a dense Jacobian being built by finite differences —
that alone was 400 s and 1 GB at Δz = 10 m in testing).

**`save_everystep=true` is the point, not an oversight.** The solve needs only
~180-280 steps for the whole benchmark, so the full dense output is 57 MB
(Gaussian) / 79 MB (Peaceman) at Δz = 10 m — small enough to keep in memory,
and the solver's own interpolant is far better than storing levels on a fixed
grid and interpolating linearly between them. Measured against a `reltol=1e-11`
reference at Δz = 10 m:

| interpolation | max error | induced error in `V` |
|---|---|---|
| **dense output (this)** | **4.4 Pa** | **0.003 %** |
| linear, hourly levels | 10.8 kPa | 7.1 % |
| linear, 300 s levels | 85 Pa | 0.05 % |

Linear interpolation is second-order and converges, but `V ~ exp(τ/(aσ̄))`
amplifies pressure error exponentially, and the error concentrates entirely at
the two kinks in the forcing (`t = 0` and `t_off`) — exactly where an adaptive
solver puts steps and a fixed grid does not.
"""
function solve_pressure_history(m::BP8Model; tspan=(0.0, m.par.t_f), alg=Rodas5P(),
                                reltol=1e-8, abstol=1e-3, verbose=false)
    par = m.par
    J = pressure_operator(m)
    n = size(J, 1)

    # Both variants force through the same eq. 20 on/off switch, so the whole
    # time dependence is one scalar times a fixed vector: the Gaussian source
    # spread over the fault (eq. 19), or the injection rate into the well bore's
    # storage (eq. 23).
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

# Interpolation happens behind a function barrier because `PressureHistory.sol`
# is untyped: everything inside `_interp_pressure!` specializes on the concrete
# solution type, so the per-call cost is one dynamic dispatch, not a
# type-unstable inner loop.
_interp_pressure!(dest, sol, t) = (sol(dest, t); dest)

"""
    well_pressure(m, t)

The Peaceman well-bore pressure `p_well` at time `t` (BP8-QD eq. 23), the extra
unknown carried by [`well_coupled_operator`](@ref). Errors for the Gaussian
variant, which has no well.

It is not a §4 reported output — it is an internal unknown — but it is what the
injected-volume balance and the eq. 25 point-source check need. After the
initial transient it sits at `p[well_cell] + Q0/WI`; measured at `t_off`, that
offset is 53.4 MPa at Δz = 50 m and 38.0 MPa at Δz = 10 m, against `Q0/WI` of
53.4 and 38.0.
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
into the model cache (no allocation). Errors rather than extrapolating if `t`
lies outside the history — silently extrapolating a diffusion solution past its
integration window would be a quiet source of wrong answers.
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

Slip and state (eq. 5, 11). Pore pressure is **not** here: it is autonomous, so
it is integrated separately by [`solve_pressure_history`](@ref) and enters only
through `evaluate!`'s `σ̄ = σ - p`.
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
    return nothing
end

"""
    run_bp8(m; tspan=(0.0, m.par.t_f), alg=Tsit5(), reltol=1e-8,
              saveat=3600.0, verbose=false, pressure_kwargs=(;), kwargs...)

Integrates slip and state. Absolute tolerances are set per block (slip, ln θ)
because they live on wildly different scales.

**Pore pressure is solved first, separately and implicitly**
([`solve_pressure_history`](@ref)), and attached to `m`; the slip integration
then interpolates it. That costs a few seconds and removes `p` from the
explicitly integrated state entirely. Pass `pressure_kwargs` to override that
solve (`alg`, `reltol`, `abstol`). An already-attached history *covering* `tspan`
is reused — `build_model` attaches one over the full `par.t_f` by default, so
sub-interval runs and parameter sweeps over elastic settings pay for it once.

`progress` (defaults to `verbose`) shows a `ProgressMeter` bar tracking `t/tspan[2]`,
updated on every accepted step (ProgressMeter throttles the redraws itself).
"""
function run_bp8(m::BP8Model; tspan=(0.0, m.par.t_f), alg=Tsit5(), reltol=1e-8,
                 saveat=3600.0, verbose=false, progress=verbose,
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

    prob = ODEProblem(rhs!, u0, tspan, m)
    t0 = time()
    solve_kwargs = (; reltol, abstol, saveat, save_everystep=false,
                     tstops=[m.par.t_off], kwargs...)
    if progress
        prog = Progress(100; desc="bp8: ")
        cb = DiscreteCallback((u, t, integrator) -> true,
            integrator -> update!(prog, floor(Int, 100 * (integrator.t - tspan[1]) / (tspan[2] - tspan[1])));
            save_positions=(false, false))
        solve_kwargs = (; solve_kwargs..., callback=cb)
    end
    # `save_everystep=false`: `saveat` already defines the output grid, and
    # storing every accepted step on top of it is what made the Peaceman variant
    # blow up. Measured at Δz = 50 m over 100 h: 31,249 accepted steps for 101
    # wanted outputs, 284 MB against 1.1 MB — and the stiffness that drives that
    # step count grows steeply as Δz shrinks (a Δz = 25 m run reached 7.9 GB
    # without finishing; see `PROGRESS.md` limitation 3).
    #
    # `sol(t)` still works: on the `saveat` grid it is bit-identical, and
    # `write_profiles`' default `profile_dt` equals the default `saveat`, so the
    # shipped path only ever asks for on-grid times. A `profile_dt` that is not
    # a multiple of `saveat` interpolates across hour-wide gaps instead of
    # actual steps — measured at 1.3e-5 relative, negligible here but not zero.
    sol = solve(prob, alg; solve_kwargs...)
    progress && finish!(prog)
    verbose && @info "integration finished" seconds = round(time() - t0, digits=1) saved = length(sol.t) steps = sol.stats.naccept retcode = sol.retcode
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

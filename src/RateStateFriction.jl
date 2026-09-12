module RateStateFriction

using StaticArrays
using LinearAlgebra: norm

export FrictionParams, radiation_damping_coefficient, friction_coefficient,
       fault_strength, aging_law_rhs, solve_slip_rate, solve_slip_velocity,
       slip_rate_derivatives, initial_state_from_strength

"""
    FrictionParams(a, b, Dc, V_star, f_star)

Rate-and-state direct-effect (`a`) and evolution-effect (`b`) parameters,
characteristic slip distance `Dc` (`D_RS` in the benchmark), reference slip
rate `V_star`, and reference friction coefficient `f_star` (BP8-QD eq. 12).
"""
struct FrictionParams
    a::Float64
    b::Float64
    Dc::Float64
    V_star::Float64
    f_star::Float64
end

"""
    radiation_damping_coefficient(μ, c_s)

`η = μ/(2c_s)`, the radiation-damping coefficient (BP8-QD eq. 8).
"""
radiation_damping_coefficient(μ, c_s) = μ / (2c_s)

"""
    friction_coefficient(V, θ, p::FrictionParams)

The regularized rate-and-state friction coefficient (BP8-QD eq. 12):
`f(V,θ) = a·asinh[ V/(2V*)·exp((f* + b·ln(V*θ/Dc))/a) ]`.
"""
function friction_coefficient(V, θ, p::FrictionParams)
    return p.a * asinh(V / (2p.V_star) * exp((p.f_star + p.b * log(p.V_star * θ / p.Dc)) / p.a))
end

"""
    fault_strength(V, θ, σ̄, p::FrictionParams)

Fault shear resistance magnitude `F = σ̄·f(V,θ)` (BP8-QD eq. 10).
"""
fault_strength(V, θ, σ̄, p::FrictionParams) = σ̄ * friction_coefficient(V, θ, p)

"""
    aging_law_rhs(V, θ, p::FrictionParams)

`dθ/dt = 1 - Vθ/Dc` (BP8-QD eq. 11, the aging law).
"""
aging_law_rhs(V, θ, p::FrictionParams) = 1 - V * θ / p.Dc

"""
    solve_slip_rate(T, θ, σ̄, η, p::FrictionParams; V0=nothing, tol=1e-12, maxiter=50)

Solves the scalar radiation-damped force balance `T = ηV + σ̄f(V,θ)` for the
slip-rate magnitude `V > 0` (BP8-QD eq. 8-10, combined and reduced to a
scalar equation in `V=|V|` — see module notes). `f(V,θ)` is smooth and
strictly increasing in `V`, so the root is unique; solved via Newton's
method in `x = ln V`, robust across the many decades of `V` this benchmark
spans. `V0`, if given, seeds the iteration (a good warm-start during time
stepping); otherwise `V_star` is used.

Newton steps are clamped to `maxstep` in log space. Undamped, the iteration
diverges whenever it is seeded *below* the root at high slip rate: there
`asinh(u) ≈ ln(2u)` is nearly linear in `x`, so the step overshoots by many
decades at once and `exp(x)` overflows to `Inf`, silently returning `NaN` or
a value wrong by many orders of magnitude. A warm start during acceleration
always approaches from below, so this is reachable in a time loop even
though BP8-QD-GS is velocity-strengthening and should not itself get near
the seismic slip rates where it first appears.

If the iteration has not converged within `maxiter` steps it never returns a
bad root: with `onfail=:error` (the default) it throws, and with
`onfail=:nan` it returns `NaN`. The latter exists for implicit time
integrators, whose Newton iterations evaluate the right-hand side at trial
states that can be arbitrarily far off the trajectory — a `NaN` there makes
the integrator reject the step and shrink it, which is the right response,
whereas an exception would abort the whole run over a state that was never
going to be accepted. Non-finite inputs (`T`, `θ`) fall through to the same
path, since no iteration converges from them.
"""
function solve_slip_rate(T, θ, σ̄, η, p::FrictionParams; V0=nothing, tol=1e-12, maxiter=50,
                         maxstep=5.0, onfail::Symbol=:error)
    C = exp((p.f_star + p.b * log(p.V_star * θ / p.Dc)) / p.a)
    x, converged = _newton_log_slip_rate(T, C, σ̄, η, p, log(V0 === nothing ? p.V_star : V0),
                                         tol, maxiter, maxstep)
    # The warm start is an optimisation, never a requirement: an implicit
    # integrator evaluates the force balance at trial states it then rejects,
    # and the seed those leave behind can be arbitrarily bad for the state it
    # actually retries from. So a seeded iteration that fails is retried once
    # from the reference rate before it counts as a failure — otherwise one
    # rejected trial poisons every later evaluation at that node and the step
    # size collapses (observed: `dt` driven below eps mid-injection).
    if !converged && V0 !== nothing
        x, converged = _newton_log_slip_rate(T, C, σ̄, η, p, log(p.V_star), tol, maxiter, maxstep)
    end
    if !converged
        onfail === :nan && return NaN
        error("solve_slip_rate failed to converge in $maxiter iterations " *
              "(T=$T, θ=$θ, σ̄=$σ̄, η=$η, V0=$V0, last V=$(exp(x)))")
    end
    return exp(x)
end

# Newton on `g(x) = ηe^x + σ̄·a·asinh(e^x·C/2V*) - T` from `x0`; returns the
# final `x` and whether `|dx| < tol` was reached.
function _newton_log_slip_rate(T, C, σ̄, η, p::FrictionParams, x0, tol, maxiter, maxstep)
    x = x0
    isfinite(x) || return x, false
    for _ in 1:maxiter
        V = exp(x)
        u = V / (2p.V_star) * C
        g = η * V + σ̄ * p.a * asinh(u) - T
        dgdx = η * V + σ̄ * p.a * u / hypot(1.0, u)
        dx = clamp(-g / dgdx, -maxstep, maxstep)
        x += dx
        abs(dx) < tol && return x, true
        isfinite(x) || return x, false
    end
    return x, false
end

"""
    slip_rate_derivatives(V, θ, σ̄, η, p::FrictionParams) -> (; dV_dT, dV_dϕ)

Partial derivatives of the slip-rate magnitude `V` — the root of the force
balance `g(V) = ηV + σ̄f(V,θ) - T = 0` that [`solve_slip_rate`](@ref) finds —
with respect to the trial-stress magnitude `T` and the log state `ϕ = ln θ`,
by implicit differentiation at an already-converged `V`:

    ∂V/∂T = 1/D,   ∂V/∂ϕ = -(∂g/∂ϕ)/D,   D = ∂g/∂V = η + σ̄·a·(C/2V*)/√(1+u²)

with `u = V·C/(2V*)` and `C = exp((f* + b·ln(V*θ/D_c))/a)`, so that
`∂g/∂ϕ = σ̄·b·u/√(1+u²)`. `D` is written with `C/(2V*)` rather than `u/V` so it
stays finite as `V → 0`.

`1/D` is the quantity behind BP8-PW's stiffness: `D ∝ σ̄` for a node whose
effective normal stress has been floored, so `K_ww/D` — the node's slip against
its own self-stiffness — is the stiff eigenvalue (PROGRESS.md "what the stiff
eigenvalue is"). These closed forms are what let `BP8.state_jacobian!` supply
that eigenvalue to an implicit integrator for the cost of one pass over the
nodes, rather than `N` right-hand-side evaluations for a finite-difference
Jacobian.
"""
function slip_rate_derivatives(V, θ, σ̄, η, p::FrictionParams)
    C = exp((p.f_star + p.b * log(p.V_star * θ / p.Dc)) / p.a)
    u = V / (2p.V_star) * C
    # `hypot` rather than `sqrt(1 + u^2)`: `u` can exceed 1e154 before `V` is
    # unphysical, and `u^2` overflows there while `u/hypot(1, u) → 1` is exact.
    h = hypot(1.0, u)
    D = η + σ̄ * p.a * (C / (2p.V_star)) / h
    return (; dV_dT=1 / D, dV_dϕ=-σ̄ * p.b * (u / h) / D)
end

"""
    solve_slip_velocity(τ0::SVector{2}, Δτ::SVector{2}, θ, σ̄, η, p::FrictionParams; kwargs...)

The vector form of the force balance (BP8-QD eq. 8-10): since both the
radiation-damping term `ηV` and the friction term `σ̄f(V,θ)V/V` are parallel
to `V`, the trial stress `τ0+Δτ` must itself be parallel to `V` — its
direction *is* `V`'s direction, and the magnitude solves `solve_slip_rate`.
Returns the slip-rate vector `V`.
"""
function solve_slip_velocity(τ0::SVector{2}, Δτ::SVector{2}, θ, σ̄, η, p::FrictionParams; kwargs...)
    Tvec = τ0 + Δτ
    T = norm(Tvec)
    # With no trial stress the direction is undefined, but `ηV + σ̄f(V,θ)` is
    # strictly positive for `V>0`, so `V=0` is the (unique) answer.
    iszero(T) && return zero(Tvec)
    V = solve_slip_rate(T, θ, σ̄, η, p; kwargs...)
    return V * Tvec / T
end

"""
    initial_state_from_strength(V, F_target, σ̄, p::FrictionParams)

Closed-form inverse of `friction_coefficient` for `θ`, given a slip rate
`V` and target fault strength `F_target` (used to set up the initial state
variable per BP8-QD eq. 28-29, where `V=V_init` and `F_target=τ_init`):
```
θ = (Dc/V*)·exp( (a·ln(2V*·sinh(f/a)/V) - f*) / b ),   f = F_target/σ̄
```
"""
function initial_state_from_strength(V, F_target, σ̄, p::FrictionParams)
    f = F_target / σ̄
    return (p.Dc / p.V_star) * exp((p.a * log(2p.V_star * sinh(f / p.a) / V) - p.f_star) / p.b)
end

end # module RateStateFriction

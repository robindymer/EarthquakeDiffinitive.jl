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
slip-rate magnitude `V > 0` (BP8 eq. 8-10 reduced to a scalar in `V=|V|`).
`f` is smooth and strictly increasing in `V`, so the root is unique; Newton in
`x = ln V` handles the many decades `V` spans. `V0` seeds the iteration
(warm start during time stepping); otherwise `V_star`.

Newton steps are clamped to `maxstep` in log space. Undamped, a seed *below*
the root at high slip rate overshoots by many decades — `asinh(u) ≈ ln(2u)` is
nearly linear there — and `exp(x)` overflows. Warm starts during acceleration
always approach from below, so this is reachable in a time loop.

On non-convergence within `maxiter` it never returns a bad root: `onfail=:error`
(default) throws, `onfail=:nan` returns `NaN`. The latter is for implicit
integrators, whose trial states can be far off the trajectory — a `NaN` makes
them reject and shrink the step, where an exception would abort the run over a
state that was never going to be accepted. Non-finite `T` or `θ` take the same
path.
"""
function solve_slip_rate(T, θ, σ̄, η, p::FrictionParams; V0=nothing, tol=1e-12, maxiter=50,
                         maxstep=5.0, onfail::Symbol=:error)
    C = exp((p.f_star + p.b * log(p.V_star * θ / p.Dc)) / p.a)
    x, converged = _newton_log_slip_rate(T, C, σ̄, η, p, log(V0 === nothing ? p.V_star : V0),
                                         tol, maxiter, maxstep)
    # A rejected implicit trial can leave a seed that is bad for the state the
    # integrator actually retries from, so a failed seeded iteration is retried
    # once from the reference rate. Otherwise one rejected trial poisons every
    # later evaluation at that node and `dt` collapses below eps.
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

`1/D` is behind BP8-PW's stiffness: `D ∝ σ̄` at a floored node, so `K_ww/D` is
the stiff eigenvalue (PROGRESS.md "what the stiff eigenvalue is"). These closed
forms let `BP8.state_jacobian!` hand that eigenvalue to an implicit integrator
in one pass over the nodes instead of `N` RHS evaluations.
"""
function slip_rate_derivatives(V, θ, σ̄, η, p::FrictionParams)
    C = exp((p.f_star + p.b * log(p.V_star * θ / p.Dc)) / p.a)
    u = V / (2p.V_star) * C
    # `hypot`, not `sqrt(1 + u^2)`: `u` exceeds 1e154 at still-physical `V`, so
    # `u^2` overflows where `u/hypot(1, u) → 1` is exact.
    h = hypot(1.0, u)
    D = η + σ̄ * p.a * (C / (2p.V_star)) / h
    return (; dV_dT=1 / D, dV_dϕ=-σ̄ * p.b * (u / h) / D)
end

"""
    solve_slip_velocity(τ0::SVector{2}, Δτ::SVector{2}, θ, σ̄, η, p::FrictionParams; kwargs...)

Vector form of the force balance (BP8 eq. 8-10). Both `ηV` and `σ̄f(V,θ)V/|V|`
are parallel to `V`, so the trial stress `τ0+Δτ` gives the direction and
[`solve_slip_rate`](@ref) the magnitude. Returns the slip-rate vector.
"""
function solve_slip_velocity(τ0::SVector{2}, Δτ::SVector{2}, θ, σ̄, η, p::FrictionParams; kwargs...)
    Tvec = τ0 + Δτ
    T = norm(Tvec)
    # Direction is undefined at zero trial stress, but `ηV + σ̄f(V,θ) > 0` for
    # `V>0`, so `V=0` is the unique answer.
    iszero(T) && return zero(Tvec)
    V = solve_slip_rate(T, θ, σ̄, η, p; kwargs...)
    return V * Tvec / T
end

"""
    initial_state_from_strength(V, F_target, σ̄, p::FrictionParams)

Closed-form inverse of `friction_coefficient` for `θ` at slip rate `V` and
target strength `F_target` — the initial state per BP8 eq. 28-29, with
`V=V_init`, `F_target=τ_init`:
```
θ = (Dc/V*)·exp( (a·ln(2V*·sinh(f/a)/V) - f*) / b ),   f = F_target/σ̄
```
"""
function initial_state_from_strength(V, F_target, σ̄, p::FrictionParams)
    f = F_target / σ̄
    return (p.Dc / p.V_star) * exp((p.a * log(2p.V_star * sinh(f / p.a) / V) - p.f_star) / p.b)
end

end # module RateStateFriction

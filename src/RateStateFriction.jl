module RateStateFriction

using StaticArrays
using LinearAlgebra: norm

export FrictionParams, radiation_damping_coefficient, friction_coefficient,
       fault_strength, aging_law_rhs, solve_slip_rate, solve_slip_velocity,
       initial_state_from_strength

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
the seismic slip rates where it first appears. Throws if the iteration has
not converged within `maxiter` steps rather than returning a bad root.
"""
function solve_slip_rate(T, θ, σ̄, η, p::FrictionParams; V0=nothing, tol=1e-12, maxiter=50, maxstep=5.0)
    C = exp((p.f_star + p.b * log(p.V_star * θ / p.Dc)) / p.a)
    x = log(V0 === nothing ? p.V_star : V0)
    converged = false
    for _ in 1:maxiter
        V = exp(x)
        u = V / (2p.V_star) * C
        g = η * V + σ̄ * p.a * asinh(u) - T
        dgdx = η * V + σ̄ * p.a * u / sqrt(1 + u^2)
        dx = clamp(-g / dgdx, -maxstep, maxstep)
        x += dx
        if abs(dx) < tol
            converged = true
            break
        end
    end
    converged || error("solve_slip_rate failed to converge in $maxiter iterations " *
                       "(T=$T, θ=$θ, σ̄=$σ̄, η=$η, V0=$V0, last V=$(exp(x)))")
    return exp(x)
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

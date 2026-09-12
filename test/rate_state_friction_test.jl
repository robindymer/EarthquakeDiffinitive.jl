using EarthquakeDiffinitive
using EarthquakeDiffinitive.RateStateFriction
using StaticArrays, LinearAlgebra
using Test

# BP8-QD-GS parameters (Table 1 of the benchmark description).
const a_rs = 0.016
const b_rs = 0.010
const Dc_rs = 0.5e-3   # m (0.5 mm)
const V_star_rs = 1e-6 # m/s
const f_star_rs = 0.6
const σ̄0 = 25.0e6      # Pa
const τ_init = 14.6e6  # Pa
const V_init = 1e-12   # m/s
const μ_rs = 32.04e9   # Pa
const c_s_rs = 3.464e3 # m/s

const params = FrictionParams(a_rs, b_rs, Dc_rs, V_star_rs, f_star_rs)
const η_rs = radiation_damping_coefficient(μ_rs, c_s_rs)

@testset "RateStateFriction" begin
    @testset "aging law steady state" begin
        for V in (1e-12, 1e-9, 1e-6, 1e-3)
            θ_ss = Dc_rs / V
            @test isapprox(aging_law_rhs(V, θ_ss, params), 0.0, atol=1e-14)
        end
    end

    @testset "fault_strength monotonically increasing in V" begin
        θ = 1e3
        Vs = 10.0 .^ range(-12, -2, length=20)
        Fs = [fault_strength(V, θ, σ̄0, params) for V in Vs]
        @test issorted(Fs)
        @test allunique(Fs)
    end

    @testset "solve_slip_rate: forward round-trip, no radiation damping" begin
        θ = 1e4
        for V_true in (1e-12, 1e-9, 1e-6, 1e-3, 1e-1)
            T = fault_strength(V_true, θ, σ̄0, params)
            V = solve_slip_rate(T, θ, σ̄0, 0.0, params)
            @test isapprox(V, V_true, rtol=1e-8)
        end
    end

    @testset "solve_slip_rate: forward round-trip with radiation damping" begin
        θ = 1e4
        for V_true in 10.0 .^ range(-12, -2, length=15)
            T = η_rs * V_true + fault_strength(V_true, θ, σ̄0, params)
            V = solve_slip_rate(T, θ, σ̄0, η_rs, params)
            @test isapprox(V, V_true, rtol=1e-8)
        end
    end

    @testset "solve_slip_velocity: direction and magnitude" begin
        θ = 1e4
        τ0 = SVector(3.0e6, -1.5e6)
        Δτ = SVector(0.5e6, 2.0e6)
        V_vec = solve_slip_velocity(τ0, Δτ, θ, σ̄0, η_rs, params)

        Tvec = τ0 + Δτ
        # direction matches τ0+Δτ
        @test isapprox(V_vec / norm(V_vec), Tvec / norm(Tvec), rtol=1e-10)

        # magnitude matches the scalar solve
        T = norm(Tvec)
        V_scalar = solve_slip_rate(T, θ, σ̄0, η_rs, params)
        @test isapprox(norm(V_vec), V_scalar, rtol=1e-10)
    end

    @testset "initial_state_from_strength: eq 28-29 consistency" begin
        θ_init = initial_state_from_strength(V_init, τ_init, σ̄0, params)
        @test θ_init > 0
        F = fault_strength(V_init, θ_init, σ̄0, params)
        @test isapprox(F, τ_init, rtol=1e-10)
    end

    @testset "solve_slip_rate: robust when seeded far below the root" begin
        # Undamped Newton overshoots into `exp` overflow here and returns NaN
        # or a value wrong by many orders of magnitude, without erroring. The
        # regime (V ≳ 1 m/s) is beyond what this velocity-strengthening
        # benchmark should reach, but a warm start during acceleration always
        # approaches from below, so it stays reachable in a time loop.
        for V_true in (1e-1, 1.0, 10.0, 100.0), σ̄ in (25.0e6, 10.0e6, 1.0e6)
            θ = Dc_rs / V_true                     # steady state
            T = η_rs * V_true + fault_strength(V_true, θ, σ̄, params)
            @test isapprox(solve_slip_rate(T, θ, σ̄, η_rs, params), V_true, rtol=1e-8)
            @test isapprox(solve_slip_rate(T, θ, σ̄, η_rs, params; V0=1e-8 * V_true),
                           V_true, rtol=1e-8)
        end
    end

    @testset "solve_slip_rate: reports non-convergence instead of a bad root" begin
        # The root sits 14 e-folds below `V_star`, so one clamped Newton step
        # (at most 5) cannot reach it from *either* seed — the warm start or
        # the `V_star` fallback a failed warm start is retried from.
        V_true = 1e-12
        θ = Dc_rs / V_true
        T = η_rs * V_true + fault_strength(V_true, θ, σ̄0, params)
        @test_throws ErrorException solve_slip_rate(T, θ, σ̄0, η_rs, params; V0=1e-20, maxiter=1)
        # `onfail=:nan` is the implicit-integrator contract: a trial state the
        # Newton iteration is about to discard must produce a rejectable NaN,
        # not an exception. Non-finite inputs take the same exit.
        @test isnan(solve_slip_rate(T, θ, σ̄0, η_rs, params; V0=1e-20, maxiter=1, onfail=:nan))
        @test isnan(solve_slip_rate(NaN, θ, σ̄0, η_rs, params; onfail=:nan))
        @test isnan(solve_slip_rate(T, Inf, σ̄0, η_rs, params; onfail=:nan))
        # ...and the default still converges on the same inputs.
        @test isapprox(solve_slip_rate(T, θ, σ̄0, η_rs, params; onfail=:nan), V_true, rtol=1e-8)
    end

    @testset "solve_slip_rate: a bad warm start is retried, not fatal" begin
        # An implicit integrator's rejected trial states leave arbitrary seeds
        # behind in the warm-start cache. A seed from which Newton cannot
        # converge (here: `exp` overflow territory) must fall back to the
        # `V_star` start and still return the root.
        V_true = 1e-6
        θ = Dc_rs / V_true
        T = η_rs * V_true + fault_strength(V_true, θ, σ̄0, params)
        for V0 in (1e300, 1e-300, Inf, NaN)
            @test isapprox(solve_slip_rate(T, θ, σ̄0, η_rs, params; V0), V_true, rtol=1e-8)
        end
    end

    @testset "slip_rate_derivatives match finite differences of the solve" begin
        # Implicit differentiation of the force balance against central
        # differences of `solve_slip_rate` itself, across the slip-rate and
        # effective-stress range BP8 spans — including the σ̄ = 1 kPa floor
        # where `1/D` is the stiff eigenvalue's scale (PROGRESS.md "what the
        # stiff eigenvalue is"). Tolerance is the FD truncation error at these
        # steps, not the formula's accuracy.
        for (V, σ̄) in ((1e-12, 25e6), (1e-9, 5e6), (1e-6, 1e3), (1e-3, 1e3), (1.0, 25e6))
            θ = 3Dc_rs / V
            T = η_rs * V + fault_strength(V, θ, σ̄, params)
            d = slip_rate_derivatives(V, θ, σ̄, η_rs, params)
            h = 1e-6 * T
            fd_T = (solve_slip_rate(T + h, θ, σ̄, η_rs, params; V0=V) -
                    solve_slip_rate(T - h, θ, σ̄, η_rs, params; V0=V)) / 2h
            hϕ = 1e-6
            fd_ϕ = (solve_slip_rate(T, θ * exp(hϕ), σ̄, η_rs, params; V0=V) -
                    solve_slip_rate(T, θ * exp(-hϕ), σ̄, η_rs, params; V0=V)) / 2hϕ
            @test isapprox(d.dV_dT, fd_T; rtol=1e-7)
            @test isapprox(d.dV_dϕ, fd_ϕ; rtol=1e-7)
            @test d.dV_dT > 0      # stronger drive, faster slip
            @test d.dV_dϕ < 0      # older contact (larger θ), stronger fault, slower slip
        end
        # `V → 0` is a legitimate input (locked nodes) and must not divide by zero.
        d0 = slip_rate_derivatives(0.0, 1e6, σ̄0, η_rs, params)
        @test isfinite(d0.dV_dT) && d0.dV_dϕ == 0
    end

    @testset "solve_slip_velocity: zero trial stress gives zero slip rate" begin
        V_vec = solve_slip_velocity(SVector(0.0, 0.0), SVector(0.0, 0.0),
                                    1e4, σ̄0, η_rs, params)
        @test all(iszero, V_vec)
    end
end

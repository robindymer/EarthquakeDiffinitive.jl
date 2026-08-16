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
end

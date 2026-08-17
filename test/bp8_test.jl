using EarthquakeDiffinitive
using EarthquakeDiffinitive.BP8
using EarthquakeDiffinitive.FaultResponse
using EarthquakeDiffinitive.RateStateFriction
using LinearAlgebra, StaticArrays
using SpecialFunctions
using Test

# The coupled model is expensive to build (a 3D elastic factorization plus one
# back-substitution per fault DOF), so the whole testset shares one coarse
# model. Δz = 100 m does not resolve the process zone — these tests check that
# the coupling, conventions and file formats are right, not that the numbers
# are benchmark-converged.
const M_GS = build_model(; Δz=100.0, L_fault=800.0, L_normal=800.0)

@testset "BP8" begin
    @testset "derived parameters match Table 1" begin
        p = benchmark_parameters()
        # ν = 0.25 makes λ = μ.
        @test EarthquakeDiffinitive.BP8.lame_lambda(p) ≈ p.μ rtol = 1e-12
        # c_s = sqrt(μ/ρ) should reproduce the tabulated 3.464 km/s.
        @test sqrt(p.μ / p.ρ) ≈ p.c_s rtol = 1e-3
        # α = k/(φβη) should reproduce the tabulated 0.05 m²/s.
        @test p.k / (p.φ * p.β * p.viscosity) ≈ p.α rtol = 1e-12
    end

    @testset "fault stiffness: slip relieves the stress driving it" begin
        K = M_GS.K
        nf = M_GS.nf
        @test size(K) == (2nf, 2nf)
        # Self-stiffness must be negative: unit slip at a node reduces the
        # shear traction there. A sign error here inverts the whole feedback.
        @test all(<(0), diag(K))
        # Reciprocity: the elastic stiffness of a self-adjoint problem is
        # symmetric. The residual is the known interface-SAT asymmetry.
        @test norm(K - K') / norm(K) < 0.01
    end

    @testset "initial conditions reproduce eq. 28-29 exactly" begin
        p = M_GS.par
        u0 = initial_state(M_GS)
        c = evaluate!(M_GS, u0, 0.0)
        act = M_GS.active
        # Uniform initial slip rate (V_init, V_zero) everywhere it is imposed.
        @test all(≈(p.V_init; rtol=1e-8), c.V2[act])
        @test all(≈(p.V_zero; rtol=1e-8), c.V3[act])
        # Uniform initial shear traction of magnitude τ_init.
        @test all(≈(p.τ_init; rtol=1e-8), hypot.(c.τ2[act], c.τ3[act]))
        # Zero slip and zero pressure change (eq. 26).
        @test all(iszero, u0[1:2M_GS.nf])
        @test all(iszero, u0[3M_GS.nf+1:4M_GS.nf])
    end

    @testset "eq. 13: locked outside the frictional domain" begin
        u0 = initial_state(M_GS)
        du = similar(u0)
        EarthquakeDiffinitive.BP8.rhs!(du, u0, M_GS, 0.0)
        nf = M_GS.nf
        locked = .!M_GS.active
        @test any(locked)
        @test all(iszero, du[1:nf][locked])       # ds2/dt = 0
        @test all(iszero, du[nf+1:2nf][locked])   # ds3/dt = 0
    end

    @testset "exponential integral against SpecialFunctions" begin
        E1 = EarthquakeDiffinitive.BP8.expint_e1
        # `expint_e1` is hand-rolled to avoid a dependency, so check it against
        # the real thing across both branches of its implementation.
        for x in (1e-6, 1e-3, 0.1, 0.5, 0.9, 1.0, 1.5, 2.0, 5.0, 10.0, 30.0, 100.0)
            @test E1(x) ≈ SpecialFunctions.expint(x) rtol = 1e-12
        end
        # Continuity across the series/continued-fraction switch at x = 1.
        @test E1(1 - 1e-9) ≈ E1(1 + 1e-9) rtol = 1e-8
        @test_throws DomainError E1(-1.0)
    end

    @testset "analytic pressure solutions are self-consistent" begin
        p = benchmark_parameters()
        # Eq. 21 at r = 0 is the closed form; approaching r → 0 must match it.
        @test analytic_pressure_gaussian(1e-4, 3600.0, p) ≈
              analytic_pressure_gaussian(0.0, 3600.0, p) rtol = 1e-6
        # Both solutions decay with distance and grow with time.
        @test analytic_pressure_gaussian(100.0, 3600.0, p) >
              analytic_pressure_gaussian(200.0, 3600.0, p) > 0
        @test analytic_pressure_point(100.0, 7200.0, p) >
              analytic_pressure_point(100.0, 3600.0, p) > 0
        @test analytic_pressure_gaussian(50.0, 0.0, p) == 0
        # Eq. 21 is exactly eq. 25 given a head start: smearing the source over
        # L_gauss is the same as letting a point source diffuse for
        # t0 = L_gauss²/(2α) first, minus that head start's own profile so
        # p(r,0) = 0. This is an identity, not an asymptotic match — the two
        # are nowhere near each other at equal t, since the Gaussian's tails
        # are already spread when the point source's are still empty.
        t0 = p.L_gauss^2 / (2p.α)
        for (r, t) in ((50.0, 3600.0), (200.0, 3600.0), (400.0, 3600.0), (300.0, 1e5))
            @test analytic_pressure_gaussian(r, t, p) ≈
                  analytic_pressure_point(r, t + t0, p) -
                  analytic_pressure_point(r, t0, p) rtol = 1e-10
        end
    end

    @testset "coupled pressure follows the eq. 21 analytic solution" begin
        # Δz = 100 m does not resolve L_gauss = 50 m, so the near-source value
        # is off; away from both the source and the no-flux edges it should
        # still track the unbounded analytic solution.
        t = 50 * 3600.0
        sol = run_bp8(M_GS; tspan=(0.0, t), saveat=t)
        nf = M_GS.nf
        p = sol.u[end][3nf+1:4nf]
        n2 = length(M_GS.x2)
        j0 = argmin(abs.(M_GS.x3))
        checked = 0
        for i in eachindex(M_GS.x2)
            r = abs(M_GS.x2[i])
            (r < 100 || r > 200) && continue
            @test p[i+(j0-1)*n2] ≈ analytic_pressure_gaussian(r, t, M_GS.par) rtol = 0.12
            checked += 1
        end
        @test checked > 0
    end

    @testset "Peaceman well conserves injected volume" begin
        m = build_model(; Δz=100.0, L_fault=800.0, L_normal=800.0, injection=:peaceman)
        t = 20 * 3600.0
        sol = run_bp8(m; tspan=(0.0, t), saveat=t)
        nf = m.nf
        p = sol.u[end][3nf+1:4nf]
        # Fluid stored in the fault plus fluid stored in the well must equal
        # the total injected volume: no-flux edges let nothing escape.
        in_fault = sum(m.weights .* p) * m.par.L_fwid * m.par.φ * m.par.β
        in_well = m.par.S_well * sol.u[end][end]
        @test in_fault + in_well ≈ m.par.Q0 * t rtol = 1e-3
    end

    @testset "Gaussian source conserves injected volume" begin
        t = 20 * 3600.0
        sol = run_bp8(M_GS; tspan=(0.0, t), saveat=t)
        nf = M_GS.nf
        p = sol.u[end][3nf+1:4nf]
        par = M_GS.par
        stored = sum(M_GS.weights .* p) * par.L_fwid * par.φ * par.β

        # Two separate properties, worth not conflating.
        # 1. The Neumann-SAT Laplacian is conservative: with no-flux edges,
        #    everything the source puts in stays in. Compare against the
        #    *discretely* integrated source, so this is exact to solver
        #    tolerance regardless of how well the Gaussian is resolved.
        source_integral = sum(M_GS.weights .* M_GS.source)
        @test stored ≈ source_integral * par.Q0 * t rtol = 1e-4
        # 2. That discrete integral approximates 1 (eq. 19 is normalized).
        #    Δz = 100 m samples L_gauss = 50 m at twice its width, so trapezoid
        #    aliasing puts this ~3% high — it shrinks fast with resolution.
        @test source_integral ≈ 1.0 rtol = 0.05
    end

    @testset "slip accelerates as injection weakens the fault" begin
        t = 40 * 3600.0
        sol = run_bp8(M_GS; tspan=(0.0, t), saveat=t)
        c = evaluate!(M_GS, sol.u[end], t)
        # Started at V_init = 1e-12 m/s; pressure has risen, so the fault must
        # be slipping faster, and slip must be positive (right-lateral, the
        # direction of τ⁰).
        @test maximum(c.Vmag) > 1e-10
        @test sol.u[end][argmax(c.Vmag)] > 0
        @test maximum(sol.u[end][3M_GS.nf+1:4M_GS.nf]) > 1e6
    end

    @testset "resolution report flags the under-resolved process zone" begin
        rr = resolution_report(M_GS)
        # L_b = μ D_RS/(b σ̄) ≈ 64 m at σ̄ = 25 MPa.
        @test rr.L_b ≈ 32.04e9 * 0.5e-3 / (0.010 * 25e6) rtol = 1e-12
        @test rr.L_b ≈ 64.08 rtol = 1e-3
        @test rr.Δz == 100.0
        @test !rr.converged           # Δz = 100 m is coarser than L_b itself
        @test rr.cells_per_L_gauss ≈ 0.5
        # L_b grows as injection reduces σ̄, so the initial σ̄ is the binding case.
        @test process_zone(M_GS.par, 10e6) > process_zone(M_GS.par, 25e6)
    end

    @testset "output files have the structure §4 specifies" begin
        t = 5 * 3600.0
        sol = run_bp8(M_GS; tspan=(0.0, t), saveat=1800.0)
        dir = mktempdir()
        write_outputs(M_GS, sol, dir; modeler="test", profile_dt=1800.0)

        files = readdir(dir)
        @test length(station_locations()) == 9
        for (name, _, _) in station_locations()
            @test "$name.dat" ∈ files
        end
        @test "global.dat" ∈ files
        for q in ("slip_2", "slip_3", "shear_stress_2", "shear_stress_3", "pore_pressure"),
            l in ("strike", "depth")
            @test "$(q)_$(l).dat" ∈ files
        end

        datarows(f) = [split(l) for l in eachline(joinpath(dir, f))
                       if !startswith(l, "#") && !isempty(strip(l)) &&
                          tryparse(Float64, first(split(l))) !== nothing]

        # §4.1: 11 fields per row, listed on one line.
        ts = datarows("fltst_strk+000dp+000.dat")
        @test !isempty(ts)
        @test all(r -> length(r) == 11, ts)
        @test issorted(parse.(Float64, first.(ts)))   # increasing time
        @test occursin("t slip_2 slip_3 slip_rate_2",
                       read(joinpath(dir, "fltst_strk+000dp+000.dat"), String))

        # §4.2: three fields.
        @test all(r -> length(r) == 3, datarows("global.dat"))

        # §4.3: (N_t+1) × (N_coord+2), first row is `0 0 <coordinates>`.
        prof = datarows("slip_2_strike.dat")
        ncoord = length(M_GS.x2)
        @test all(r -> length(r) == ncoord + 2, prof)
        @test parse.(Float64, prof[1][1:2]) == [0.0, 0.0]
        @test parse.(Float64, prof[1][3:end]) ≈ M_GS.x2
        # Field list is four separate lines for the profiles.
        @test occursin("\nx2\nt\nmax_slip_rate\nslip_2\n",
                       read(joinpath(dir, "slip_2_strike.dat"), String))
    end
end

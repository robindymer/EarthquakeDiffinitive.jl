using EarthquakeDiffinitive
using EarthquakeDiffinitive.BP8
using EarthquakeDiffinitive.FaultResponse
using EarthquakeDiffinitive.RateStateFriction
using LinearAlgebra, StaticArrays, SparseArrays
using SpecialFunctions
using OrdinaryDiffEq: Tsit5, ReturnCode
using Test

# The coupled model is expensive to build (a 3D elastic factorization plus one
# back-substitution per fault DOF), so the whole testset shares one coarse
# model. Δz = 100 m does not resolve the process zone — these tests check that
# the coupling, conventions and file formats are right, not that the numbers
# are benchmark-converged.
const M_GS = build_model(; Δz=100.0, L_fault=800.0, L_normal=800.0)

# The Peaceman variant at Δz = 50 m, 100 h — the configuration PROGRESS.md's
# limitation-3 table was measured on, and the coarsest at which the well cell
# actually reaches the σ̄ floor and makes the problem stiff. One explicit
# (Tsit5) reference solution serves both the Jacobian check and the implicit
# integrator comparison; it is the expensive part (~29,000 steps, ~15 s).
const M_PW = build_model(; Δz=50.0, L_fault=800.0, L_normal=400.0, injection=:peaceman,
                         stiffness=:toeplitz)
const PW_TSPAN = (0.0, 100 * 3600.0)
const PW_REF = run_bp8(M_PW; tspan=PW_TSPAN, saveat=3600.0, alg=Tsit5())

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
        # Zero slip (eq. 26). Pressure is no longer part of the integrated
        # state — it lives in the separately solved history — so eq. 27's zero
        # initial pressure change is checked there instead.
        @test all(iszero, u0[1:2M_GS.nf])
        @test length(u0) == 3M_GS.nf
        @test all(iszero, pressure_at!(M_GS, 0.0))
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
        run_bp8(M_GS; tspan=(0.0, t), saveat=t)
        p = pressure_at!(M_GS, t)
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
        run_bp8(m; tspan=(0.0, t), saveat=t)
        p = pressure_at!(m, t)
        # Fluid stored in the fault plus fluid stored in the well must equal
        # the total injected volume: no-flux edges let nothing escape. This is
        # the conservation check on the `[p; p_well]` subsystem that
        # `solve_pressure_history` now integrates on its own, so it also pins
        # that the split did not drop the well coupling.
        in_fault = sum(m.weights .* p) * m.par.L_fwid * m.par.φ * m.par.β
        in_well = m.par.S_well * well_pressure(m, t)
        @test in_fault + in_well ≈ m.par.Q0 * t rtol = 1e-3
    end

    @testset "Gaussian source conserves injected volume" begin
        t = 20 * 3600.0
        run_bp8(M_GS; tspan=(0.0, t), saveat=t)
        p = pressure_at!(M_GS, t)
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
        @test maximum(pressure_at!(M_GS, t)) > 1e6
    end

    @testset "state_jacobian! matches finite differences of rhs!" begin
        # The block-diagonal Jacobian is what makes an implicit integrator
        # affordable for BP8-PW (see the comment block above `state_jacobian!`).
        # Its promise is that every per-node 3×3 block equals the corresponding
        # block of the true `∂rhs!/∂u` — the *off*-diagonal blocks are dropped
        # by design and are not checked here. Taken mid-injection on the
        # Peaceman model so a node is actually at the σ̄ floor, where `1/D` is
        # large and the block is far from trivial.
        #
        # FD step: at the floored node `V ~ exp(τ/(aσ̄))` with `aσ̄ = 16 Pa`, so
        # a slip perturbation of `1e-5·|s|` already moves traction by several
        # Pa and the difference quotient is dominated by curvature; `1e-8`
        # relative is where it has converged onto the analytic value.
        m = M_PW
        @test effective_stress_report(m).nodes ≥ 1
        j85 = findfirst(==(85 * 3600.0), PW_REF.t)
        u = copy(PW_REF.u[j85]); t = PW_REF.t[j85]
        nf = m.nf; N = 3nf

        J = state_jacobian_prototype(m)
        @test size(J) == (N, N) && nnz(J) == 9nf
        state_jacobian!(J, u, m, t)

        fp = zeros(N); fm = zeros(N)
        function fd_column!(col, j)
            h = 1e-8 * (j <= 2nf ? max(abs(u[j]), 1e-5) : 1.0)
            up = copy(u); up[j] += h; BP8.rhs!(fp, up, m, t)
            um = copy(u); um[j] -= h; BP8.rhs!(fm, um, m, t)
            col .= (fp .- fm) ./ 2h
        end
        col = zeros(N)
        worst = 0.0
        for i in 1:nf
            idx = (i, nf + i, 2nf + i)
            A = zeros(3, 3); B = zeros(3, 3)
            for (c, j) in enumerate(idx)
                fd_column!(col, j)
                for (r, k) in enumerate(idx)
                    A[r, c] = J[k, j]; B[r, c] = col[k]
                end
            end
            scale = maximum(abs, B)
            scale == 0 && continue
            worst = max(worst, maximum(abs, A .- B) / scale)
        end
        @test worst < 1e-4

        # Locked ring: only the aging law's own `-e^{-ϕ}` on the diagonal.
        i = findfirst(!, m.active)
        @test J[i, i] == 0 && J[nf + i, nf + i] == 0
        @test J[2nf + i, 2nf + i] ≈ -exp(-u[2nf + i])
    end

    @testset "implicit integration of BP8-PW agrees with the explicit reference" begin
        # The point of the Jacobian: a stiff solver takes far fewer steps than
        # `Tsit5` on the Peaceman variant while landing on the same answer.
        # Δz = 50 m, 100 h is the configuration PROGRESS.md's limitation-3
        # table was measured on (Tsit5 ~29,000 steps).
        m = M_PW
        ref = PW_REF
        @test default_integrator(m) isa EarthquakeDiffinitive.BP8.QNDF
        sol = run_bp8(m; tspan=PW_TSPAN, saveat=3600.0)          # the PW default
        @test sol.retcode == ReturnCode.Success
        @test sol.stats.naccept < ref.stats.naccept / 5
        nf = m.nf
        vmax(s) = maximum(maximum(evaluate!(m, s.u[j], s.t[j]).Vmag) for j in eachindex(s.t))
        @test isapprox(vmax(sol), vmax(ref); rtol=1e-5)
        @test maximum(abs, sol.u[end][1:2nf] .- ref.u[end][1:2nf]) <
              1e-6 * maximum(abs, ref.u[end][1:2nf])
        @test maximum(abs, sol.u[end][2nf+1:end] .- ref.u[end][2nf+1:end]) < 1e-4
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

#!/usr/bin/env julia
#
# 2D elastic wave equation simulation using EarthquakeDiffinitive.Elasticity's
# constant-coefficient isotropic Navier operator, in the same spirit as
# context/notebooks/elastic.jl's operator tests: here used to qualitatively
# validate the operator by watching it actually propagate P- and S-waves,
# rather than checking pointwise accuracy against polynomials.
#
# ρ ∂²u/∂t² = ∇·σ = E*u, leapfrog (central-difference) time integration,
# clamped (injected u=0) boundaries, animated with CairoMakie.
#
# Run with: julia --project=scripts scripts/elastic_wave_2d.jl

using EarthquakeDiffinitive
using EarthquakeDiffinitive.Elasticity
using Diffinitive.Grids
using Diffinitive.SbpOperators
using StaticArrays
using LinearAlgebra
using CairoMakie

gaussian(x; x0, σ) = exp(-sum(abs2, x - x0) / (2σ^2))
mag(u) = map(norm, u)
u1(u) = map(x -> x[1], u)

function clamp_boundary!(u, g, boundaries)
    for bid in boundaries
        u[boundary_indices(g, bid)] .= Ref(SVector(0.0, 0.0))
    end
end

function main()
    # ---- Material parameters (constant, isotropic) ----
    λ, μ, ρ = 1.0, 0.05, 1.0
    cp = sqrt((λ + 2μ) / ρ)   # P-wave (dilatational) speed
    cs = sqrt(μ / ρ)          # S-wave (shear) speed
    println("P-wave speed cp = $cp, S-wave speed cs = $cs")

    # ---- Grid and operator ----
    stencil_set = read_stencil_set(SbpOperators.sbp_operators_path() * "standard_diagonal.toml", order=4)
    n = 161
    g = equidistant_grid((0.0, 0.0), (2.0, 2.0), n, n)
    E = elastic_operator(g, λ, μ, stencil_set)
    boundaries = boundary_identifiers(g)

    # ---- Initial condition: a Gaussian pulse in the x-displacement, at
    # rest. Not purely irrotational or divergence-free, so it excites both
    # P- and S-waves — the two wavefronts separating as they propagate is
    # the main visual check that the operator is doing the right physics.
    x0 = SVector(1.0, 1.0)
    σ0 = 0.06
    u_prev = map(x -> SVector(gaussian(x; x0, σ=σ0), 0.0), g)
    v0 = map(x -> SVector(0.0, 0.0), g)
    clamp_boundary!(u_prev, g, boundaries)

    # ---- Leapfrog time stepping ----
    Δt = 0.3 * min_spacing(g) / cp
    n_frames = 150
    steps_per_frame = 6
    n_steps = n_frames * steps_per_frame
    println("Δt = $Δt, $n_steps steps total (t_final ≈ $(round(n_steps * Δt, digits=3)))")

    # Second-order-accurate leapfrog start: u₁ = u₀ + Δt v₀ + ½Δt² a₀.
    a0 = (E * u_prev) ./ ρ
    u_curr = u_prev .+ Δt .* v0 .+ (Δt^2 / 2) .* a0
    clamp_boundary!(u_curr, g, boundaries)

    u_next = similar(u_curr)
    Eu = similar(u_curr)

    # ---- Animate and save ----
    fig = Figure(size=(700, 650))
    ax = Axis(fig[1, 1]; aspect=DataAspect())
    frame = Observable(u1(u_curr))
    plt = plot!(ax, g, frame; colorrange=(-0.3, 0.3), colormap=:balance)
    Colorbar(fig[1, 2], plt)

    t = 0.0
    snapshot_times = (0.2, 0.5, 0.8, 1.2)
    snapshot_idx = 1
    outpath = joinpath(@__DIR__, "elastic_wave_2d.mp4")
    record(fig, outpath, 1:n_frames; framerate=25) do _
        for _ in 1:steps_per_frame
            Eu .= E * u_curr
            @. u_next = 2 * u_curr - u_prev + (Δt^2 / ρ) * Eu
            clamp_boundary!(u_next, g, boundaries)
            u_prev, u_curr, u_next = u_curr, u_next, u_prev
            t += Δt
        end
        frame[] = u1(u_curr)
        ax.title = "Elastic wave u₁,  t = $(round(t, digits=3))  (cp=$(round(cp, digits=2)), cs=$(round(cs, digits=2)))"

        if snapshot_idx <= length(snapshot_times) && t >= snapshot_times[snapshot_idx]
            png_path = joinpath(@__DIR__, "elastic_wave_2d_t$(snapshot_times[snapshot_idx]).png")
            save(png_path, fig)
            println("Saved snapshot to $png_path")
            snapshot_idx += 1
        end
    end

    println("Saved animation to $outpath")
end

main()

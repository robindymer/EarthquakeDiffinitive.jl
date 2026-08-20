# End-to-end validation of the block-Toeplitz K reconstruction.
#
# `k_toeplitz_structure.jl` shows 4 corner sources (8 CG solves) rebuild K to
# ~0.03% in Frobenius norm. That is a statement about the MATRIX. What matters
# is the SOLUTION: K feeds a friction law where V ~ exp(τ/aσ̄), so a small matrix
# error is amplified — measured at ~370× here. This sweeps the number of source
# columns and reports what each buys in the outputs the benchmark asks for.
#
# Run:  julia --project=. -t auto scripts/k_toeplitz_validate.jl [Δz] [L_fault] [L_normal]
using EarthquakeDiffinitive
using EarthquakeDiffinitive.BP8
using Printf, Statistics, LinearAlgebra

Δz       = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 50.0
L_fault  = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 800.0
L_normal = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 400.0
# Hours. 100 h is through the injection period; 720 h is the benchmark's full
# 30 days, which is what the submitted V_max(t) series actually covers — the
# approximation has to hold over the shut-in and relaxation phases too, not
# just while injection is driving the system.
const T_END = (length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 100.0) * 3600.0

@info "building model" Δz L_fault L_normal
m = build_model(; Δz, L_fault, L_normal, stiffness=:exact)
const n2, n3, nf = length(m.x2), length(m.x3), m.nf
node_ab(k) = (mod1(k, n2), (k - 1) ÷ n2 + 1)
corner(a, b) = (b - 1) * n2 + a

function toeplitz_block(B, sources)
    acc = Dict{Tuple{Int,Int},Vector{Float64}}()
    for j in sources
        aj, bj = node_ab(j)
        for i in 1:nf
            ai, bi = node_ab(i)
            push!(get!(acc, (ai - aj, bi - bj), Float64[]), B[i, j])
        end
    end
    kmean = Dict(k => mean(v) for (k, v) in acc)
    out = zeros(nf, nf)
    for j in 1:nf, i in 1:nf
        ai, bi = node_ab(i); aj, bj = node_ab(j)
        out[i, j] = get(kmean, (ai - aj, bi - bj), 0.0)
    end
    return out
end

function rebuild_K(K, sources)
    Kt = similar(K)
    Kt[1:nf, 1:nf]         = toeplitz_block(K[1:nf, 1:nf], sources)
    Kt[nf+1:2nf, nf+1:2nf] = toeplitz_block(K[nf+1:2nf, nf+1:2nf], sources)
    Kt[1:nf, nf+1:2nf]     = toeplitz_block(K[1:nf, nf+1:2nf], sources)
    Kt[nf+1:2nf, 1:nf]     = toeplitz_block(K[nf+1:2nf, 1:nf], sources)
    return Kt
end

with_K(Kt) = BP8.BP8Model(m.par, Kt, m.Ap, m.source, m.Q2, m.Q3, m.weights,
                          m.active, m.x2, m.x3, m.nf, m.τ0, m.injection,
                          m.well_cell, m.WI, m.S_e, BP8.Cache(m.nf), m.grid_info)

vmax_series(mm, sol) = [maximum(evaluate!(mm, sol.u[j], sol.t[j]).Vmag) for j in eachindex(sol.t)]

# reference run with the exact K
sol_ref = run_bp8(m; tspan=(0.0, T_END), saveat=3600.0)
v_ref = vmax_series(m, sol_ref)
ic = argmin(abs.(m.x2)) + (argmin(abs.(m.x3)) - 1) * length(m.x2)
slip_ref = sol_ref.u[end][ic]
@printf("\nreference (exact K, %d solves) at t = %.0f h: V_max = %.6e, slip(0,0) = %.6e\n",
        2nf, T_END / 3600, v_ref[end], slip_ref)

# Source sets: a k×k grid of sources spread over Ω_f. k=2 is the four corners.
grid_sources(k) = k == 1 ? [corner((n2+1)÷2, (n3+1)÷2)] :
    unique([corner(round(Int, 1 + (n2-1)*(i-1)/(k-1)),
                   round(Int, 1 + (n3-1)*(j-1)/(k-1))) for i in 1:k, j in 1:k])

println("\n", "="^92)
@printf("%-14s %7s  %12s  %12s  %12s  %12s\n",
        "sources", "solves", "K Frob.err", "V_max err", "V_max(t) worst", "slip err")
println("="^92)
for k in 1:6
    srcs = grid_sources(k)
    length(srcs) > nf && continue
    Kt = rebuild_K(m.K, srcs)
    mt = with_K(Kt)
    sol = run_bp8(mt; tspan=(0.0, T_END), saveat=3600.0)
    v = vmax_series(mt, sol)
    n = min(length(v), length(v_ref))
    worst = maximum(100abs.(v[1:n] .- v_ref[1:n]) ./ abs.(v_ref[1:n]))
    @printf("%-14s %7d  %11.4f%%  %11.4f%%  %13.4f%%  %11.4f%%\n",
            "$(k)×$(k) grid", 2length(srcs),
            100norm(Kt - m.K)/norm(m.K),
            100abs(v[end] - v_ref[end])/abs(v_ref[end]),
            worst,
            100abs(sol.u[end][ic] - slip_ref)/abs(slip_ref))
    flush(stdout)
end

# ---------------------------------------------------------------------------
# PRIORITY FILL. The sweep above averages every source's kernel together, which
# lets boundary-contaminated corner columns corrupt the clean near-field one.
# Instead: take the centre source's kernel wherever it reaches, and fall back to
# outer sources ONLY for separations the centre cannot cover. That should keep
# the centre's accuracy while closing its ~44% coverage gap.
# ---------------------------------------------------------------------------
function toeplitz_block_priority(B, ordered_sources)
    kmean = Dict{Tuple{Int,Int},Float64}()
    for j in ordered_sources                      # earlier sources win
        aj, bj = node_ab(j)
        for i in 1:nf
            ai, bi = node_ab(i)
            k = (ai - aj, bi - bj)
            haskey(kmean, k) || (kmean[k] = B[i, j])
        end
    end
    out = zeros(nf, nf)
    for j in 1:nf, i in 1:nf
        ai, bi = node_ab(i); aj, bj = node_ab(j)
        out[i, j] = get(kmean, (ai - aj, bi - bj), 0.0)
    end
    return out
end

function rebuild_K_priority(K, srcs)
    Kt = similar(K)
    Kt[1:nf, 1:nf]         = toeplitz_block_priority(K[1:nf, 1:nf], srcs)
    Kt[nf+1:2nf, nf+1:2nf] = toeplitz_block_priority(K[nf+1:2nf, nf+1:2nf], srcs)
    Kt[1:nf, nf+1:2nf]     = toeplitz_block_priority(K[1:nf, nf+1:2nf], srcs)
    Kt[nf+1:2nf, 1:nf]     = toeplitz_block_priority(K[nf+1:2nf, 1:nf], srcs)
    return Kt
end

const CENTRE = corner((n2+1)÷2, (n3+1)÷2)
priority_sets = (("centre only", [CENTRE]),
                 ("centre + 4 corners", vcat(CENTRE, grid_sources(2))),
                 ("centre + 3×3 ring", unique(vcat(CENTRE, grid_sources(3)))))

println("\nPRIORITY FILL (centre kernel wins; outer sources only fill its gaps):")
@printf("%-22s %7s  %12s  %12s  %12s  %11s\n",
        "sources", "solves", "K Frob.err", "V_max err", "V_max(t) worst", "slip err")
println("-"^92)
for (label, srcs) in priority_sets
    Kt = rebuild_K_priority(m.K, srcs)
    mt = with_K(Kt)
    sol = run_bp8(mt; tspan=(0.0, T_END), saveat=3600.0)
    v = vmax_series(mt, sol)
    n = min(length(v), length(v_ref))
    worst = maximum(100abs.(v[1:n] .- v_ref[1:n]) ./ abs.(v_ref[1:n]))
    @printf("%-22s %7d  %11.4f%%  %11.4f%%  %13.4f%%  %10.4f%%\n",
            label, 2length(srcs), 100norm(Kt - m.K)/norm(m.K),
            100abs(v[end] - v_ref[end])/abs(v_ref[end]), worst,
            100abs(sol.u[end][ic] - slip_ref)/abs(slip_ref))
    flush(stdout)
end

println("="^92)
println("""
The K error is amplified into V_max by ~2-3 orders of magnitude, because
V ~ exp(τ/aσ̄). Slip and moment rate are not amplified. So the question is not
whether K is Toeplitz (it very nearly is) but how many sources are needed
before the AMPLIFIED error is acceptable — compare against the ~53% domain
bias and the resolution error, and note that a full build costs $(2nf) solves
here but 2·N_Ωf ∝ Δz⁻² in general.""")

# Does the Toeplitz K's error grow with resolution?
#
# This is the last open gate on `:toeplitz` (TODO §2). Everything about the
# approximation has been validated at Δz = 50 m; the doubt is whether the
# centre source's ~44% coverage gap — resolution-independent as a FRACTION —
# stays harmless as the grid refines, since the far-field-decay argument behind
# it is not proven to be.
#
# **Matched pair at fixed domain.** Run this at Δz = 50 m and Δz = 25 m with the
# SAME (small) domain and compare. Holding the domain fixed is the point: the
# converged domain at Δz = 25 m is 4.9 M DOF and ~77 h (PERFORMANCE.md §4), and
# comparing a fine small-domain run against a coarse converged-domain one would
# confound resolution with truncation exactly the way the old L_normal
# comparison did.
#
# Reference at Δz = 50 m, small domain, 30 days: 5.7767%.
#
# Run:  julia --project=. -t auto scripts/k_toeplitz_resolution.jl [Δz] [L_fault] [L_normal] [T_hours]
using EarthquakeDiffinitive
using EarthquakeDiffinitive.BP8
using Printf, LinearAlgebra

Δz       = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 25.0
L_fault  = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 800.0
L_normal = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 400.0
const T_END = (length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 720.0) * 3600.0

phase(label, f) = begin
    t = time(); r = f()
    @printf("[%7.1f s] %s\n", time() - t, label); flush(stdout)
    r
end

@info "matched-resolution Toeplitz check" Δz L_fault L_normal T_hours = T_END/3600
m = phase("build_model (assembly + exact K)",
          () -> build_model(; Δz, L_fault, L_normal, stiffness=:exact, verbose=true))

const n2, n3, nf = length(m.x2), length(m.x3), m.nf
node_ab(k) = (mod1(k, n2), (k - 1) ÷ n2 + 1)
node_idx(a, b) = (b - 1) * n2 + a
@printf("Ω_f is %d×%d = %d nodes ⇒ exact build used %d solves\n", n2, n3, nf, 2nf)

# The shipped 5-source priority reconstruction, expressed on an already-built K.
# Identical to fault_stiffness_toeplitz's 10 solves — its sources ARE these
# columns — and the exact K is needed as the reference anyway.
const SOURCES = [node_idx((n2+1)÷2, (n3+1)÷2),          # centre wins wherever it reaches
                 node_idx(1, 1), node_idx(n2, 1),
                 node_idx(1, n3), node_idx(n2, n3)]

function toeplitz_block_priority(B)
    kern = Dict{Tuple{Int,Int},Float64}()
    for j in SOURCES                                     # earlier source wins
        aj, bj = node_ab(j)
        for i in 1:nf
            ai, bi = node_ab(i)
            k = (ai - aj, bi - bj)
            haskey(kern, k) || (kern[k] = B[i, j])
        end
    end
    out = zeros(nf, nf)
    for j in 1:nf, i in 1:nf
        ai, bi = node_ab(i); aj, bj = node_ab(j)
        out[i, j] = get(kern, (ai - aj, bi - bj), 0.0)   # unreachable → 0
    end
    return out
end

K_toep = phase("Toeplitz reconstruction", function ()
    K = similar(m.K)
    K[1:nf, 1:nf]         = toeplitz_block_priority(m.K[1:nf, 1:nf])
    K[nf+1:2nf, nf+1:2nf] = toeplitz_block_priority(m.K[nf+1:2nf, nf+1:2nf])
    K[1:nf, nf+1:2nf]     = toeplitz_block_priority(m.K[1:nf, nf+1:2nf])
    K[nf+1:2nf, 1:nf]     = toeplitz_block_priority(m.K[nf+1:2nf, 1:nf])
    K
end)

with_K(Kt) = BP8.BP8Model(m.par, Kt, m.Ap, m.source, m.Q2, m.Q3, m.weights,
                          m.active, m.x2, m.x3, m.nf, m.τ0, m.injection,
                          m.well_cell, m.WI, m.S_e, BP8.Cache(m.nf), m.grid_info)
vmax_series(mm, sol) = [maximum(evaluate!(mm, sol.u[j], sol.t[j]).Vmag) for j in eachindex(sol.t)]

ic = argmin(abs.(m.x2)) + (argmin(abs.(m.x3)) - 1) * n2
sol_ref = phase("reference run (exact K), 30 d",
                () -> run_bp8(m; tspan=(0.0, T_END), saveat=3600.0))
v_ref = vmax_series(m, sol_ref)

mt = with_K(K_toep)
sol = phase("Toeplitz run, 30 d", () -> run_bp8(mt; tspan=(0.0, T_END), saveat=3600.0))
v = vmax_series(mt, sol)

n = min(length(v), length(v_ref))
worst = maximum(100abs.(v[1:n] .- v_ref[1:n]) ./ abs.(v_ref[1:n]))
println("\n", "="^72)
@printf("Δz = %g m, domain (%g, %g), %.0f days, %d nodes on Ω_f\n",
        Δz, L_fault, L_normal, T_END/86400, nf)
@printf("  K Frobenius error      %10.4f%%\n", 100norm(K_toep - m.K)/norm(m.K))
@printf("  V_max at end           %10.4f%%\n", 100abs(v[end]-v_ref[end])/abs(v_ref[end]))
@printf("  V_max(t) WORST         %10.4f%%   <- the acceptance number\n", worst)
@printf("  slip(0,0) at end       %10.4f%%\n",
        100abs(sol.u[end][ic]-sol_ref.u[end][ic])/abs(sol_ref.u[end][ic]))
println("="^72)
println("""
Compare against Δz = 50 m at this same domain: 5.7767%. If the worst-case
number is comparable, the approximation is resolution-transferable and
:toeplitz is defensible as the production route. If it climbs materially,
it is not, and multi-node parallelism on the exact build is the answer.""")

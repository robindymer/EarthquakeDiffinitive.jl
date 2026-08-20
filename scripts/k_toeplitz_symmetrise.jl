# Does enforcing reciprocity on the Toeplitz K help?
#
# TODO.md §2: the true whole-space kernel is even, kernel(−d) = kernel(d), so the
# exact K is symmetric — measured ~0.2% here. The Toeplitz reconstruction takes
# the kernel from a handful of source columns and does NOT enforce that, so
# K_toep is only as even as the sampled columns happen to be. Replacing it with
# (K + Kᵀ)/2 costs nothing. Whether it HELPS is the open question.
#
# It has to be answered end-to-end, not in Frobenius norm: PERFORMANCE.md §4b's
# central lesson is that matrix-norm error and V_max error are amplified apart by
# 2-3 orders of magnitude and can move in opposite directions. So every variant
# below is run through the full coupled benchmark and scored on worst relative
# error in V_max(t), the quantity §4.2 asks for.
#
# The exact K is built ONCE and every variant is derived from it — the Toeplitz
# sources are literally columns of the exact K, so deriving them is identical to
# running the 10 CG solves, and the reference is needed anyway.
#
# Run:  julia --project=. -t auto scripts/k_toeplitz_symmetrise.jl [Δz] [L_fault] [L_normal] [T_hours]
using EarthquakeDiffinitive
using EarthquakeDiffinitive.BP8
using Printf, LinearAlgebra

Δz       = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 50.0
L_fault  = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 800.0
L_normal = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 400.0
# 720 h = the benchmark's 30 days. §4b: validating through the 100 h injection
# phase alone hid a 97% error, so the default here is the full run.
const T_END = (length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 720.0) * 3600.0

@info "building model" Δz L_fault L_normal T_hours = T_END / 3600
t0 = time()
m = build_model(; Δz, L_fault, L_normal, verbose=true)
@info "exact model ready" seconds = round(time() - t0, digits=1)

const n2, n3, nf = length(m.x2), length(m.x3), m.nf
node_ab(k) = (mod1(k, n2), (k - 1) ÷ n2 + 1)
node_idx(a, b) = (b - 1) * n2 + a

# ---------------------------------------------------------------------------
# The shipped reconstruction, expressed on an already-built K. Identical to
# fault_stiffness_toeplitz's 10 solves: its sources ARE these columns.
# ---------------------------------------------------------------------------
const SOURCES = [node_idx((n2+1)÷2, (n3+1)÷2),          # centre first — it wins
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

function rebuild_K_priority(K)
    Kt = similar(K)
    Kt[1:nf, 1:nf]         = toeplitz_block_priority(K[1:nf, 1:nf])
    Kt[nf+1:2nf, nf+1:2nf] = toeplitz_block_priority(K[nf+1:2nf, nf+1:2nf])
    Kt[1:nf, nf+1:2nf]     = toeplitz_block_priority(K[1:nf, nf+1:2nf])
    Kt[nf+1:2nf, 1:nf]     = toeplitz_block_priority(K[nf+1:2nf, 1:nf])
    return Kt
end

# Full symmetrisation is the reciprocity statement on the whole 2nf×2nf: it
# symmetrises K22 and K33 AND replaces the cross blocks by (K23 + K32ᵀ)/2.
sym_full(K) = (K + K') / 2
# Diagonal-blocks-only, in case the cross-coupling is the contaminated part and
# averaging it in does harm. Cross blocks left exactly as reconstructed.
function sym_diag(K)
    Ks = copy(K)
    Ks[1:nf, 1:nf]         = sym_full(K[1:nf, 1:nf])
    Ks[nf+1:2nf, nf+1:2nf] = sym_full(K[nf+1:2nf, nf+1:2nf])
    return Ks
end

with_K(Kt) = BP8.BP8Model(m.par, Kt, m.Ap, m.source, m.Q2, m.Q3, m.weights,
                          m.active, m.x2, m.x3, m.nf, m.τ0, m.injection,
                          m.well_cell, m.WI, m.S_e, BP8.Cache(m.nf), m.grid_info)

vmax_series(mm, sol) = [maximum(evaluate!(mm, sol.u[j], sol.t[j]).Vmag) for j in eachindex(sol.t)]
asym(K) = 100norm(K - K') / norm(K)

# ---------------------------------------------------------------------------
K_exact = m.K
K_toep  = rebuild_K_priority(K_exact)

# Where the asymmetry actually lives. Reciprocity on the full 2nf×2nf means
# K22 and K33 each symmetric AND K23 = K32ᵀ; those are three separate claims and
# they do not have to fail together. Measured per block rather than inferred from
# the aggregate, because the diagonal-block-only variant below is a no-op if the
# diagonal blocks are already even — and a no-op that LOOKS like a null result is
# worth telling apart from a real one.
function asym_blocks(K)
    K22 = K[1:nf, 1:nf]; K33 = K[nf+1:2nf, nf+1:2nf]
    K23 = K[1:nf, nf+1:2nf]; K32 = K[nf+1:2nf, 1:nf]
    return (full  = asym(K),
            K22   = 100norm(K22 - K22') / norm(K22),
            K33   = 100norm(K33 - K33') / norm(K33),
            cross = 100norm(K23 - K32') / norm(K23))
end

println("\nAsymmetry, before any symmetrisation (% of each block's norm):")
@printf("%-12s %9s %9s %9s %9s\n", "", "full", "K22", "K33", "K23−K32ᵀ")
for (label, K) in (("exact K", K_exact), ("Toeplitz K", K_toep))
    a = asym_blocks(K)
    @printf("%-12s %8.4f%% %8.4f%% %8.4f%% %8.4f%%\n", label, a.full, a.K22, a.K33, a.cross)
end
println("If the Toeplitz K is MORE asymmetric than the exact one, the")
println("reconstruction is discarding reciprocity the sampled columns carried.")

sol_ref = run_bp8(m; tspan=(0.0, T_END), saveat=3600.0)
v_ref = vmax_series(m, sol_ref)
ic = argmin(abs.(m.x2)) + (argmin(abs.(m.x3)) - 1) * n2
slip_ref = sol_ref.u[end][ic]
@printf("\nreference (exact K, %d solves), t = %.0f h: V_max = %.6e, slip(0,0) = %.6e\n",
        2nf, T_END / 3600, v_ref[end], slip_ref)

variants = (("Toeplitz (shipped)",      K_toep),
            ("Toeplitz + sym full",     sym_full(K_toep)),
            ("Toeplitz + sym diag only", sym_diag(K_toep)),
            # Control: symmetrising the EXACT K. Bounds what reciprocity
            # enforcement can do on its own, with no reconstruction error in
            # play — if this alone moves V_max by more than the Toeplitz error,
            # symmetrisation is not a free correction whatever it does above.
            ("exact + sym full",        sym_full(K_exact)))

println("\n", "="^94)
@printf("%-26s %12s %12s  %12s  %14s  %11s\n",
        "K variant", "asymmetry", "K Frob.err", "V_max err", "V_max(t) worst", "slip err")
println("="^94)
for (label, Kv) in variants
    sol = run_bp8(with_K(Kv); tspan=(0.0, T_END), saveat=3600.0)
    v = vmax_series(with_K(Kv), sol)
    n = min(length(v), length(v_ref))
    worst = maximum(100abs.(v[1:n] .- v_ref[1:n]) ./ abs.(v_ref[1:n]))
    @printf("%-26s %11.4f%% %11.4f%%  %11.4f%%  %13.4f%%  %10.4f%%\n",
            label, asym(Kv), 100norm(Kv - K_exact) / norm(K_exact),
            100abs(v[end] - v_ref[end]) / abs(v_ref[end]), worst,
            100abs(sol.u[end][ic] - slip_ref) / abs(slip_ref))
    flush(stdout)
end
println("="^94)
println("""
Accept a change only on the 'V_max(t) worst' column, and only at 30 days —
the K Frob.err column is shown to be compared against it, not trusted.""")

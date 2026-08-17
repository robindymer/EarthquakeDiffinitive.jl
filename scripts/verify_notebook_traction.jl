# Does the notebook's traction operator preserve BP8's interface conditions?
#
# `scripts/symmetry_decomposition.jl` shows that restoring the narrow/wide split
# in the traction operator (the way `context/notebooks/elastic_clean.jl`'s
# `IsotropicTractionOperator` does) makes `-HP(D+SAT)P` symmetric. Symmetry is
# worthless if it costs the physics, so this script re-runs
# `test/elasticity_split_node_test.jl`'s interface checks against BOTH traction
# operators and compares, plus the slip→traction stiffness `K`.
#
# `elastic_blocks` is NOT touched by this fix, so `D` is identical in both
# columns; only the SAT's traction operator and the traction extraction change.
#
# Run with:  julia --project=. scripts/verify_notebook_traction.jl [n ...]
using EarthquakeDiffinitive
using EarthquakeDiffinitive.Elasticity: elastic_blocks, traction_blocks,
                                       dof_index
using EarthquakeDiffinitive.ElasticitySplitNode
using Diffinitive
using Diffinitive.Grids
using Diffinitive.SbpOperators
using LinearAlgebra, SparseArrays, Printf

const λ_ = 2.0
const μ_ = 1.0
const DIM = 3

to_sparse(M) = reduce(vcat, [reduce(hcat, [sparse(M[j][k]) for k in 1:length(M)])
                             for j in 1:length(M)])
volume_H(g, set) = blockdiag(fill(sparse(inner_product(g, set)), DIM)...)
is_upper(bid) = occursin("Upper", string(typeof(bid)))
outward_sign(bid) = is_upper(bid) ? 1.0 : -1.0

"""
    traction_blocks_nb(g, λ, μ, set, bid)

`traction_blocks` with the narrow/wide split restored, per
`context/notebooks/elastic_clean.jl`: λ terms and μ's tangential terms use
`first_derivative`; **μ's diagonal terms use the boundary derivative**
`s·normal_derivative`, because they pair with the narrow `second_derivative`.
"""
function traction_blocks_nb(g, λ, μ, set, bid)
    n = grid_id(bid)
    e = sparse(boundary_restriction(g, set, bid))
    d̂ = outward_sign(bid) .* sparse(normal_derivative(g, set, bid))
    ∂ = [sparse(first_derivative(g, set, k)) for k in 1:DIM]
    T = [spzeros(size(e, 1), length(g)) for _ in 1:DIM, _ in 1:DIM]
    T[n, n] = λ * (e * ∂[n]) + 2μ * d̂
    for k in 1:DIM
        k == n || (T[n, k] = λ * (e * ∂[k]))
    end
    for i in 1:DIM
        i == n && continue
        T[i, n] = μ * (e * ∂[i])
        T[i, i] = μ * d̂
    end
    return T
end

"The interface SAT, replicating `split_node_system` verbatim except for `tb`."
function interface_sat(g_minus, g_plus, λ, μ, set, tb)
    Nm, Np = length(g_minus), length(g_plus)
    Ntot = DIM * (Nm + Np)
    bid_m = CartesianBoundary{1,UpperBoundary}()
    bid_p = CartesianBoundary{1,LowerBoundary}()
    Tm = tb(g_minus, λ, μ, set, bid_m)
    Tp = tb(g_plus, λ, μ, set, bid_p)
    Pm = ElasticitySplitNode._prolongation(g_minus, set, bid_m)
    Pp = ElasticitySplitNode._prolongation(g_plus, set, bid_p)
    S = spzeros(Ntot, Ntot)
    for j in 1:DIM
        rm = (j-1)*Nm+1:j*Nm
        rp = DIM * Nm .+ ((j-1)*Np+1:j*Np)
        Tm_j = reduce(hcat, Tm[j, :])
        Tp_j = reduce(hcat, Tp[j, :])
        S[rm, 1:DIM*Nm] .+= 0.5 .* (Pm * Tm_j)
        S[rm, DIM*Nm+1:Ntot] .+= -0.5 .* (Pm * Tp_j)
        S[rp, DIM*Nm+1:Ntot] .+= -0.5 .* (Pp * Tp_j)
        S[rp, 1:DIM*Nm] .+= 0.5 .* (Pp * Tm_j)
    end
    return S
end

"""
    solve_gaussian_slip(n, tb; L, w, amp)

Mirrors `test/elasticity_split_node_test.jl`'s `solve_gaussian_slip`, but with
the traction operator `tb` used BOTH in the SAT and for extracting σ_i1 — a
consistent swap, as the real change would be.
"""
function solve_gaussian_slip(n, tb; L=1.0, w=0.25, amp=1.0)
    set = read_stencil_set(SbpOperators.sbp_operators_path() * "standard_diagonal.toml"; order=4)
    g_minus = equidistant_grid((-L, -L, -L), (0.0, L, L), n, n, n)
    g_plus = equidistant_grid((0.0, -L, -L), (L, L, L), n, n, n)
    slip_fn(x2, x3) = (amp * exp(-(x2^2 + x3^2) / (2w^2)), 0.0)

    _, _, P = split_node_system(g_minus, g_plus, λ_, μ_, set)
    D = blockdiag(to_sparse(elastic_blocks(g_minus, λ_, μ_, set)),
                  to_sparse(elastic_blocks(g_plus, λ_, μ_, set)))
    H = blockdiag(volume_H(g_minus, set), volume_H(g_plus, set))
    DSAT = D + interface_sat(g_minus, g_plus, λ_, μ_, set, tb)
    A = -H * P * DSAT * P
    HP_DSAT = H * P * DSAT
    rs = factorize_reduced(A, P)

    Nm = length(g_minus)
    bid_m = CartesianBoundary{1,UpperBoundary}()
    bid_p = CartesianBoundary{1,LowerBoundary}()
    Tm = tb(g_minus, λ_, μ_, set, bid_m)
    Tp = tb(g_plus, λ_, μ_, set, bid_p)
    Tm_j = [reduce(hcat, Tm[j, :]) for j in 1:DIM]
    Tp_j = [reduce(hcat, Tp[j, :]) for j in 1:DIM]
    pairs = fault_node_pairs(g_minus, g_plus)

    function solve_for(slip_of)
        χ = build_chi(g_minus, g_plus, slip_of)
        return P * reduced_solve(rs, HP_DSAT * χ) .+ χ
    end

    U = solve_for(slip_fn)
    τ_minus = [Tm_j[j] * U[1:DIM*Nm] for j in 1:DIM]
    τ_plus = [Tp_j[j] * U[DIM*Nm+1:end] for j in 1:DIM]
    jump(c) = [U[dof_index_plus(g_minus, g_plus, c, Ip)] -
               U[dof_index_minus(g_minus, c, Im)] for (Im, Ip) in pairs]
    slip = [slip_fn(g_minus[Im][2], g_minus[Im][3])[1] for (Im, _) in pairs]

    # --- slip→τ₂ stiffness on the fault pairs, one unit-slip column at a time.
    # `boundary_selection`-style ordering is not needed here: both sides share
    # the same boundary grid, so τ indices line up with `pairs` via `sel`.
    e_m = sparse(boundary_restriction(g_minus, set, bid_m))
    sel = zeros(Int, size(e_m, 1))
    for col in 1:size(e_m, 2), idx in nzrange(e_m, col)
        sel[rowvals(e_m)[idx]] = col
    end
    lin = LinearIndices(size(g_minus))
    brow = [findfirst(==(lin[Im]), sel) for (Im, _) in pairs]
    K = zeros(length(pairs), length(pairs))
    for (c, (Im, _)) in enumerate(pairs)
        x2c, x3c = g_minus[Im][2], g_minus[Im][3]
        unit(x2, x3) = ((x2 ≈ x2c && x3 ≈ x3c) ? 1.0 : 0.0, 0.0)
        Uc = solve_for(unit)
        τ2 = Tm_j[2] * Uc[1:DIM*Nm]
        K[:, c] .= τ2[brow]
    end

    return (; jump, slip, τ_minus, τ_plus, U, amp, K, npairs=length(pairs))
end

function report(n)
    @printf("\n%s  n = %d\n", "="^70, n)
    @printf("%-42s %14s %14s\n", "", "traction_blocks", "notebook")
    results = Dict{Symbol,Any}()
    for (name, tb) in ((:current, traction_blocks), (:notebook, traction_blocks_nb))
        results[name] = solve_gaussian_slip(n, tb)
    end
    r1, r2 = results[:current], results[:notebook]

    checks = [
        ("eq 3  no opening  max|[u₁]|/amp",
         r -> maximum(abs, r.jump(1)) / r.amp, 1e-10),
        ("eq 4  max|[u₂]-slip|/amp",
         r -> maximum(abs, r.jump(2) .- r.slip) / r.amp, 1e-10),
        ("eq 4  max|[u₃]|/amp",
         r -> maximum(abs, r.jump(3)) / r.amp, 1e-10),
        ("eq 6b σ₂₁ continuity  max|Δ|/scale",
         r -> maximum(abs, r.τ_plus[2] .- r.τ_minus[2]) / maximum(abs, r.τ_minus[2]), 1e-10),
        ("eq 6c σ₃₁ continuity  max|Δ|/scale",
         r -> maximum(abs, r.τ_plus[3] .- r.τ_minus[3]) / maximum(abs, r.τ_minus[3]), 1e-10),
        ("antisym split |max‖U‖ - amp/2|",
         r -> abs(maximum(abs, r.U) - r.amp / 2), 1e-8),
        ("σ₁₁ / σ₂₁  (→ 0 under refinement)",
         r -> maximum(abs, r.τ_minus[1]) / maximum(abs, r.τ_minus[2]), Inf),
        ("K: max diag (must be < 0)",
         r -> maximum(diag(r.K)), 0.0),
        ("K: asymmetry ‖K-Kᵀ‖/‖K‖",
         r -> norm(r.K - r.K') / norm(r.K), Inf),
    ]
    for (label, f, tol) in checks
        v1, v2 = f(r1), f(r2)
        mark(v) = isfinite(tol) ? (v < tol ? "" : "  ← FAIL") : ""
        @printf("  %-40s %14.3e %14.3e%s\n", label, v1, v2, mark(v2))
    end
    @printf("  %-40s %14d %14d\n", "fault pairs in K", r1.npairs, r2.npairs)
    return (r1, r2)
end

ns = isempty(ARGS) ? [11, 15] : parse.(Int, ARGS)
out = [report(n) for n in ns]

if length(out) > 1
    println("\nσ₁₁/σ₂₁ under refinement (must decrease):")
    for (i, n) in enumerate(ns)
        c, nb = out[i]
        @printf("  n=%2d   current %.4e   notebook %.4e\n", n,
                maximum(abs, c.τ_minus[1]) / maximum(abs, c.τ_minus[2]),
                maximum(abs, nb.τ_minus[1]) / maximum(abs, nb.τ_minus[2]))
    end
end
println("""

The interface conditions are imposed by P and χ, not by the traction operator,
so eq 3/4 and the antisymmetric split should be unchanged. The traction
CONTINUITY checks (eq 6b,c) and σ₁₁ do depend on the extraction operator, as
does K — those are the columns to scrutinise.""")

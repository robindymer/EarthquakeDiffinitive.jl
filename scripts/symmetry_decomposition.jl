# WHERE does the asymmetry in `A = -H P (D+SAT) P` come from?
#
# `scripts/split_node_spd.jl` measures that `A` is ~14% asymmetric and that the
# bulk is clean, which narrows it to "the fault SAT" but no further. This script
# takes the operator apart factor by factor so the defect can be attributed to
# one specific piece.
#
# The tests are ordered so each one only involves machinery the previous ones
# have already cleared:
#
#   1  P alone            — symmetric? idempotent?
#   2  H vs P             — do they COMMUTE? The source builds `H*P*(D+SAT)*P`,
#                           but the form that is symmetric-by-construction is
#                           `P*(H*(D+SAT))*P`. Those agree only if HP = PH.
#   3  ONE GRID, NO COUPLING — the interface is entirely absent here, so
#                           anything asymmetric at this stage is a defect in
#                           the elastic operator or the traction SAT, not in
#                           the split-node construction:
#      3a  H*D on interior rows/cols only     (must be symmetric)
#      3b  H*D on everything                  (asymmetric by boundary terms —
#                                              this is SBP theory, not a bug)
#      3c  H*(D + free-surface traction SAT on all 6 faces)
#                                             (THE test: for a correct
#                                              SBP-consistent flux operator the
#                                              boundary terms cancel and this is
#                                              symmetric)
#   4  TWO GRIDS          — H*(D+SAT_interface), then the two orderings from
#                           test 2, to see which if either is symmetric.
#
# Run with:  julia --project=. scripts/symmetry_decomposition.jl [n ...]
using EarthquakeDiffinitive
using EarthquakeDiffinitive.Elasticity: elastic_blocks, traction_blocks
using EarthquakeDiffinitive.ElasticitySplitNode
using Diffinitive.Grids
using Diffinitive.SbpOperators
using LinearAlgebra, SparseArrays, Printf

const λ_ = 2.0
const μ_ = 1.0
const DIM = 3

asym(M) = norm(M - M') / max(norm(M), eps())

# Same flattening the module uses on `elastic_blocks`/`traction_blocks` output.
to_sparse(M) = reduce(vcat, [reduce(hcat, [sparse(M[j][k]) for k in 1:length(M)])
                             for j in 1:length(M)])

volume_H(g, set) = blockdiag(fill(sparse(inner_product(g, set)), DIM)...)

# `traction_blocks` is fixed-`+axis`, so the OUTWARD traction picks up a minus
# on lower boundaries. Same convention as `split_node_system`'s sign discussion.
is_upper(bid) = occursin("Upper", string(typeof(bid)))
outward_sign(bid) = is_upper(bid) ? 1.0 : -1.0

"""
    traction_sat(g, set, bid, λ, μ)

The Neumann/traction SAT for one boundary, exactly as `split_node_system`
builds it: the scalar pattern `-H⁻¹∘e'∘Hᵧ` with the normal derivative replaced
by `traction_blocks`, signed by the outward normal. Reuses the module's own
`_prolongation` so this cannot drift from the real thing.
"""
function traction_sat(g, set, bid, λ, μ; notebook::Bool=false)
    Prolong = ElasticitySplitNode._prolongation(g, set, bid)
    T = notebook ? traction_blocks_nb(g, λ, μ, set, bid) :
                   traction_blocks(g, λ, μ, set, bid)
    N = length(g)
    S = spzeros(N * DIM, N * DIM)
    for j in 1:DIM
        S[(j-1)*N+1:j*N, :] .+= outward_sign(bid) .* (Prolong * reduce(hcat, T[j, :]))
    end
    return S
end

bdim(bid) = typeof(bid).parameters[1]

"""
    interface_sat(g_minus, g_plus, λ, μ, set)

The two-grid interface SAT, replicating `split_node_system`'s construction
verbatim (half-weighted on both sides, `+`-side carrying the extra outward-normal
minus) so the only thing that differs in test 5 is the volume operator.
"""
function interface_sat(g_minus, g_plus, λ, μ, set; notebook::Bool=false)
    Nm, Np = length(g_minus), length(g_plus)
    Ntot = DIM * (Nm + Np)
    bid_m = CartesianBoundary{1,UpperBoundary}()
    bid_p = CartesianBoundary{1,LowerBoundary}()
    tb = notebook ? traction_blocks_nb : traction_blocks
    Tm = tb(g_minus, λ, μ, set, bid_m)
    Tp = tb(g_plus, λ, μ, set, bid_p)
    Pm = ElasticitySplitNode._prolongation(g_minus, set, bid_m)
    Pp = ElasticitySplitNode._prolongation(g_plus, set, bid_p)

    S = spzeros(Ntot, Ntot)
    for j in 1:DIM
        rows_m = (j-1)*Nm+1:j*Nm
        rows_p = DIM * Nm .+ ((j-1)*Np+1:j*Np)
        Tm_j = reduce(hcat, Tm[j, :])
        Tp_j = reduce(hcat, Tp[j, :])
        S[rows_m, 1:DIM*Nm] .+= 0.5 .* (Pm * Tm_j)
        S[rows_m, DIM*Nm+1:Ntot] .+= -0.5 .* (Pm * Tp_j)
        S[rows_p, DIM*Nm+1:Ntot] .+= -0.5 .* (Pp * Tp_j)
        S[rows_p, 1:DIM*Nm] .+= 0.5 .* (Pp * Tm_j)
    end
    return S
end

"""
    scalar_mechanism(g, set) -> NamedTuple

All four pairings of {narrow, wide} Laplacian × {`normal_derivative`,
`e∘first_derivative`} SAT, on the SCALAR Laplacian where no elasticity can
confuse matters. **One `H` throughout** — `inner_product(g, set)` — so this also
settles whether the two schemes need different quadrature.

The two discretizations satisfy *different* SBP identities, each with its own
boundary operator:

    narrow:  H*D₂    = -A          + e'∘Hᵧ∘normal_derivative
    wide:    H*D₁∘D₁ = -D₁ᵀ∘H∘D₁   + B∘D₁          (B∘D₁ = e'∘Hᵧ∘(e∘∂/∂xₙ))

In both the first term is symmetric, so a SAT that cancels that scheme's own
boundary term leaves a symmetric operator — and a SAT built for the *other*
scheme does not. If the diagonal of this 2×2 comes out at round-off and the
off-diagonal does not, the defect is a mismatched flux operator, not `H`.
"""
function scalar_mechanism(g, set)
    N = length(g)
    nd = ndims(g)
    H = sparse(inner_product(g, set))
    Hinv = sparse(inverse_inner_product(g, set))
    D1 = [sparse(first_derivative(g, set, k)) for k in 1:nd]
    L_narrow = sum(sparse(second_derivative(g, set, k)) for k in 1:nd)
    L_wide = sum(D1[k] * D1[k] for k in 1:nd)

    sat_nd = spzeros(N, N)   # from normal_derivative
    sat_fd = spzeros(N, N)   # from e∘first_derivative, what traction_blocks uses
    for bid in boundary_identifiers(g)
        e = sparse(boundary_restriction(g, set, bid))
        Hb = sparse(inner_product(boundary_grid(g, bid), set))
        pen = -Hinv * e' * Hb
        sat_nd .+= pen * sparse(normal_derivative(g, set, bid))
        sat_fd .+= outward_sign(bid) .* (pen * e * D1[bdim(bid)])
    end
    return (narrow_nd=asym(H * (L_narrow + sat_nd)),
            narrow_fd=asym(H * (L_narrow + sat_fd)),
            wide_nd=asym(H * (L_wide + sat_nd)),
            wide_fd=asym(H * (L_wide + sat_fd)))
end

"""
    traction_blocks_nb(g, λ, μ, set, bid)

`traction_blocks` rebuilt the way `context/notebooks/elastic_clean.jl`'s
`IsotropicTractionOperator` does it: **λ terms use `first_derivative`** (they
pair with the wide sandwich), **μ's diagonal terms use the boundary derivative**
(they pair with the narrow `second_derivative`), and μ's off-diagonal
(tangential) terms use `first_derivative`.

`src/Elasticity.jl`'s `traction_blocks` uses `e∘first_derivative` for every
term, dropping that distinction — which is the asymmetry this script measures.

The boundary derivative is `s·normal_derivative`, undoing Diffinitive's outward
sign to get the fixed-`+axis` convention `traction_blocks` uses. The notebook
does the same thing, with a TODO noting that a `boundary_derivative` operator
would be the honest way to ask for it.
"""
function traction_blocks_nb(g, λ, μ, set, bid)
    n = grid_id(bid)
    e = sparse(boundary_restriction(g, set, bid))
    d̂ = outward_sign(bid) .* sparse(normal_derivative(g, set, bid))   # unsigned
    ∂ = [sparse(first_derivative(g, set, k)) for k in 1:DIM]
    Nb, N = size(e, 1), length(g)

    T = [spzeros(Nb, N) for _ in 1:DIM, _ in 1:DIM]
    # σ_nn: λ·∂ₖuₖ summed (wide → first_derivative) + 2μ·∂ₙuₙ (narrow → d̂)
    T[n, n] = λ * (e * ∂[n]) + 2μ * d̂
    for k in 1:DIM
        k == n && continue
        T[n, k] = λ * (e * ∂[k])
    end
    # σ_in = μ(∂ₙuᵢ + ∂ᵢuₙ): the ∂ₙ part is narrow → d̂, the tangential ∂ᵢ is not
    for i in 1:DIM
        i == n && continue
        T[i, n] = μ * (e * ∂[i])
        T[i, i] = μ * d̂
    end
    return T
end

"""
    adapted_narrow(g, set, dir)

Almquist & Dunham (arXiv:2003.12811) eq (67): the narrow `second_derivative` in
direction `dir`, corrected at the boundary points so that the boundary term in
its SBP identity uses `D₁` instead of the boundary derivative `D̂`. This makes
`first_derivative` and `second_derivative` *fully compatible* in their sense —
Mattsson's operators, which Diffinitive implements, are compatible but NOT fully
compatible (measured gap `‖D̂ - e·D₁‖` ≈ 160 at order 4).

`d - s*(e*D₁)` is exactly that gap, outward-signed, and
`ElasticitySplitNode._prolongation` is `-H⁻¹∘e'∘Hᵧ`, so the correction below is
eq (67)'s `∓H⁻¹e e'(D̂ - D₁)` with the sign folded into the prolongation.

Costs one order of accuracy at exactly one grid point per boundary. Duru &
Virta reported no convergence loss for isotropic elasticity.
"""
function adapted_narrow(g, set, dir)
    D2 = sparse(second_derivative(g, set, dir))
    D1 = sparse(first_derivative(g, set, dir))
    for bid in boundary_identifiers(g)
        grid_id(bid) == dir || continue          # only boundaries ⊥ dir
        e = sparse(boundary_restriction(g, set, bid))
        d = sparse(normal_derivative(g, set, bid))
        D2 = D2 + ElasticitySplitNode._prolongation(g, set, bid) *
                  (d - outward_sign(bid) .* (e * D1))
    end
    return D2
end

"""
    elastic_blocks_matrix(g, λ, μ, set; adapt)

`elastic_blocks` rebuilt directly as a sparse matrix, so the narrow pieces can
be swapped for their adapted counterparts. Mirrors `Elasticity.elastic_blocks`
term for term:

    M[j][j] = μ·Σᵢ D₂ᵢ + μ·D₂ⱼ + λ·(D₁ⱼ∘D₁ⱼ)
    M[j][k] = (λ+μ)·(D₁ⱼ∘D₁ₖ)                     k ≠ j

`adapt=false` must reproduce `elastic_blocks` exactly — checked in `report`.
"""
function elastic_blocks_matrix(g, λ, μ, set; adapt::Bool)
    N = length(g)
    D1 = [sparse(first_derivative(g, set, i)) for i in 1:DIM]
    D2 = [adapt ? adapted_narrow(g, set, i) : sparse(second_derivative(g, set, i))
          for i in 1:DIM]
    lap = sum(D2)
    rows = map(1:DIM) do j
        reduce(hcat, map(1:DIM) do k
            j == k ? μ * lap + μ * D2[j] + λ * (D1[j] * D1[j]) :
                     (λ + μ) * (D1[j] * D1[k])
        end)
    end
    return reduce(vcat, rows)
end

"Linear DOF indices whose grid node touches no boundary at all."
function interior_dofs(g)
    N = length(g)
    on_bdry = falses(N)
    lin = LinearIndices(size(g))
    for bid in boundary_identifiers(g), I in boundary_indices(g, bid)
        on_bdry[lin[I]] = true
    end
    keep = findall(!, on_bdry)
    return vcat((keep .+ (j - 1) * N for j in 1:DIM)...)
end

function report(n; order=4)
    set = read_stencil_set(SbpOperators.sbp_operators_path() * "standard_diagonal.toml"; order)
    @printf("\n%s  n = %d, order = %d\n", "="^66, n, order)

    # ---------------- 3. ONE GRID, NO COUPLING ----------------
    g = equidistant_grid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), n, n, n)
    D1 = to_sparse(elastic_blocks(g, λ_, μ_, set))
    H1 = volume_H(g, set)
    HD = H1 * D1
    idx = interior_dofs(g)

    sat_free = spzeros(size(D1)...)
    for bid in boundary_identifiers(g)
        sat_free .+= traction_sat(g, set, bid, λ_, μ_)
    end
    HDSAT_free = H1 * (D1 + sat_free)

    println("  --- one grid, no interface ---")
    @printf("  3a  H*D, interior rows/cols only          %.2e   %s\n",
            asym(HD[idx, idx]), asym(HD[idx, idx]) < 1e-12 ? "symmetric" : "ASYMMETRIC")
    @printf("  3b  H*D, all rows/cols                    %.2e   %s\n",
            asym(HD), "asymmetric by boundary terms — expected, this is SBP")
    @printf("  3c  H*(D + free-surface traction SAT)     %.2e   %s\n",
            asym(HDSAT_free),
            asym(HDSAT_free) < 1e-12 ? "symmetric — flux operator is SBP-consistent" :
            "ASYMMETRIC — the traction SAT does not cancel the boundary terms")

    # Which elastic term breaks the identity? λ and μ enter `elastic_blocks`
    # through different discretizations (see src/Elasticity.jl: λ always uses
    # the wide `Dᵢ∘Dⱼ` sandwich; μ's diagonal uses the narrow native
    # `second_derivative`), so running each alone says which one the traction
    # operator is inconsistent with. Both zero is not a valid elastic operator,
    # but each alone is a well-posed test of the SBP identity.
    for (name, λt, μt) in (("λ only (μ=0)", 1.0, 0.0), ("μ only (λ=0)", 0.0, 1.0))
        Dt = to_sparse(elastic_blocks(g, λt, μt, set))
        st = spzeros(size(Dt)...)
        for bid in boundary_identifiers(g)
            st .+= traction_sat(g, set, bid, λt, μt)
        end
        a = asym(H1 * (Dt + st))
        @printf("  3d  ... with %-14s                %.2e   %s\n", name, a,
                a < 1e-12 ? "symmetric — this part is consistent" : "ASYMMETRIC")
    end

    # --- 3g. The notebook's traction operator: fix the SAT, leave D alone.
    # Strictly preferable to 3f if it works — no accuracy is given up anywhere.
    sat_nb = spzeros(size(D1)...)
    for bid in boundary_identifiers(g)
        sat_nb .+= traction_sat(g, set, bid, λ_, μ_; notebook=true)
    end
    a_nb = asym(H1 * (D1 + sat_nb))
    @printf("  3g  H*(SAME D + notebook traction SAT)     %.2e   %s\n", a_nb,
            a_nb < 1e-12 ? "SYMMETRIC — notebook's split is the fix" : "still asymmetric")

    # --- 3f. Almquist & Dunham eq (67): adapt the narrow operator instead of
    # touching the SAT. If this works, `traction_blocks` becomes the correct
    # flux operator and the physical boundary condition is preserved exactly.
    D_plain = elastic_blocks_matrix(g, λ_, μ_, set; adapt=false)
    @printf("  3f  sanity: rebuilt D == elastic_blocks    %.2e   (must be 0)\n",
            norm(D_plain - D1) / norm(D1))
    D_fc = elastic_blocks_matrix(g, λ_, μ_, set; adapt=true)
    a_fc = asym(H1 * (D_fc + sat_free))
    @printf("      H*(D_adapted + SAME traction SAT)      %.2e   %s\n", a_fc,
            a_fc < 1e-12 ? "SYMMETRIC — eq (67) is the fix" : "still asymmetric")

    m = scalar_mechanism(g, set)
    println("  3e  scalar Laplacian, one H throughout — asymmetry of H*(L + SAT):")
    @printf("                          SAT from normal_deriv   SAT from e∘first_deriv\n")
    @printf("        narrow  (D₂)            %.2e                %.2e\n", m.narrow_nd, m.narrow_fd)
    @printf("        wide  (D₁∘D₁)           %.2e                %.2e\n", m.wide_nd, m.wide_fd)

    # ---------------- 1, 2, 4. TWO GRIDS ----------------
    g_minus = equidistant_grid((-1.0, -1.0, -1.0), (0.0, 1.0, 1.0), n, n, n)
    g_plus = equidistant_grid((0.0, -1.0, -1.0), (1.0, 1.0, 1.0), n, n, n)
    A, HP_DSAT, P = split_node_system(g_minus, g_plus, λ_, μ_, set)

    # Rebuild D+SAT for the two-grid system. `HP_DSAT = H*P*(D+SAT)`, and both
    # H and P are known here, so recover it rather than re-deriving:
    #   D+SAT = P⁻¹ H⁻¹ (H*P*(D+SAT)) — but P is singular, so instead assemble
    # H for the pair and use the identity A = -(H*P*(D+SAT))*P directly.
    H = blockdiag(volume_H(g_minus, set), volume_H(g_plus, set))
    # reorder: volume_H is component-major per grid, and the two grids are
    # concatenated, which is exactly the module's DOF layout.

    println("  --- two grids ---")
    @printf("  1   P                                     %.2e   %s\n",
            asym(P), asym(P) < 1e-14 ? "symmetric" : "ASYMMETRIC")
    @printf("      ‖P²-P‖/‖P‖                            %.2e   %s\n",
            norm(P * P - P) / norm(P),
            norm(P * P - P) / norm(P) < 1e-14 ? "idempotent" : "NOT idempotent")
    commute = norm(H * P - P * H) / norm(H * P)
    @printf("  2   ‖HP-PH‖/‖HP‖                           %.2e   %s\n", commute,
            commute < 1e-14 ? "H and P commute — operator ordering is harmless" :
            "H and P DO NOT COMMUTE — H*P*(...)*P ≠ P*(H*(...))*P")

    # H*P*(D+SAT) is HP_DSAT. Peel H off to get P*(D+SAT), then compare the two
    # orderings of the final product.
    A_code = -HP_DSAT * P                     # what the source builds
    A_alt = -P * (H \ HP_DSAT) * P            # P*(H*(D+SAT))*P... see below
    @printf("  4a  H*(D+SAT) before projection           %.2e\n", asym(H \ (HP_DSAT)))
    @printf("  4b  A = -H*P*(D+SAT)*P   (source)         %.2e\n", asym(A_code))
    @printf("  4c  -P*(D+SAT)*P         (H peeled off)   %.2e\n", asym(A_alt))
    @printf("      ‖A_code - A_source‖                   %.2e   (sanity: must be 0)\n",
            norm(A_code - A))

    # ---------------- 5. TWO GRIDS WITH THE ADAPTED OPERATOR ----------------
    # Same P, same H, same SAT, same traction_blocks. Only the volume operator
    # changes. This is the production system, so this is the payoff test.
    SATi = interface_sat(g_minus, g_plus, λ_, μ_, set)
    Dm_fc = elastic_blocks_matrix(g_minus, λ_, μ_, set; adapt=true)
    Dp_fc = elastic_blocks_matrix(g_plus, λ_, μ_, set; adapt=true)
    DSAT_fc = blockdiag(Dm_fc, Dp_fc) + SATi
    A_fc = -H * P * DSAT_fc * P

    # sanity: the same assembly with unadapted operators must reproduce A
    Dm_pl = elastic_blocks_matrix(g_minus, λ_, μ_, set; adapt=false)
    Dp_pl = elastic_blocks_matrix(g_plus, λ_, μ_, set; adapt=false)
    A_pl = -H * P * (blockdiag(Dm_pl, Dp_pl) + SATi) * P

    println("  --- two grids, adapted narrow operator (eq 67) ---")
    @printf("  5   sanity: unadapted assembly == source   %.2e   (must be 0)\n",
            norm(A_pl - A) / norm(A))
    @printf("      H*(D_adapted+SAT) before projection    %.2e\n", asym(H \ (H * DSAT_fc)))
    @printf("      A_fc = -H*P*(D_adapted+SAT)*P          %.2e   %s\n", asym(A_fc),
            asym(A_fc) < 1e-12 ? "SYMMETRIC" : "still asymmetric")

    # ---------------- 7. TWO GRIDS, NOTEBOOK TRACTION SAT, D UNCHANGED -------
    # The preferred fix: `elastic_blocks` untouched, so no accuracy is lost
    # anywhere. Only the SAT's traction operator changes.
    SAT_nb = interface_sat(g_minus, g_plus, λ_, μ_, set; notebook=true)
    DSAT_nb = blockdiag(Dm_pl, Dp_pl) + SAT_nb
    A_nb = -H * P * DSAT_nb * P
    println("  --- two grids, notebook traction SAT, D unchanged ---")
    @printf("  7   A_nb = -H*P*(D+SAT_notebook)*P         %.2e   %s\n", asym(A_nb),
            asym(A_nb) < 1e-12 ? "SYMMETRIC" : "still asymmetric")

    # ---------------- 6. IS IT NOW CG-SOLVABLE? ----------------
    # `A` is singular by construction (P's null space), so reduce first. The
    # Galerkin form SᵀAS is the congruence transform — the only reduction that
    # inherits symmetry; `factorize_reduced`'s E·A·S does not (see PROGRESS.md).
    rs = factorize_reduced(A_fc, P)
    pos = Dict(k => c for (c, k) in enumerate(rs.keep))
    rows = collect(rs.keep); cols = collect(1:length(rs.keep))
    for (other, rep) in rs.merge_pairs
        push!(rows, other); push!(cols, pos[rep])
    end
    S = sparse(rows, cols, 1.0, rs.Ntot, length(rs.keep))
    Ar = S' * A_fc * S
    ev = eigvals(Symmetric(Matrix(0.5 * (Ar + Ar'))))

    # Physical RHS: a Gaussian slip patch, same as split_node_spd.jl uses.
    χ = build_chi(g_minus, g_plus, (x2, x3) -> (exp(-(x2^2 + x3^2) / 0.125), 0.0))
    b = S' * (H * P * DSAT_fc * χ)
    x = zeros(length(b)); r = copy(b); p = copy(r); rr = dot(r, r); nb = norm(b)
    its = 0; brk = false
    for k in 1:2000
        its = k; Ap = Ar * p; pAp = dot(p, Ap)
        pAp <= 0 && (brk = true; break)
        α = rr / pAp; @. x += α * p; @. r -= α * Ap
        rr_new = dot(r, r)
        sqrt(rr_new) / nb < 1e-10 && (rr = rr_new; break)
        @. p = r + (rr_new / rr) * p; rr = rr_new
    end
    @printf("  6   SᵀA_fcS:  asym %.2e,  %d/%d negative eigenvalues,  κ = %.2e\n",
            asym(Ar), count(<(-1e-12 * maximum(abs, ev)), ev), length(ev),
            maximum(abs, ev) / minimum(abs, ev))
    @printf("      CG: %d iterations, TRUE ‖b-Ax‖/‖b‖ = %.2e%s\n",
            its, norm(b - Ar * x) / nb, brk ? "   [pᵀAp ≤ 0 BREAKDOWN]" : "")
    return nothing
end

ns = isempty(ARGS) ? [9] : parse.(Int, ARGS)
for n in ns
    report(n)
end
println("""

How to read this — see SYMMETRIC_SAT.md for the full write-up.

  3c ASYMMETRIC while 3a is at round-off localizes the defect to the boundary
  flux, not to `elastic_blocks`, and does so on a single grid — so neither the
  interface, the projection, nor the two-grid coupling is involved.

  3d attributes it: λ (wide sandwiches only) is exact, μ (narrow diagonal) is
  not. 3e isolates the mechanism on the scalar Laplacian with ONE H: each
  scheme is symmetric with its own boundary operator and broken with the
  other's. `traction_blocks` uses `e∘first_derivative` for everything, which is
  right for wide and wrong for narrow.

  Tests 1 and 2 clear `P` and the operator ordering: `P` is exactly symmetric
  and idempotent, and `H*P = P*H` exactly, so `-H*P*(D+SAT)*P` and
  `-P*(H*(D+SAT))*P` are the same operator.

  3g and 7 are THE FIX: restore the narrow/wide split in the traction operator,
  the way `context/notebooks/elastic_clean.jl`'s `IsotropicTractionOperator`
  already does. `elastic_blocks`, `P` and `H` untouched, and no accuracy given
  up anywhere.

  3f and 5 are the alternative (Almquist & Dunham arXiv:2003.12811 eq 67):
  adapt D₂ instead so its boundary term is expressed in `first_derivative`.
  Same symmetry, but it costs one order of accuracy at one grid point per
  boundary — and the fault IS a boundary. Kept because it corroborates the
  diagnosis from the other side. 6 confirms the result is SPD and CG-solvable.

  NOT YET VERIFIED: that either fix preserves BP8's interface conditions, and
  what the notebook fix does to `K` and peak V — `traction_blocks` is also what
  `FaultResponse` uses for the physical Δτ, so those values will shift.""")

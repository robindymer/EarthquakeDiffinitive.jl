# ==============================================================================
# Convergence-order study.
#
# STANDALONE — deliberately not included from `runtests.jl`. Run it directly:
#
#     julia --project=. test/convergence_test.jl
#
# It costs minutes rather than seconds (the split-node section alone solves a
# 215k-DOF system), which is why it is not in the suite. Run it after touching
# `Elasticity.jl`, `ElasticitySplitNode.jl`, or after an upstream Diffinitive
# stencil change.
#
# WHY THIS EXISTS
#
# Every refinement check in the suite proper is a *monotonicity* assertion —
# `e5 < e10 < e20` in `pore_pressure_test.jl`, `σ11(fine) < σ11(coarse)` in
# `elasticity_split_node_test.jl`. A scheme converging at order 1 where it
# should be order 2, or at `h^0.56`, passes all of them. That matters here
# specifically: the wide/narrow `traction_blocks` bug (SYMMETRIC_SAT.md) was a
# *consistency-order* defect whose asymmetry decayed as `h^0.56`. Order
# collapse is the failure mode this codebase has actually had, and rates are
# what detect it.
#
# THE THREE ORDERS, WHICH ARE NOT THE SAME NUMBER
#
# The recurring confusion is between *truncation* order (how well an operator
# reproduces a derivative pointwise) and *solution* order (how fast the answer
# converges). They differ here by almost two orders, and in the helpful
# direction. Measured:
#
#   interior truncation, elastic operator     4      (§2)
#   boundary truncation, D1 and D2 alone      2      (§1) — textbook SBP
#   boundary truncation, elastic operator     1      (§1,§2) — one term, §2b
#   solution error, displacement            ~3       (§4)
#   fault shear traction σ21                ~2       (§4) — the one that matters
#
# WHY THE BOUNDARY IS ORDER 1 AND NOT ORDER 2
#
# Diagonal-norm SBP of interior order 2p has boundary closure order p, so
# order 4 → order 2 at the boundary. §1 confirms that for `first_derivative`
# and the narrow `second_derivative` individually.
#
# The elastic operator drops to order 1 there, but NOT because it composes
# operators — that would be too broad a claim, and §2b measures it false.
# Composition costs an order only when the OUTER derivative is taken in the
# same direction as the boundary being crossed:
#
#   D1x∘D1x  normal-normal          order 1     ← the only term that loses it
#   D1x∘D1y  normal-tangential      order 2
#   D1y∘D1y  tangential-tangential  order 4
#   D2x      narrow, same direction order 2
#
# The mechanism. `D1` is order-p accurate on a SMOOTH grid function. The inner
# `D1`'s truncation error τ is not one: it holds distinct O(h²) values on its
# closure rows and O(h⁴) beyond, i.e. a boundary layer of fixed width in GRID
# POINTS, so it varies by O(h²) across O(h) of the axis. Differentiating that
# ALONG the layer's own direction divides by h and gives O(h) — the operator is
# not being inaccurate, it is accurately differentiating something that becomes
# discontinuous as h→0. Differentiating across it costs nothing, because τ is
# smooth in the other direction; hence the D1x∘D1y row above.
#
# Splitting the composed error into `A = D1·u' - u''` (smooth input) and
# `B = D1·τ` (error input), which sum to it exactly, measures A at 1.99 and
# B at 1.01. The same split on the ORDER 2 operator is provable by hand:
# rows `(-1,1)/h` and `(-1,0,1)/2h` give (D1(D1u))₁ = (u₁-2u₂+u₃)/(2h²) → u''/2,
# a factor of two, i.e. not merely a lost order but inconsistent.
#
# DO NOT "FIX" THIS. Note from the table that the one term which loses the
# order is exactly the one with a narrow equivalent, so `isotropic_lambda_mu`
# could use `i == j ? λ*D2[i] : λ*(D1[i]∘D1[j])` and recover order 2. It would
# buy nothing and cost two things:
#
#   * σ21 would not improve. It is `traction_blocks` applied to the solution,
#     so its error is T·(U_h - u) + (T·u - σ21). The second term is the
#     traction operator's OWN closure truncation error, order 2 (§3), and
#     raising the elastic closure does not touch it. σ21 stays order 2 — and
#     σ21 is the only quantity that reaches `K`, hence all of BP8.
#   * It discards the λ/μ dispersion consistency `Elasticity.jl` documents as
#     deliberate, and narrow `D2`'s SBP identity carries `normal_derivative`
#     rather than `e∘D1` — so λ's normal terms in `traction_blocks` must be
#     re-derived in lockstep or `-HP(D+SAT)P` goes asymmetric again. That is
#     exactly the bug in SYMMETRIC_SAT.md.
#
# Raising σ21 past order 2 needs order 6 (closure order p=3), which is blocked
# upstream — see TODO.md §3.
#
# The order-1 closure also does not propagate: §4 shows the displacement still
# converging at ~3. The closure occupies O(1) points, i.e. a region of width
# O(h), so it costs far less in the solution than in the truncation error.
#
# HOW THE ASSERTIONS ARE SET
#
# Floors, never values. Rates here scatter by ±0.3 between grid pairs, and the
# affordable grids are PRE-ASYMPTOTIC — §4's σ21 rate measures 1.52 on the
# n = 9,17,33 triple used below but 1.91 on n = 13,25,49, which costs ~5 min
# for the n=49 solve alone. The floors deliberately sit below the asymptotic
# rates. Do not tighten them to match the numbers printed on a good day; the
# next grid triple will scatter the other way.
# ==============================================================================

using EarthquakeDiffinitive
using EarthquakeDiffinitive.Elasticity
using EarthquakeDiffinitive.ElasticitySplitNode
using Diffinitive.Grids
using Diffinitive.SbpOperators
using Tokens  # loads the sparse(::LazyTensor) extension
using StaticArrays, LinearAlgebra, SparseArrays, Printf
using Test

const λc = 2.0
const μc = 1.0

stencils() = read_stencil_set(SbpOperators.sbp_operators_path() * "standard_diagonal.toml", order=4)

# Observed order between two grids. `h` decreasing, `e` the error on each.
obs_order(e1, e2, h1, h2) = log(e1 / e2) / log(h1 / h2)

# Prints one row of errors and the rates between consecutive columns, and
# returns the rates so they can be asserted on.
function report(label, hs, es)
    rates = [obs_order(es[i-1], es[i], hs[i-1], hs[i]) for i in 2:lastindex(es)]
    @printf("  %-26s", label)
    for e in es
        @printf("%11.2e", e)
    end
    print("   ")
    for r in rates
        @printf("%7.2f", r)
    end
    println()
    return rates
end

header(cols) = (@printf("  %-26s", ""); for c in cols; @printf("%11s", c); end;
                print("      rates"); println())

# ==============================================================================
# §1. The primitive SBP operators, in 1D.
#
# The cheapest and sharpest test in the file — milliseconds, no solve, no
# assembly. It isolates the wide-sandwich order loss from everything else, so
# when §2 reports an order-1 boundary this section says whether the cause is
# the composition (expected) or a broken stencil (not).
#
# 1D can only exhibit the normal-normal case, which is the one that loses the
# order. §2b separates it from the directions that do not.
#
# The manufactured field must have no zeros and no symmetry: `exp∘sin` is used
# rather than a bare trig product because a field vanishing at a sample point
# turns a max-norm error into round-off and the rate into nonsense. See §2's
# note for what that looks like when it goes wrong.
# ==============================================================================

f1d(x)   = exp(sin(3.0x + 0.7))
df1d(x)  = 3.0cos(3.0x + 0.7) * f1d(x)
d2f1d(x) = f1d(x) * (9.0cos(3.0x + 0.7)^2 - 9.0sin(3.0x + 0.7))

# Order-4 diagonal-norm closures occupy the first/last 4 rows. `interior_1d`
# also skips the next 4, so that no sampled point's stencil — including the
# composed `D1∘D1`, whose reach is twice `D1`'s — touches a closure row.
const NB = 4

function operator_errors_1d(n)
    set = stencils()
    g = equidistant_grid(0.0, 1.0, n)
    x = collect(g)
    D1 = sparse(first_derivative(g, set, 1))
    D2 = sparse(second_derivative(g, set, 1))
    u = f1d.(x)

    e_D1 = abs.(D1 * u .- df1d.(x))
    e_D2 = abs.(D2 * u .- d2f1d.(x))
    e_D11 = abs.(D1 * (D1 * u) .- d2f1d.(x))

    boundary(e) = maximum(e[1:NB])
    interior(e) = maximum(e[NB+5:end-NB-4])

    return (h=1 / (n - 1),
        D1_int=interior(e_D1), D1_bnd=boundary(e_D1),
        D2_int=interior(e_D2), D2_bnd=boundary(e_D2),
        D11_int=interior(e_D11), D11_bnd=boundary(e_D11))
end

@testset "§1 primitive SBP operators (1D truncation error)" begin
    # n = 801 is deliberately NOT used: `D2`'s interior error reaches the
    # round-off floor there (eps/h² ≈ 1.4e-10 against a 1.7e-9 signal) and its
    # rate degrades to ~3.1 for reasons that have nothing to do with the
    # discretization.
    ns = (101, 201, 401)
    res = [operator_errors_1d(n) for n in ns]
    hs = [r.h for r in res]

    println("\n§1  Primitive SBP operators, 1D truncation error")
    header(["h=1/$(n-1)" for n in ns])
    get(k) = [getfield(r, k) for r in res]

    r_D1_int = report("D1        interior", hs, get(:D1_int))
    r_D1_bnd = report("D1        boundary", hs, get(:D1_bnd))
    r_D2_int = report("D2 narrow interior", hs, get(:D2_int))
    r_D2_bnd = report("D2 narrow boundary", hs, get(:D2_bnd))
    r_D11_int = report("D1∘D1     interior", hs, get(:D11_int))
    r_D11_bnd = report("D1∘D1     boundary", hs, get(:D11_bnd))

    # Interior: design order 4 for all three.
    @test minimum(r_D1_int) > 3.8
    @test minimum(r_D2_int) > 3.8
    @test minimum(r_D11_int) > 3.8

    # Boundary: order p = 2 for the primitives. This is the textbook SBP
    # result, and it is what makes the next assertion meaningful — the order
    # loss below is composition, not a defective closure.
    @test minimum(r_D1_bnd) > 1.8
    @test minimum(r_D2_bnd) > 1.8

    # Boundary: order 1 for the composition, and BOUNDED ABOVE. The upper
    # bound is the point of the test: if this ever reads 2, the wide sandwich
    # is no longer being composed the way `isotropic_lambda_mu` intends, and
    # §2's order-1 result becomes a real regression rather than a known cost.
    @test minimum(r_D11_bnd) > 0.8
    @test maximum(r_D11_bnd) < 1.4
end

# ==============================================================================
# §2. The assembled elastic operator, in 2D.
#
# `elastic_blocks` is dimension-generic (`ntuple(Val(D))` throughout
# `isotropic_lambda_mu`), so 2D exercises the same wide/narrow logic as the
# production 3D operator at a small fraction of the cost. That is what makes
# an interior-order measurement affordable at all: the interior sample must
# stay a fixed number of points clear of every closure, and in 3D the grids
# where that leaves a meaningful sample are ~10 min of assembly.
#
# §3 confirms the 2D result transfers to 3D.
#
# THE TRAP, recorded because it cost a wrong answer: the interior error must be
# sampled on a fixed PHYSICAL subdomain, and the manufactured field must not
# vanish on it. Sampling a fixed *point count* from the boundary at these grid
# sizes leaves only the domain centre, and `elasticity_test.jl`'s `ua` is
# exactly zero at (0.5,0.5,0.5) — that combination produced apparent rates of
# 2.8, 1.22 and −82 depending on the strip width, none of them real.
# ==============================================================================

const K2 = (3.0, 2.5, 2.0, 3.5)   # wavenumbers
const P2 = (0.7, 0.3, 1.1, 0.4)   # phases — generic, so the field has no zeros
                                  # or symmetry planes on the sample region

ua2(x) = SVector(sin(K2[1] * x[1] + P2[1]) * cos(K2[2] * x[2] + P2[2]),
                 cos(K2[3] * x[1] + P2[3]) * sin(K2[4] * x[2] + P2[4]))

# ∇·σ = (λ+μ)∇(∇·u) + μ∇²u for constant coefficients.
function divsigma2(x)
    u = ua2(x)
    dx_div = -K2[1]^2 * sin(K2[1] * x[1] + P2[1]) * cos(K2[2] * x[2] + P2[2]) -
             K2[3] * K2[4] * sin(K2[3] * x[1] + P2[3]) * cos(K2[4] * x[2] + P2[4])
    dy_div = -K2[1] * K2[2] * cos(K2[1] * x[1] + P2[1]) * sin(K2[2] * x[2] + P2[2]) -
             K2[4]^2 * cos(K2[3] * x[1] + P2[3]) * sin(K2[4] * x[2] + P2[4])
    return SVector((λc + μc) * dx_div - μc * (K2[1]^2 + K2[2]^2) * u[1],
                   (λc + μc) * dy_div - μc * (K2[3]^2 + K2[4]^2) * u[2])
end

function elastic_residual_2d(n)
    set = stencils()
    g = equidistant_grid((0.0, 0.0), (1.0, 1.0), n, n)
    M = elastic_blocks(g, λc, μc, set)
    A = reduce(vcat, [reduce(hcat, [sparse(M[j][k]) for k in 1:2]) for j in 1:2])
    R = reshape(A * flatten(map(ua2, g)) .- flatten(map(divsigma2, g)), length(g), 2)

    h = 1 / (n - 1)
    idx = LinearIndices((n, n))
    lo = findfirst(i -> (i - 1) * h >= 0.3, 1:n)
    hi = findlast(i -> (i - 1) * h <= 0.7, 1:n)
    # The subdomain is physical, so this margin only grows with n; assert it
    # rather than trust it.
    @assert lo - 1 >= 8 "interior sample is too close to the closure at n=$n"

    interior = vec([idx[i, j] for i in lo:hi, j in lo:hi])
    boundary = vec([idx[i, j] for i in 1:NB, j in 1:n])
    return (h=h, interior=maximum(abs, R[interior, :]),
            boundary=maximum(abs, R[boundary, :]), full=maximum(abs, R))
end

@testset "§2 elastic operator truncation order (2D)" begin
    ns = (41, 81, 161)
    res = [elastic_residual_2d(n) for n in ns]
    hs = [r.h for r in res]

    println("\n§2  elastic_blocks truncation error, 2D")
    header(["h=1/$(n-1)" for n in ns])
    r_int = report("interior [0.3,0.7]²", hs, [r.interior for r in res])
    r_bnd = report("closure rows", hs, [r.boundary for r in res])
    r_all = report("whole domain", hs, [r.full for r in res])

    # The headline: the interior really is fourth order.
    @test minimum(r_int) > 3.8

    # The closure is order 1, inherited from the normal-normal composition
    # alone — §2b isolates which term. Bounded above for the same reason as
    # §1: a jump to 2 means the operator changed.
    @test minimum(r_bnd) > 0.8
    @test maximum(r_bnd) < 1.4

    # And max-norm over the whole domain is therefore boundary-dominated. This
    # is the number a naive "is it fourth order?" check would report, and it is
    # order 1 — which is correct, and not a defect.
    @test minimum(r_all) > 0.8
    @test maximum(r_all) < 1.4
end

# ==============================================================================
# §2b. WHICH composed term loses the order.
#
# §2 reports an order-1 closure for the whole operator; this says which of its
# terms is responsible, and — more usefully — which are not. The answer is that
# composition per se is innocent: the order is lost only when the OUTER
# derivative is taken in the same direction as the boundary being crossed. See
# the header for the mechanism and for why this is not worth "fixing".
#
# Measured at the x1 closure rows, sampled ≥8 points clear of the x2 closure so
# that exactly one boundary is in play. Anyone tempted to reach for a narrow
# `D2` after reading §2 should read the header's "DO NOT FIX THIS" first: the
# term that loses the order is precisely the term that has a narrow equivalent,
# which is what makes the temptation credible and the change useless.
#
# Note also the magnitudes, not just the rates — the normal-normal term is an
# order of magnitude larger than the others as well as one order lower, which
# is why it sets §2's whole-operator closure rate single-handedly.
# ==============================================================================

# f = gx(x)·hy(y). Separable, so every mixed derivative is exact in closed
# form; generic phases, so no zeros or symmetry planes on the sample region.
gx2(x) = exp(sin(3.0x + 0.7))
gx2p(x) = gx2(x) * 3.0cos(3.0x + 0.7)
gx2pp(x) = gx2(x) * ((3.0cos(3.0x + 0.7))^2 - 9.0sin(3.0x + 0.7))
hy2(y) = exp(cos(2.0y + 0.3))
hy2p(y) = hy2(y) * (-2.0sin(2.0y + 0.3))
hy2pp(y) = hy2(y) * ((-2.0sin(2.0y + 0.3))^2 - 4.0cos(2.0y + 0.3))

function directional_terms(n)
    set = stencils()
    g = equidistant_grid((0.0, 0.0), (1.0, 1.0), n, n)
    D1x = sparse(first_derivative(g, set, 1))
    D1y = sparse(first_derivative(g, set, 2))
    D2x = sparse(second_derivative(g, set, 1))

    F = vec(map(p -> gx2(p[1]) * hy2(p[2]), g))
    Fxx = vec(map(p -> gx2pp(p[1]) * hy2(p[2]), g))
    Fxy = vec(map(p -> gx2p(p[1]) * hy2p(p[2]), g))
    Fyy = vec(map(p -> gx2(p[1]) * hy2pp(p[2]), g))

    closure(e) = maximum(abs, reshape(e, n, n)[1:NB, 9:n-8])
    return (h=1 / (n - 1),
        xx_wide=closure(D1x * (D1x * F) .- Fxx),
        xx_narrow=closure(D2x * F .- Fxx),
        xy=closure(D1x * (D1y * F) .- Fxy),
        yy_wide=closure(D1y * (D1y * F) .- Fyy))
end

@testset "§2b directional decomposition of the closure loss" begin
    ns = (41, 81, 161)
    res = [directional_terms(n) for n in ns]
    hs = [r.h for r in res]

    println("\n§2b Which composed term loses the order (x1 closure rows)")
    header(["h=1/$(n-1)" for n in ns])
    r_xx_wide = report("D1x∘D1x normal-normal", hs, [r.xx_wide for r in res])
    r_xx_narrow = report("D2x     normal, narrow", hs, [r.xx_narrow for r in res])
    r_xy = report("D1x∘D1y normal-tangential", hs, [r.xy for r in res])
    r_yy = report("D1y∘D1y tangential-tangl.", hs, [r.yy_wide for r in res])

    # The culprit, bounded above so that a change restoring order 2 here is
    # caught rather than silently welcomed — if this reads 2, the header's
    # analysis and §2's expectations both need revisiting.
    @test minimum(r_xx_wide) > 0.7
    @test maximum(r_xx_wide) < 1.4

    # Same derivative, narrow operator: keeps the textbook closure order. This
    # is the pairing that makes the loss above a property of composition-in-
    # the-normal-direction rather than of the derivative being second order.
    @test minimum(r_xx_narrow) > 1.5

    # Composition ACROSS directions costs nothing — the claim that would be
    # false if "composition destroys the closure order" were the real rule.
    @test minimum(r_xy) > 1.8

    # Purely tangential: no x1 closure is involved at all, so full interior
    # order survives.
    @test minimum(r_yy) > 3.8

    # Magnitude, not just rate: the normal-normal term dominates the closure
    # error outright, which is why §2's whole-operator rate tracks it.
    @test res[end].xx_wide > 10 * res[end].xx_narrow
end

# ==============================================================================
# §3. Confirmation in 3D, and the traction operator.
#
# Two things:
#
#  (a) that §2's 2D proxy transfers to the production dimension, and
#  (b) `traction_blocks`' own order.
#
# (b) is the important one. `elasticity_test.jl` checks traction against a
# manufactured field with a single-grid 5% absolute tolerance — a bound loose
# enough for a wrong-order operator to pass, and the operator that had to be
# fixed was wrong by a fractional power of h. The order is design 2 (it is a
# boundary operator, so §1's closure order applies directly, with no
# composition to lose an order to).
#
# It also caps §4: σ21 on the fault is `traction_blocks` applied to the
# solution, so the fault traction cannot converge faster than this, however
# accurate the displacement gets.
# ==============================================================================

ua3(x) = SVector(sin(π * x[1]) * cos(π * x[2]) * cos(π * x[3]),
                 cos(π * x[1]) * sin(π * x[2]) * cos(π * x[3]),
                 cos(π * x[1]) * cos(π * x[2]) * sin(π * x[3]))

# For this field ∇(∇·u) and ∇²u are both -3π²u, so ∇·σ = -3π²(λ+2μ)u.
divsigma3(x) = -3π^2 * (λc + 2μc) * ua3(x)

# J[i,k] = ∂u_i/∂x_k, built column-major to match `SMatrix`. Symmetric for this
# field, but written out in full rather than relying on that.
function jacobian_ua3(x)
    S = (sin(π * x[1]), sin(π * x[2]), sin(π * x[3]))
    C = (cos(π * x[1]), cos(π * x[2]), cos(π * x[3]))
    return π * SMatrix{3,3,Float64}(
        C[1] * C[2] * C[3], -S[1] * S[2] * C[3], -S[1] * C[2] * S[3],   # k=1
        -S[1] * S[2] * C[3], C[1] * C[2] * C[3], -C[1] * S[2] * S[3],   # k=2
        -S[1] * C[2] * S[3], -C[1] * S[2] * S[3], C[1] * C[2] * C[3])   # k=3
end

# Same convention as `elasticity_test.jl`'s `traction_analytic`: fixed +dim
# axis, outward sign not applied.
function traction_true3(x, dim)
    J = jacobian_ua3(x)
    return SVector(ntuple(3) do i
        i == dim ? λc * tr(J) + 2μc * J[i, dim] : μc * (J[i, dim] + J[dim, i])
    end)
end

function elastic_residual_3d(n)
    set = stencils()
    g = equidistant_grid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), n, n, n)
    M = elastic_blocks(g, λc, μc, set)
    A = reduce(vcat, [reduce(hcat, [sparse(M[j][k]) for k in 1:3]) for j in 1:3])
    R = reshape(A * flatten(map(ua3, g)) .- flatten(map(divsigma3, g)), length(g), 3)
    return (h=1 / (n - 1), full=maximum(abs, R))
end

function traction_error_3d(n)
    set = stencils()
    g = equidistant_grid((0.0, -1.0, -1.0), (1.0, 1.0, 1.0), n, n, n)
    bid = CartesianBoundary{1,LowerBoundary}()
    T = traction_blocks(g, λc, μc, set, bid)
    v = flatten(map(ua3, g))
    N = length(g)
    τ = [sum(T[i, k] * v[(k-1)*N+1:k*N] for k in 1:3) for i in 1:3]
    τ_true = map(x -> traction_true3(x, 1), boundary_grid(g, bid))
    err = maximum(maximum(abs, τ[i] .- vec(getindex.(τ_true, i))) for i in 1:3)
    return (h=1 / (n - 1), max=err)
end

@testset "§3 3D confirmation and traction operator order" begin
    println("\n§3  3D operators")

    # Only two grids: n = 41 in 3D is ~68k nodes and the assembly dominates
    # this file's runtime. Two grids give one rate, which is all that is needed
    # to confirm §2 transfers.
    ns_e = (11, 21)
    res_e = [elastic_residual_3d(n) for n in ns_e]
    header(["h=1/$(n-1)" for n in ns_e])
    r_e = report("elastic_blocks, max norm", [r.h for r in res_e], [r.full for r in res_e])

    ns_t = (11, 21, 41)
    res_t = [traction_error_3d(n) for n in ns_t]
    header(["h=1/$(n-1)" for n in ns_t])
    r_t = report("traction_blocks, max norm", [r.h for r in res_t], [r.max for r in res_t])

    # Same order-1 closure as 2D, so the cheap proxy is faithful. Looser
    # bounds than §2: one grid pair at coarse n, so more scatter.
    @test minimum(r_e) > 0.8
    @test maximum(r_e) < 1.6

    # Design order 2, and this is the sharp version of `elasticity_test.jl`'s
    # 5% bound. The last rate is the trustworthy one — the coarse pair is
    # pre-asymptotic.
    @test last(r_t) > 1.7
    @test minimum(r_t) > 1.5
end

# ==============================================================================
# §4. Solution error, via self-convergence.
#
# The question §1-§3 cannot answer: does the order-1 closure reach the answer?
#
# There is no manufactured solution here, deliberately. `Elasticity.jl`'s
# module docstring records that the earlier half-space BVP was removed because
# deriving true forcing/boundary data from continuum theory is easy to get
# subtly wrong, and a wrong derivation produces a confident wrong order. Self-
# convergence needs no continuum data at all: solve on nested grids n, 2n-1,
# 4n-3 (whose coarse points coincide exactly, so no interpolation is needed
# either) and measure the rate of ‖U_h − U_{h/2}‖.
#
# The answer is that it does not propagate — displacement converges at ~3
# against an order-1 closure, because the closure occupies a region of width
# O(h). But σ21 on the fault converges at ~2, capped by §3's traction operator:
# it is a boundary functional evaluated exactly where the closure error lives,
# so it collects none of the interior's gain.
#
# σ21 IS THE NUMBER THAT MATTERS. It is the only output that reaches `K`, and
# `K` is the entire coupling to BP8. Order ~2, not 4, means refining Δz
# 50 → 20 m buys ~2.5² ≈ 6× in fault-traction accuracy, not 2.5⁴ ≈ 39×. Worth
# remembering when reading PERFORMANCE.md §4's cost table.
# ==============================================================================

function solve_split_node(n; L=1.0, w=0.25)
    set = stencils()
    g_minus = equidistant_grid((-L, -L, -L), (0.0, L, L), n, n, n)
    g_plus = equidistant_grid((0.0, -L, -L), (L, L, L), n, n, n)
    slip_fn(x2, x3) = (exp(-(x2^2 + x3^2) / (2w^2)), 0.0)

    A, HP_DSAT, P = split_node_system(g_minus, g_plus, λc, μc, set)
    χ = build_chi(g_minus, g_plus, slip_fn)
    U = reconstruct_U(P, split_node_solve(CGSolver(A), HP_DSAT * χ), χ)

    Nm = length(g_minus)
    Tm = traction_blocks(g_minus, λc, μc, set, CartesianBoundary{1,UpperBoundary}())
    τ = [reduce(hcat, Tm[j, :]) * U[1:3Nm] for j in 1:3]
    return (n=n, h=2L / (n - 1), U=U, Nm=Nm, Np=length(g_plus),
            σ11=reshape(τ[1], n, n), σ21=reshape(τ[2], n, n))
end

# Gathers the displacement at a grid's own points, in a fixed order, so that a
# coarse solution and a restricted fine one can be subtracted entry-wise.
function gather(r, nc, stride)
    out = Float64[]
    lin = LinearIndices((r.n, r.n, r.n))
    for side in 1:2, c in 1:3
        offset = side == 1 ? (c - 1) * r.Nm : 3 * r.Nm + (c - 1) * r.Np
        for k in 1:nc, j in 1:nc, i in 1:nc
            push!(out, r.U[offset+lin[stride*(i-1)+1, stride*(j-1)+1, stride*(k-1)+1]])
        end
    end
    return out
end
coarse_points(rc) = gather(rc, rc.n, 1)
restrict_to(rf, nc) = (@assert rf.n == 2nc - 1; gather(rf, nc, 2))

@testset "§4 solution self-convergence" begin
    # Nested triple. n = 33 is 215k DOF and ~85 s; the next nested grid, 65, is
    # 1.65M DOF and ~35 min, which is why the assertions below tolerate
    # pre-asymptotic rates (see the header).
    ns = (9, 17, 33)
    println("\n§4  Split-node solution, self-convergence on nested grids $(ns)")
    sols = Dict{Int,Any}()
    for n in ns
        t = @elapsed sols[n] = solve_split_node(n)
        @printf("  n=%-3d %8d DOF  %6.1f s\n", n, 6n^3, t)
        flush(stdout)
    end

    du(nc, nf) = coarse_points(sols[nc]) .- restrict_to(sols[nf], nc)
    e1, e2 = du(ns[1], ns[2]), du(ns[2], ns[3])
    rms(v) = norm(v) / sqrt(length(v))
    hs = [sols[ns[1]].h, sols[ns[2]].h]   # successive halvings

    header(["|U₁-U₂|", "|U₂-U₃|"])
    r_umax = report("displacement, max", hs, [maximum(abs, e1), maximum(abs, e2)])
    r_urms = report("displacement, rms", hs, [rms(e1), rms(e2)])

    dσ(nc, nf) = sols[nc].σ21 .- sols[nf].σ21[1:2:end, 1:2:end]
    t1, t2 = dσ(ns[1], ns[2]), dσ(ns[2], ns[3])
    r_tmax = report("fault σ21, max", hs, [maximum(abs, t1), maximum(abs, t2)])
    r_trms = report("fault σ21, rms", hs, [rms(t1), rms(t2)])

    # The order-1 closure does not propagate into the solution.
    @test last(r_umax) > 2.2
    @test last(r_urms) > 2.5

    # σ21 is capped by §3's traction operator, not by the displacement. The
    # floor is 1.2 and NOT ~1.9: this triple measures 1.52 while n = 13,25,49
    # measures 1.91. Tightening this to match a good run makes it flaky.
    @test last(r_tmax) > 1.2
    @test last(r_trms) > 1.2

    # ------------------------------------------------------------------
    # σ11 → 0 on the fault, the one error with an exactly known answer.
    #
    # Mirror symmetry x1 → -x1 makes σ11 odd, and eq. 6a makes it continuous,
    # so it is identically zero on the fault. Unusually for this problem that
    # holds for the TRUNCATED domain too, so this is a clean discretization
    # error with no far-field contamination — unlike every other quantity
    # here, which carries the domain bias of TODO §1.
    #
    # This is the rate version of `elasticity_split_node_test.jl`'s
    # `σ11(fine) < σ11(coarse)`.
    # ------------------------------------------------------------------
    hs3 = [sols[n].h for n in ns]
    println()
    header(["n=$n" for n in ns])
    r_s11 = report("fault σ11 → 0", hs3, [maximum(abs, sols[n].σ11) for n in ns])
    @test minimum(r_s11) > 1.2
end

# ==============================================================================
# §5. The fault-edge ring, and why §4 is allowed to take a whole-face maximum.
#
# `elasticity_split_node_test.jl`'s σ11 check passes for a reason its own
# comments do not state, and the reason is a property of its slip patch rather
# than of the discretization: at w = 0.25 on L = 1 the Gaussian is exp(-8) ≈
# 3e-4 of amplitude at the fault edge, i.e. numerically zero there.
#
# Widen it to w = 0.4 — still an unremarkable patch, 4.4% of amplitude at the
# edge — and max|σ11| GROWS under refinement, at rates of −0.55, −0.74, −0.76.
# The cause is structural, not a bug: `fault_node_pairs` excludes the ring
# where the fault plane meets the truncated boundary, because those nodes carry
# the far-field u=0 Dirichlet (see the docstring there, and lore.md). Slip that
# has not decayed by the ring therefore meets a genuine discontinuity, and
# refining resolves that singularity better rather than removing it.
#
# So this section asserts the structural fact — the maximum sits ON the ring —
# rather than pinning the divergence itself, which would cement a wart. Drop
# one node of border and convergence returns at ~1.2-1.8.
#
# The practical consequence for §4: a whole-face maximum is only meaningful for
# a patch that has decayed at the edge. §4's w = 0.25 qualifies. A future test
# that widens the patch must exclude the ring or it will measure the ring.
# ==============================================================================

@testset "§5 fault-edge ring dominates a non-decayed slip patch" begin
    ns = (9, 13, 17)
    println("\n§5  Non-decayed slip patch (w=0.4), fault σ11")
    sols = [solve_split_node(n; w=0.4) for n in ns]
    hs = [s.h for s in sols]

    # Excluding one node of border: `M[2:end-1, 2:end-1]` is the fault face
    # minus exactly the ring.
    inner(M) = M[2:end-1, 2:end-1]

    header(["n=$n" for n in ns])
    r_full = report("whole face (diverges)", hs, [maximum(abs, s.σ11) for s in sols])
    r_inner = report("ring excluded", hs, [maximum(abs, inner(s.σ11)) for s in sols])

    # The structural claim: the maximum is attained on the ring, at every
    # resolution. This is what makes the divergence above an artefact of where
    # the maximum is taken rather than a property of the scheme.
    for s in sols
        i, j = Tuple(argmax(abs.(s.σ11)))
        @test i == 1 || i == s.n || j == 1 || j == s.n
    end

    # Remove the ring and it converges, at the same rate as §4's decayed patch.
    @test minimum(r_inner) > 0.8

    # Recorded, not asserted: `r_full` is negative. Left un-asserted on
    # purpose — it is a consequence of the ring's Dirichlet condition, and a
    # future change that made it converge would be an improvement, not a
    # failure.
    @printf("  (whole-face rates %s — negative, see the section comment)\n",
            join([@sprintf("%.2f", r) for r in r_full], ", "))
end

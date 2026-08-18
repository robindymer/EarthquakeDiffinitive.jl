using EarthquakeDiffinitive
using EarthquakeDiffinitive.Elasticity
using EarthquakeDiffinitive.ElasticitySplitNode
using Diffinitive.Grids
using Diffinitive.SbpOperators
using StaticArrays, LinearAlgebra, SparseArrays
using Test

const λ_sn = 2.0
const μ_sn = 1.0

# Algebraic self-consistency check: rather than deriving a manufactured
# solution's true forcing/boundary data from continuum PDE theory (easy to
# get subtly wrong — see src/Elasticity.jl's module docstring for exactly
# such a mistake found this session), pick ARBITRARY smooth, localized
# (Gaussian, decaying to ~0 at the far-field boundaries so they don't fight
# P's far-field zeroing) fields on each side, and derive the forcing that
# makes them an exact solution *directly from the assembled discrete
# operator itself*: for U = P*u + χ, the governing equation P(D+SAT)U = f
# gives A*u = HP_DSAT*(χ - U) (see derivation in PR/commit notes). Solving
# and reconstructing U should then exactly recover the manufactured field,
# for ANY smooth field — this tests that D, SAT, P, χ, and the solve are
# all self-consistent, without needing any continuum-mechanics derivation.

gaussian(x, x0, σ) = exp(-sum(abs2, x - x0) / (2σ^2))

# `u1` has to be continuous across the fault for the field to be reachable at
# all: `P` averages the fault-normal DOF pair, so `P*U + χ = U` (which is what
# makes the manufactured forcing below exact) holds only if `u1(0⁺)=u1(0⁻)`.
# That is not a limitation of the test — it is the no-opening condition, BP8
# eq. 3. Mirroring in `x1` is the cheapest way to get it while keeping the
# field smooth on each side separately (each side only ever sees its own half).
# `u2`,`u3` stay free to jump arbitrarily; that jump is the slip.
u1_common(x) = 0.10 * gaussian(SVector(abs(x[1]), x[2], x[3]), SVector(0.3, 0.2, 0.1), 0.15)

u_plus_true(x) = SVector(
    u1_common(x),
    0.08 * gaussian(x, SVector(0.2, -0.1, 0.2), 0.15),
    0.05 * gaussian(x, SVector(0.4, 0.1, -0.2), 0.15),
)
u_minus_true(x) = SVector(
    u1_common(x),
    0.07 * gaussian(x, SVector(-0.25, 0.05, 0.15), 0.15),
    0.06 * gaussian(x, SVector(-0.35, -0.1, 0.2), 0.15),
)

function split_node_stencil_set()
    return read_stencil_set(SbpOperators.sbp_operators_path() * "standard_diagonal.toml", order=4)
end

@testset "ElasticitySplitNode" begin
    @testset "self-consistent: recovers an arbitrary smooth two-sided field" begin
        set = split_node_stencil_set()
        n = 11
        g_minus = equidistant_grid((-1.0, -1.0, -1.0), (0.0, 1.0, 1.0), n, n, n)
        g_plus = equidistant_grid((0.0, -1.0, -1.0), (1.0, 1.0, 1.0), n, n, n)
        Nm, Np = length(g_minus), length(g_plus)

        U_minus = map(u_minus_true, g_minus)
        U_plus = map(u_plus_true, g_plus)
        U_true = vcat(flatten(U_minus), flatten(U_plus))

        # slip = actual jump of the manufactured fields at the fault.
        bid_minus = CartesianBoundary{1,UpperBoundary}()
        bid_plus = CartesianBoundary{1,LowerBoundary}()
        slip_fn(x2, x3) = begin
            up = u_plus_true(SVector(0.0, x2, x3))
            um = u_minus_true(SVector(0.0, x2, x3))
            (up[2] - um[2], up[3] - um[3])
        end

        A, HP_DSAT, P = split_node_system(g_minus, g_plus, λ_sn, μ_sn, set)
        χ = build_chi(g_minus, g_plus, slip_fn)

        rhs = HP_DSAT * (χ - U_true)
        u_sol = reduced_solve(A, rhs, P)
        U_reconstructed = reconstruct_U(P, u_sol, χ)

        # A direct solve on the (still somewhat ill-conditioned, given H's
        # small quadrature weights) reduced system doesn't reach full
        # machine precision — check relative to the field's own scale.
        scale = maximum(abs.(U_true))
        @test maximum(abs.(U_reconstructed - U_true)) / scale < 1e-3
    end

    # The self-consistency check above validates D/SAT/P/χ/solve *against each
    # other*, so it is blind to the model being the wrong model — exactly the
    # failure mode that killed `halfspace_system`. These tests instead check
    # the reconstructed field against BP8's interface conditions and against
    # the whole-space symmetry, none of which the solver is told about:
    #
    #   * eq. 3    no opening,  [u1] = 0
    #   * eq. 4    the imposed slip really is the jump in u2,u3
    #   * eq. 6    tractions σ_i1 continuous across the fault
    #   * mirror symmetry x1 → -x1 (u1 even, u2/u3 odd) forces σ11 to be odd;
    #     combined with eq. 6a's continuity that means σ11 ≡ 0 on the fault,
    #     so it must vanish under refinement. It is also why the field splits
    #     antisymmetrically: max‖U‖ = max|s|/2.
    function solve_gaussian_slip(n; L=1.0, w=0.25, amp=1.0)
        set = split_node_stencil_set()
        g_minus = equidistant_grid((-L, -L, -L), (0.0, L, L), n, n, n)
        g_plus = equidistant_grid((0.0, -L, -L), (L, L, L), n, n, n)
        slip_fn(x2, x3) = (amp * exp(-(x2^2 + x3^2) / (2w^2)), 0.0)

        A, HP_DSAT, P = split_node_system(g_minus, g_plus, λ_sn, μ_sn, set)
        χ = build_chi(g_minus, g_plus, slip_fn)
        U = reconstruct_U(P, reduced_solve(A, HP_DSAT * χ, P), χ)

        Nm = length(g_minus)
        # `traction_blocks` returns boundary-ordered vectors, and both fault
        # boundaries share the same (x2,x3) boundary grid, so σ_i1 from the
        # two sides can be compared entry-wise.
        Tm = traction_blocks(g_minus, λ_sn, μ_sn, set, CartesianBoundary{1,UpperBoundary}())
        Tp = traction_blocks(g_plus, λ_sn, μ_sn, set, CartesianBoundary{1,LowerBoundary}())
        τ_minus = [reduce(hcat, Tm[j, :]) * U[1:3Nm] for j in 1:3]
        τ_plus = [reduce(hcat, Tp[j, :]) * U[3Nm+1:end] for j in 1:3]

        pairs = fault_node_pairs(g_minus, g_plus)
        jump(c) = [U[dof_index_plus(g_minus, g_plus, c, Ip)] - U[dof_index_minus(g_minus, c, Im)]
                   for (Im, Ip) in pairs]
        slip = [slip_fn(g_minus[Im][2], g_minus[Im][3])[1] for (Im, _) in pairs]

        return (; U, jump, slip, τ_minus, τ_plus, amp)
    end

    @testset "interface conditions for a prescribed slip patch" begin
        r = solve_gaussian_slip(13)
        τ_scale = maximum(abs, r.τ_minus[2])

        # eq. 3: no opening.
        @test maximum(abs, r.jump(1)) / r.amp < 1e-10
        # eq. 4: the jump is the prescribed slip.
        @test maximum(abs, r.jump(2) .- r.slip) / r.amp < 1e-10
        @test maximum(abs, r.jump(3)) / r.amp < 1e-10
        # eq. 6b,c: shear tractions continuous (an SAT sign error here shows
        # up as an O(1) relative jump).
        @test maximum(abs, r.τ_plus[2] .- r.τ_minus[2]) / τ_scale < 1e-10
        @test maximum(abs, r.τ_plus[3] .- r.τ_minus[3]) / τ_scale < 1e-10
        # antisymmetric split of the field.
        @test maximum(abs, r.U) ≈ r.amp / 2 rtol = 1e-8
    end

    # `factorize_reduced` defaults to a Cholesky of the Galerkin form `SᵀAS`,
    # which is only legal if that matrix really is symmetric positive definite.
    # It is SPD only because `traction_blocks` pairs each of `elastic_blocks`'
    # two SBP schemes with its own boundary operator (`test/elasticity_test.jl`'s
    # "SBP property" test) — Cholesky *fails* on the pre-fix operator. So this
    # is the downstream half of that same check, and it is what would catch a
    # silent regression into the LU fallback.
    #
    # The `:lu` comparison is the part that has teeth: LU assumes nothing about
    # symmetry, so if the two agree, Cholesky's answer is not an artefact of
    # `Symmetric` having discarded a real lower triangle.
    @testset "Galerkin reduction is SPD and Cholesky agrees with LU" begin
        set = split_node_stencil_set()
        n = 11
        g_minus = equidistant_grid((-1.0, -1.0, -1.0), (0.0, 1.0, 1.0), n, n, n)
        g_plus = equidistant_grid((0.0, -1.0, -1.0), (1.0, 1.0, 1.0), n, n, n)
        A, HP_DSAT, P = split_node_system(g_minus, g_plus, λ_sn, μ_sn, set)

        rs_chol = factorize_reduced(A, P)                    # default
        rs_lu = factorize_reduced(A, P; method=:lu)
        # A silent PosDefException fallback would leave an LU here too.
        @test rs_chol.fact isa SparseArrays.CHOLMOD.Factor

        S = prolongation(rs_chol)
        A_gal = S' * A * S
        @test norm(A_gal - A_gal') / norm(A_gal) < 1e-12

        χ = build_chi(g_minus, g_plus, (x2, x3) -> (exp(-(x2^2 + x3^2) / 0.125), 0.0))
        rhs = HP_DSAT * χ
        x_chol = reduced_solve(rs_chol, rhs)
        x_lu = reduced_solve(rs_lu, rhs)
        @test norm(x_chol - x_lu) / norm(x_lu) < 1e-10

        # The Petrov form `E·A·S` the reduction used to build is nonsymmetric by
        # construction even now that `A` is symmetric — asserted so that the
        # Galerkin form is not quietly swapped back for it.
        A_pet = (A*S)[rs_chol.keep, :]
        @test norm(A_pet - A_pet') / norm(A_pet) > 1e-3
    end

    # `CGSolver` runs CG on the SINGULAR `A` with no reduction. That is only
    # legitimate because of three properties, so each is asserted directly
    # rather than inferred from the answer coming out right — an unconverged or
    # null-space-polluted solve can still look plausible.
    @testset "CG on the singular system" begin
        set = split_node_stencil_set()
        n = 11
        g_minus = equidistant_grid((-1.0, -1.0, -1.0), (0.0, 1.0, 1.0), n, n, n)
        g_plus = equidistant_grid((0.0, -1.0, -1.0), (1.0, 1.0, 1.0), n, n, n)
        A, HP_DSAT, P = split_node_system(g_minus, g_plus, λ_sn, μ_sn, set)
        χ = build_chi(g_minus, g_plus, (x2, x3) -> (exp(-(x2^2 + x3^2) / 0.125), 0.0))
        b = HP_DSAT * χ

        # 1. Consistency: b ⊥ null(A). null(P) ⊆ null(A) and (I-P) projects
        #    onto null(P), so this is `b ∈ range(P)`. Holds exactly, because
        #    `HP = PH` and `P = Pᵀ` — not to a tolerance.
        @test norm(b - P * b) / norm(b) < 1e-14

        # 2. A really is symmetric — CG is meaningless otherwise, and this is
        #    what `traction_blocks` had to be fixed to achieve.
        @test norm(A - A') / norm(A) < 1e-12

        cg = split_node_solver(A, P; method=:cg)
        u_cg = split_node_solve(cg, b)
        direct = split_node_solver(A, P)
        u_dir = split_node_solve(direct, b)

        @test solver_report(cg).unconverged == 0
        @test norm(b - A * u_cg) / norm(b) < 1e-8

        # 3. The iterates stay in range(A), so `u` carries no null-space
        #    component — and even if it did, `P` in the reconstruction removes
        #    it. Both halves are checked.
        @test norm(u_cg - P * u_cg) / norm(u_cg) < 1e-12
        @test norm(P * u_cg - P * u_dir) / norm(P * u_dir) < 1e-8

        # The reconstruction is what everything downstream consumes.
        U_cg = reconstruct_U(P, u_cg, χ)
        U_dir = reconstruct_U(P, u_dir, χ)
        @test norm(U_cg - U_dir) / norm(U_dir) < 1e-8
    end

    @testset "fault-normal stress vanishes under refinement" begin
        coarse = solve_gaussian_slip(11)
        fine = solve_gaussian_slip(15)
        σ11(r) = maximum(abs, r.τ_minus[1])
        # Small compared with the shear traction it sits alongside, and
        # shrinking with h — the exact value on the fault is zero.
        @test σ11(coarse) < 0.1 * maximum(abs, coarse.τ_minus[2])
        @test σ11(fine) < σ11(coarse)
    end
end

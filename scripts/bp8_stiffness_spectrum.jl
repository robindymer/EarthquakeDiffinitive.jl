# Where does BP8-PW's stiffness actually live?
#
# `PROGRESS.md` "BP8-PW stiffness: what the stiff eigenvalue is". Builds the
# dense Jacobian of the coupled RHS by finite differences at several times,
# and answers three questions with it:
#
#   1. what is the stiff eigenvalue, and does `K_ww/D` predict it?
#   2. is it local — does a per-node 3x3 block-diagonal reproduce it?
#   3. is the pore-pressure diffusion block stiff? (no)
#
# Only affordable because the ODE state is small: N = 4*nf+1 is 1157 at
# Δz = 50 m, so a full eigendecomposition is seconds. It is a diagnostic, not
# something the solver path uses.
#
# Run: julia --project=scripts scripts/bp8_stiffness_spectrum.jl [Δz]

using EarthquakeDiffinitive
using EarthquakeDiffinitive.BP8
using LinearAlgebra, Printf

Δz = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 50.0

m = build_model(; Δz, injection=:peaceman, verbose=true)
nf = m.nf; p = m.par; w = m.well_cell; N = 4nf + 1
fp = BP8.friction_params(p); η = BP8.damping(p)
@printf("\nnf=%d  N=%d  well_cell=%d  K[w,w]=%.4e Pa/m  eta=%.4e  a=%.4g\n",
        nf, N, w, m.K[w, w], η, fp.a)

sol = run_bp8(m; tspan=(0.0, 100 * 3600.0), saveat=1800.0)
@printf("Tsit5 over 100 h: %d accepted steps\n\n", sol.stats.naccept)

"""Dense Jacobian of `rhs!` at `(u,t)`, by one-sided differences with
per-block step sizes (slip, ln θ and pressure live on very different scales)."""
function jacobian(u, t)
    f0 = similar(u); BP8.rhs!(f0, u, m, t)
    J = zeros(N, N); fp_ = similar(u); up = similar(u)
    h = [fill(1e-9, 2nf); fill(1e-6, nf); fill(1e2, nf); 1e2]
    for j in 1:N
        copyto!(up, u); up[j] += h[j]
        BP8.rhs!(fp_, up, m, t)
        @. J[:, j] = (fp_ - f0) / h[j]
    end
    return J
end

# `K_ww/D` predicts λ only where the floor actually binds (`floored` >= 1).
# Before that the stiffest mode is a different, non-well-cell one and the ratio
# column is meaningless — at Δz = 100 m the floor never binds at all (σ̄ stays
# ~6 MPa, 613 steps over 100 h) and every ratio there is ~0.27.
@printf("  t[h] | floored | sigmabar[Pa] |     V_w    |   D=dg/dV   |  K_ww/D    | lambda(J)   | ratio\n")
for th in (60.0, 75.0, 85.0, 95.0, 99.5)
    t = th * 3600; u = Array(sol(t))
    σ̄ = max(p.σ0 - u[3nf+w], p.σ̄_min)
    nfl = count(<(p.σ̄_min), p.σ0 .- u[3nf+1:4nf])
    V = BP8.evaluate!(m, u, t).Vmag[w]
    θ = exp(u[2nf+w])

    # D = dg/dV for the scalar force balance g(V) = ηV + σ̄·a·asinh(u) - T,
    # i.e. exactly what `solve_slip_rate`'s `dgdx` computes, divided by V.
    C = exp((fp.f_star + fp.b * log(fp.V_star * θ / fp.Dc)) / fp.a)
    uu = V / (2fp.V_star) * C
    D = η + σ̄ * fp.a * (C / (2fp.V_star)) / sqrt(1 + uu^2)

    λ = minimum(real.(eigvals(jacobian(u, t))))
    @printf("%6.1f | %7d | %12.4e | %.4e | %11.4e | %10.4e | %11.4e | %.4f\n",
            th, nfl, σ̄, V, D, m.K[w, w] / D, λ, (m.K[w, w] / D) / λ)
end

# --- locality: does a per-node 3x3 block (s2, s3, ϕ) carry the stiff mode? ---
t = 85 * 3600.0; u = Array(sol(t))
J = jacobian(u, t)
λfull = minimum(real.(eigvals(J)))

Jb = zeros(N, N)
for i in 1:nf
    idx = (i, nf + i, 2nf + i)
    for a in idx, b in idx
        Jb[a, b] = J[a, b]
    end
end
Jb[3nf+1:4nf, 3nf+1:4nf] .= J[3nf+1:4nf, 3nf+1:4nf]   # pressure block untouched
Jb[end, end] = J[end, end]
λblk = minimum(real.(eigvals(Jb)))
λp = minimum(real.(eigvals(J[3nf+1:4nf, 3nf+1:4nf])))

@printf("\nfull J stiffest eigenvalue         : %.6e\n", λfull)
@printf("per-node 3x3 block-diagonal only   : %.6e   ratio=%.4f\n", λblk, λblk / λfull)
@printf("pressure block alone (Ap)          : %.6e   (not stiff)\n", λp)

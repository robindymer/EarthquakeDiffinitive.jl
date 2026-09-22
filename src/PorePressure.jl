module PorePressure

using Diffinitive
using Diffinitive.Grids
using Diffinitive.SbpOperators
using SparseArrays
using Tokens # loads DiffinitiveSparseArraysExt, which defines sparse(::LazyTensor)
using OrdinaryDiffEq
using LinearAlgebra: mul!

export pore_pressure_operator, gaussian_source, injection_rate,
       solve_pore_pressure, darcy_operators, well_index, well_cell_index,
       SBP4_RE_FACTOR,
       peaceman_cell_volume, well_coupled_operator

"""
    pore_pressure_operator(l_f, Δz, α, stencil_set) -> (g, A)
    pore_pressure_operator(g, α, stencil_set) -> A

Fault-plane grid `g = [-l_f,l_f]²` and the constant sparse `A` with
`dp/dt = A*p + forcing(t)`, discretizing `∂p/∂t = α∇²p` with zero-flux Neumann
on all four sides (BP8 eq. 17-18). `α` is constant, so `A` is built once.

The second form takes a grid directly: the coupled driver passes the elastic
solver's own `Ω_f` nodes so pressure and slip share an index.
"""
function pore_pressure_operator(l_f, Δz, α, stencil_set)
    n = round(Int, 2l_f / Δz) + 1
    g = equidistant_grid((-l_f, -l_f), (l_f, l_f), n, n)
    return g, pore_pressure_operator(g, α, stencil_set)
end

function pore_pressure_operator(g, α, stencil_set)
    Δ = Laplace(g, stencil_set)
    D = Δ
    for id ∈ boundary_identifiers(g)
        D = D + foldl(∘, sat_tensors(Δ, g, NeumannCondition(0.0, id)))
    end
    return α .* sparse(D)
end

"""
    darcy_operators(g, stencil_set; k, viscosity) -> (Q2, Q3)

Darcy velocity components from the pressure field, `q_j = Q_j * p =
-(k/η) ∂p/∂x_j` (BP8 eq. 16), for the `darcy_vel_2`/`darcy_vel_3` outputs.
"""
function darcy_operators(g, stencil_set; k, viscosity)
    c = -k / viscosity
    return (c .* sparse(first_derivative(g, stencil_set, 1)),
            c .* sparse(first_derivative(g, stencil_set, 2)))
end

"""
    well_cell_index(g)

Linear index of the node closest to the injection point `(0,0)` — the Peaceman
model's "well cell" (BP8 §2.1.2).
"""
function well_cell_index(g)
    pts = vec(collect(map(x -> x[1]^2 + x[2]^2, g)))
    return argmin(pts)
end

"""
    SBP4_RE_FACTOR

Measured equivalent-radius factor for this package's SBP order-4 Laplacian:
`r_e ≈ 0.268Δz`.

Peaceman's `0.198` is derived for a centred five-point stencil; the wider SBP4
operator's well cell represents a different radius. Measured by inverting
eq. 25 against the numeric well-cell pressure, the factor collapses on
`Δz²/(4αt)` and approaches 0.268 as that group → 0 — a property of the stencil
alone, so `r_e/Δz` is resolution-independent.

That is what §2.1.2 asks for: it calls `r_e` "discretization-dependent (and
consequently, the model output mesh-independent)" and gives `0.198Δx` only as a
five-point example. Using `0.198` with a wider operator would break the
mesh-independence. Measurements in `PROGRESS.md`.
"""
const SBP4_RE_FACTOR = 0.268

"""
    well_index(; k, L_fwid, viscosity, Δz, r_well, r_e_factor=SBP4_RE_FACTOR)

Peaceman well index `WI = 2πkL_fwid / (η ln(r_e/r_well))` (BP8 eq. 22), in
m³/(Pa·s), with `r_e = r_e_factor*Δz`.

Defaults to [`SBP4_RE_FACTOR`](@ref), not Peaceman's 0.198 — see there for why.
Pass `r_e_factor=0.198` to compare against a run made the five-point way.
"""
function well_index(; k, L_fwid, viscosity, Δz, r_well, r_e_factor=SBP4_RE_FACTOR)
    r_e = r_e_factor * Δz
    r_e > r_well || error("equivalent radius r_e=$r_e must exceed r_well=$r_well; " *
                          "the cell is too small for the Peaceman model")
    return 2π * k * L_fwid / (viscosity * log(r_e / r_well))
end

"""
    peaceman_cell_volume(Δz, L_fwid)

Well-cell volume `V_e = Δz²·L_fwid`, for its storage `S_e = V_e·φ·β`
(BP8 eq. 23).
"""
peaceman_cell_volume(Δz, L_fwid) = Δz^2 * L_fwid

"""
    well_coupled_operator(Ap, well_cell, WI, S_e, S_well) -> J

The Peaceman pressure subsystem (BP8 eq. 22-23) as one constant sparse
operator on the stacked unknown `[p; p_well]`:

    d/dt [p; p_well] = J * [p; p_well] + [0; q_inj(t)/S_well].

This is §2.1.2's **option 2** — the well pressure as one extra unknown in the
same linear system, rather than operator-split (option 1) or eliminated (option
3). The exchange `WI*(p_well - p[well_cell])` enters `S_e` and `S_well` with
opposite signs, adding a 2x2 block at `(well_cell, end)` to `Ap`.

That block, not diffusion, is the stiffest part of the subsystem: its
eigenvalue `-WI*(1/S_well + 1/S_e)` is constant in `σ̄` and the friction state,
and is ~10x stiffer than diffusion at Δz = 100 m. It is why `p` is integrated
implicitly and separately ([`BP8.solve_pressure_history`](@ref)) rather than
carried in the coupled state. It is *not* the `K_ww/D ≈ -1.7` stiffness that
dominates BP8-PW once the `σ̄_min` floor binds; see
`scripts/extra/bp8_stiffness_spectrum.jl`.
"""
function well_coupled_operator(Ap, well_cell, WI, S_e, S_well)
    nf = size(Ap, 1)
    J = [Ap spzeros(nf, 1); spzeros(1, nf + 1)]
    return J + sparse([well_cell, well_cell, nf + 1, nf + 1],
                      [well_cell, nf + 1, well_cell, nf + 1],
                      [-WI / S_e, WI / S_e, WI / S_well, -WI / S_well],
                      nf + 1, nf + 1)
end

"""
    gaussian_source(g, L_gauss)

Spatial part of the Gaussian-source injection term (BP8 eq. 19),
`exp(-(x2²+x3²)/(2L_gauss²)) / (2π L_gauss²)`, as a vector on `g`.
"""
function gaussian_source(g, L_gauss)
    return vec(collect(map(g) do x
        exp(-(x[1]^2 + x[2]^2) / (2L_gauss^2)) / (2π * L_gauss^2)
    end))
end

"""
    injection_rate(t; q0, t_off)

`q_inj(t)` (BP8 eq. 20): constant `q0` for `t < t_off`, zero afterwards.
"""
injection_rate(t; q0, t_off) = t < t_off ? q0 : zero(q0)

"""
    solve_pore_pressure(g, A, source_grid; q0, t_off, β, φ, tspan, alg = Rodas5P())

Solves `dp/dt = A*p + (injection_rate(t; q0, t_off) / (β*φ)) * source_grid`
over `tspan` with a stiff Rosenbrock solver, returning the `ODESolution`. `A`
is treated as a constant, pre-factorizable linear operator.
"""
function solve_pore_pressure(g, A, source_grid; q0, t_off, β, φ, tspan, alg=Rodas5P())
    p0 = zeros(length(source_grid))

    function rhs!(dp, p, params, t)
        mul!(dp, A, p)
        dp .+= (injection_rate(t; q0, t_off) / (β * φ)) .* source_grid
        return nothing
    end
    # The system is linear, so the Jacobian is exactly A. Supplying it avoids
    # OrdinaryDiffEq's sparse-differencing fallback, which was unstable here.
    jac!(J, p, params, t) = copyto!(J, A)

    f = ODEFunction(rhs!; jac=jac!, jac_prototype=A)
    prob = ODEProblem(f, p0, tspan)
    return solve(prob, alg; tstops=[t_off])
end

end # module PorePressure

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
       peaceman_cell_volume

"""
    pore_pressure_operator(l_f, Δz, α, stencil_set) -> (g, A)
    pore_pressure_operator(g, α, stencil_set) -> A

Builds the fault-plane grid `g = [-l_f,l_f]×[-l_f,l_f]` and the constant
sparse operator `A` such that `dp/dt = A*p + forcing(t)` discretizes
`∂p/∂t = α∇²p` with homogeneous Neumann (zero-flux) boundary conditions on
all four sides (BP8-QD eq. 17-18). `α` is assumed constant (homogeneous
medium), so `A` only needs to be built once.

The second form takes a grid directly, for when it has to line up with
something else — the coupled driver builds it from the elastic solver's own
`Ω_f` fault nodes so pressure and slip share an index.
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

Sparse operators giving the Darcy velocity components from the pressure
field, `q_j = Q_j * p = -(k/η) ∂p/∂x_j` (BP8-QD eq. 16), for the benchmark's
`darcy_vel_2`/`darcy_vel_3` outputs.
"""
function darcy_operators(g, stencil_set; k, viscosity)
    c = -k / viscosity
    return (c .* sparse(first_derivative(g, stencil_set, 1)),
            c .* sparse(first_derivative(g, stencil_set, 2)))
end

"""
    well_cell_index(g)

Linear index of the grid node closest to the injection point `(0,0)` — the
"well cell" of the Peaceman model (BP8-QD §2.1.2).
"""
function well_cell_index(g)
    pts = vec(collect(map(x -> x[1]^2 + x[2]^2, g)))
    return argmin(pts)
end

"""
    well_index(; k, L_fwid, viscosity, Δz, r_well, r_e_factor=0.198)

The Peaceman well index `WI = 2πkL_fwid / (η ln(r_e/r_well))` (BP8-QD eq. 22),
in m³/(Pa·s). `r_e = r_e_factor*Δz` is the equivalent radius; the benchmark
gives `0.198Δz` for a centred five-point finite-difference stencil on a square
mesh, which is the default.
"""
function well_index(; k, L_fwid, viscosity, Δz, r_well, r_e_factor=0.198)
    r_e = r_e_factor * Δz
    r_e > r_well || error("equivalent radius r_e=$r_e must exceed r_well=$r_well; " *
                          "the cell is too small for the Peaceman model")
    return 2π * k * L_fwid / (viscosity * log(r_e / r_well))
end

"""
    peaceman_cell_volume(Δz, L_fwid)

Volume `V_e = Δz²·L_fwid` of the well cell, used for its storage
`S_e = V_e·φ·β` in BP8-QD eq. 23.
"""
peaceman_cell_volume(Δz, L_fwid) = Δz^2 * L_fwid

"""
    gaussian_source(g, L_gauss)

The spatial part of the Gaussian-source injection term (BP8-QD eq. 19),
`exp(-(x2²+x3²)/(2L_gauss²)) / (2π L_gauss²)`, as a vector on `g`.
"""
function gaussian_source(g, L_gauss)
    return vec(collect(map(g) do x
        exp(-(x[1]^2 + x[2]^2) / (2L_gauss^2)) / (2π * L_gauss^2)
    end))
end

"""
    injection_rate(t; q0, t_off)

The injection rate `q_inj(t)` (BP8-QD eq. 20): constant `q0` for `t < t_off`,
zero afterwards.
"""
injection_rate(t; q0, t_off) = t < t_off ? q0 : zero(q0)

"""
    solve_pore_pressure(g, A, source_grid; q0, t_off, β, φ, tspan, alg = Rodas5P())

Solves `dp/dt = A*p + (injection_rate(t; q0, t_off) / (β*φ)) * source_grid`
over `tspan`, returning the `ODESolution`. `A` is treated as a constant
(pre-factorizable) linear operator.
"""
function solve_pore_pressure(g, A, source_grid; q0, t_off, β, φ, tspan, alg=Rodas5P())
    p0 = zeros(length(source_grid))

    function rhs!(dp, p, params, t)
        mul!(dp, A, p)
        dp .+= (injection_rate(t; q0, t_off) / (β * φ)) .* source_grid
        return nothing
    end
    # The system is linear (dp/dt = A*p + forcing(t)), so the Jacobian is
    # exactly A — supplying it explicitly avoids OrdinaryDiffEq falling back
    # to sparse-differencing, which was numerically unstable here.
    jac!(J, p, params, t) = copyto!(J, A)

    f = ODEFunction(rhs!; jac=jac!, jac_prototype=A)
    prob = ODEProblem(f, p0, tspan)
    return solve(prob, alg; tstops=[t_off])
end

end # module PorePressure

module PorePressure

using Diffinitive
using Diffinitive.Grids
using Diffinitive.SbpOperators
using SparseArrays
using Tokens # loads DiffinitiveSparseArraysExt, which defines sparse(::LazyTensor)
using OrdinaryDiffEq
using LinearAlgebra: mul!

export pore_pressure_operator, gaussian_source, injection_rate,
       solve_pore_pressure

"""
    pore_pressure_operator(l_f, Δz, α, stencil_set) -> (g, A)

Builds the fault-plane grid `g = [-l_f,l_f]×[-l_f,l_f]` and the constant
sparse operator `A` such that `dp/dt = A*p + forcing(t)` discretizes
`∂p/∂t = α∇²p` with homogeneous Neumann (zero-flux) boundary conditions on
all four sides (BP8-QD eq. 17-18). `α` is assumed constant (homogeneous
medium), so `A` only needs to be built once.
"""
function pore_pressure_operator(l_f, Δz, α, stencil_set)
    n = round(Int, 2l_f / Δz) + 1
    g = equidistant_grid((-l_f, -l_f), (l_f, l_f), n, n)

    Δ = Laplace(g, stencil_set)
    D = Δ
    for id ∈ boundary_identifiers(g)
        D = D + foldl(∘, sat_tensors(Δ, g, NeumannCondition(0.0, id)))
    end

    A = α .* sparse(D)
    return g, A
end

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

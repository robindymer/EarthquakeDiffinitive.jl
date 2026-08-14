using Diffinitive
using Diffinitive.Grids
using Diffinitive.SbpOperators
using Diffinitive.LazyTensors
using BenchmarkTools



stencil_set = let
    fn = SbpOperators.sbp_operators_path()*"standard_diagonal.toml"
    
    read_stencil_set(fn, order = 4)
end

g = equidistant_grid((0,-1),(10,0), 100, 15)

Δ = laplace(g, stencil_set)

H⁻¹ = inverse_inner_product(g, stencil_set)

b_bottom = CartesianBoundary{2, LowerBoundary}()
b_surface = CartesianBoundary{2, UpperBoundary}()
b_left = CartesianBoundary{1, LowerBoundary}()
b_right = CartesianBoundary{1, UpperBoundary}()

e_bottom = boundary_restriction(g, stencil_set, b_bottom)
d_bottom = normal_derivative(g, stencil_set, b_bottom)
# Hᵧ_bottom = inner_product(boundary_grid(g,b_bottom), stencil_set)

L = Δ# - H⁻¹∘e_bottom'∘Hᵧ_bottom∘d_bottom


u = map(x->rand(),g)
u̇ = similar(u)
ü = similar(u)

function fullapply1!(Au, A, u)
    Au .= A*u
end


############################


LazyTensors.apply(Δ,u,10,10)

@edit LazyTensors.apply(Δ,u,10,10)

@code_warntype LazyTensors.apply(Δ,u,10,10)


@benchmark fullapply1!($ü, $Δ, $u)

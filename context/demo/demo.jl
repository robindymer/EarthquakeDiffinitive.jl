### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 70441495-6903-4929-a322-90419ff19042
begin
	import Pkg
	Pkg.activate(".")

	using PlutoUI
	
	using Diffinitive
	using Diffinitive.Grids
	using Diffinitive.SbpOperators
	
	using StaticArrays
	using LinearAlgebra
	using ForwardDiff
	using OrdinaryDiffEq
	using BenchmarkTools

	# using CairoMakie
	using WGLMakie
end

# ╔═╡ ba825d17-2914-49bd-841a-e5c262308796
begin
	using Tokens
	using SparseArrays
end

# ╔═╡ ab2fed8f-9869-4f14-8d42-8f56b5c254df
using TiledIteration

# ╔═╡ 7fe19dc3-8c4a-4722-8c4e-ab4f6892bebb
md"""
# Diffinitive.jl demo
"""

# ╔═╡ 5c41b08e-373e-4da9-beac-fd0238df6efd
md"""
* What are the goals of Diffinitive?
   - Easy to experiment
   - Easy to compose different parts
   - Performant (how performant? at the moment vs potential)
* Ideas
   - Lazy operators
   - Cell based indexing
* Why use Diffinitive?
   * Designed to be flexible
   * Designed to be performant
"""

# ╔═╡ 5f21c1c8-f0c9-4d25-9c5f-e5a0892022b5
md"""
## Grids
"""

# ╔═╡ d7493595-2d8f-4a9a-a7ac-0d8d9f31217f
md"""
### 1D
"""

# ╔═╡ d885f8b5-fc6b-48f5-8a9d-a6c24823ac90
md"""
### 2D
"""

# ╔═╡ d3de3989-16b1-4b38-852c-297eb9bddef2
md"""
### 2D curvilinear
"""

# ╔═╡ e75701fe-8e91-4acf-8542-203499826bdd
md"""
We forward unknown derviative computations to `ForwardDiff`:
"""

# ╔═╡ 387b67e1-87a4-472a-a560-a8f44d077826
Grids.jacobian(c::Function, x::Float64) = ForwardDiff.derivative(c, x)

# ╔═╡ f2a53605-5e33-46fe-8cf9-5f604d5e2a7b
md"""
### Grid properties
"""

# ╔═╡ e05fa744-83e2-486c-9409-3d4e1679fc26
md"""
## SbpOperators
"""

# ╔═╡ 55f726bc-5e4b-466c-876a-a744d6b383f8
md"""
### Stencil sets
"""

# ╔═╡ b0bde7dc-0193-48c9-8a23-fc5741d26b05
edit(SbpOperators.sbp_operators_path()*"standard_diagonal.toml")

# ╔═╡ aad9cc2f-b181-4094-baa0-032cc758e2e6
stencil_set = let
	fn = SbpOperators.sbp_operators_path()*"standard_diagonal.toml"
	
	read_stencil_set(fn, order = 4)
end

# ╔═╡ 5fbdcd21-b2d2-47c5-a0e8-5dc7f402649e
md"""
### Creation
"""

# ╔═╡ 6d161e75-d7c3-4a59-852e-cfc59ec451da
let
	g = equidistant_grid(0,10, 11)
	∂ = first_derivative(g, stencil_set)
end

# ╔═╡ e44bd4eb-5f37-4b45-b7ea-5d05cc8559cc
let
	g = equidistant_grid(0,10, 11)
	∂² = second_derivative(g, stencil_set)
end

# ╔═╡ c68b294c-13a2-4d0b-8a08-29495436bf11
md"""
Both operators are represented as the same kind of object with different interior and closure stencils.
"""

# ╔═╡ df62465e-2e84-4d9e-a659-0e82145f9c34
let
	g = equidistant_grid(0,10, 11)
	∂² = second_derivative(g, stencil_set)
	sparse(∂²)
end

# ╔═╡ 909536ac-5504-4b98-8a38-acf282936964
md"""
### Application
"""

# ╔═╡ 3122c6af-d6b0-48fe-bf57-3aaf5f72f94b
md"""
The application of `∂` on `u` is lazy and returns a `TensorApplication`. Nothing is computed until we index the object. This gives important control over memory access order.
"""

# ╔═╡ 9928d201-7961-4d3d-bb89-90884fb569c9
md"""
### Higher dimensions
"""

# ╔═╡ 2acab9cf-94f9-46dc-9f25-36494571f57e
let
	g = equidistant_grid((0,0), (1,1), 40, 40)
	∂₁ = first_derivative(g, stencil_set, 1)
	∂₂ = first_derivative(g, stencil_set, 2)

	u = map(g) do x
		sin(20norm(x))
	end

	v = ∂₁*u

	
	fig = Figure()

	ax = Axis(fig[1,1];
		title="u",
		aspect=DataAspect(),
		limits=((0,1),(0,1)),
	)
	plot!(ax, g, u, colorrage=(-1,1))

	ax = Axis(fig[1,2];
		title="∂₁u",
		aspect=DataAspect(),
		limits=((0,1),(0,1)),
	)
	plot!(ax, g,v, colorrage=(-1,1))

	fig
end

# ╔═╡ 720c9f3a-06cd-4768-b050-c5b19bf76bc2
md"""
### Other operators
"""

# ╔═╡ 38ba2fdc-d2a1-4aec-858d-f61c0695a7f9
let
	g = equidistant_grid((0,0), (1,1), 40, 40)

	(
		inner_product(g, stencil_set),
		laplace(g, stencil_set),
		normal_derivative(g, stencil_set, CartesianBoundary{1,LowerBoundary}())
	)
end

# ╔═╡ 2ca95a49-8869-4b9d-a05e-7e1b727b17b7
md"""
All operators needed to implement a scheme can be constructed as LazyTensors, either as a specialized struct or as a composition of already defined operators.
"""

# ╔═╡ a56fb5bd-1b90-471d-8fc3-dbb882fd6e01
# @edit laplace(g,stencil_set)

# ╔═╡ bcec3a56-4728-43f9-82b3-408ed8769c03
# @edit undivided_skewed04(equidistant_grid(0,1,10),2)

# ╔═╡ f502399d-e04e-4146-be6f-474f55b80977
md"""
## Implementing the wave equation
"""

# ╔═╡ 917691bc-77e9-4d73-9b6f-39577a5cd425
md"""
* Dirchlet at left, right, and surface.
* Neumann at bottom.
"""

# ╔═╡ 1ad1c41e-a09e-48e9-8bbb-f2c0a6c3b66f
md"""
### Needed operators
"""

# ╔═╡ 05fddef1-530b-4965-950b-761bfecf3a22
md"""
Here, the details of the curvilinear mapping are hidden in the implementation of these methods.
"""

# ╔═╡ de920e8e-1594-4a5e-a58a-65519149c933
# @edit normal_derivative(g, stencil_set, b_bottom)

# ╔═╡ b3762de6-6262-4a94-aa58-a66cd3545dcf
md"""
### Time integration
"""

# ╔═╡ f29d3fe6-10ac-4af4-9162-c884d8899c34
md"""
### Initial conditions
"""

# ╔═╡ da7cb259-0ae6-406c-9972-632e1fb71f0d
@bind animate CounterButton("Animate")

# ╔═╡ 6e84cdb8-eebc-47e1-8890-ff6a0fac8a7f
md"""
Hidden cell:
"""

# ╔═╡ 052455c2-0b96-460e-8d66-ae188918b54d
md"""
## Performance
   - The compiler inlines everything! (hmm...)
   - There are no allocations
   - Threading?
"""

# ╔═╡ cd6affc4-1926-4e4f-a850-47ebc048784f
md"""
## Conclusion
* What are we working on?
    - Elastic operator
    - Elastic-Acoustic coupling
    - Moving Multiblock grids/grid functions into the package.
* What could be contributed?
   - Users
   - Performance, GPU
   - More operators
      - Optimized grids
      - Block inner products
   - Domain truncation SuperGrid/PML
   - Different couplings
   - Projection operators
"""

# ╔═╡ d0d05031-b49f-4f59-8de9-9213d038a00a
md"""
# The End
"""

# ╔═╡ 9fbb9753-4ca6-4095-a3c9-b9ed0b98aa5a
md"""
## Functions
"""

# ╔═╡ b1a9bb25-e1f2-4142-854e-419ab3d1b0a1
begin
	gaussian(x; x₀, σ=1, A=1) = exp(-norm(x-x₀)^2/σ^2)
	gaussian(; x₀, σ=1, A=1) = x -> gaussian(x; x₀, σ, A)
end

# ╔═╡ 7c2667cd-aefd-458f-aff4-44e8559db226
let
	g = equidistant_grid(0, 2, 41)
	gf = map(gaussian(x₀=1.2, σ=0.25), g)
	scatter(g,gf)
end

# ╔═╡ 03c27b5a-78f1-45db-8126-da91b2a511bf
let
	g = equidistant_grid((0,0), (3,2), 41,31)
	gf = map(gaussian(x₀=@SVector[2.2,1.3], σ=0.25), g)
	
	plot(g,gf, axis=(;aspect=DataAspect()))
end

# ╔═╡ de36950b-a1e8-4a4b-b078-7daf698cecb1
let
	g = equidistant_grid(0,1, 40)
	u = map(gaussian(x₀=0.4, σ=0.2), g)
	∂ = first_derivative(g, stencil_set)
	∂u = ∂*u
end

# ╔═╡ c35802c5-a364-4d5e-a3a4-9acaf4b182fe
let
	g = equidistant_grid(0,1, 40)
	u = map(gaussian(x₀=0.4, σ=0.2), g)
	∂ = first_derivative(g, stencil_set)
	∂u = ∂*u
	
	fig = Figure()
	ax = Axis(fig[1,1])

	scatter!(ax,g,u, label="u")
	scatter!(ax,g,∂u, label="∂u")

	Legend(fig[1,2], ax)
	fig
end

# ╔═╡ 18659bfe-03ec-4fc0-8a28-a11ef6f382b5
bathymetry(x) = -1 +0.1sin(4x) + 0.07*sin(2x) + 0.2sin(x+7) + 0.7gaussian(x; x₀=4, σ=1)

# ╔═╡ ef987bd8-a160-4d5c-8869-f074a9512ff0
g_example = let
	c₁(t) = @SVector[t*10, bathymetry(t*10)]
	A = @SVector[c₁(0)[1],0]
	B = @SVector[c₁(1)[1],0]
	c₂,c₃,c₄ = Grids.linesegments(c₁(1),B,A,c₁(0))
	
	chart = Chart(
		Grids.TransfiniteInterpolationSurface(c₁,c₂,c₃,c₄),
		unitsquare(),
	)

	chart isa Chart
	equidistant_grid(chart, 100,15)
end

# ╔═╡ 669a569d-0188-44b0-9ff4-ffac82bd6d0c
let
	gf = map(g_example) do x
		x₀ = @SVector[2,-0.5]
		sin(5norm(x-x₀))
	end

	fig = Figure(size=(900,400))
	ax = Axis(fig[1,1];
		aspect=DataAspect(),
		limits = ((-0.1, 10.1),(-1.5,0.1)),
	)
	scatter!(ax,g_example, markersize=6)

	ax = Axis(fig[2,1];
		aspect=DataAspect(),
		limits = ((-0.1, 10.1),(-1.5,0.1)),
	)
	plot!(ax,g_example,gf)

	fig
end |> WideCell

# ╔═╡ 93f31307-f4e0-4ee2-927e-e9215f5a8fa0
let
	bid =  Grids.CartesianBoundary{2,LowerBoundary}()
	n = normal(g_example,bid)
	I = boundary_indices(g_example, bid)

	x = map(i->g_example[i], I)


	fig = Figure(size=(900,220))
	ax = Axis(fig[1,1];
		aspect=DataAspect(),
		limits = ((-0.1, 10.1),(-1.5,0.1)),
	)
	# scatter!(ax,g_example, markersize=6)
	plot!(ax,g_example, map(det,jacobian(g_example)))
	arrows2d!(x,0.2n)

	fig
end |> WideCell

# ╔═╡ 85bd6c2f-f6ea-44e7-a612-fb242dc58f94
g = let
	c₁(t) = @SVector[t*6, bathymetry(t*6)]
	A = @SVector[c₁(0)[1],0]
	B = @SVector[c₁(1)[1],0]
	c₂,c₃,c₄ = Grids.linesegments(c₁(1),B,A,c₁(0))
	
	chart = Chart(
		Grids.TransfiniteInterpolationSurface(c₁,c₂,c₃,c₄),
		unitsquare(),
	)

	chart isa Chart
	equidistant_grid(chart, 400,60)

	# equidistant_grid((0,-1),(10,0), 100, 15)
	# equidistant_grid((0,-1),(10,0), 400, 60)
	# equidistant_grid(0,10, 100)
end

# ╔═╡ 38526021-01d8-4bf6-b308-2bbbb46714af
let
	fig = Figure(size=(900,200))
	ax = Axis(fig[1,1];
		aspect=DataAspect(),
		limits=((-0.1,6.1), nothing),
	)
	scatter!(ax,g, markersize=3)
	fig
end |> WideCell

# ╔═╡ 5527e5a6-1054-4f2e-b341-fa74c6ce16f3
begin
	Δ = laplace(g, stencil_set)

	H⁻¹ = inverse_inner_product(g, stencil_set)

	b_bottom = CartesianBoundary{2, LowerBoundary}()
	b_surface = CartesianBoundary{2, UpperBoundary}()
	b_left = CartesianBoundary{1, LowerBoundary}()
	b_right = CartesianBoundary{1, UpperBoundary}()
	
	e_bottom = boundary_restriction(g, stencil_set, b_bottom)
	d_bottom = normal_derivative(g, stencil_set, b_bottom)
	Hᵧ_bottom = inner_product(boundary_grid(g,b_bottom), stencil_set)

	L = Δ - H⁻¹∘e_bottom'∘Hᵧ_bottom∘d_bottom

	L̃ = sparse(L) # CHEATING?!??!?
end;

# ╔═╡ fefb611d-d89a-44ea-a29e-49ab734d9519
begin
	u₀ = map(g) do x
		gaussian(x, x₀=@SVector[2.5,-0.5], σ=0.05)
	end
	u̇₀ = map(x->0., g)
end;

# ╔═╡ 696385cb-48dc-4914-99fa-ebda006b4b5a
let
	v = copy(u₀)

	v .= L*u₀
end

# ╔═╡ 6f161a1c-f9eb-4d1f-84f3-22e97bf0b342
let
	v = copy(u₀)

	for I∈ CartesianIndices(v)
		v[I] = (L*u₀)[I]
	end
end

# ╔═╡ 9057d18f-7d48-4c43-be92-c0c0e5bb5146
begin
	Δt = 0.5*min_spacing(g)
	N = 1200

	ts = (0:N)*Δt

	u = Vector{Matrix{Float64}}(undef, N+1)
	
	u[1] = u₀
	u[2] = u₀ + Δt*u̇₀

	Luₙ = copy(u₀)
	for n ∈ 2:N
		mul!(reshape(Luₙ,:), L̃, reshape(u[n],:)) 
		u[n+1] = 2u[n] - u[n-1] + Δt^2*Luₙ

		for bid ∈ (b_surface, b_left, b_right)
			Is = boundary_indices(g, bid)
			u[n+1][Is] .= 0.
		end
	end
end

# ╔═╡ d0ac23fa-260b-4d21-b5c2-06a2e817eba9
u2 = let
	Δt = 0.5*min_spacing(g)
	N = 1200

	ts = (0:N)*Δt

	u2 = Vector{Matrix{Float64}}(undef, N+1)
	
	u2[1] = u₀
	u2[2] = u₀ + Δt*u̇₀

	Luₙ = copy(u₀)
	for n ∈ 2:N
		u2[n+1] = similar(u₀)
		u2[n+1] .= 2u2[n] - u2[n-1] + Δt^2*L*u2[n]

		for bid ∈ (b_surface, b_left, b_right)
			Is = boundary_indices(g, bid)
			u2[n+1][Is] .= 0.
		end
	end

	u2
end

# ╔═╡ b160165e-6f2a-4346-81a8-29c5f7d23e53
n_anim, fig_solution = let
	DMExt = Base.get_extension(Diffinitive, :DiffinitiveMakieExt)
	
	n = Observable(1)

	uₙ = @lift u2[$n]

	fig = Figure(size=(900,400))
	ax = Axis(fig[1,1];
		aspect=DataAspect(),
		limits=((-0.1,6.1),(-1.1,0.1))
	)


	r = @lift DMExt.verticies_and_faces_and_values(g, $uₙ)
    v,f,c = (@lift $r[1]), (@lift $r[2]), (@lift $r[3])
    mesh!(ax, v, f;
        color=c,
		colorrange=(-0.7,0.7),
		colormap = :seismic,
        shading = NoShading,
    )

	foreach(boundary_identifiers(g)) do bid
		gb = boundary_grid(g, bid)
		lines!(ax,collect(gb);
			color=:black,
		)
	end


	
	# plot!(ax,g,uₙ, colorrange=(-1,1))

	time_slider = Makie.Slider(fig[2,1], range=1:N+1, startvalue=1)
	connect!(n, time_slider.value)

	n,fig
end;

# ╔═╡ 9e11a227-2aca-4c44-bb56-a914ae2a3a34
 fig_solution |> WideCell

# ╔═╡ bd79c82f-64d0-473b-9be6-a3db3cee78e3
if animate > 0
	Threads.@spawn for n ∈ 1:length(u)
		n_anim[] = n
		sleep(6/length(u))
	end
end

# ╔═╡ e3c3dd63-beff-4204-9524-56219b75067d
let
	xs = range(0,10,200)
	ys = map(bathymetry,xs)

	fig = Figure()
	ax = Axis(fig[1,1];
		aspect=DataAspect(),
		limits=(nothing, (-1.3, 0.1)),
	)
	lines!(xs,ys)

	fig
end

# ╔═╡ ba9a5f3c-116f-11f1-9e9b-95257ff43cba
md"""
## Appendix
"""

# ╔═╡ 28e8ac28-60e8-41f9-bc21-13b7f9fd7772
PlutoUI.TableOfContents()

# ╔═╡ a7a6942a-3959-4de3-ab19-2ed33f50cba6
md"""
### Inference experiments
"""

# ╔═╡ 100fcf7a-0078-4c59-af91-170f3a072941
let
	sz(d) = ntuple(i->100, d)
	ll(d) = ntuple(i->0., d)
	lu(d) = ntuple(i->1., d)
	
	g1 = equidistant_grid(ll(1)[1], lu(1)[1], sz(1)...)
	g2 = equidistant_grid(ll(2), lu(2), sz(2)...)
	g3 = equidistant_grid(ll(3), lu(3), sz(3)...)
	
	v1 = rand(sz(1)...)
	v2 = rand(sz(2)...)
	v3 = rand(sz(3)...)
	
	u1 = rand(sz(1)...)
	u2 = rand(sz(2)...)
	u3 = rand(sz(3)...)
	
	stencil_set = read_stencil_set(joinpath(sbp_operators_path(),"standard_diagonal.toml"); order=4)

end

# ╔═╡ 542ce447-2512-4878-a696-9a1ef1df25dc


# ╔═╡ ce741462-87f4-4dd0-89e1-e191b5e44899
# @edit LazyTensors.apply(L,u, 1, 1)

# ╔═╡ 81d57b0f-1482-43a8-93a3-19d014c09952
function fullapply1!(Au, A, u)
	Au .= A*u
end

# ╔═╡ d667457d-1ed3-4acd-8a9c-864be22c18a8
LazyTensors.apply(Δ,u₀,10,10)

# ╔═╡ 482a9c45-678f-48e4-aa10-2f0b5110dc9e
# @code_warntype LazyTensors.apply(Δ,u₀,10,10)

# ╔═╡ a5171d1a-a6f9-4a57-a28b-5b029b33174a
function fullapply2!(Au, A, u)
	for I ∈ CartesianIndices(Au)
		Au[I] = (A*u)[I]
	end
end

# ╔═╡ e6b7b20a-d51b-460e-b4b5-783a4b46da85
v₀ = copy(u₀);

# ╔═╡ 656f21a7-85b6-40c2-b84b-4f86bac7e630
@benchmark fullapply1!($v₀, $L, $u₀)

# ╔═╡ 1d4d59bc-14f2-40f4-aba0-66106a9a22ee
@benchmark mul!($(reshape(v₀,:)), $L̃, $(reshape(u₀,:)))

# ╔═╡ 18b5a43c-9e01-45a4-9daf-b4e1e3ed3b1b
@benchmark fullapply2!($v₀, $L, $u₀)

# ╔═╡ Cell order:
# ╟─7fe19dc3-8c4a-4722-8c4e-ab4f6892bebb
# ╟─5c41b08e-373e-4da9-beac-fd0238df6efd
# ╟─5f21c1c8-f0c9-4d25-9c5f-e5a0892022b5
# ╟─d7493595-2d8f-4a9a-a7ac-0d8d9f31217f
# ╠═7c2667cd-aefd-458f-aff4-44e8559db226
# ╟─d885f8b5-fc6b-48f5-8a9d-a6c24823ac90
# ╠═03c27b5a-78f1-45db-8126-da91b2a511bf
# ╟─d3de3989-16b1-4b38-852c-297eb9bddef2
# ╠═ef987bd8-a160-4d5c-8869-f074a9512ff0
# ╟─e75701fe-8e91-4acf-8542-203499826bdd
# ╠═387b67e1-87a4-472a-a560-a8f44d077826
# ╠═669a569d-0188-44b0-9ff4-ffac82bd6d0c
# ╟─f2a53605-5e33-46fe-8cf9-5f604d5e2a7b
# ╠═93f31307-f4e0-4ee2-927e-e9215f5a8fa0
# ╟─e05fa744-83e2-486c-9409-3d4e1679fc26
# ╟─55f726bc-5e4b-466c-876a-a744d6b383f8
# ╠═b0bde7dc-0193-48c9-8a23-fc5741d26b05
# ╠═aad9cc2f-b181-4094-baa0-032cc758e2e6
# ╟─5fbdcd21-b2d2-47c5-a0e8-5dc7f402649e
# ╠═6d161e75-d7c3-4a59-852e-cfc59ec451da
# ╠═e44bd4eb-5f37-4b45-b7ea-5d05cc8559cc
# ╟─c68b294c-13a2-4d0b-8a08-29495436bf11
# ╠═ba825d17-2914-49bd-841a-e5c262308796
# ╠═df62465e-2e84-4d9e-a659-0e82145f9c34
# ╟─909536ac-5504-4b98-8a38-acf282936964
# ╠═de36950b-a1e8-4a4b-b078-7daf698cecb1
# ╟─3122c6af-d6b0-48fe-bf57-3aaf5f72f94b
# ╠═c35802c5-a364-4d5e-a3a4-9acaf4b182fe
# ╟─9928d201-7961-4d3d-bb89-90884fb569c9
# ╠═2acab9cf-94f9-46dc-9f25-36494571f57e
# ╟─720c9f3a-06cd-4768-b050-c5b19bf76bc2
# ╠═38ba2fdc-d2a1-4aec-858d-f61c0695a7f9
# ╟─2ca95a49-8869-4b9d-a05e-7e1b727b17b7
# ╠═a56fb5bd-1b90-471d-8fc3-dbb882fd6e01
# ╠═bcec3a56-4728-43f9-82b3-408ed8769c03
# ╟─f502399d-e04e-4146-be6f-474f55b80977
# ╠═85bd6c2f-f6ea-44e7-a612-fb242dc58f94
# ╟─38526021-01d8-4bf6-b308-2bbbb46714af
# ╟─917691bc-77e9-4d73-9b6f-39577a5cd425
# ╟─1ad1c41e-a09e-48e9-8bbb-f2c0a6c3b66f
# ╠═5527e5a6-1054-4f2e-b341-fa74c6ce16f3
# ╠═696385cb-48dc-4914-99fa-ebda006b4b5a
# ╠═6f161a1c-f9eb-4d1f-84f3-22e97bf0b342
# ╟─05fddef1-530b-4965-950b-761bfecf3a22
# ╠═de920e8e-1594-4a5e-a58a-65519149c933
# ╟─b3762de6-6262-4a94-aa58-a66cd3545dcf
# ╟─f29d3fe6-10ac-4af4-9162-c884d8899c34
# ╠═fefb611d-d89a-44ea-a29e-49ab734d9519
# ╠═9057d18f-7d48-4c43-be92-c0c0e5bb5146
# ╠═d0ac23fa-260b-4d21-b5c2-06a2e817eba9
# ╟─9e11a227-2aca-4c44-bb56-a914ae2a3a34
# ╟─da7cb259-0ae6-406c-9972-632e1fb71f0d
# ╟─6e84cdb8-eebc-47e1-8890-ff6a0fac8a7f
# ╠═bd79c82f-64d0-473b-9be6-a3db3cee78e3
# ╠═b160165e-6f2a-4346-81a8-29c5f7d23e53
# ╟─052455c2-0b96-460e-8d66-ae188918b54d
# ╟─cd6affc4-1926-4e4f-a850-47ebc048784f
# ╟─d0d05031-b49f-4f59-8de9-9213d038a00a
# ╟─9fbb9753-4ca6-4095-a3c9-b9ed0b98aa5a
# ╠═b1a9bb25-e1f2-4142-854e-419ab3d1b0a1
# ╠═18659bfe-03ec-4fc0-8a28-a11ef6f382b5
# ╠═e3c3dd63-beff-4204-9524-56219b75067d
# ╟─ba9a5f3c-116f-11f1-9e9b-95257ff43cba
# ╠═70441495-6903-4929-a322-90419ff19042
# ╠═28e8ac28-60e8-41f9-bc21-13b7f9fd7772
# ╟─a7a6942a-3959-4de3-ab19-2ed33f50cba6
# ╠═100fcf7a-0078-4c59-af91-170f3a072941
# ╠═542ce447-2512-4878-a696-9a1ef1df25dc
# ╠═ce741462-87f4-4dd0-89e1-e191b5e44899
# ╠═81d57b0f-1482-43a8-93a3-19d014c09952
# ╠═d667457d-1ed3-4acd-8a9c-864be22c18a8
# ╠═482a9c45-678f-48e4-aa10-2f0b5110dc9e
# ╠═a5171d1a-a6f9-4a57-a28b-5b029b33174a
# ╠═ab2fed8f-9869-4f14-8d42-8f56b5c254df
# ╠═e6b7b20a-d51b-460e-b4b5-783a4b46da85
# ╠═656f21a7-85b6-40c2-b84b-4f86bac7e630
# ╠═1d4d59bc-14f2-40f4-aba0-66106a9a22ee
# ╠═18b5a43c-9e01-45a4-9daf-b4e1e3ed3b1b

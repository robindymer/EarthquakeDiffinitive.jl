### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ ac8de83a-17d3-450f-9f8f-3f4239232ebc
begin
	using Pkg
	Pkg.Registry.add(url = "https://github.com/Diffinitive/diffinitive_registry.git")
end

# ╔═╡ 7424297a-0722-11f1-802e-73c5586716fa
using Diffinitive, Diffinitive.Grids, Diffinitive.SbpOperators, Diffinitive.LazyTensors, Diffinitive.RegionIndices

# ╔═╡ 8fc174a5-c063-46a9-8a6b-63b95ca8734b
using StaticArrays, StructArrays

# ╔═╡ b453ef6f-e571-48be-ac69-4c7ed462edae
md"""
This notebook contains an example implementation of a discrete elastic operator for isotropic materials based on summation-by-parts finite difference operators available in the library [Diffinitive.jl](https://github.com/Diffinitive). A related implementation of the traction operator is also included.

Tests include accuracy, and verification of that the summation-by-parts property. Furthermore the operators are profiled for performance.

TODO:s

	- Operators on mapped grids
	- Interfaces
	- Get the general D-dimensional operator applications to be performant. Currently inference fails resulting in allocations, trashing performance. Specialialized non-allocating functions are handcoded for the 2D and 3D-case.
"""

# ╔═╡ 1312c3cc-1275-44ac-8669-545cdd9bd32e
md"""
Either add the [Diffinitive registry](https://github.com/Diffinitive/diffinitive_registry) to your Julia installation, or enable and run the cell below.
"""

# ╔═╡ 1d5a0c7a-d24b-4d28-b26d-fb1b16e81d81
operator_path = sbp_operators_path()*"standard_diagonal.toml"

# ╔═╡ 8ccb065f-1829-4efd-aea9-0e4200a28a69
stencil_set = read_stencil_set(operator_path, order = 4)

# ╔═╡ 74a212a8-3311-4fed-9d43-29de07d34206
md"""
# IsotropicElasticOperator

A discrete elastic operator for isotropic materials, implemented using the SBP finite difference operators in Diffinitive.jl.
"""

# ╔═╡ 63c5b893-1edd-4d7d-b132-7f5f9c344b7c
md"""
## Accuracy test

Check that we differentiate polynomial functions (including variable material parameters) exactly
"""

# ╔═╡ bc9f841e-9a6f-4f98-b999-651f6179521f
md"""
### 2D
"""

# ╔═╡ 83492e6d-c00e-42f9-872e-d74684cee180
md"""
### 3D
"""

# ╔═╡ 7d8938f7-3d8a-4ac3-a2f4-46445e3a4ec2
md"""
## Test using `StructArrays` for a different storage format for the grid functions 
"""

# ╔═╡ ac977d73-4803-433b-9738-8b16170d0d89
md"""
Implement `componentview` on `StructArray` to test if we can use other data layouts for  vector-valued grid functions
"""

# ╔═╡ 12b4aa2b-ea9b-4072-b4c9-79d08f7f1967
Grids.componentview(v::StructArray, component_index...) = components(v)[component_index...]

# ╔═╡ 859df5f4-0941-4406-b5f5-ae362ec535fd
begin
	struct IsotropicElasticOperator{T, D, TM1 <: NTuple{D, NTuple{D,LazyTensor{T,D,D}}}, TM2 <: NTuple{D, NTuple{D,LazyTensor{T,D,D}}}} <: LazyTensor{T, D, D}
		second_derivatives_lambda::TM1 # Dᵢⱼ(λ)
		second_derivatives_mu::TM2 # Dᵢⱼ(μ)
		size::NTuple{D,Int}
		stencil_set::StencilSet
	end
	
	function IsotropicElasticOperator(g, lambda, mu, stencil_set)
		λ = DiagonalTensor(lambda)
		μ = DiagonalTensor(mu)

		# Dᵢⱼ(λ)
		# Note: The diagonal terms Dᵢᵢ(λ) should still use the wide stencil
		# second derivative Dᵢ∘λ∘Dᵢ to avoid dispersion errors for problems
		# with large discrepancies between λ and μ.
		Dᵢⱼλ = ntuple(Val(ndims(g))) do i
						ntuple(Val(ndims(g))) do j
							Dᵢ = first_derivative(g, stencil_set, i)
							Dⱼ = first_derivative(g, stencil_set, j)
							return Dᵢ ∘ λ ∘ Dⱼ
						end
		end

		# Dᵢⱼ(μ)
		# Note: Here we use narrow-stencil second derivatives for the diagonal
		# terms.
		Dᵢⱼμ = ntuple(Val(ndims(g))) do i
						ntuple(Val(ndims(g))) do j
							if i == j
								return second_derivative_variable(g, mu, stencil_set, i)
							else
								Dᵢ = first_derivative(g, stencil_set, i)
								Dⱼ = first_derivative(g, stencil_set, j)
								return Dᵢ ∘ μ ∘ Dⱼ
							end
						end
		end
		
		return IsotropicElasticOperator(Dᵢⱼλ, Dᵢⱼμ, size(g), stencil_set)
	end
	
	IsotropicElasticOperator(g, stencil_set) =  IsotropicElasticOperator(g, ones(size(g)), ones(size(g)), stencil_set)
	
	LazyTensors.range_size(op::IsotropicElasticOperator) = op.size
	LazyTensors.domain_size(op::IsotropicElasticOperator) = op.size

	# TODO: T and Tu should be the same type but currently Tu can be e.g.
	# SVector{D,T}.
	@inline function LazyTensors.apply(op::IsotropicElasticOperator{T,D}, u::AbstractArray{Tu,D}, I...) where {T,Tu,D}
		dim = domain_dim(op)
		return ntuple(Val(dim)) do j
			@inline
			uⱼ = componentview(u, j)
			
			## Diagonal terms, i = j
			# μ term
			Dⱼⱼμ = op.second_derivatives_mu[j][j] 
			res = T(2)*apply(Dⱼⱼμ, uⱼ, I...) # ∂ⱼμ∂ⱼuⱼ
			# λ term
			for i in 1:dim
				Dⱼᵢλ = op.second_derivatives_lambda[j][i] # ∂ⱼλ∂ₖuₖ
				uᵢ = componentview(u, i)
				res += apply(Dⱼᵢλ, uᵢ, I...)
			end
			
			## Off-diagonal terms, i != j
			# TODO: Loop split due to worse inference
			# 		Should be able to merge the loops using
			# 		Iterators.flatten((1:j-1, j+1:dim))
			for i in 1:j-1
				Dᵢᵢμ = op.second_derivatives_mu[i][i]
				Dᵢⱼμ = op.second_derivatives_mu[i][j]
				uᵢ = componentview(u, i)
				# ∂ᵢμ∂ᵢuⱼ + ∂ᵢμ∂ⱼuᵢ
				res += apply(Dᵢᵢμ, uⱼ, I...) + apply(Dᵢⱼμ, uᵢ, I...)
			end
			for i in j+1:dim
				Dᵢᵢμ = op.second_derivatives_mu[i][i]
				Dᵢⱼμ = op.second_derivatives_mu[i][j]
				uᵢ = componentview(u, i)
				# ∂ᵢμ∂ᵢuⱼ + ∂ᵢμ∂ⱼuᵢ
				res += apply(Dᵢᵢμ, uⱼ, I...) + apply(Dᵢⱼμ, uᵢ, I...)
			end

			# # Alternative, compact, but probably less performant implementation
			# comp_res = ntuple(Val(dim)) do i
			# 	@inline
			# 	uᵢ = componentview(u, i)
			# 	@inbounds Dⱼᵢλ = op.second_derivatives_lambda[j][i]
			# 	@inbounds Dᵢᵢμ = op.second_derivatives_mu[i][i]
			# 	@inbounds Dᵢⱼμ = op.second_derivatives_mu[i][j]
			# 	return @inline apply(Dⱼᵢλ, uᵢ, I...) + @inline apply(Dᵢᵢμ, uⱼ, I...) + @inline apply(Dᵢⱼμ, uᵢ, I...)
			# end
			# res = +(comp_res...)
			
			return res
		end |> SVector
	end

	## NOTE:
	# The below functions are specializations for 2D and 3D cases.
	# These should NOT really be necessary, but for some reason
	# inference fails in the general case, leading to allocations
	# and poor performance.
	@inline function LazyTensors.apply(op::IsotropicElasticOperator{T,2}, u::AbstractMatrix, I...) where T
		u1 =  componentview(u, 1)
		u2 = componentview(u, 2)

		D₁₁λ = op.second_derivatives_lambda[1][1]
		D₁₂λ = op.second_derivatives_lambda[1][2]
		D₂₁λ = op.second_derivatives_lambda[2][1]
		D₂₂λ = op.second_derivatives_lambda[2][2]

		D₁₁μ = op.second_derivatives_mu[1][1]
		D₁₂μ = op.second_derivatives_mu[1][2]
		D₂₁μ = op.second_derivatives_mu[2][1]
		D₂₂μ = op.second_derivatives_mu[2][2]
	
		# Component 1
		res1 =  apply(D₁₁λ, u1, I...) +
				apply(D₁₂λ, u2, I...) +
			  2*apply(D₁₁μ, u1, I...) +
			 	apply(D₂₂μ, u1, I...) + 
			 	apply(D₂₁μ, u2, I...)
		
		# Component 2
		res2 =  apply(D₂₁λ, u1, I...) +
				apply(D₂₂λ, u2, I...) +
			  2*apply(D₂₂μ, u2, I...) +
			 	apply(D₁₁μ, u2, I...) + 
			 	apply(D₁₂μ, u1, I...)
		
		return SVector{2}(res1, res2)
	end

	@inline function LazyTensors.apply(op::IsotropicElasticOperator{T,3}, 			u::AbstractArray{Tu,3}, I...) where {T,Tu}
		u1 = componentview(u, 1)
		u2 = componentview(u, 2)
		u3 = componentview(u, 3)

		D₁₁λ = op.second_derivatives_lambda[1][1]
		D₁₂λ = op.second_derivatives_lambda[1][2]
		D₁₃λ = op.second_derivatives_lambda[1][3]
		D₂₁λ = op.second_derivatives_lambda[2][1]
		D₂₂λ = op.second_derivatives_lambda[2][2]
		D₂₃λ = op.second_derivatives_lambda[2][3]
		D₃₁λ = op.second_derivatives_lambda[3][1]
		D₃₂λ = op.second_derivatives_lambda[3][2]
		D₃₃λ = op.second_derivatives_lambda[3][3]

		D₁₁μ = op.second_derivatives_mu[1][1]
		D₁₂μ = op.second_derivatives_mu[1][2]
		D₁₃μ = op.second_derivatives_mu[1][3]
		D₂₁μ = op.second_derivatives_mu[2][1]
		D₂₂μ = op.second_derivatives_mu[2][2]
		D₂₃μ = op.second_derivatives_mu[2][3]
		D₃₁μ = op.second_derivatives_mu[3][1]
		D₃₂μ = op.second_derivatives_mu[3][2]
		D₃₃μ = op.second_derivatives_mu[3][3]
		
		# Component 1
		res1 =  apply(D₁₁λ, u1, I...) +
				apply(D₁₂λ, u2, I...) +
				apply(D₁₃λ, u3, I...) +
		   2*apply(D₁₁μ, u1, I...) +
				apply(D₂₂μ, u1, I...) +
				apply(D₂₁μ, u2, I...) +
				apply(D₃₁μ, u3, I...) +
				apply(D₃₃μ, u1, I...)
				
		# Component 2
		res2 =  apply(D₂₁λ, u1, I...) +
				apply(D₂₂λ, u2, I...) +
				apply(D₂₃λ, u3, I...) +
			    apply(D₁₁μ, u2, I...) +
				apply(D₁₂μ, u1, I...) +
		   2*apply(D₂₂μ, u2, I...) +
				apply(D₃₂μ, u3, I...) +
			 	apply(D₃₃μ, u2, I...)

		# Component 3
		res3 =  apply(D₃₁λ, u1, I...) +
				apply(D₃₂λ, u2, I...) +
				apply(D₃₃λ, u3, I...) +
			    apply(D₁₁μ, u3, I...) +
				apply(D₁₃μ, u1, I...) +
			    apply(D₂₂μ, u3, I...) +
				apply(D₂₃μ, u2, I...) +
		   2*apply(D₃₃μ, u3, I...)
		
		return SVector{3,T}(res1, res2, res3)
	end
end

# ╔═╡ 76fb4bec-8445-42ef-b0a6-cb9e96918f24
let 
	g = equidistant_grid(unitsquare(Float64),41,41)
	## Test accuracy
	u_res = map(zero, g) # Initialize result vector
	# u = [x, y²] with λ = μ = 1
	u = map(x -> @SVector[x[1],x[2]^2], g)
	# Analytic solution is [0; 6]
	u_true = map(x -> @SVector[0., 6.], g)
	# Create elastic operator and compute approximate solution
	E = IsotropicElasticOperator(g, stencil_set)
	u_res .= E*u
	println("Correct results for u = [x, y²] with λ = 1, μ = 1: " * string(u_res ≈ u_true ))

	# u = [x, y²] with λ = y, μ = x
	λ = map(x -> x[2], g)
	μ = map(x -> x[1], g)
	# Analytic solution is [2, 1 + 4y + 4x]
	u_true = map(x -> @SVector[2., 1. + 4. * x[2] + 4. *x[1]], g)
	E = IsotropicElasticOperator(g, λ, μ, stencil_set)
	u_res .= E*u
	println("Correct results for u = [x, y²] with λ = y, μ = x: " * string(u_res ≈ u_true ))

	# u = [x, y²] with λ = x², μ = x
	λ = map(x -> x[1]^2, g)
	μ = map(x -> x[2], g)
	# Analytic solution is [2x + 4xy, 2x² + 8y]
	u_true = map(x -> @SVector[2. * x[1] + 4. * x[1] * x[2], 2. * x[1]^2 + 8. *x[2]], g)
	E = IsotropicElasticOperator(g, λ, μ, stencil_set)
	u_res .= E*u
	println("Correct results for u = [x, y²] with λ = x², μ = x: " * string(u_res ≈ u_true ))

	# u = [y, x] with λ = x, μ = y
	u = map(x -> @SVector[x[2],x[1]], g)
	λ = map(x -> x[1], g)
	μ = map(x -> x[2], g)
	# Analytic solution is [2., 0.]
	u_true = map(x -> @SVector[2. , 0.], g)
	E = IsotropicElasticOperator(g, λ, μ, stencil_set)
	u_res .= E*u
	println("Correct results for u = [y, x] with λ = x, μ = y: " * string(u_res ≈ u_true ))

	# u = [y, x] with λ = y, μ = xy
	λ = map(x -> x[2], g)
	μ = map(x -> x[1]*x[2], g)
	# Analytic solution is [2x, 2y]
	u_true = map(x -> @SVector[2*x[1] , 2*x[2]], g)
	E = IsotropicElasticOperator(g, λ, μ, stencil_set)
	u_res .= E*u
	println("Correct results for u = [y, x] with λ = y, μ = xy: " * string(u_res ≈ u_true ))
end

# ╔═╡ 229e9c52-ebc3-485c-a000-d493fc5ea942
let 
	g = equidistant_grid(unitcube(Float64),21,21,21)
	# u = [x, y², xz] with λ = μ = 1
	u = map(x -> @SVector[x[1], x[2]^2, x[1]*x[3]], g)
	u_res = copy(u)
	# Analytic solution is [0; 6]
	u_true = map(x -> @SVector[2., 6., 0.], g)
	# Create elastic operator and compute approximate solution
	E = IsotropicElasticOperator(g, stencil_set)
	u_res .= E*u
	println("Correct results for u = [x, y², xz] with λ = 1, μ = 1: " * string(u_res ≈ u_true ))
end

# ╔═╡ 67deb1fc-cfd5-43cf-9f40-0accbd4eb25a
let
	g = equidistant_grid(unitsquare(Float64),41,41)
	## Test storage format for solution
	λ = map(x -> x[2], g)
	μ = map(x -> x[1]*x[2], g)
	E = IsotropicElasticOperator(g, λ, μ, stencil_set)

	# Standard format
	u = map(x -> @SVector[x[2],x[1]], g)
	u_res = map(zero, g) # Initialize result vector
	u_res .= E*u

	# StructArray format
	v = StructArray{eltype(g)}((map(x -> x[2], g), map(x -> x[1], g)))
	v_res = map(zero, g)
	v_res .= E*v

	# Check if results are equal
	print("Different array formats give same results: " * string(u_res ≈ v_res ))
end

# ╔═╡ 390deec4-1496-44ee-99ab-a5e243672360
md"""
# IsotropicTractionOperator

A discrete traction operator for isotropic materials
"""

# ╔═╡ dbd40490-058f-4c66-80cd-1c6bcc69d427
md"""
To start with, we implement `normal` for TensorGrids, which currently doesnt exist in Diffinitive.jl
"""

# ╔═╡ 65a6863b-8bdc-4165-b067-da23ceb990f1
begin
	function Grids.normal(g::TensorGrid, boundary::TensorGridBoundary)
		return map(I -> tensor_grid_normal(g, boundary), boundary_indices(g, boundary))
	end

	function tensor_grid_normal(g, boundary)
		s = Grids._boundary_sign(component_type(g), boundary)
		return ntuple(Val(ndims(g))) do i
			return i == grid_id(boundary) ? s : zero(eltype(s))
		end |> SVector
	end
end

# ╔═╡ 4d66242a-ddcc-4a1a-8602-75c8b20fd7a1
md"""
Now we can implement the traction operator for isotropic materials, using a similar pattern as for `IsotropicElasticOperator`. 

Note that for TensorGrids, the normal will be zero apart from in the dimension given by `grid_id(boundary)`, so this could (should?) be simplified. Moreover, we don't really need the normal as a grid function in this case.
"""

# ╔═╡ 75edd61d-6f0d-4fad-9a18-917ef69547e6
begin
	struct IsotropicTractionOperator{T, R, D, TM1 <: NTuple{D, NTuple{D,LazyTensor{T,R,D}}}, TM2 <: NTuple{D, NTuple{D,LazyTensor{T,R,D}}}} <: LazyTensor{T, R, D}
	 	boundary_derivatives_lambda::TM1
	 	boundary_derivatives_mu::TM2
		range_size::NTuple{R,Int}
		domain_size::NTuple{D,Int}
	 	stencil_set::StencilSet
	end

	function IsotropicTractionOperator(g, lambda, mu, stencil_set, b)
		e = boundary_restriction(g, stencil_set, b)
		n = normal(g, b)
		λ = DiagonalTensor(collect(e*lambda))
		μ = DiagonalTensor(collect(e*mu))
		
		# λnᵢdⱼ
		# Note: The diagonal terms Dᵢᵢ(λ) should still use the wide stencil
		# second derivative Dᵢ∘λ∘Dᵢ to avoid dispersion errors for problems
		# with large discrepancies between λ and μ.
		λnᵢdⱼ = ntuple(Val(ndims(g))) do i
					ntuple(Val(ndims(g))) do j
						nᵢ = componentview(n, i)
						Nᵢ = DiagonalTensor(nᵢ)
						Dⱼ = first_derivative(g, stencil_set, j)
						return λ ∘ Nᵢ ∘ e ∘ Dⱼ
					end
		end

		# μnᵢdⱼ
		# Note: Here we use narrow-stencil second derivatives for the diagonal
		# terms.
		μnᵢdⱼ = ntuple(Val(ndims(g))) do i
					nᵢ = componentview(n, i) 
					Nᵢ = DiagonalTensor(nᵢ)
					ntuple(Val(ndims(g))) do j
						if i == j
							# TODO: On TensorGrids, normal_derivative is already
							# scaled with the sign of the normal, which means that
							# Nᵢdᵢ will have the incorrect sign. We should really use
							# the boundary derivative here, rather than the normal
							# derivative, since the operator nᵢdⱼ, is something else
							# than the normal derivative for i == j.
							dᵢ = normal_derivative(g, stencil_set, b)
							s = Grids._boundary_sign(component_type(g), b)
							return s * μ ∘ Nᵢ ∘ dᵢ # Remove multiplication with s, once we have changed to boundary_derivative
						else
							Dⱼ = first_derivative(g, stencil_set, j)
							return μ ∘ Nᵢ ∘ e ∘ Dⱼ
						end
					end
		end

		
		return IsotropicTractionOperator(λnᵢdⱼ,
										 μnᵢdⱼ,
										 size(boundary_grid(g, b)),
										 size(g),
										 stencil_set)
	end

	function IsotropicTractionOperator(g, stencil_set, boundary)
		return IsotropicTractionOperator(g, ones(size(g)), ones(size(g)), stencil_set, boundary)
	end

	LazyTensors.range_size(op::IsotropicTractionOperator) = op.range_size
	LazyTensors.domain_size(op::IsotropicTractionOperator) = op.domain_size
	
	# TODO: T and Tu should be the same type but currently Tu can be e.g.
	# SVector{D,T}.
	@inline function LazyTensors.apply(op::IsotropicTractionOperator{T,R,D}, u::AbstractArray{Tu,D}, I...) where {T,Tu,R,D}
		dim = domain_dim(op)
		return ntuple(Val(dim)) do j
			@inline
			uⱼ = componentview(u, j)
			## Diagonal terms, i = j
			# μ term
			μnⱼdⱼ = op.boundary_derivatives_mu[j][j]
			res = T(2)*apply(μnⱼdⱼ, uⱼ, I...)
			# λ term
			for i in 1:dim
				λnⱼdᵢ = op.boundary_derivatives_lambda[j][i]
				uₖ = componentview(u, i)
				res += apply(λnⱼdᵢ, uₖ, I...)
			end
			## Off-diagonal terms, i != j
			# TODO: Loop split due to worse inference
			# 		Should be able to merge the loops using
			# 		Iterators.flatten((1:j-1, j+1:dim))
			for i in 1:j-1
				μnᵢdᵢ = op.boundary_derivatives_mu[i][i]
				μnᵢdⱼ = op.boundary_derivatives_mu[i][j]
				uᵢ = componentview(u, i)
				res += apply(μnᵢdᵢ, uⱼ, I...) + apply(μnᵢdⱼ, uᵢ, I...)
			end
			for i in j+1:dim
				μnᵢdᵢ = op.boundary_derivatives_mu[i][i]
				μnᵢdⱼ = op.boundary_derivatives_mu[i][j]
				uᵢ = componentview(u, i)
				res += apply(μnᵢdᵢ, uⱼ, I...) + apply(μnᵢdⱼ, uᵢ, I...)
			end

			# # Alternative, compact, but probably less performant implementation
			# comp_res = ntuple(Val(dim)) do i
			# 	@inline
			# 	uᵢ = componentview(u, i)
			# 	@inbounds λnⱼdᵢ = op.boundary_derivatives_lambda[j][i]
			# 	@inbounds μnᵢdᵢ = op.boundary_derivatives_mu[i][i]
			# 	@inbounds μnᵢdⱼ = op.boundary_derivatives_mu[i][j]
			# 	return apply(λnⱼdᵢ, uᵢ, I...) + apply(μnᵢdᵢ, uⱼ, I...) + apply(μnᵢdⱼ, uᵢ, I...)
			# end
			# res = +(comp_res...)
			return res
		end |> SVector
	end

	## NOTE:
	# The below functions are specializations for 2D and 3D cases.
	# These should NOT really be necessary, but for some reason
	# inference fails in the general case, leading to allocations
	# and poor performance.
	@inline function LazyTensors.apply(op::IsotropicTractionOperator{T,1,2}, u::AbstractMatrix, I...) where T

		u1 = componentview(u, 1)
		u2 = componentview(u, 2)

		λn₁d₁ = op.boundary_derivatives_lambda[1][1]
		λn₁d₂ = op.boundary_derivatives_lambda[1][2]
		λn₂d₁ = op.boundary_derivatives_lambda[2][1]
		λn₂d₂ = op.boundary_derivatives_lambda[2][2]
			
		μn₁d₁ = op.boundary_derivatives_mu[1][1]
		μn₁d₂ = op.boundary_derivatives_mu[1][2]
		μn₂d₁ = op.boundary_derivatives_mu[2][1]
		μn₂d₂ = op.boundary_derivatives_mu[2][2]

		# Component 1
		res1 =  apply(λn₁d₁, u1, I...) +
				apply(λn₁d₂, u2, I...) +
			  2*apply(μn₁d₁, u1, I...) +
			 	apply(μn₂d₁, u2, I...) +
				apply(μn₂d₂, u1, I...)
		
		# Component 2
		res2 =  apply(λn₂d₁, u1, I...) +
				apply(λn₂d₂, u2, I...) +
			 	apply(μn₁d₁, u2, I...) + 
			 	apply(μn₁d₂, u1, I...) +
			  2*apply(μn₂d₂, u2, I...)
		
		return SVector{2}(res1, res2)
	end

	@inline function LazyTensors.apply(op::IsotropicTractionOperator{T,2,3}, u::AbstractArray{Tu,3}, I...) where {T,Tu}

		u1 = componentview(u, 1)
		u2 = componentview(u, 2)
		u3 = componentview(u, 3)

		λn₁d₁ = op.boundary_derivatives_lambda[1][1]
		λn₁d₂ = op.boundary_derivatives_lambda[1][2]
		λn₁d₃ = op.boundary_derivatives_lambda[1][3]
		λn₂d₁ = op.boundary_derivatives_lambda[2][1]
		λn₂d₂ = op.boundary_derivatives_lambda[2][2]
		λn₂d₃ = op.boundary_derivatives_lambda[2][3]
		λn₃d₁ = op.boundary_derivatives_lambda[3][1]
		λn₃d₂ = op.boundary_derivatives_lambda[3][2]
		λn₃d₃ = op.boundary_derivatives_lambda[3][3]

		μn₁d₁ = op.boundary_derivatives_mu[1][1]
		μn₁d₂ = op.boundary_derivatives_mu[1][2]
		μn₁d₃ = op.boundary_derivatives_mu[1][3]
		μn₂d₁ = op.boundary_derivatives_mu[2][1]
		μn₂d₂ = op.boundary_derivatives_mu[2][2]
		μn₂d₃ = op.boundary_derivatives_mu[2][3]
		μn₃d₁ = op.boundary_derivatives_mu[3][1]
		μn₃d₂ = op.boundary_derivatives_mu[3][2]
		μn₃d₃ = op.boundary_derivatives_mu[3][3]
		
		# Component 1
		res1 =  apply(λn₁d₁, u1, I...) +
				apply(λn₁d₂, u2, I...) +
				apply(λn₁d₃, u3, I...) +
		   2*apply(μn₁d₁, u1, I...) +
				apply(μn₂d₂, u1, I...) +
				apply(μn₂d₁, u2, I...) +
				apply(μn₃d₁, u3, I...) +
				apply(μn₃d₃, u1, I...)
				
		# Component 2
		res2 =  apply(λn₂d₁, u1, I...) +
				apply(λn₂d₂, u2, I...) +
				apply(λn₂d₃, u3, I...) +
			    apply(μn₁d₁, u2, I...) +
				apply(μn₁d₂, u1, I...) +
		   2*apply(μn₂d₂, u2, I...) +
				apply(μn₃d₂, u3, I...) +
			 	apply(μn₃d₃, u2, I...)

		# Component 3
		res3 =  apply(λn₃d₁, u1, I...) +
				apply(λn₃d₂, u2, I...) +
				apply(λn₃d₃, u3, I...) +
			    apply(μn₁d₁, u3, I...) +
				apply(μn₁d₃, u1, I...) +
			    apply(μn₂d₂, u3, I...) +
				apply(μn₂d₃, u2, I...) +
		   2*apply(μn₃d₃, u3, I...)
		
		return SVector{3,T}(res1, res2, res3)
	end
	
end

# ╔═╡ aa346240-3150-4923-ae13-1e84e86d0a7b
md"""
## Accuracy test

Check that we obtain the correct traction vector when applied to polynomial functions (including variable material parameters) exactly
"""

# ╔═╡ 6c9b06f7-8da1-43f1-8712-fe7ad5eaa8a9
md"""
### 2D
"""

# ╔═╡ 6ef92e9c-56df-4886-a25f-568669f71ba3
let 
	g = equidistant_grid(unitsquare(Float64),41,41)
	## Test accuracy
	boundaries = boundary_identifiers(g)

	# u = [x, y²] with λ = μ = 1
	u = map(x -> @SVector[x[1],x[2]^2], g)
	
	# West boundary: Analytic solution is -[2y + 3; 0]
	b_w = boundaries[1]
	t_res = map(zero, boundary_grid(g, b_w)) # Initialize result vector
	t_true = map(x -> @SVector[-2.0*x[2] - 3., 0.], boundary_grid(g, b_w))
	# Create elastic operator and compute approximate solution
	T_w = IsotropicTractionOperator(g, stencil_set, b_w)
	t_res .= T_w*u
	println("Correct results for t = [x, y²] with λ = 1, μ = 1 on west boundary: " * string(t_res ≈ t_true ))

	# North boundary: Analytic solution is [0; 6y + 1]
	b_n = boundaries[4]
	t_res = map(zero, boundary_grid(g, b_n)) # Initialize result vector
	t_true = map(x -> @SVector[0., 6.0*x[2] + 1.], boundary_grid(g, b_n))
	# Create elastic operator and compute approximate solution
	T_n = IsotropicTractionOperator(g, stencil_set, b_n)
	t_res .= T_n*u
	println("Correct results for t = [x, y²] with λ = 1, μ = 1 on north boundary: " * string(t_res ≈ t_true ))


	# u = [y, x] with λ = y, μ = xy
	λ = map(x -> x[2], g)
	μ = map(x -> x[1]*x[2], g)
	u = map(x -> @SVector[x[2],x[1]], g)
	
	# East boundary: Analytic solution is [2x, 2y]
	b_e = boundaries[2]
	t_res = map(zero, boundary_grid(g, b_e)) # Initialize result vector
	t_true = map(x -> @SVector[0. , 2*x[1]*x[2]], boundary_grid(g, b_e))
	T_e = IsotropicTractionOperator(g, λ, μ, stencil_set, b_e)
	t_res .= T_e*u
	println("Correct results for u = [y, x] with λ = y, μ = xy on east boundary: " * string(t_res ≈ t_true ))

	# East boundary: Analytic solution is [2x, 2y]
	b_e = boundaries[3]
	t_res = map(zero, boundary_grid(g, b_e)) # Initialize result vector
	t_true = map(x -> @SVector[-2*x[1]*x[2], 0.], boundary_grid(g, b_e))
	T_e = IsotropicTractionOperator(g, λ, μ, stencil_set, b_e)
	t_res .= T_e*u
	println("Correct results for u = [y, x] with λ = y, μ = xy on south boundary: " * string(t_res ≈ t_true ))
end

# ╔═╡ 51a06c87-e4be-48fb-a1de-3d3009495fc0
md"""
# SBP property test

Check that for random vectors ``u``, ``v``, and elastic/traction operators ``E``, ``T``:
```math
( v_i, (Eu)_i)_{\Omega} - ((Ev)_i, u_i)_{\Omega} = (v_i, (Tu)_i )_{\partial \Omega} - \left( (Tv)_i, u_i \right)_{\partial \Omega}
```
"""

# ╔═╡ 6d9acaae-c071-481a-8eff-d5903f1b6fd4
let
	g = equidistant_grid(unitcube(Float64),21,21,21)
	E = IsotropicElasticOperator(g, stencil_set) # Elastic Operators
	H = inner_product(g, stencil_set) # Inner product operator for Ω
	
	bs = boundary_identifiers(g)
	nb = length(bs)
	# Traction operators:
	T = ntuple(d -> IsotropicTractionOperator(g, stencil_set, bs[d]), Val(nb)) 
	# Boundary restriction operators:
	e = ntuple(d -> boundary_restriction(g, stencil_set, bs[d]), Val(nb)) 
	# Boundary inner product operators
	Hb = ntuple(d -> inner_product(boundary_grid(g, bs[d]), stencil_set), Val(nb))

	# Random vectors u ,v
	u = map(x -> @SVector(rand(ndims(g))), g)
	v = map(x -> @SVector(rand(ndims(g))), g)

	# Lazy apply of elastic/traction operators
	Eu = E*u
	Ev = E*v
	Tu = ntuple(d -> T[d]*u, Val(nb)) # For each boundary
	Tv = ntuple(d -> T[d]*v, Val(nb)) # For each boundary

	# Compute difference between volume term and boundary term for each dimesion and add the results
	diff = mapreduce(+, 1:ndims(g)) do i # for each dimension
		uᵢ = componentview(u, i)
		vᵢ = componentview(v, i)
		Euᵢ = componentview(Eu, i)
		Evᵢ = componentview(Ev, i)
		volume_term = sum( vᵢ .* (H*Euᵢ) - uᵢ .* (H*Evᵢ) )
		boundary_term = mapreduce(+, 1:nb) do j # for each boundary
			Tuᵢⱼ = componentview(Tu[j], i)
			Tvᵢⱼ = componentview(Tv[j], i)
			Hⱼ = Hb[j]
			eⱼ = e[j]
			return sum( (eⱼ*vᵢ) .* (Hⱼ*Tuᵢⱼ) - (eⱼ*uᵢ) .* (Hⱼ*Tvᵢⱼ) )
		end
		return volume_term - boundary_term # Difference along dimension i
	end
	println("Checking SBP property: ")
	println("Difference between volume terms and boundary terms: " * string(diff))
	println("(Should be zero in exact arithmetic)")
end

# ╔═╡ de193d02-33d8-4bc7-bb2e-739130ff4c53
md"""
# Profiling
"""

# ╔═╡ c742b160-6689-4874-b5a8-7b7f0c05145e
md"""
The profiling takes a bit of time to compile and run and are therefore initially disabled. Enable cells as you go about. Make sure to enable the `using` statements below first.
"""

# ╔═╡ 4d205e89-d560-43bc-8d41-ff8666c9b931
# ╠═╡ disabled = true
#=╠═╡
using BenchmarkTools, Profile, ProfileCanvas
  ╠═╡ =#

# ╔═╡ 3e456e9e-db79-46e4-a894-a270da3fc160
md"""
## IsotropicElasticOperator
"""

# ╔═╡ 8618d99b-a7c3-40c5-ae3e-ce9b64e3c7d8
md"""
### 2D apply w. region index
"""

# ╔═╡ ca44632d-c56d-40a1-9cc7-1051be519261
# ╠═╡ disabled = true
#=╠═╡
let
	g = equidistant_grid(unitsquare(Float64),21,21)
	u = map(x -> @SVector[x[1],x[2]^2], g)
	E = IsotropicElasticOperator(g, stencil_set)
	@benchmark LazyTensors.apply($E, $u, Index{Interior}(11), Index{Interior}(11))
end
  ╠═╡ =#

# ╔═╡ 555701fc-35cf-41fd-8c4f-9decea0b8ed0
md"""
### 2D apply w. Int index
"""

# ╔═╡ 36a43d04-d7e5-42a5-ac04-bc3616709d8d
# ╠═╡ disabled = true
#=╠═╡
let
	g = equidistant_grid(unitsquare(Float64),21,21)
	u = map(x -> @SVector[x[1],x[2]^2], g)
	E = IsotropicElasticOperator(g, stencil_set)
	@benchmark LazyTensors.apply($E, $u, 11, 11)
end
  ╠═╡ =#

# ╔═╡ 2bc37446-d8c0-491a-9853-f44db2050890
md"""
### 2D broadcast
"""

# ╔═╡ fe419612-3c9a-4e6b-a57f-c46aed41e873
# ╠═╡ disabled = true
#=╠═╡
let
	g = equidistant_grid(unitsquare(Float64),21,21)
	u = map(x -> @SVector[x[1],x[2]^2], g)
	v = copy(u)
	E = IsotropicElasticOperator(g, stencil_set)
	function profile_rhs!(v, E, u) 
	 	v .= E*u
	end
	@benchmark $profile_rhs!($v, $E, $u)
end
  ╠═╡ =#

# ╔═╡ cb0a07a8-9517-42ce-addf-edeb592e9226
md"""
### 2D broadcast w. StructArray
"""

# ╔═╡ c3b22830-6a5b-499d-8611-b0d7b819e4f2
# ╠═╡ disabled = true
#=╠═╡
let
	g = equidistant_grid(unitsquare(Float64),21,21)
	u = StructArray{eltype(g)}((map(x -> x[1], g), map(x -> x[2]^2, g)))
	v = copy(u)
	E = IsotropicElasticOperator(g, stencil_set)
	function profile_rhs!(v, E, u) 
	 	v .= E*u
	end
	@benchmark $profile_rhs!($v, $E, $u)
end
  ╠═╡ =#

# ╔═╡ 5fdb3825-56c4-41d6-b1f8-6c99b7db128a
md"""
### 3D apply w. region index
"""

# ╔═╡ 4aba61d8-849f-4eac-8c38-7527fe3b3bb9
# ╠═╡ disabled = true
#=╠═╡
let
	g = equidistant_grid(unitcube(Float64),21,21,21)
	u = map(x -> @SVector[x[1],x[2]^2, x[3]], g)
	E = IsotropicElasticOperator(g, stencil_set)
	@benchmark LazyTensors.apply($E, $u, Index{Interior}(11), Index{Interior}(11), Index{Interior}(11))
end
  ╠═╡ =#

# ╔═╡ 96647b06-fa51-48da-b124-7f6a66e63583
md"""
### 3D apply w. Int index
"""

# ╔═╡ cdedf7d0-94be-4470-8354-6c5ab549690a
# ╠═╡ disabled = true
#=╠═╡
let
	g = equidistant_grid(unitcube(Float64),21,21,21)
	u = map(x -> @SVector[x[1],x[2]^2, x[3]], g)
	E = IsotropicElasticOperator(g, stencil_set)
	@benchmark LazyTensors.apply($E, $u, 11, 11, 11)
end
  ╠═╡ =#

# ╔═╡ 12c1a799-3bfe-4bb2-b5a3-c12038e8de5a
md"""
### 3D broadcast
"""

# ╔═╡ 4be501b2-8440-4445-af57-683da23ad09e
# ╠═╡ disabled = true
#=╠═╡
let
	g = equidistant_grid(unitcube(Float64),21,21,21)
	u = map(x -> @SVector[x[1],x[2]^2, x[3]], g)
	v = copy(u)
	E = IsotropicElasticOperator(g, stencil_set)
	function profile_rhs!(v, E, u) 
	 	v .= E*u
	end
	@benchmark $profile_rhs!($v, $E, $u)
end
  ╠═╡ =#

# ╔═╡ 32947508-f74e-4502-9149-fa38de6db8b7
md"""
### 3D broadcast StructArray
"""

# ╔═╡ 44444fe1-3cea-4f0a-a3e9-c444b04f7da6
# ╠═╡ disabled = true
#=╠═╡
let
	g = equidistant_grid(unitcube(Float64),21,21,21)
	u = StructArray{eltype(g)}((map(x -> x[1], g), map(x -> x[2]^2, g), map(x -> x[3], g)))
	v = copy(u)
	E = IsotropicElasticOperator(g, stencil_set)
	function profile_rhs!(v, E, u) 
	 	v .= E*u
	end
	@benchmark $profile_rhs!($v, $E, $u)
end
  ╠═╡ =#

# ╔═╡ c1795732-1e42-4cf7-91c9-a98bd75b4ba8
md"""
### 3D Profiling with flame graph
"""

# ╔═╡ 2a0d83f3-733a-47ce-9fc7-5634598c67b9
# ╠═╡ disabled = true
#=╠═╡
let
	g = equidistant_grid(unitcube(Float64), 61,61,61)
	u = map(x -> @SVector[x[1],x[2]^2, x[3]], g)
	E = IsotropicElasticOperator(g, stencil_set)
	function profile_apply(E, u, iter)
		for i in 1:iter
			LazyTensors.apply(E, u, 31, 31, 31)
		end
	end
	Profile.clear()
	profile_apply(E,u,1)
	@profview profile_apply(E,u,10000)
end
  ╠═╡ =#

# ╔═╡ a4d9b55f-bfa4-4c5e-8e6d-fcac1b33fc4e
md"""
## IsotropicTractionOperator
"""

# ╔═╡ 37eb0fad-a227-4754-ad39-a215009ad5ca
md"""
### 2D apply
"""

# ╔═╡ d42dbfd2-4ae8-4623-9d04-c456a372771d
# ╠═╡ disabled = true
#=╠═╡
let
	g = equidistant_grid(unitsquare(Float64),21,21)
	boundaries = boundary_identifiers(g)
	b_w = boundaries[1]
	u = map(x -> @SVector[x[1], x[2]], g)
	T = IsotropicTractionOperator(g, stencil_set, b_w)
	@benchmark LazyTensors.apply($T, $u, 11)
end
  ╠═╡ =#

# ╔═╡ 1a8dc2ce-a0e4-4f49-a1f0-38abfc96c5b0
md"""
### 3D apply
"""

# ╔═╡ 9f6cce56-fd39-4440-b93e-d39217b81001
# ╠═╡ disabled = true
#=╠═╡
let
	g = equidistant_grid(unitcube(Float64),21,21,21)
	boundaries = boundary_identifiers(g)
	b_w = boundaries[1]
	u = map(x -> @SVector[x[1], x[2], x[3]], g)
	T = IsotropicTractionOperator(g, stencil_set, b_w)
	@benchmark LazyTensors.apply($T, $u, 11, 11)
end
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Diffinitive = "5a373a26-915f-4769-bcab-bf03835de17b"
Pkg = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
Profile = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
ProfileCanvas = "efd6af41-a80b-495e-886c-e51b0c7d77a3"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"

[compat]
BenchmarkTools = "~1.6.3"
Diffinitive = "~0.1.7"
ProfileCanvas = "~0.1.7"
StaticArrays = "~1.9.17"
StructArrays = "~0.7.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.6"
manifest_format = "2.0"
project_hash = "eaa5f8b6f40bd9c1d9b4abcbb9727e8000c044c5"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "7fecfb1123b8d0232218e2da0c213004ff15358d"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.3"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Diffinitive]]
deps = ["LinearAlgebra", "StaticArrays", "TOML"]
git-tree-sha1 = "7028a57cd8362f06edc7f79f84ada85ece72b935"
uuid = "5a373a26-915f-4769-bcab-bf03835de17b"
version = "0.1.7"

    [deps.Diffinitive.extensions]
    DiffinitiveMakieExt = "Makie"
    DiffinitivePlotsExt = "Plots"
    DiffinitiveSparseArrayKitExt = ["SparseArrayKit", "Tokens"]
    DiffinitiveSparseArraysExt = ["SparseArrays", "Tokens"]

    [deps.Diffinitive.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
    SparseArrayKit = "a9a3c162-d163-4c15-8926-b8794fbefed2"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Tokens = "040c2ec2-8d69-4aca-bf03-7d3a7092f2f6"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "c7345ab1a7ca4dc8a02c9f6510da0d9857bbe513"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.7.1"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OrderedCollections]]
git-tree-sha1 = "05f45c2e0de6259db764adbfd2f1dc6d3f8de13c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "2.0.1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "3de8f5e6e90ebfa8d6d1f86997d6cdcd6a912ff3"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.7"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.ProfileCanvas]]
deps = ["Base64", "JSON", "Pkg", "Profile", "REPL"]
git-tree-sha1 = "990016fb1508b0726a70039f39569720d054c78d"
uuid = "efd6af41-a80b-495e-886c-e51b0c7d77a3"
version = "0.1.7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "0f529006004a8be48f1be25f3451186579392d47"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.17"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "a2c37d815bf00575332b7bd0389f771cb7987214"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.2"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "2d0fc55c61321ba245c47be599570d11bac50303"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.5"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "0f38a06c83f0007bbab3cf911262841c9a0f07e0"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.13.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─b453ef6f-e571-48be-ac69-4c7ed462edae
# ╟─1312c3cc-1275-44ac-8669-545cdd9bd32e
# ╠═ac8de83a-17d3-450f-9f8f-3f4239232ebc
# ╠═7424297a-0722-11f1-802e-73c5586716fa
# ╠═8fc174a5-c063-46a9-8a6b-63b95ca8734b
# ╠═1d5a0c7a-d24b-4d28-b26d-fb1b16e81d81
# ╠═8ccb065f-1829-4efd-aea9-0e4200a28a69
# ╟─74a212a8-3311-4fed-9d43-29de07d34206
# ╠═859df5f4-0941-4406-b5f5-ae362ec535fd
# ╟─63c5b893-1edd-4d7d-b132-7f5f9c344b7c
# ╟─bc9f841e-9a6f-4f98-b999-651f6179521f
# ╠═76fb4bec-8445-42ef-b0a6-cb9e96918f24
# ╟─83492e6d-c00e-42f9-872e-d74684cee180
# ╠═229e9c52-ebc3-485c-a000-d493fc5ea942
# ╟─7d8938f7-3d8a-4ac3-a2f4-46445e3a4ec2
# ╟─ac977d73-4803-433b-9738-8b16170d0d89
# ╠═12b4aa2b-ea9b-4072-b4c9-79d08f7f1967
# ╠═67deb1fc-cfd5-43cf-9f40-0accbd4eb25a
# ╟─390deec4-1496-44ee-99ab-a5e243672360
# ╟─dbd40490-058f-4c66-80cd-1c6bcc69d427
# ╠═65a6863b-8bdc-4165-b067-da23ceb990f1
# ╟─4d66242a-ddcc-4a1a-8602-75c8b20fd7a1
# ╠═75edd61d-6f0d-4fad-9a18-917ef69547e6
# ╟─aa346240-3150-4923-ae13-1e84e86d0a7b
# ╟─6c9b06f7-8da1-43f1-8712-fe7ad5eaa8a9
# ╠═6ef92e9c-56df-4886-a25f-568669f71ba3
# ╟─51a06c87-e4be-48fb-a1de-3d3009495fc0
# ╠═6d9acaae-c071-481a-8eff-d5903f1b6fd4
# ╟─de193d02-33d8-4bc7-bb2e-739130ff4c53
# ╟─c742b160-6689-4874-b5a8-7b7f0c05145e
# ╠═4d205e89-d560-43bc-8d41-ff8666c9b931
# ╟─3e456e9e-db79-46e4-a894-a270da3fc160
# ╟─8618d99b-a7c3-40c5-ae3e-ce9b64e3c7d8
# ╠═ca44632d-c56d-40a1-9cc7-1051be519261
# ╟─555701fc-35cf-41fd-8c4f-9decea0b8ed0
# ╠═36a43d04-d7e5-42a5-ac04-bc3616709d8d
# ╟─2bc37446-d8c0-491a-9853-f44db2050890
# ╠═fe419612-3c9a-4e6b-a57f-c46aed41e873
# ╟─cb0a07a8-9517-42ce-addf-edeb592e9226
# ╠═c3b22830-6a5b-499d-8611-b0d7b819e4f2
# ╟─5fdb3825-56c4-41d6-b1f8-6c99b7db128a
# ╠═4aba61d8-849f-4eac-8c38-7527fe3b3bb9
# ╟─96647b06-fa51-48da-b124-7f6a66e63583
# ╠═cdedf7d0-94be-4470-8354-6c5ab549690a
# ╟─12c1a799-3bfe-4bb2-b5a3-c12038e8de5a
# ╠═4be501b2-8440-4445-af57-683da23ad09e
# ╟─32947508-f74e-4502-9149-fa38de6db8b7
# ╠═44444fe1-3cea-4f0a-a3e9-c444b04f7da6
# ╟─c1795732-1e42-4cf7-91c9-a98bd75b4ba8
# ╠═2a0d83f3-733a-47ce-9fc7-5634598c67b9
# ╟─a4d9b55f-bfa4-4c5e-8e6d-fcac1b33fc4e
# ╟─37eb0fad-a227-4754-ad39-a215009ad5ca
# ╠═d42dbfd2-4ae8-4623-9d04-c456a372771d
# ╟─1a8dc2ce-a0e4-4f49-a1f0-38abfc96c5b0
# ╠═9f6cce56-fd39-4440-b93e-d39217b81001
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
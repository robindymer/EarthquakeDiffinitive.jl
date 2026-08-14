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

# ╔═╡ e54a7592-795f-4f6a-9ee9-c52001e64820
using GLMakie

# ╔═╡ e50b25b5-e146-4ac4-b065-56a0ba4e7957
using LinearAlgebra

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

# ╔═╡ d61bb790-a95e-4d62-91a6-6bc6f40cb2f8
begin
    # ==============================================================================
    # 1. Define the spatial operator (from provided Diffinitive examples)
    # ==============================================================================
    
    struct IsotropicElasticOperator{T, D, TM1 <: NTuple{D, NTuple{D,LazyTensor{T,D,D}}}, TM2 <: NTuple{D, NTuple{D,LazyTensor{T,D,D}}}} <: LazyTensor{T, D, D}
        second_derivatives_lambda::TM1 # Dᵢⱼ(λ)
        second_derivatives_mu::TM2     # Dᵢⱼ(μ)
        size::NTuple{D,Int}            #
        stencil_set::StencilSet        #
    end
    
    function IsotropicElasticOperator(g, lambda, mu, stencil_set)
        λ = DiagonalTensor(lambda) #
        μ = DiagonalTensor(mu)     #
    
        # Dᵢⱼ(λ) using wide stencils to avoid dispersion
        Dᵢⱼλ = ntuple(Val(ndims(g))) do i
            ntuple(Val(ndims(g))) do j
                Dᵢ = first_derivative(g, stencil_set, i) #
                Dⱼ = first_derivative(g, stencil_set, j) #
                return Dᵢ ∘ λ ∘ Dⱼ #
            end
        end
    
        # Dᵢⱼ(μ) using narrow-stencil second derivatives for the diagonal
        Dᵢⱼμ = ntuple(Val(ndims(g))) do i
            ntuple(Val(ndims(g))) do j
                if i == j
                    return second_derivative_variable(g, mu, stencil_set, i) #
                else
                    Dᵢ = first_derivative(g, stencil_set, i) #
                    Dⱼ = first_derivative(g, stencil_set, j) #
                    return Dᵢ ∘ μ ∘ Dⱼ #
                end
            end
        end
        
        return IsotropicElasticOperator(Dᵢⱼλ, Dᵢⱼμ, size(g), stencil_set) #
    end
    
    # Fallback for constant λ=1, μ=1
    IsotropicElasticOperator(g, stencil_set) = IsotropicElasticOperator(g, ones(size(g)), ones(size(g)), stencil_set) #
    
    LazyTensors.range_size(op::IsotropicElasticOperator) = op.size #
    LazyTensors.domain_size(op::IsotropicElasticOperator) = op.size #
    
    # Specialized fast application for 2D grids (avoids inference issues)
    @inline function LazyTensors.apply(op::IsotropicElasticOperator{T,2}, u::AbstractMatrix, I...) where T
        u1 = Diffinitive.Grids.componentview(u, 1) #[cite: 1, 8]
        u2 = Diffinitive.Grids.componentview(u, 2) #[cite: 1, 8]
    
        D₁₁λ = op.second_derivatives_lambda[1][1] #
        D₁₂λ = op.second_derivatives_lambda[1][2] #
        D₂₁λ = op.second_derivatives_lambda[2][1] #
        D₂₂λ = op.second_derivatives_lambda[2][2] #
    
        D₁₁μ = op.second_derivatives_mu[1][1] #
        D₁₂μ = op.second_derivatives_mu[1][2] #
        D₂₁μ = op.second_derivatives_mu[2][1] #
        D₂₂μ = op.second_derivatives_mu[2][2] #
    
        # Component 1
        res1 =  apply(D₁₁λ, u1, I...) +
                apply(D₁₂λ, u2, I...) +
              2*apply(D₁₁μ, u1, I...) +
                apply(D₂₂μ, u1, I...) + 
                apply(D₂₁μ, u2, I...) #
        
        # Component 2
        res2 =  apply(D₂₁λ, u1, I...) +
                apply(D₂₂λ, u2, I...) +
              2*apply(D₂₂μ, u2, I...) +
                apply(D₁₁μ, u2, I...) + 
                apply(D₁₂μ, u1, I...) #
        
        return SVector{2}(res1, res2) #
    end
end

# ╔═╡ 9660c238-2c9e-4b12-8b2a-fb1594201df4
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

# ╔═╡ 964c70ae-f9e8-4029-90dc-96501e119399
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

# ╔═╡ cf492ac3-9958-4bf4-ac4e-920f7450f6bc
md"""
Same as above, a bit more clean implementation
"""

# ╔═╡ 9a4b05bc-9013-4e1f-8669-c61c2d0e7ba7
# ╠═╡ disabled = true
#=╠═╡
begin
    # ==============================================================================
    # 2. Main Simulation Setup & Loop
    # ==============================================================================
    println("Step 2...")
    
    # ------------------------------------------------------------------------------
    # Custom Operator Definition
    # ------------------------------------------------------------------------------
    struct PenalizedElasticOperator{EOp, TOps, HBOps, HInv, BInds, Buffers}
        E::EOp
        T_ops::TOps
        H_b_ops::HBOps
        H_inv::HInv
        b_indices::BInds
        work::Buffers
    end
    
    function (op::PenalizedElasticOperator)(u_accel, u_curr)
        # 1. Evaluate interior volume operator
        u_accel .= op.E * u_curr
        
        # 2. Add boundary penalties
        for i in 1:length(op.T_ops)
            T_b = op.T_ops[i]
            H_b = op.H_b_ops[i]
            b_idx = op.b_indices[i]
            
            trac = op.work.trac[i]
            wpen1 = op.work.wpen1[i]
            wpen2 = op.work.wpen2[i]
            
            trac .= T_b * u_curr
            
            wpen1 .= H_b * (-Diffinitive.Grids.componentview(trac, 1))
            wpen2 .= H_b * (-Diffinitive.Grids.componentview(trac, 2))
            
            fill!(op.work.lifted1, 0.0)
            fill!(op.work.lifted2, 0.0)
            
            @inbounds for (local_idx, vol_idx) in enumerate(b_idx)
                op.work.lifted1[vol_idx] = wpen1[local_idx]
                op.work.lifted2[vol_idx] = wpen2[local_idx]
            end
            
            op.work.sat1 .= op.H_inv * op.work.lifted1
            op.work.sat2 .= op.H_inv * op.work.lifted2
            
            @inbounds for j in eachindex(u_accel)
                u_accel[j] += @SVector[op.work.sat1[j], op.work.sat2[j]]
            end
        end
    end
    
    # ------------------------------------------------------------------------------
    # Main Solver Function
    # ------------------------------------------------------------------------------
    function solve_2d_elastic_wave()
        # Basic Grid and Operator Setup
        Nx, Ny = 201, 201
        Lx = 3.0
        Ly = 3.0
        g = equidistant_grid((0, 0), (Lx, Ly), Nx, Ny) 
        
        operator_path = sbp_operators_path() * "standard_diagonal.toml" 
        stencil_set = read_stencil_set(operator_path, order = 2) 
    
        lambda_const = 1.0
        mu_const = 0.05
        λ = map(x -> lambda_const, g) 
        μ = map(x -> mu_const, g) 
        ρ = 1.0              
        
        E = IsotropicElasticOperator(g, λ, μ, stencil_set) 
    
        u_old = map(x -> @SVector[0.0, 0.0], g) 
        u_curr = map(g) do x
            r2 = (x[1] - Lx/2)^2 + (x[2] - Ly/2)^2
            @SVector[ exp(-100*r2), exp(-100*r2) ]
        end
        u_new = map(x -> @SVector[0.0, 0.0], g)
        u_accel = map(x -> @SVector[0.0, 0.0], g)
    
        dx = 1.0 / (Nx - 1)
        cp = sqrt((lambda_const + 2*mu_const)/ρ) 
        dt = 0.5 * dx / cp         
        T_final = 3.0
        n_steps = ceil(Int, T_final / dt)
    
        # Plotting setup
        u1_obs = Observable(map(v -> v[1], u_curr))
        fig, ax, plt = plot(g, u1_obs, clim=(-0.01, 0.01))
        display(fig)
    
        # ==============================================================================
        # SAT BOUNDARY SETUP
        # ==============================================================================
    
        boundaries = boundary_identifiers(g)
        H_inv = inverse_inner_product(g, stencil_set)
        E_ops = [boundary_restriction(g, stencil_set, b) for b in boundaries]
        T_ops = [IsotropicTractionOperator(g, λ, μ, stencil_set, b) for b in boundaries]
        H_b_ops = [inner_product(boundary_grid(g, b), stencil_set) for b in boundaries]
        b_indices_list = [boundary_indices(g, b) for b in boundaries]
        
        # Pre-allocate buffers
        trac_buffers = [map(x -> @SVector[0.0, 0.0], boundary_grid(g, b)) for b in boundaries]
        wpen_1_buffers = [zeros(Float64, size(boundary_grid(g, b))) for b in boundaries]
        wpen_2_buffers = [zeros(Float64, size(boundary_grid(g, b))) for b in boundaries]
        lifted_1 = zeros(Float64, size(g))
        lifted_2 = zeros(Float64, size(g))
        sat_1_buf = zeros(Float64, size(g))
        sat_2_buf = zeros(Float64, size(g))
    
        # --- THE MISSING STEP: Bundle buffers and instantiate the operator ---
        work_buffers = (
            trac = trac_buffers,
            wpen1 = wpen_1_buffers,
            wpen2 = wpen_2_buffers,
            lifted1 = lifted_1,
            lifted2 = lifted_2,
            sat1 = sat_1_buf,
            sat2 = sat_2_buf
        )
        
        L = PenalizedElasticOperator(E, T_ops, H_b_ops, H_inv, b_indices_list, work_buffers)
        # ---------------------------------------------------------------------
    
        println("Starting simulation for $n_steps steps...")
        
        for step in 1:n_steps
            # Computes both interior derivatives and all SAT boundary penalties in one call
            L(u_accel, u_curr)
            
            # Leapfrog step
            for i in eachindex(g)
                u_new[i] = 2.0 * u_curr[i] - u_old[i] + (dt^2 / ρ) * u_accel[i]
            end
            
            u_old .= u_curr
            u_curr .= u_new
    
            if step % 5 == 0
                u1_obs[] = map(v -> v[1], u_curr) 
                sleep(0.001)
            end
        end
        
        println("Simulation finished.")
        return u_curr, g
    end
    
    # Run the solver
    final_u, domain_grid = solve_2d_elastic_wave()
end
  ╠═╡ =#

# ╔═╡ 4e2f78b4-e2dd-418f-bba3-c2db01bbc827
md"""
Works, but slowly
"""

# ╔═╡ 9701d870-4e6f-4185-944c-192ad0eb109e
# ╠═╡ disabled = true
#=╠═╡
begin
	struct FreeSurfaceSAT{TP,TT}
        penalty_tensor::TP   # scalar LazyTensor: H⁻¹ ∘ e' ∘ H_Γ
        T_b::TT              # traction operator for this boundary
    end
    
    function FreeSurfaceSAT(g, λ, μ, stencil_set, b)
        e     = boundary_restriction(g, stencil_set, b)
        H_inv = inverse_inner_product(g, stencil_set)
        H_Γ   = inner_product(boundary_grid(g, b), stencil_set)
        T_b   = IsotropicTractionOperator(g, λ, μ, stencil_set, b)
        penalty_tensor = H_inv ∘ e' ∘ H_Γ
        return FreeSurfaceSAT(penalty_tensor, T_b)
    end
    
    # accel .-= penalty(T_b*u) componentwise (target traction = 0)
    #
    # NOTE: componentview on a plain Array{SVector} is read-only (ArrayComponentView
    # doesn't support setindex!) -- only StructArray storage (as in your notebook's
    # Grids.componentview extension) gives a real mutable view. So here we read
    # each component (fine), build the penalty as a full SVector field, and
    # subtract that from accel as a whole rather than writing into a component view.
    function apply_sat!(accel, u, sat::FreeSurfaceSAT)
        t  = sat.T_b * u
        p1 = sat.penalty_tensor * componentview(t, 1)
        p2 = sat.penalty_tensor * componentview(t, 2)
        accel .-= SVector.(p1, p2)
    end

    function solve_2d_elastic_wave()
        Nx, Ny = 51, 51
        Lx, Ly = 3.0, 3.0
        g = equidistant_grid((0, 0), (Lx, Ly), Nx, Ny)
    
        operator_path = sbp_operators_path() * "standard_diagonal.toml"
        stencil_set   = read_stencil_set(operator_path, order = 2)
    
        lambda_const, mu_const, ρ = 1.0, 0.05, 1.0
        λ = map(x -> lambda_const, g)
        μ = map(x -> mu_const, g)
    
        E = IsotropicElasticOperator(g, λ, μ, stencil_set)
    
        boundaries = boundary_identifiers(g)
        sats = [FreeSurfaceSAT(g, λ, μ, stencil_set, b) for b in boundaries]
    
        u_old  = map(x -> @SVector[0.0, 0.0], g)
        u_curr = map(g) do x
            r2 = (x[1] - Lx/2)^2 + (x[2] - Ly/2)^2
            @SVector[exp(-100r2), exp(-100r2)]
        end
        u_new   = similar(u_curr)
        u_accel = similar(u_curr)
    
        dx = 1.0 / (Nx - 1)
        cp = sqrt((lambda_const + 2mu_const) / ρ)
        dt = 0.5 * dx / cp
        n_steps = ceil(Int, 1.0 / dt)
    
        println("Starting simulation for $n_steps steps...")
        for step in 1:n_steps
            u_accel .= E * u_curr
            for sat in sats
                apply_sat!(u_accel, u_curr, sat)
            end
    
            @. u_new = 2u_curr - u_old + (dt^2 / ρ) * u_accel
            u_old, u_curr, u_new = u_curr, u_new, u_old
    
            step % 50 == 0 && println("Step $step / $n_steps completed.")
        end
        println("Simulation finished.")
        return u_curr, g
    end
    
    final_u, domain_grid = solve_2d_elastic_wave()
end
  ╠═╡ =#

# ╔═╡ 74a212a8-3311-4fed-9d43-29de07d34206
md"""
# IsotropicElasticOperator

A discrete elastic operator for isotropic materials, implemented using the SBP finite difference operators in Diffinitive.jl.
"""

# ╔═╡ 859df5f4-0941-4406-b5f5-ae362ec535fd
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ 63c5b893-1edd-4d7d-b132-7f5f9c344b7c
md"""
## Accuracy test

Check that we differentiate polynomial functions (including variable material parameters) exactly
"""

# ╔═╡ bc9f841e-9a6f-4f98-b999-651f6179521f
md"""
### 2D
"""

# ╔═╡ 76fb4bec-8445-42ef-b0a6-cb9e96918f24
# ╠═╡ disabled = true
#=╠═╡
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
	Grids.plot(g)

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
  ╠═╡ =#

# ╔═╡ 83492e6d-c00e-42f9-872e-d74684cee180
md"""
### 3D
"""

# ╔═╡ 229e9c52-ebc3-485c-a000-d493fc5ea942
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ 7d8938f7-3d8a-4ac3-a2f4-46445e3a4ec2
md"""
## Test using `StructArrays` for a different storage format for the grid functions 
"""

# ╔═╡ ac977d73-4803-433b-9738-8b16170d0d89
md"""
Implement `componentview` on `StructArray` to test if we can use other data layouts for  vector-valued grid functions
"""

# ╔═╡ 12b4aa2b-ea9b-4072-b4c9-79d08f7f1967
# ╠═╡ disabled = true
#=╠═╡
Grids.componentview(v::StructArray, component_index...) = components(v)[component_index...]
  ╠═╡ =#

# ╔═╡ 67deb1fc-cfd5-43cf-9f40-0accbd4eb25a
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

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
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ 4d66242a-ddcc-4a1a-8602-75c8b20fd7a1
md"""
Now we can implement the traction operator for isotropic materials, using a similar pattern as for `IsotropicElasticOperator`. 

Note that for TensorGrids, the normal will be zero apart from in the dimension given by `grid_id(boundary)`, so this could (should?) be simplified. Moreover, we don't really need the normal as a grid function in this case.
"""

# ╔═╡ 75edd61d-6f0d-4fad-9a18-917ef69547e6
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

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
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ 51a06c87-e4be-48fb-a1de-3d3009495fc0
md"""
# SBP property test

Check that for random vectors ``u``, ``v``, and elastic/traction operators ``E``, ``T``:
```math
( v_i, (Eu)_i)_{\Omega} - ((Ev)_i, u_i)_{\Omega} = (v_i, (Tu)_i )_{\partial \Omega} - \left( (Tv)_i, u_i \right)_{\partial \Omega}
```
"""

# ╔═╡ 6d9acaae-c071-481a-8eff-d5903f1b6fd4
#=╠═╡
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
  ╠═╡ =#

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

# ╔═╡ 23dd5798-3416-44b6-830b-d9fda7df92c1
# ╠═╡ disabled = true
#=╠═╡
begin
    struct PenalizedElasticOperator{EOp, TOps, HBOps, HInv, BInds, Buffers}
        E::EOp
        T_ops::TOps
        H_b_ops::HBOps
        H_inv::HInv
        b_indices::BInds
        work::Buffers
    end

    function (op::PenalizedElasticOperator)(u_accel, u_curr)
        u_accel .= op.E * u_curr
        
        for i in 1:length(op.T_ops)
            T_b = op.T_ops[i]
            H_b = op.H_b_ops[i]
            b_idx = op.b_indices[i]
            
            trac = op.work.trac[i]
            wpen1 = op.work.wpen1[i]
            wpen2 = op.work.wpen2[i]
            
            trac .= T_b * u_curr
            
            wpen1 .= H_b * (-Diffinitive.Grids.componentview(trac, 1))
            wpen2 .= H_b * (-Diffinitive.Grids.componentview(trac, 2))
            
            fill!(op.work.lifted1, 0.0)
            fill!(op.work.lifted2, 0.0)
            
            @inbounds for (local_idx, vol_idx) in enumerate(b_idx)
                op.work.lifted1[vol_idx] = wpen1[local_idx]
                op.work.lifted2[vol_idx] = wpen2[local_idx]
            end
            
            op.work.sat1 .= op.H_inv * op.work.lifted1
            op.work.sat2 .= op.H_inv * op.work.lifted2
            
            @inbounds for j in eachindex(u_accel)
                u_accel[j] += @SVector[op.work.sat1[j], op.work.sat2[j]]
            end
        end
    end

    function solve_2d_elastic_wave()
        Nx, Ny = 101, 201
        Lx = 1.5
        Ly = 3.0
        
        g_L = equidistant_grid((-Lx, -Ly/2), (0.0, Ly/2), Nx, Ny)
        g_R = equidistant_grid((0.0, -Ly/2), (Lx, Ly/2), Nx, Ny)
        
        operator_path = sbp_operators_path() * "standard_diagonal.toml" 
        stencil_set = read_stencil_set(operator_path, order = 2) 

        lambda_const = 1.0
        mu_const = 0.05
        λ_L = map(x -> lambda_const, g_L) 
        μ_L = map(x -> mu_const, g_L) 
        λ_R = map(x -> lambda_const, g_R) 
        μ_R = map(x -> mu_const, g_R) 
        ρ = 1.0              
        
        E_L = IsotropicElasticOperator(g_L, λ_L, μ_L, stencil_set) 
        E_R = IsotropicElasticOperator(g_R, λ_R, μ_R, stencil_set) 

        u_old_L = map(x -> @SVector[0.0, 0.0], g_L) 
        u_old_R = map(x -> @SVector[0.0, 0.0], g_R) 

        u_curr_L = map(g_L) do x
            r2 = (x[1] + Lx/2)^2 + (x[2])^2
            @SVector[ exp(-100*r2), exp(-100*r2) ]
        end
        u_curr_R = map(g_R) do x
            @SVector[ 0.0, 0.0 ]
        end

        u_new_L = map(x -> @SVector[0.0, 0.0], g_L)
        u_new_R = map(x -> @SVector[0.0, 0.0], g_R)
        u_accel_L = map(x -> @SVector[0.0, 0.0], g_L)
        u_accel_R = map(x -> @SVector[0.0, 0.0], g_R)
        
        v_curr_L = map(x -> @SVector[0.0, 0.0], g_L)
        v_curr_R = map(x -> @SVector[0.0, 0.0], g_R)

        dx = 1.0 / (Nx - 1)
        cp = sqrt((lambda_const + 2*mu_const)/ρ) 
        dt = 0.5 * dx / cp         
        T_final = 3.0
        n_steps = ceil(Int, T_final / dt)

        tau_p = 2.0 * (lambda_const + 2.0*mu_const) / dx

        a_rsf = 0.015
        b_rsf = 0.02
        Dc = 0.1
        f0 = 0.6
        V0 = 1e-6
        sigma_n = 50.0

        fault_b_L = CartesianBoundary{1, UpperBoundary}()
        fault_b_R = CartesianBoundary{1, LowerBoundary}()
        
        b_L_all = boundary_identifiers(g_L)
        b_R_all = boundary_identifiers(g_R)
        
        outer_b_L = filter(b -> b != fault_b_L, b_L_all)
        outer_b_R = filter(b -> b != fault_b_R, b_R_all)

        H_inv_L = inverse_inner_product(g_L, stencil_set)
        H_inv_R = inverse_inner_product(g_R, stencil_set)

        E_ops_L = [boundary_restriction(g_L, stencil_set, b) for b in outer_b_L]
        T_ops_L = [IsotropicTractionOperator(g_L, λ_L, μ_L, stencil_set, b) for b in outer_b_L]
        H_b_ops_L = [inner_product(boundary_grid(g_L, b), stencil_set) for b in outer_b_L]
        b_idx_L = [boundary_indices(g_L, b) for b in outer_b_L]
        
        work_L = (
            trac = [map(x -> @SVector[0.0, 0.0], boundary_grid(g_L, b)) for b in outer_b_L],
            wpen1 = [zeros(Float64, size(boundary_grid(g_L, b))) for b in outer_b_L],
            wpen2 = [zeros(Float64, size(boundary_grid(g_L, b))) for b in outer_b_L],
            lifted1 = zeros(Float64, size(g_L)),
            lifted2 = zeros(Float64, size(g_L)),
            sat1 = zeros(Float64, size(g_L)),
            sat2 = zeros(Float64, size(g_L))
        )
        Op_L = PenalizedElasticOperator(E_L, T_ops_L, H_b_ops_L, H_inv_L, b_idx_L, work_L)

        E_ops_R = [boundary_restriction(g_R, stencil_set, b) for b in outer_b_R]
        T_ops_R = [IsotropicTractionOperator(g_R, λ_R, μ_R, stencil_set, b) for b in outer_b_R]
        H_b_ops_R = [inner_product(boundary_grid(g_R, b), stencil_set) for b in outer_b_R]
        b_idx_R = [boundary_indices(g_R, b) for b in outer_b_R]
        
        work_R = (
            trac = [map(x -> @SVector[0.0, 0.0], boundary_grid(g_R, b)) for b in outer_b_R],
            wpen1 = [zeros(Float64, size(boundary_grid(g_R, b))) for b in outer_b_R],
            wpen2 = [zeros(Float64, size(boundary_grid(g_R, b))) for b in outer_b_R],
            lifted1 = zeros(Float64, size(g_R)),
            lifted2 = zeros(Float64, size(g_R)),
            sat1 = zeros(Float64, size(g_R)),
            sat2 = zeros(Float64, size(g_R))
        )
        Op_R = PenalizedElasticOperator(E_R, T_ops_R, H_b_ops_R, H_inv_R, b_idx_R, work_R)

        E_int_L = boundary_restriction(g_L, stencil_set, fault_b_L)
        T_int_L = IsotropicTractionOperator(g_L, λ_L, μ_L, stencil_set, fault_b_L)
        H_b_int_L = inner_product(boundary_grid(g_L, fault_b_L), stencil_set)
        idx_int_L = boundary_indices(g_L, fault_b_L)
        
        E_int_R = boundary_restriction(g_R, stencil_set, fault_b_R)
        T_int_R = IsotropicTractionOperator(g_R, λ_R, μ_R, stencil_set, fault_b_R)
        H_b_int_R = inner_product(boundary_grid(g_R, fault_b_R), stencil_set)
        idx_int_R = boundary_indices(g_R, fault_b_R)
        
        trac_buf_L = map(x -> @SVector[0.0, 0.0], boundary_grid(g_L, fault_b_L))
        trac_buf_R = map(x -> @SVector[0.0, 0.0], boundary_grid(g_R, fault_b_R))
        
        wpen1_L = zeros(Float64, size(boundary_grid(g_L, fault_b_L)))
        wpen2_L = zeros(Float64, size(boundary_grid(g_L, fault_b_L)))
        wpen1_R = zeros(Float64, size(boundary_grid(g_R, fault_b_R)))
        wpen2_R = zeros(Float64, size(boundary_grid(g_R, fault_b_R)))

        theta_curr = fill(Dc / V0, size(boundary_grid(g_L, fault_b_L)))
        theta_new = zeros(Float64, size(boundary_grid(g_L, fault_b_L)))

        for step in 1:n_steps
            Op_L(u_accel_L, u_curr_L)
            Op_R(u_accel_R, u_curr_R)
            
            trac_buf_L .= T_int_L * u_curr_L
            trac_buf_R .= T_int_R * u_curr_R
            
            u_int_L = collect(E_int_L * u_curr_L)
            u_int_R = collect(E_int_R * u_curr_R)
            v_int_L = collect(E_int_L * v_curr_L)
            v_int_R = collect(E_int_R * v_curr_R)
            
            for i in eachindex(u_int_L)
                s1 = u_int_R[i][1] - u_int_L[i][1]
                V_s = v_int_R[i][2] - v_int_L[i][2]
                
                trac_1_L = trac_buf_L[i][1]
                trac_1_R = trac_buf_R[i][1]
                wpen1_L[i] = -0.5 * (trac_1_L + trac_1_R) + tau_p * s1
                wpen1_R[i] = -0.5 * (trac_1_L + trac_1_R) - tau_p * s1
                
                trac_2_L = trac_buf_L[i][2]
                trac_2_R = trac_buf_R[i][2]
                
                Q = exp((f0 + b_rsf * log(V0 * max(theta_curr[i], 1e-14) / Dc)) / a_rsf)
                tau_fault = sigma_n * a_rsf * asinh((V_s / (2.0 * V0)) * Q)
                
                wpen2_L[i] = tau_fault - trac_2_L
                wpen2_R[i] = -tau_fault - trac_2_R
                
                theta_new[i] = max(1e-14, theta_curr[i] + dt * (1.0 - (abs(V_s) * theta_curr[i]) / Dc))
            end
            
            wpen1_L .= H_b_int_L * wpen1_L
            wpen2_L .= H_b_int_L * wpen2_L
            wpen1_R .= H_b_int_R * wpen1_R
            wpen2_R .= H_b_int_R * wpen2_R
            
            fill!(work_L.lifted1, 0.0)
            fill!(work_L.lifted2, 0.0)
            @inbounds for (local_idx, vol_idx) in enumerate(idx_int_L)
                work_L.lifted1[vol_idx] = wpen1_L[local_idx]
                work_L.lifted2[vol_idx] = wpen2_L[local_idx]
            end
            work_L.sat1 .= H_inv_L * work_L.lifted1
            work_L.sat2 .= H_inv_L * work_L.lifted2
            @inbounds for j in eachindex(u_accel_L)
                u_accel_L[j] += @SVector[work_L.sat1[j], work_L.sat2[j]]
            end
            
            fill!(work_R.lifted1, 0.0)
            fill!(work_R.lifted2, 0.0)
            @inbounds for (local_idx, vol_idx) in enumerate(idx_int_R)
                work_R.lifted1[vol_idx] = wpen1_R[local_idx]
                work_R.lifted2[vol_idx] = wpen2_R[local_idx]
            end
            work_R.sat1 .= H_inv_R * work_R.lifted1
            work_R.sat2 .= H_inv_R * work_R.lifted2
            @inbounds for j in eachindex(u_accel_R)
                u_accel_R[j] += @SVector[work_R.sat1[j], work_R.sat2[j]]
            end
            
            for i in eachindex(g_L)
                u_new_L[i] = 2.0 * u_curr_L[i] - u_old_L[i] + (dt^2 / ρ) * u_accel_L[i]
                v_curr_L[i] = (u_new_L[i] - u_old_L[i]) / (2.0 * dt)
            end
            for i in eachindex(g_R)
                u_new_R[i] = 2.0 * u_curr_R[i] - u_old_R[i] + (dt^2 / ρ) * u_accel_R[i]
                v_curr_R[i] = (u_new_R[i] - u_old_R[i]) / (2.0 * dt)
            end
            
            u_old_L .= u_curr_L
            u_curr_L .= u_new_L
            u_old_R .= u_curr_R
            u_curr_R .= u_new_R
            
            theta_curr .= theta_new
        end
        
        return u_curr_L, u_curr_R, g_L, g_R
    end

    final_u_L, final_u_R, g_L, g_R = solve_2d_elastic_wave()
end
  ╠═╡ =#

# ╔═╡ 435f8e95-b498-413d-a543-5e53f6e6bd3f
begin
    # using GLMakie
    # using StaticArrays
    # using Diffinitive
    # using Diffinitive.Grids
    # using Diffinitive.SbpOperators

    struct PenalizedElasticOperator{EOp, TOps, HBOps, HInv, BInds, Buffers}
        E::EOp
        T_ops::TOps
        H_b_ops::HBOps
        H_inv::HInv
        b_indices::BInds
        work::Buffers
    end

    function (op::PenalizedElasticOperator)(u_accel, u_curr)
        u_accel .= op.E * u_curr
        
        for i in 1:length(op.T_ops)
            T_b = op.T_ops[i]
            H_b = op.H_b_ops[i]
            b_idx = op.b_indices[i]
            
            trac = op.work.trac[i]
            wpen1 = op.work.wpen1[i]
            wpen2 = op.work.wpen2[i]
            
            trac .= T_b * u_curr
            
            wpen1 .= H_b * (-Diffinitive.Grids.componentview(trac, 1))
            wpen2 .= H_b * (-Diffinitive.Grids.componentview(trac, 2))
            
            fill!(op.work.lifted1, 0.0)
            fill!(op.work.lifted2, 0.0)
            
            @inbounds for (local_idx, vol_idx) in enumerate(b_idx)
                op.work.lifted1[vol_idx] = wpen1[local_idx]
                op.work.lifted2[vol_idx] = wpen2[local_idx]
            end
            
            op.work.sat1 .= op.H_inv * op.work.lifted1
            op.work.sat2 .= op.H_inv * op.work.lifted2
            
            @inbounds for j in eachindex(u_accel)
                u_accel[j] += @SVector[op.work.sat1[j], op.work.sat2[j]]
            end
        end
    end

    function solve_2d_elastic_wave(test_mode::Symbol)
        Nx, Ny = 101, 201
        Lx = 1.5
        Ly = 3.0
        
        g_L = equidistant_grid((-Lx, -Ly/2), (0.0, Ly/2), Nx, Ny)
        g_R = equidistant_grid((0.0, -Ly/2), (Lx, Ly/2), Nx, Ny)
        
        operator_path = sbp_operators_path() * "standard_diagonal.toml" 
        stencil_set = read_stencil_set(operator_path, order = 2) 

        lambda_const = 1.0
        mu_const = 0.05
        λ_L = map(x -> lambda_const, g_L) 
        μ_L = map(x -> mu_const, g_L) 
        λ_R = map(x -> lambda_const, g_R) 
        μ_R = map(x -> mu_const, g_R) 
        ρ = 1.0              
        
        E_L = IsotropicElasticOperator(g_L, λ_L, μ_L, stencil_set) 
        E_R = IsotropicElasticOperator(g_R, λ_R, μ_R, stencil_set) 

        u_old_L = map(x -> @SVector[0.0, 0.0], g_L) 
        u_old_R = map(x -> @SVector[0.0, 0.0], g_R) 

        if test_mode == :locked
            u_curr_L = map(g_L) do x
                r2 = (x[1] + Lx/2)^2 + (x[2])^2
                @SVector[ exp(-100*r2), exp(-100*r2) ]
            end
            u_curr_R = map(x -> @SVector[0.0, 0.0], g_R)
        else
            u_curr_L = map(x -> @SVector[0.0, 0.0], g_L)
            u_curr_R = map(x -> @SVector[0.0, 0.0], g_R)
        end

        u_new_L = map(x -> @SVector[0.0, 0.0], g_L)
        u_new_R = map(x -> @SVector[0.0, 0.0], g_R)
        u_accel_L = map(x -> @SVector[0.0, 0.0], g_L)
        u_accel_R = map(x -> @SVector[0.0, 0.0], g_R)
        
        v_curr_L = map(x -> @SVector[0.0, 0.0], g_L)
        v_curr_R = map(x -> @SVector[0.0, 0.0], g_R)

        dx = 1.0 / (Nx - 1)
        cp = sqrt((lambda_const + 2*mu_const)/ρ) 
        dt = 0.5 * dx / cp         
        T_final = 3.0
        n_steps = ceil(Int, T_final / dt)

        tau_p = 2.0 * (lambda_const + 2.0*mu_const) / dx

        a_rsf = 0.015
        b_rsf = 0.02
        Dc = 0.1
        f0 = test_mode == :locked ? 1000.0 : 0.6
        V0 = 1e-6
        sigma_n = 50.0

        if test_mode == :nucleation
            u1_obs_L = Observable(map(v -> v[2], v_curr_L))
            u1_obs_R = Observable(map(v -> v[2], v_curr_R))
        else
            u1_obs_L = Observable(map(v -> v[1], u_curr_L))
            u1_obs_R = Observable(map(v -> v[1], u_curr_R))
        end
        
        fig = Figure()
        ax = Axis(fig[1, 1], aspect=DataAspect(), title="Test Mode: $(String(test_mode))")
        plot!(ax, g_L, u1_obs_L, colormap=:RdBu, colorrange=(-0.01, 0.01))
        plot!(ax, g_R, u1_obs_R, colormap=:RdBu, colorrange=(-0.01, 0.01))
        display(fig)

        fault_b_L = CartesianBoundary{1, UpperBoundary}()
        fault_b_R = CartesianBoundary{1, LowerBoundary}()
        
        b_L_all = boundary_identifiers(g_L)
        b_R_all = boundary_identifiers(g_R)
        
        outer_b_L = filter(b -> b != fault_b_L, b_L_all)
        outer_b_R = filter(b -> b != fault_b_R, b_R_all)

        H_inv_L = inverse_inner_product(g_L, stencil_set)
        H_inv_R = inverse_inner_product(g_R, stencil_set)

        E_ops_L = [boundary_restriction(g_L, stencil_set, b) for b in outer_b_L]
        T_ops_L = [IsotropicTractionOperator(g_L, λ_L, μ_L, stencil_set, b) for b in outer_b_L]
        H_b_ops_L = [inner_product(boundary_grid(g_L, b), stencil_set) for b in outer_b_L]
        b_idx_L = [boundary_indices(g_L, b) for b in outer_b_L]
        
        work_L = (
            trac = [map(x -> @SVector[0.0, 0.0], boundary_grid(g_L, b)) for b in outer_b_L],
            wpen1 = [zeros(Float64, size(boundary_grid(g_L, b))) for b in outer_b_L],
            wpen2 = [zeros(Float64, size(boundary_grid(g_L, b))) for b in outer_b_L],
            lifted1 = zeros(Float64, size(g_L)),
            lifted2 = zeros(Float64, size(g_L)),
            sat1 = zeros(Float64, size(g_L)),
            sat2 = zeros(Float64, size(g_L))
        )
        Op_L = PenalizedElasticOperator(E_L, T_ops_L, H_b_ops_L, H_inv_L, b_idx_L, work_L)

        E_ops_R = [boundary_restriction(g_R, stencil_set, b) for b in outer_b_R]
        T_ops_R = [IsotropicTractionOperator(g_R, λ_R, μ_R, stencil_set, b) for b in outer_b_R]
        H_b_ops_R = [inner_product(boundary_grid(g_R, b), stencil_set) for b in outer_b_R]
        b_idx_R = [boundary_indices(g_R, b) for b in outer_b_R]
        
        work_R = (
            trac = [map(x -> @SVector[0.0, 0.0], boundary_grid(g_R, b)) for b in outer_b_R],
            wpen1 = [zeros(Float64, size(boundary_grid(g_R, b))) for b in outer_b_R],
            wpen2 = [zeros(Float64, size(boundary_grid(g_R, b))) for b in outer_b_R],
            lifted1 = zeros(Float64, size(g_R)),
            lifted2 = zeros(Float64, size(g_R)),
            sat1 = zeros(Float64, size(g_R)),
            sat2 = zeros(Float64, size(g_R))
        )
        Op_R = PenalizedElasticOperator(E_R, T_ops_R, H_b_ops_R, H_inv_R, b_idx_R, work_R)

        E_int_L = boundary_restriction(g_L, stencil_set, fault_b_L)
        T_int_L = IsotropicTractionOperator(g_L, λ_L, μ_L, stencil_set, fault_b_L)
        H_b_int_L = inner_product(boundary_grid(g_L, fault_b_L), stencil_set)
        idx_int_L = boundary_indices(g_L, fault_b_L)
        
        E_int_R = boundary_restriction(g_R, stencil_set, fault_b_R)
        T_int_R = IsotropicTractionOperator(g_R, λ_R, μ_R, stencil_set, fault_b_R)
        H_b_int_R = inner_product(boundary_grid(g_R, fault_b_R), stencil_set)
        idx_int_R = boundary_indices(g_R, fault_b_R)
        
        trac_buf_L = map(x -> @SVector[0.0, 0.0], boundary_grid(g_L, fault_b_L))
        trac_buf_R = map(x -> @SVector[0.0, 0.0], boundary_grid(g_R, fault_b_R))
        
        wpen1_L = zeros(Float64, size(boundary_grid(g_L, fault_b_L)))
        wpen2_L = zeros(Float64, size(boundary_grid(g_L, fault_b_L)))
        wpen1_R = zeros(Float64, size(boundary_grid(g_R, fault_b_R)))
        wpen2_R = zeros(Float64, size(boundary_grid(g_R, fault_b_R)))

        theta_curr = fill(Dc / V0, size(boundary_grid(g_L, fault_b_L)))
        theta_new = zeros(Float64, size(boundary_grid(g_L, fault_b_L)))
        
        bg_stress = test_mode == :nucleation ? 25.0 : 0.0

        for step in 1:n_steps
            Op_L(u_accel_L, u_curr_L)
            Op_R(u_accel_R, u_curr_R)
            
            trac_buf_L .= T_int_L * u_curr_L
            trac_buf_R .= T_int_R * u_curr_R
            
            u_int_L = collect(E_int_L * u_curr_L)
            u_int_R = collect(E_int_R * u_curr_R)
            v_int_L = collect(E_int_L * v_curr_L)
            v_int_R = collect(E_int_R * v_curr_R)
            
            grid_b_L = collect(boundary_grid(g_L, fault_b_L))

            for i in eachindex(u_int_L)
                s1 = u_int_R[i][1] - u_int_L[i][1]
                V_s = v_int_R[i][2] - v_int_L[i][2]
                
                trac_1_L = trac_buf_L[i][1]
                trac_1_R = trac_buf_R[i][1]
                wpen1_L[i] = -0.5 * (trac_1_L + trac_1_R) + tau_p * s1
                wpen1_R[i] = -0.5 * (trac_1_L + trac_1_R) - tau_p * s1
                
                trac_2_L = trac_buf_L[i][2]
                trac_2_R = trac_buf_R[i][2]
                
                Q = exp((f0 + b_rsf * log(V0 * max(theta_curr[i], 1e-14) / Dc)) / a_rsf)
                tau_fault = sigma_n * a_rsf * asinh((V_s / (2.0 * V0)) * Q)
                
                if test_mode == :nucleation
                    y_coord = grid_b_L[i][2]
                    local_stress = abs(y_coord) < 0.2 ? 35.0 : bg_stress
                    
                    wpen2_L[i] = tau_fault - trac_2_L - local_stress
                    wpen2_R[i] = -tau_fault - trac_2_R + local_stress
                else
                    wpen2_L[i] = tau_fault - trac_2_L
                    wpen2_R[i] = -tau_fault - trac_2_R
                end
                
                theta_new[i] = max(1e-14, theta_curr[i] + dt * (1.0 - (abs(V_s) * theta_curr[i]) / Dc))
            end
            
            wpen1_L .= H_b_int_L * wpen1_L
            wpen2_L .= H_b_int_L * wpen2_L
            wpen1_R .= H_b_int_R * wpen1_R
            wpen2_R .= H_b_int_R * wpen2_R
            
            fill!(work_L.lifted1, 0.0)
            fill!(work_L.lifted2, 0.0)
            @inbounds for (local_idx, vol_idx) in enumerate(idx_int_L)
                work_L.lifted1[vol_idx] = wpen1_L[local_idx]
                work_L.lifted2[vol_idx] = wpen2_L[local_idx]
            end
            work_L.sat1 .= H_inv_L * work_L.lifted1
            work_L.sat2 .= H_inv_L * work_L.lifted2
            @inbounds for j in eachindex(u_accel_L)
                u_accel_L[j] += @SVector[work_L.sat1[j], work_L.sat2[j]]
            end
            
            fill!(work_R.lifted1, 0.0)
            fill!(work_R.lifted2, 0.0)
            @inbounds for (local_idx, vol_idx) in enumerate(idx_int_R)
                work_R.lifted1[vol_idx] = wpen1_R[local_idx]
                work_R.lifted2[vol_idx] = wpen2_R[local_idx]
            end
            work_R.sat1 .= H_inv_R * work_R.lifted1
            work_R.sat2 .= H_inv_R * work_R.lifted2
            @inbounds for j in eachindex(u_accel_R)
                u_accel_R[j] += @SVector[work_R.sat1[j], work_R.sat2[j]]
            end
            
            for i in eachindex(g_L)
                u_new_L[i] = 2.0 * u_curr_L[i] - u_old_L[i] + (dt^2 / ρ) * u_accel_L[i]
                v_curr_L[i] = (u_new_L[i] - u_old_L[i]) / (2.0 * dt)
            end
            for i in eachindex(g_R)
                u_new_R[i] = 2.0 * u_curr_R[i] - u_old_R[i] + (dt^2 / ρ) * u_accel_R[i]
                v_curr_R[i] = (u_new_R[i] - u_old_R[i]) / (2.0 * dt)
            end
            
            u_old_L .= u_curr_L
            u_curr_L .= u_new_L
            u_old_R .= u_curr_R
            u_curr_R .= u_new_R
            
            theta_curr .= theta_new

            if step % 5 == 0
                if test_mode == :nucleation
                    u1_obs_L[] = map(v -> v[2], v_curr_L)
                    u1_obs_R[] = map(v -> v[2], v_curr_R)
                else
                    u1_obs_L[] = map(v -> v[1], u_curr_L)
                    u1_obs_R[] = map(v -> v[1], u_curr_R)
                end
                sleep(0.001)
            end
        end
        
        return u_curr_L, u_curr_R, g_L, g_R
    end

    # Run the locked transmission test
    # final_u_L, final_u_R, g_L, g_R = solve_2d_elastic_wave(:locked)

    # Run the spontaneous nucleation test
    final_u_L, final_u_R, g_L, g_R = solve_2d_elastic_wave(:nucleation)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Diffinitive = "5a373a26-915f-4769-bcab-bf03835de17b"
GLMakie = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Pkg = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"

[compat]
Diffinitive = "~0.1.7"
GLMakie = "~0.13.13"
StaticArrays = "~1.9.17"
StructArrays = "~0.7.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.6"
manifest_format = "2.0"
project_hash = "1e2cb622d79a0490d64e572de5da44e4145331f9"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

    [deps.AbstractFFTs.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "7063ad1083578215c7c4bf410368150abe8d5524"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.45"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "daa72978cd7a624246e894a4f4f067706d4e17e2"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.7.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Automa]]
deps = ["PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "94eab0b3ccdcac361188cc661daf69d4433c1818"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.2.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "4126b08903b777c88edf1754288144a0492c05ad"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.8"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BaseDirs]]
git-tree-sha1 = "8c290a1b223deaeea9aea44b235d24546da8eb98"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.4.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "66188d9d103b92b6cd705214242e27f5737a1e5e"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.2"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "12177ad6b3cad7fd50c8b3825ce24a99ad61c18f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodecZstd]]
deps = ["TranscodingStreams", "Zstd_jll"]
git-tree-sha1 = "da54a6cd93c54950c15adf1d336cfd7d71f51a56"
uuid = "6b39b394-51ab-5f42-8807-6242bab2b4c2"
version = "0.8.7"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "07da79661b919001e6863b81fc572497daa58349"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.CommonSolve]]
git-tree-sha1 = "cf963add2340ad9960e5eb22844e61ad8f931fe1"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.13"

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

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ComputePipeline]]
deps = ["Observables", "Preferences"]
git-tree-sha1 = "7bc84b769c1d384315e7b5c4ac03a6c303e6cf35"
uuid = "95dc2771-c249-4cd0-9c9f-1f3b4330693c"
version = "0.1.8"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CoreMath]]
deps = ["CoreMath_jll"]
git-tree-sha1 = "8c0480f92b1b1796239156a1b9b1bfb1b39499b4"
uuid = "b7a15901-be09-4a0e-87d2-2e66b0e09b5a"
version = "0.1.0"

[[deps.CoreMath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a692a4c1dc59a4b8bc0b6403876eb3250fde2bc3"
uuid = "a38c48d9-6df1-5ac9-9223-b6ada3b5572b"
version = "0.1.0+0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "c55f5a9fd67bdbc8e089b5a3111fe4292986a8e8"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.6"

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

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "Roots", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "d2facc77c08c1c2bfb1a77c148edd05b3db5410b"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.130"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "83231673ea4d3d6008ac74dc5079e77ab2209d8f"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.9"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e6c4a6407a949e79a9d3f249bf49e6987c80e01f"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.2+0"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "7a58e45171b63ed4782f2d36fdee8713a469e6e0"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.2+0"

[[deps.FFTA]]
deps = ["AbstractFFTs", "DocStringExtensions", "LinearAlgebra", "MuladdMacro", "Primes", "Random", "Reexport"]
git-tree-sha1 = "65e55303b72f4a567a51b174dd2c47496efeb95a"
uuid = "b86e33f2-c0db-4aa1-a6e0-ab43e668529e"
version = "0.3.1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "6621fef488e496356c9c9625d0562c12a6070819"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.20.0"

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

    [deps.FileIO.weakdeps]
    HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport"]
git-tree-sha1 = "a1b2fbfe98503f15b665ed45b3d149e5d8895e4c"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.9.0"

    [deps.FilePaths.extensions]
    FilePathsGlobExt = "Glob"
    FilePathsURIParserExt = "URIParser"
    FilePathsURIsExt = "URIs"

    [deps.FilePaths.weakdeps]
    Glob = "c27321d9-0574-5035-807b-f59d2c89b15c"
    URIParser = "30578b45-9adc-5946-b283-645ec420af67"
    URIs = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

    [deps.FilePathsBase.weakdeps]
    Mmap = "a63ad114-7e13-5084-954f-fe012c677804"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "5bad39456d9f0166184fce2248783dd9862645c1"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.17.0"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FreeTypeAbstraction]]
deps = ["BaseDirs", "ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "Mmap"]
git-tree-sha1 = "4ebb930ef4a43817991ba35db6317a05e59abd11"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.8"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GLFW]]
deps = ["GLFW_jll"]
git-tree-sha1 = "af06f66cca2b698ab9c482de55977ff8178d025e"
uuid = "f7f18e0c-5ee9-5ccd-a5bf-e8befd85ed98"
version = "3.4.6"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "9e0fb9e54594c47f278d75063980e43066e26e20"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.1+1"

[[deps.GLMakie]]
deps = ["ColorTypes", "Colors", "FileIO", "FixedPointNumbers", "FreeTypeAbstraction", "GLFW", "GeometryBasics", "LinearAlgebra", "Makie", "Markdown", "MeshIO", "ModernGL", "Observables", "PrecompileTools", "Printf", "ShaderAbstractions", "StaticArrays"]
git-tree-sha1 = "3e1770a9d85b8bd1767431b959d657dc8825d78d"
uuid = "e9467ef8-e4e7-5192-8a1a-b1aee30e663a"
version = "0.13.13"

[[deps.Gamma]]
deps = ["LogExpFunctions"]
git-tree-sha1 = "becc397f7cfb06e343496ae6ffb04818a851da51"
uuid = "a0844989-3bd2-4988-8bea-c9407ab0941b"
version = "1.2.0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "364685f5ffde25deb1bbcfd5bb278a5c6b7a9b37"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.11"

    [deps.GeometryBasics.extensions]
    ExtentsExt = "Extents"
    GeometryBasicsGeoInterfaceExt = "GeoInterface"
    IntervalSetsExt = "IntervalSets"

    [deps.GeometryBasics.weakdeps]
    Extents = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
    GeoInterface = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "090526e65de8f69648ac156daae153de8b56df62"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.88.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "93d5c27c8de51687a2c70ec0716e6e76f298416f"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.2"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HypergeometricFunctions]]
deps = ["Gamma", "LinearAlgebra"]
git-tree-sha1 = "31bb6c92405c084617facc1d7ed9eb6c402d061e"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.30"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dcc8d0cd653e55213df9b75ebc6fe4a8d3254c65"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.2.2+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "c72458f1962faeb003bf23cbdb75164fe6280906"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.4"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "48922d06068130f87e43edef52382e6a94305ae6"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.3"

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "CoreMath", "MacroTools", "OpenBLASConsistentFPCSR_jll", "Printf", "Random", "RoundingEmulator"]
git-tree-sha1 = "c3ee408ae340565f41699e3a3fa1053698c7626e"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "1.0.10"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticArblibExt = "Arblib"
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticIrrationalConstantsExt = "IrrationalConstants"
    IntervalArithmeticLinearAlgebraExt = "LinearAlgebra"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"
    IntervalArithmeticSparseArraysExt = "SparseArrays"

    [deps.IntervalArithmetic.weakdeps]
    Arblib = "fb37089c-8514-4489-9461-98f9c8763369"
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    IrrationalConstants = "92d709cd-6900-40b7-9082-c6be49f344b6"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.IntervalSets]]
git-tree-sha1 = "79d6bd28c8d9bccc2229784f1bd637689b256377"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.14"

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

    [deps.IntervalSets.weakdeps]
    Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

    [deps.InverseFunctions.weakdeps]
    Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "c7345ab1a7ca4dc8a02c9f6510da0d9857bbe513"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.7.1"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "037babc10853eeb8e585418922246cb97b8e5b74"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.0+1"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTA", "Interpolations", "StatsBase"]
git-tree-sha1 = "9eda8292dd3268b3b7ec9df21bbfac24e177ec52"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.12"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "17b94ecafcfa45e8360a4fc9ca6b583b049e4e37"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.1.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b7970cef8ae1c990ba0c09cd8bdc1145e006632f"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "22.1.7+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

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

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "aebd334d06cee9f24cea70bd19a39749daf73881"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "bba2d9aa057d8f126415de240573e86a8f39d2a1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.1"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "ComputePipeline", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "Pkg", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "f2c8715d05bf10f9d4dc354e69dee30b6be53239"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.24.13"

    [deps.Makie.extensions]
    MakieDynamicQuantitiesExt = "DynamicQuantities"

    [deps.Makie.weakdeps]
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"

[[deps.MappedArrays]]
git-tree-sha1 = "0ee4497a4e80dbd29c058fcee6493f5219556f40"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.3"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "aa1078778be5a8e5259ff04fbc3d258b3e78d464"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.MeshIO]]
deps = ["ColorTypes", "FileIO", "GeometryBasics", "Printf"]
git-tree-sha1 = "c009236e222df68e554c7ce5c720e4a33cc0c23f"
uuid = "7269a6da-0436-5bbc-96c2-40638cbb6118"
version = "0.5.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.ModernGL]]
deps = ["Libdl"]
git-tree-sha1 = "ac6cb1d8807a05cf1acc9680e09d2294f9d33956"
uuid = "66fc600b-dfda-50eb-8b99-91cfa97b1301"
version = "1.1.8"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.MuladdMacro]]
deps = ["PrecompileTools"]
git-tree-sha1 = "283bf85d4a767481dd924dff0eee1735e95f449e"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.7"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLASConsistentFPCSR_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "38a93f17e431141c6470bb67a88952a7c4f0e928"
uuid = "6cdc7f73-28fd-5e50-80fb-958a8875b1af"
version = "0.3.34+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "0d621a4beb5e48d195f907c3c5b0bea285d9ff9d"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.4.13+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.5+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d8cce34295c55f47be683580f44791716045b8fe"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.7+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05f45c2e0de6259db764adbfd2f1dc6d3f8de13c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "2.0.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "123266c25174ef6c8d4718920abc206452cf8de6"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.41"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "32b657a0d57c310a1a172bfc8c8cf68c5e674323"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.5"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7126b66b721a605a2fec966a2874c5ed53258eb3"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.58.0+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "3de8f5e6e90ebfa8d6d1f86997d6cdcd6a912ff3"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.7"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

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

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "472daaa816895cb7aee81658d4e7aec901fa1106"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d40b2fe70437b01397d2a4d5b020008da4e7019"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.2+0"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "7fb25a964849d90a0446366cdefca822e0e84900"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "3.0.6"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"
    RootsUnitfulExt = "Unitful"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "e24dc23107d426a096d3eae6c165b921e74c18e4"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.2"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays"]
git-tree-sha1 = "818554664a2e01fc3784becb2eb3a82326a604b6"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.5.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.SignedDistanceFields]]
deps = ["Statistics"]
git-tree-sha1 = "3949ad92e1c9d2ff0cd4a1317d5ecbba682f4b92"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "7ddb0b49c109481b046972c0e4ab02b2127d6a75"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.6"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "0494aed9501e7fb65daba895fb7fd57cc38bc743"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.5"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "c3ac026e735264e9bdc6a9bcbd1b1e781b36e3bc"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.8.3"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "0f529006004a8be48f1be25f3451186579392d47"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.17"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "e4d7a1a0edc20af42689ea6f4f3587a2175d50ee"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.12"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91a5737baed20ee31f3faea0e51f57461f6a689e"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "2.2.1"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

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

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

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

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.TiffImages]]
deps = ["CodecZstd", "ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "PrecompileTools", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "9ca5f1f2d42f80df4b8c9f6ab5a64f438bbd9976"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.9"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "57e1b2c9de4bd6f40ecb9de4ac1797b81970d008"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.28.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    LatexifyExt = ["Latexify", "LaTeXStrings"]
    NaNMathExt = "NaNMath"
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
    Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
    NaNMath = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "248a7031b3da79a127f14e5dc5f417e26f9f6db7"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.1.0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b29c22e245d092b8b4e8d3c09ad7baa586d9f573"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "dcb316b3ce0941f195537dda56bea4517fcd3ff5"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.4+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "2e59214e017a55cb87474a00fa76035c82ac0e17"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.47.0+2"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "852795ceb802e0f375e749d07ae97ba1179ebfcb"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.3+1"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "28e57478e8a160d346a19c28b3fffb9273bcc9c2"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.134+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "4e4282c4d846e11dce56d74fa8040130b7a95cb3"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.6.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"
"""

# ╔═╡ Cell order:
# ╟─b453ef6f-e571-48be-ac69-4c7ed462edae
# ╟─1312c3cc-1275-44ac-8669-545cdd9bd32e
# ╠═ac8de83a-17d3-450f-9f8f-3f4239232ebc
# ╠═7424297a-0722-11f1-802e-73c5586716fa
# ╠═8fc174a5-c063-46a9-8a6b-63b95ca8734b
# ╠═e54a7592-795f-4f6a-9ee9-c52001e64820
# ╠═e50b25b5-e146-4ac4-b065-56a0ba4e7957
# ╠═1d5a0c7a-d24b-4d28-b26d-fb1b16e81d81
# ╠═8ccb065f-1829-4efd-aea9-0e4200a28a69
# ╠═d61bb790-a95e-4d62-91a6-6bc6f40cb2f8
# ╠═9660c238-2c9e-4b12-8b2a-fb1594201df4
# ╠═964c70ae-f9e8-4029-90dc-96501e119399
# ╠═cf492ac3-9958-4bf4-ac4e-920f7450f6bc
# ╠═9a4b05bc-9013-4e1f-8669-c61c2d0e7ba7
# ╠═23dd5798-3416-44b6-830b-d9fda7df92c1
# ╠═435f8e95-b498-413d-a543-5e53f6e6bd3f
# ╟─4e2f78b4-e2dd-418f-bba3-c2db01bbc827
# ╠═9701d870-4e6f-4185-944c-192ad0eb109e
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

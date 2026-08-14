### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 3ccff4e8-1649-11f1-bd69-5968d5dac04b
begin
	using Diffinitive
	using Diffinitive.Grids
	using Diffinitive.SbpOperators
	using Tokens
	using SparseArrays
	using BenchmarkTools
	using LinearAlgebra
	using Diffinitive.RegionIndices
end

# ╔═╡ c2b0969f-54fd-46e2-bca3-3e51c49948a5
begin
	g = equidistant_grid((0,0),(1,1), 100,100)
	stencil_set = read_stencil_set(SbpOperators.sbp_operators_path()*"standard_diagonal.toml", order = 4)
	Δ = laplace(g, stencil_set)
end

# ╔═╡ 7af1ad52-af16-44f6-acb1-b64d6ac289da


# ╔═╡ 8f65bcd0-2836-42c5-8ddc-8138fa3c46c4


# ╔═╡ eda5ff59-7b69-4d02-8ade-849cc3195f4e
closure_sizes = map(Δ.tms) do tm
	closure_size(tm.tm)
end

# ╔═╡ 0da8d11c-a8d1-4327-af9c-3a5eab4385a3
C = CartesianIndex(1,2)[2]

# ╔═╡ 882888a5-d9f8-4c3f-b44c-057b7e3785fc


# ╔═╡ 9ec00f7e-c0d4-43c7-8ba4-fe77eee4f8b2
v = rand(size(g)...);

# ╔═╡ d4435923-f815-46d3-b5bc-6f8c3085626f
u = rand(size(g)...);

# ╔═╡ 64f77629-6086-41eb-bbf1-bc50fdd9975f
Δ̄ = sparse(Δ)

# ╔═╡ 2f84cae6-56b1-4a35-9fe8-553e4287f255
function region_apply!(u, L, v)
	closure_sizes = map(L.tms) do tm
		closure_size(tm.tm)
	end

	# R1 = Lower()
	# R2 = Lower()
	# for I ∈ regionindices(size(v), closure_sizes, (R1,R2))
	# 	u[I] = (L*v)[Index{typeof(R1),Int}(I[1]),Index{typeof(R2),Int}(I[2])]
	# end

	# R1 = Lower()
	# R2 = Interior()
	# for I ∈ regionindices(size(v), closure_sizes, (R1,R2))
	# 	u[I] = (L*v)[Index{typeof(R1),Int}(I[1]),Index{typeof(R2),Int}(I[2])]
	# end

	# R1 = Lower()
	# R2 = Upper()
	# for I ∈ regionindices(size(v), closure_sizes, (R1,R2))
	# 	u[I] = (L*v)[Index{typeof(R1),Int}(I[1]),Index{typeof(R2),Int}(I[2])]
	# end

	# R1 = Interior()
	# R2 = Lower()
	# for I ∈ regionindices(size(v), closure_sizes, (R1,R2))
	# 	u[I] = (L*v)[Index{typeof(R1),Int}(I[1]),Index{typeof(R2),Int}(I[2])]
	# end

	for j ∈ 5:96, i ∈ 5:96
		@inbounds u[i,j] = @inbounds (L*v)[Index{Interior,Int}(i),Index{Interior,Int}(j)]
	end

	# R1 = Interior()
	# R2 = Upper()
	# for I ∈ regionindices(size(v), closure_sizes, (R1,R2))
	# 	u[I] = (L*v)[Index{typeof(R1),Int}(I[1]),Index{typeof(R2),Int}(I[2])]
	# end

	
	# R1 = Upper()
	# R2 = Lower()
	# for I ∈ regionindices(size(v), closure_sizes, (R1,R2))
	# 	u[I] = (L*v)[Index{typeof(R1),Int}(I[1]),Index{typeof(R2),Int}(I[2])]
	# end

	# R1 = Upper()
	# R2 = Interior()
	# for I ∈ regionindices(size(v), closure_sizes, (R1,R2))
	# 	u[I] = (L*v)[Index{typeof(R1),Int}(I[1]),Index{typeof(R2),Int}(I[2])]
	# end

	# R1 = Upper()
	# R2 = Upper()
	# for I ∈ regionindices(size(v), closure_sizes, (R1,R2))
	# 	u[I] = (L*v)[Index{typeof(R1),Int}(I[1]),Index{typeof(R2),Int}(I[2])]
	# end


	return u
end

# ╔═╡ 977feee2-37f8-4f8b-a983-454070e20224
function loop_apply!(u,L,v)
	for I ∈ CartesianIndices(u)
		u[I] = LazyTensors.apply(L,v,Tuple(I)...)
	end
end

# ╔═╡ d399ec7a-7b2a-49e2-8ba4-51f692cefba0
function loop_apply_inbounds!(u,L,v)
	for I ∈ CartesianIndices(u)
		@inbounds u[I] = @inbounds LazyTensors.apply(L,v,Tuple(I)...)
	end
end

# ╔═╡ 03670946-8f52-4607-b8be-fea4a108dc62
function loop_apply2!(u,L,v)
	for i ∈ 1:100, j ∈ 1:100
		@inbounds u[i,j] = @inbounds LazyTensors.apply(L,v,i,j)
	end
end

# ╔═╡ 2086df21-08d1-45b8-85f6-3bc481746666
function broadcast_apply!(u,L,v)
	u .= L*v
end

# ╔═╡ 73bc721d-991d-40a1-b41f-63ea32913c00
function getrange(gridsize::Integer, closuresize::Integer, region::Region)
    if region == Lower()
        r = 1:closuresize
    elseif region == Interior()
        r = (closuresize+1):(gridsize - closuresize)
    elseif region == Upper()
        r = (gridsize - closuresize + 1):gridsize
    end
    return r
end

# ╔═╡ a9323e3b-72c1-4f36-9619-755acbea5b5d
getrange(100, closure_sizes[1], Lower())

# ╔═╡ 712212aa-a282-4cc3-90fc-50ba093d1428
getrange(100, closure_sizes[1], Interior())

# ╔═╡ e20fe8e1-f29f-45e0-ad46-9d3e44be8cb7
getrange(100, closure_sizes[1], Upper())

# ╔═╡ 96b2ba3f-6c62-41f4-89df-3addfe1a401f
getrange(10, 3, Lower())

# ╔═╡ fb85c652-0ec6-42d2-bb41-42eb5cd21919
function regionindices(gridsize, closuresize, region::NTuple{N,Region} where N)
    regions = map(getrange,gridsize,closuresize,region)
    return CartesianIndices(regions)
end

# ╔═╡ 9637603e-d184-4f6e-9ed6-242e91a432f3
@code_warntype region_apply!(u,Δ,v)

# ╔═╡ 137e03e8-62db-497c-89a8-a5280c5725d8
@code_native region_apply!(u,Δ,v)

# ╔═╡ a1f8b0a7-6064-4bbe-960b-a3190bb9863a
let
	u = rand(size(g)...)
	v1 = similar(u)
	v2 = similar(u)

	v1 .= Δ*u
	region_apply!(v2,Δ,u)
	
	v1 ≈ v2
end

# ╔═╡ e34afe35-fba9-42fe-a65d-10348138e79c
@benchmark broadcast_apply!($u,$Δ,$v)

# ╔═╡ 53839612-4c9a-478d-b764-9f535b717341
@benchmark loop_apply!($u,$Δ,$v)

# ╔═╡ c5f585ed-5b9c-4632-bae6-5e6bb7de6c3c
@benchmark loop_apply2!($u,$Δ,$v)

# ╔═╡ fc02566c-7efa-4b75-a0c7-71e49492c0b2
@benchmark region_apply!($u,$Δ,$v)

# ╔═╡ 9f2d20c3-828f-4c91-ba4f-308e4d2d2a6d
@benchmark mul!($(reshape(v,:)), $Δ̄, $(reshape(u,:)))

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Diffinitive = "5a373a26-915f-4769-bcab-bf03835de17b"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
Tokens = "040c2ec2-8d69-4aca-bf03-7d3a7092f2f6"

[compat]
BenchmarkTools = "~1.8.0"
Diffinitive = "~0.1.8"
Tokens = "~0.1.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "5dfd84c15a4c630bc80bc36c7e8df23a1834fd7c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "PrecompileTools", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "9670d3febc2b6da60a0ae57846ba74670290653f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.8.0"

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
version = "1.3.0+1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Diffinitive]]
deps = ["LinearAlgebra", "StaticArrays", "TOML"]
git-tree-sha1 = "5438e4e201401f7f7e080f6f48182c64bc9c96fe"
uuid = "5a373a26-915f-4769-bcab-bf03835de17b"
version = "0.1.8"

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

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "c7345ab1a7ca4dc8a02c9f6510da0d9857bbe513"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.7.1"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.LRUCache]]
git-tree-sha1 = "5519b95a490ff5fe629c4a7aa3b3dfc9160498b3"
uuid = "8ac3fa9e-de4c-5943-b1dc-09c6b5f20637"
version = "1.6.2"
weakdeps = ["Serialization"]

    [deps.LRUCache.extensions]
    SerializationExt = ["Serialization"]

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "fb28e33b8a95c4cee25ce296c817d89cc2e53518"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.2"

    [deps.PackageExtensionCompat.weakdeps]
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    TOML = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "3de8f5e6e90ebfa8d6d1f86997d6cdcd6a912ff3"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.7"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

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
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SparseArrayKit]]
deps = ["LinearAlgebra", "PackageExtensionCompat", "TensorOperations", "TupleTools", "VectorInterface"]
git-tree-sha1 = "ca73c56565f410f54841eadd13145516cab2c02d"
uuid = "a9a3c162-d163-4c15-8926-b8794fbefed2"
version = "0.4.3"
weakdeps = ["SparseArrays"]

    [deps.SparseArrayKit.extensions]
    SparseArrayKitSparseArrays = "SparseArrays"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "246a8bb2e6667f832eea063c3a56aef96429a3db"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.18"

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
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.Strided]]
deps = ["LinearAlgebra", "PrecompileTools", "StridedViews", "TupleTools"]
git-tree-sha1 = "5fa7f6845c91e6e351880cee67a9efc3b892bd3b"
uuid = "5e0ebb24-38b0-5f93-81fe-25c709ecae67"
version = "2.6.4"

    [deps.Strided.extensions]
    StridedAMDGPUExt = "AMDGPU"
    StridedGPUArraysExt = "GPUArrays"
    StridedcuBLASExt = "cuBLAS"

    [deps.Strided.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    cuBLAS = "182d3088-87b7-4494-8cad-fc6afaa545bc"

[[deps.StridedViews]]
deps = ["LinearAlgebra", "PrecompileTools"]
git-tree-sha1 = "21dc3942c478661f72c527ff5d67baa98e555372"
uuid = "4db3bf67-4bd7-4b4e-b153-31dc3fb37143"
version = "0.5.2"

    [deps.StridedViews.extensions]
    StridedViewsAMDGPUExt = "AMDGPU"
    StridedViewsAdaptExt = "Adapt"
    StridedViewsCUDACoreExt = "CUDACore"
    StridedViewsJLArraysExt = "JLArrays"
    StridedViewsPtrArraysExt = "PtrArrays"

    [deps.StridedViews.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    CUDACore = "bd0ed864-bdfe-4181-a5ed-ce625a5fdea2"
    JLArrays = "27aeb0d3-9eb9-45fb-866b-73c2ecf80fcb"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    PtrArrays = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"

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

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TensorOperations]]
deps = ["LRUCache", "LinearAlgebra", "PackageExtensionCompat", "PrecompileTools", "Preferences", "PtrArrays", "Strided", "StridedViews", "TupleTools", "VectorInterface"]
git-tree-sha1 = "6cae1bb6d94db05972ccb3b8d83e422df3224d7a"
uuid = "6aa20fa7-93e2-5fca-9bc0-fbd0db3c71a2"
version = "5.8.0"

    [deps.TensorOperations.extensions]
    TensorOperationsAMDGPUExt = "AMDGPU"
    TensorOperationsBumperExt = "Bumper"
    TensorOperationsCUDACoreExt = "CUDACore"
    TensorOperationsChainRulesCoreExt = "ChainRulesCore"
    TensorOperationsEnzymeExt = "Enzyme"
    TensorOperationsGPUArraysExt = "GPUArrays"
    TensorOperationsJLArraysExt = "JLArrays"
    TensorOperationsMooncakeExt = "Mooncake"
    TensorOperationsTBLISExt = "TBLIS"
    TensorOperationscuTENSORExt = "cuTENSOR"

    [deps.TensorOperations.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    Bumper = "8ce10254-0962-460f-a3d8-1f77fea1446e"
    CUDACore = "bd0ed864-bdfe-4181-a5ed-ce625a5fdea2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    JLArrays = "27aeb0d3-9eb9-45fb-866b-73c2ecf80fcb"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    TBLIS = "48530278-0828-4a49-9772-0f3830dfa1e9"
    cuTENSOR = "011b41b2-24ef-40a8-b3eb-fa098493e9e1"

[[deps.Tokens]]
deps = ["SparseArrayKit", "SparseArrays"]
git-tree-sha1 = "c4f40125383ce3bfcfcd49a1b206080b7afd9a34"
uuid = "040c2ec2-8d69-4aca-bf03-7d3a7092f2f6"
version = "0.1.1"

[[deps.TupleTools]]
git-tree-sha1 = "41e43b9dc950775eac654b9f845c839cd2f1821e"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.6.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.VectorInterface]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "949dd28df19a5bf0973214e4a9d36c19079d4d45"
uuid = "409d34a3-91d5-4945-b6ec-7529ddf182d8"
version = "0.6.0"

    [deps.VectorInterface.extensions]
    VectorInterfaceChainRulesCoreExt = "ChainRulesCore"
    VectorInterfaceEnzymeExt = "Enzyme"
    VectorInterfaceMooncakeExt = "Mooncake"
    VectorInterfaceStaticArraysExt = "StaticArrays"

    [deps.VectorInterface.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"
"""

# ╔═╡ Cell order:
# ╠═3ccff4e8-1649-11f1-bd69-5968d5dac04b
# ╠═c2b0969f-54fd-46e2-bca3-3e51c49948a5
# ╠═7af1ad52-af16-44f6-acb1-b64d6ac289da
# ╠═8f65bcd0-2836-42c5-8ddc-8138fa3c46c4
# ╠═eda5ff59-7b69-4d02-8ade-849cc3195f4e
# ╠═0da8d11c-a8d1-4327-af9c-3a5eab4385a3
# ╠═a9323e3b-72c1-4f36-9619-755acbea5b5d
# ╠═712212aa-a282-4cc3-90fc-50ba093d1428
# ╠═e20fe8e1-f29f-45e0-ad46-9d3e44be8cb7
# ╠═882888a5-d9f8-4c3f-b44c-057b7e3785fc
# ╠═9ec00f7e-c0d4-43c7-8ba4-fe77eee4f8b2
# ╠═d4435923-f815-46d3-b5bc-6f8c3085626f
# ╠═64f77629-6086-41eb-bbf1-bc50fdd9975f
# ╠═96b2ba3f-6c62-41f4-89df-3addfe1a401f
# ╠═2f84cae6-56b1-4a35-9fe8-553e4287f255
# ╠═977feee2-37f8-4f8b-a983-454070e20224
# ╠═d399ec7a-7b2a-49e2-8ba4-51f692cefba0
# ╠═03670946-8f52-4607-b8be-fea4a108dc62
# ╠═2086df21-08d1-45b8-85f6-3bc481746666
# ╠═fb85c652-0ec6-42d2-bb41-42eb5cd21919
# ╠═73bc721d-991d-40a1-b41f-63ea32913c00
# ╠═9637603e-d184-4f6e-9ed6-242e91a432f3
# ╠═137e03e8-62db-497c-89a8-a5280c5725d8
# ╠═a1f8b0a7-6064-4bbe-960b-a3190bb9863a
# ╠═e34afe35-fba9-42fe-a65d-10348138e79c
# ╠═53839612-4c9a-478d-b764-9f535b717341
# ╠═c5f585ed-5b9c-4632-bae6-5e6bb7de6c3c
# ╠═fc02566c-7efa-4b75-a0c7-71e49492c0b2
# ╠═9f2d20c3-828f-4c91-ba4f-308e4d2d2a6d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002

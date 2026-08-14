### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 1f8a7cfa-94cc-41bf-a8c8-2dc5218741e0
begin
	using Pkg
	Pkg.activate(".")
	
	using Diffinitive
	using Diffinitive.Grids
	using Diffinitive.LazyTensors
	using Diffinitive.SbpOperators
	using PlutoUI
	using StaticArrays
end

# ╔═╡ 885c60d7-d33c-4741-ae49-6a57510ec7b5
md"""
# Display tests
"""

# ╔═╡ 9ee3372a-e78d-4f74-84ce-e04208d1558d
repl_show(v) = repr(MIME("text/plain"), v) |> println

# ╔═╡ 51c02ced-f684-417f-83f1-cade4edda43f
md"""
Common julia objects to compare with:
"""

# ╔═╡ 25c90528-22cd-41ca-8572-ccd946928318
1 |> repl_show

# ╔═╡ e7f3e466-9833-428c-99ad-20bc9d88d951
[1,1] |> repl_show

# ╔═╡ 2e74f9b5-5b4f-4887-8a30-4655d560a45c
[1;; 2;;] |> repl_show

# ╔═╡ 365524b5-3182-4691-9817-1bbec1492c14
[1; 2;;] |> repl_show

# ╔═╡ d5725e1b-bc4f-4a95-975d-179c193908c9
[1; 2;; 3; 4;;] |> repl_show

# ╔═╡ b824ef8d-5026-4861-9a23-45a7939fd38c
"hej" |> repl_show

# ╔═╡ fbe365a2-f95e-4297-8326-c18d22932869
Dict("A" => 1, "B"=> 2) |> repl_show

# ╔═╡ 56670aff-0343-41cb-a653-35a61376dda4
1//2 |> repl_show

# ╔═╡ b5a6491e-a93e-4058-8ceb-be1dc4d4c100
BigInt(30) |> repl_show

# ╔═╡ 828d57a1-ee58-4204-8050-78127821a4c6
1:10 |> repl_show

# ╔═╡ 127d34f6-69f7-4082-a74b-0be86942f153
range(0,1,10) |> repl_show

# ╔═╡ c46a278e-a102-4544-82d8-7df816440410
rand(2,2,2,2) |> repl_show

# ╔═╡ 5aa7079c-8005-47f1-bb82-c35f3aa54b42
md"""
## Parameter spaces
"""

# ╔═╡ 08f493ed-189c-43f3-86f2-95fc475ec0e7
Interval(1,2) |> repl_show

# ╔═╡ e2dc581f-0949-4be7-8e4b-74deeaffc68d
md"""
## Stencil Set
"""

# ╔═╡ 272e564f-bd21-4ba2-8d6e-ae4714dc30bc
read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=2) |> repl_show

# ╔═╡ f7244bf7-8266-469f-b07f-30c203d9af48
md"""
## Grids
"""

# ╔═╡ 0e14bd28-5dd1-44c4-abf4-23b70546bd49
equidistant_grid(0,1,11) |> repl_show

# ╔═╡ fcb74341-6b03-4ada-8f5d-bc245c23679b
equidistant_grid((0,0),(1,1),10,20) |> repl_show

# ╔═╡ 8dec053b-eaae-463d-800b-b8d89d5d550b
ZeroDimGrid(@SVector[1,2]) |> repl_show

# ╔═╡ c1172a36-c5d7-47dc-bc79-af0d43a8f6ee
let
	x̄((ξ, η)) = @SVector[2ξ + η*(1-η), 3η+(1+η/2)*ξ^2]
    J((ξ, η)) = @SMatrix[
        2       1-2η;
        (2+η)*ξ 3+1/2*ξ^2;
    ]

	mapped_grid(x̄, J, 10,10) |> repl_show
end

# ╔═╡ 9c889176-865b-402d-81b5-71957d2878f7
md"""
## LazyArrays
"""

# ╔═╡ 85e8e748-e575-4a29-80c7-22d110578343
LazyTensors.LazyConstantArray(10, (5,)) |> repl_show

# ╔═╡ 804ad722-9081-4d1d-b0d2-c536a26fe20d
LazyTensors.LazyFunctionArray((i,j)->10*i+j, (3,4)) |> repl_show

# ╔═╡ a70c689d-0851-497f-938a-e5c92ce59ddb
md"""
## LazyTensors
"""

# ╔═╡ 68c7a1d8-729e-4f38-abf2-26deb7a90cb1
md"""
### Basic tensors
"""

# ╔═╡ 2afde3fe-96ed-4d7e-a79b-fc880e0da268
LazyTensors.IdentityTensor(5) |> repl_show

# ╔═╡ 5451a071-14ae-47ae-99c5-4d65508d280f
LazyTensors.IdentityTensor(4,3) |> repl_show

# ╔═╡ b6b06fe9-de16-41ca-ad45-eef6dd038485
LazyTensors.ScalingTensor(2., (4,3)) |> repl_show

# ╔═╡ d1c8c3a0-76ec-4c32-853e-0471d71e5cf0
LazyTensors.DiagonalTensor([1,2,3,4]) |> repl_show

# ╔═╡ 6051c144-9982-4bd9-92f9-d0aaf3961872
LazyTensors.DenseTensor(rand(2,2,2,2), (1,2), (3,4)) |> repl_show

# ╔═╡ 83ed7f7e-c88d-4ee4-a53a-1b91e775ff52
md"""
### Simple SBP-operators
"""

# ╔═╡ 12a9f430-f96b-43f2-bf63-149b5a028fd7
begin
	g1 = equidistant_grid(0,1,10)
	g2 = equidistant_grid((0,0),(1,1),10, 11)
	x̄((ξ, η)) = @SVector[2ξ + η*(1-η), 3η+(1+η/2)*ξ^2]
    J((ξ, η)) = @SMatrix[
        2       1-2η;
        (2+η)*ξ 3+1/2*ξ^2;
    ]
	mg = mapped_grid(x̄, J, 10,10) 
	stencil_set2 = stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=2)
	stencil_set4 = stencil_set = read_stencil_set(sbp_operators_path()*"standard_diagonal.toml"; order=4)
end;

# ╔═╡ e44f8d91-1cbf-44be-bef1-40e60c4a777f
first_derivative(g1, stencil_set2) |> repl_show

# ╔═╡ c378b88a-1d74-44a2-bdc1-b371da478de8
first_derivative(g1, stencil_set4) |> repl_show

# ╔═╡ 9614cda7-b48c-4925-89b5-113cf514f20f
first_derivative(g2, stencil_set2, 1) |> repl_show

# ╔═╡ 725e9430-4821-4939-bfef-6c186d2dc500
first_derivative(g2, stencil_set4, 2) |> repl_show

# ╔═╡ e8ca54a1-a6db-40a1-b44a-73e175894df4
second_derivative(g1, stencil_set2) |> repl_show

# ╔═╡ 5d1f10fe-f620-469c-822c-55955a5541ad
second_derivative(g1, stencil_set4) |> repl_show

# ╔═╡ fdc531d4-9d27-41dc-ba79-6febefde223a
second_derivative(g2, stencil_set2, 1) |> repl_show

# ╔═╡ fff2e04a-357f-4254-996e-d5ccd9ff31f8
second_derivative(g2, stencil_set4, 2) |> repl_show

# ╔═╡ 959b071e-1ef6-4f29-aa8b-d88bfef80c00
second_derivative_variable(g1, map(x->2x, g1), stencil_set2) |> repl_show

# ╔═╡ 0a0f7e77-a789-4fb3-a4c2-7853b67788ec
second_derivative_variable(g1, map(x->2x, g1), stencil_set4) |> repl_show

# ╔═╡ 177e0893-fbb1-4bb5-a108-e5990e943ab7
 second_derivative_variable(g2, map(x->x[1]+x[2], g2), stencil_set2, 1) |> repl_show

# ╔═╡ f1b6bd54-baf6-4360-aec0-bd8d52497894
second_derivative_variable(g2, map(x->x[1]+x[2], g2), stencil_set4, 2) |> repl_show

# ╔═╡ a8f8343e-cd6f-451d-a2b3-e5f0f561f8af
undivided_skewed04(g1,4)[1] |> repl_show

# ╔═╡ 50291998-7194-4a0d-9c19-eacc48b3f5da
undivided_skewed04(g1,4)[2] |> repl_show

# ╔═╡ 1720af08-85e4-4502-b37e-9fa73008e221
undivided_skewed04(g2,4,1)[1] |> repl_show

# ╔═╡ 4a72f217-2423-4bc1-8452-eb28dde36689
undivided_skewed04(g2,4,2)[2] |> repl_show

# ╔═╡ d51b6bd7-0235-4a51-a989-4f7858363d02
md"""
### Inner products
"""

# ╔═╡ 48a28ade-73bb-461d-ab96-82f92ed199c8
inner_product(g1, stencil_set2) |> repl_show

# ╔═╡ 3c681a63-94a6-4677-aef7-df903c463896
inner_product(g1, stencil_set4) |> repl_show

# ╔═╡ db735370-153a-40f9-b77f-9f60e30a35c4
inner_product(g2, stencil_set2) |> repl_show

# ╔═╡ 613ebac7-50bc-424c-8fa2-064b64c93319
inner_product(g2, stencil_set4) |> repl_show

# ╔═╡ 9e7d7667-960e-491f-8b83-e3b01a0db5b0
inverse_inner_product(g2, stencil_set4) |> repl_show

# ╔═╡ 29488e48-1d42-4232-9d49-1ee77fb869d8
md"""
### Boundary operators
"""

# ╔═╡ c3005e74-5b96-4b0c-9c57-b7d02968ed94
boundary_restriction(g1, stencil_set, LowerBoundary()) |> repl_show

# ╔═╡ fd019973-0d54-4e31-b43b-f53a704cb01c
boundary_restriction(g1, stencil_set, UpperBoundary()) |> repl_show

# ╔═╡ df2e8af0-9ca5-4972-8771-f8bea1591f85
boundary_restriction(g2, stencil_set, CartesianBoundary{1,LowerBoundary}()) |> repl_show

# ╔═╡ 8e538109-16a1-4af7-a986-9dc1455b7de7
boundary_restriction(g2, stencil_set, CartesianBoundary{2,UpperBoundary}()) |> repl_show

# ╔═╡ 5f3744a4-72d8-4448-820f-a928bfaaf825
normal_derivative(g1, stencil_set, LowerBoundary()) |> repl_show

# ╔═╡ 4529b5c4-4905-4fd3-9aaa-5f88faa841c8
normal_derivative(g1, stencil_set, UpperBoundary()) |> repl_show

# ╔═╡ c3bb0450-a7d5-44c1-9ca9-9e1ecf2db9f8
normal_derivative(g2, stencil_set, CartesianBoundary{1,LowerBoundary}()) |> repl_show

# ╔═╡ da63e57a-0794-4cd8-9941-ada0e5c1c40e
normal_derivative(g2, stencil_set, CartesianBoundary{2,UpperBoundary}()) |> repl_show

# ╔═╡ 0b951425-979c-4ac9-8581-f690b729bab4
md"""
## Tensor operations
"""

# ╔═╡ a39bb6e2-f1fe-4206-8ee3-88ff0c075233
begin
	Dx = first_derivative(g2, stencil_set2, 1)
	Dy = first_derivative(g2, stencil_set4, 2)

	v = map(x->sin(x[1]^2+x[2]^2), g2)
end;

# ╔═╡ 8cd052d9-f40e-4796-aeef-52c02b3bf156
Dx+Dy |> repl_show

# ╔═╡ 1b1b7d12-50ef-4c4e-9376-a353a56540c3
Dx∘Dy |> repl_show

# ╔═╡ 57e48b4c-eef6-433e-98a1-1006e844b368
Dx*v |> repl_show

# ╔═╡ e2bf2649-4770-4f52-9752-b61ce03c6f82
(Dx+Dy)*v |> repl_show

# ╔═╡ d163d363-853d-4d83-a2d3-f8dd6e8f552d
(Dx∘Dy)*v |> repl_show

# ╔═╡ 1c38d3f9-1839-468c-a368-4ef101bd4f18
laplace(g2, stencil_set2) |> repl_show

# ╔═╡ cf84cefb-2dbd-4b8d-880b-47cc350a7c43
laplace(g2, stencil_set4) |> repl_show

# ╔═╡ 67c73667-1f41-47b5-b59a-459787767f29
# laplace(mg, stencil_set2) |> repl_show

# ╔═╡ 4634c1a6-0520-4b0f-8d32-a1fdf2ebaea5
md"""
## Appendix
"""

# ╔═╡ 24788161-b29a-450a-bd35-f9c29e7ded9a
PlutoUI.TableOfContents()

# ╔═╡ Cell order:
# ╟─885c60d7-d33c-4741-ae49-6a57510ec7b5
# ╠═9ee3372a-e78d-4f74-84ce-e04208d1558d
# ╟─51c02ced-f684-417f-83f1-cade4edda43f
# ╠═25c90528-22cd-41ca-8572-ccd946928318
# ╠═e7f3e466-9833-428c-99ad-20bc9d88d951
# ╠═2e74f9b5-5b4f-4887-8a30-4655d560a45c
# ╠═365524b5-3182-4691-9817-1bbec1492c14
# ╠═d5725e1b-bc4f-4a95-975d-179c193908c9
# ╠═b824ef8d-5026-4861-9a23-45a7939fd38c
# ╠═fbe365a2-f95e-4297-8326-c18d22932869
# ╠═56670aff-0343-41cb-a653-35a61376dda4
# ╠═b5a6491e-a93e-4058-8ceb-be1dc4d4c100
# ╠═828d57a1-ee58-4204-8050-78127821a4c6
# ╠═127d34f6-69f7-4082-a74b-0be86942f153
# ╠═c46a278e-a102-4544-82d8-7df816440410
# ╟─5aa7079c-8005-47f1-bb82-c35f3aa54b42
# ╠═08f493ed-189c-43f3-86f2-95fc475ec0e7
# ╟─e2dc581f-0949-4be7-8e4b-74deeaffc68d
# ╠═272e564f-bd21-4ba2-8d6e-ae4714dc30bc
# ╟─f7244bf7-8266-469f-b07f-30c203d9af48
# ╠═0e14bd28-5dd1-44c4-abf4-23b70546bd49
# ╠═fcb74341-6b03-4ada-8f5d-bc245c23679b
# ╠═8dec053b-eaae-463d-800b-b8d89d5d550b
# ╠═c1172a36-c5d7-47dc-bc79-af0d43a8f6ee
# ╟─9c889176-865b-402d-81b5-71957d2878f7
# ╠═85e8e748-e575-4a29-80c7-22d110578343
# ╠═804ad722-9081-4d1d-b0d2-c536a26fe20d
# ╟─a70c689d-0851-497f-938a-e5c92ce59ddb
# ╟─68c7a1d8-729e-4f38-abf2-26deb7a90cb1
# ╠═2afde3fe-96ed-4d7e-a79b-fc880e0da268
# ╠═5451a071-14ae-47ae-99c5-4d65508d280f
# ╠═b6b06fe9-de16-41ca-ad45-eef6dd038485
# ╠═d1c8c3a0-76ec-4c32-853e-0471d71e5cf0
# ╠═6051c144-9982-4bd9-92f9-d0aaf3961872
# ╟─83ed7f7e-c88d-4ee4-a53a-1b91e775ff52
# ╠═12a9f430-f96b-43f2-bf63-149b5a028fd7
# ╠═e44f8d91-1cbf-44be-bef1-40e60c4a777f
# ╠═c378b88a-1d74-44a2-bdc1-b371da478de8
# ╠═9614cda7-b48c-4925-89b5-113cf514f20f
# ╠═725e9430-4821-4939-bfef-6c186d2dc500
# ╠═e8ca54a1-a6db-40a1-b44a-73e175894df4
# ╠═5d1f10fe-f620-469c-822c-55955a5541ad
# ╠═fdc531d4-9d27-41dc-ba79-6febefde223a
# ╠═fff2e04a-357f-4254-996e-d5ccd9ff31f8
# ╠═959b071e-1ef6-4f29-aa8b-d88bfef80c00
# ╠═0a0f7e77-a789-4fb3-a4c2-7853b67788ec
# ╠═177e0893-fbb1-4bb5-a108-e5990e943ab7
# ╠═f1b6bd54-baf6-4360-aec0-bd8d52497894
# ╠═a8f8343e-cd6f-451d-a2b3-e5f0f561f8af
# ╠═50291998-7194-4a0d-9c19-eacc48b3f5da
# ╠═1720af08-85e4-4502-b37e-9fa73008e221
# ╠═4a72f217-2423-4bc1-8452-eb28dde36689
# ╟─d51b6bd7-0235-4a51-a989-4f7858363d02
# ╠═48a28ade-73bb-461d-ab96-82f92ed199c8
# ╠═3c681a63-94a6-4677-aef7-df903c463896
# ╠═db735370-153a-40f9-b77f-9f60e30a35c4
# ╠═613ebac7-50bc-424c-8fa2-064b64c93319
# ╠═9e7d7667-960e-491f-8b83-e3b01a0db5b0
# ╟─29488e48-1d42-4232-9d49-1ee77fb869d8
# ╠═c3005e74-5b96-4b0c-9c57-b7d02968ed94
# ╠═fd019973-0d54-4e31-b43b-f53a704cb01c
# ╠═df2e8af0-9ca5-4972-8771-f8bea1591f85
# ╠═8e538109-16a1-4af7-a986-9dc1455b7de7
# ╠═5f3744a4-72d8-4448-820f-a928bfaaf825
# ╠═4529b5c4-4905-4fd3-9aaa-5f88faa841c8
# ╠═c3bb0450-a7d5-44c1-9ca9-9e1ecf2db9f8
# ╠═da63e57a-0794-4cd8-9941-ada0e5c1c40e
# ╟─0b951425-979c-4ac9-8581-f690b729bab4
# ╠═a39bb6e2-f1fe-4206-8ee3-88ff0c075233
# ╠═8cd052d9-f40e-4796-aeef-52c02b3bf156
# ╠═1b1b7d12-50ef-4c4e-9376-a353a56540c3
# ╠═57e48b4c-eef6-433e-98a1-1006e844b368
# ╠═e2bf2649-4770-4f52-9752-b61ce03c6f82
# ╠═d163d363-853d-4d83-a2d3-f8dd6e8f552d
# ╠═1c38d3f9-1839-468c-a368-4ef101bd4f18
# ╠═cf84cefb-2dbd-4b8d-880b-47cc350a7c43
# ╠═67c73667-1f41-47b5-b59a-459787767f29
# ╟─4634c1a6-0520-4b0f-8d32-a1fdf2ebaea5
# ╠═1f8a7cfa-94cc-41bf-a8c8-2dc5218741e0
# ╠═24788161-b29a-450a-bd35-f9c29e7ded9a

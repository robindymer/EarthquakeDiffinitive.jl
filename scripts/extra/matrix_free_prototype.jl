# Matrix-free (Kronecker 1D-operator) prototype of the split-node system A = -H P (D+SAT) P.
#   julia --project=scripts scripts/extra/matrix_free_prototype.jl <Δz> <L_fault> <L_normal> [gpu]
using EarthquakeDiffinitive
using EarthquakeDiffinitive.BP8
using EarthquakeDiffinitive.BP8: BP8Params, lame_lambda, fault_grid_sizes
using EarthquakeDiffinitive.Elasticity
using EarthquakeDiffinitive.ElasticitySplitNode
using EarthquakeDiffinitive.ElasticitySplitNode: _prolongation, _append_block!, _to_sparse_matrix
using Diffinitive, Diffinitive.Grids, Diffinitive.SbpOperators, Diffinitive.LazyTensors
using SparseArrays, LinearAlgebra, Printf, Random
using CUDA, CUDA.CUSPARSE
using EarthquakeDiffinitive.ElasticitySplitNode: CgWorkspace, cg!

Δz = parse(Float64, ARGS[1]); L_fault = parse(Float64, ARGS[2]); L_normal = parse(Float64, ARGS[3])
use_gpu = length(ARGS) >= 4 && ARGS[4] == "gpu"

par = BP8Params()
order = 4
set = read_stencil_set(SbpOperators.sbp_operators_path() * "standard_diagonal.toml"; order)
n1, n23 = fault_grid_sizes(par, Δz, L_fault, L_normal, order)
λ, μ = lame_lambda(par), par.μ
g_minus = equidistant_grid((-L_normal, -L_fault, -L_fault), (0.0, L_fault, L_fault), n1, n23, n23)
g_plus = equidistant_grid((0.0, -L_fault, -L_fault), (L_normal, L_fault, L_fault), n1, n23, n23)
Nm, Np = length(g_minus), length(g_plus)
Ntot = 3(Nm + Np)
@printf("grid %d x %d x %d per side, Ntot = %d\n", n1, n23, n23, Ntot)

# ---------------------------------------------------------------- 1. profile the assembly
function profiled_split_node_system(g_minus, g_plus, λ, μ, stencil_set)
    D = 3
    Nm, Np = length(g_minus), length(g_plus)
    Ntot = D * (Nm + Np)
    bid_minus = CartesianBoundary{1,UpperBoundary}()
    bid_plus = CartesianBoundary{1,LowerBoundary}()
    tt = Dict{String,Float64}()
    tt["D  (sparse of lazy blocks)"] = @elapsed begin
        Dm = _to_sparse_matrix(elastic_blocks(g_minus, λ, μ, stencil_set))
        Dp = _to_sparse_matrix(elastic_blocks(g_plus, λ, μ, stencil_set))
    end
    Dmat = blockdiag(Dm, Dp)
    tt["SAT (traction_blocks etc.)"] = @elapsed begin
        Tm = traction_blocks(g_minus, λ, μ, stencil_set, bid_minus)
        Tp = traction_blocks(g_plus, λ, μ, stencil_set, bid_plus)
        Prolong_p = _prolongation(g_plus, stencil_set, bid_plus)
        Prolong_m = _prolongation(g_minus, stencil_set, bid_minus)
        Isat, Jsat, Vsat = Int[], Int[], Float64[]
        for j in 1:D
            row0_m = (j-1) * Nm; row0_p = D * Nm + (j-1) * Np
            Tp_j = reduce(hcat, Tp[j, :]); Tm_j = reduce(hcat, Tm[j, :])
            _append_block!(Isat, Jsat, Vsat, 0.5 .* (Prolong_m * Tm_j), row0_m, 0)
            _append_block!(Isat, Jsat, Vsat, -0.5 .* (Prolong_m * Tp_j), row0_m, D * Nm)
            _append_block!(Isat, Jsat, Vsat, -0.5 .* (Prolong_p * Tp_j), row0_p, D * Nm)
            _append_block!(Isat, Jsat, Vsat, 0.5 .* (Prolong_p * Tm_j), row0_p, 0)
        end
        SATmat = sparse(Isat, Jsat, Vsat, Ntot, Ntot)
    end
    DSAT = Dmat + SATmat
    tt["P  (row loops on CSC)"] = @elapsed begin
        P = sparse(1.0I, Ntot, Ntot)
        far_field_minus = filter(!=(bid_minus), boundary_identifiers(g_minus))
        far_field_plus = filter(!=(bid_plus), boundary_identifiers(g_plus))
        for bid in far_field_minus, I in boundary_indices(g_minus, bid), comp in 1:D
            P[dof_index_minus(g_minus, comp, I), :] .= 0.0
        end
        for bid in far_field_plus, I in boundary_indices(g_plus, bid), comp in 1:D
            P[dof_index_plus(g_minus, g_plus, comp, I), :] .= 0.0
        end
        for (Im, Ip) in fault_node_pairs(g_minus, g_plus), comp in 1:3
            rm = dof_index_minus(g_minus, comp, Im)
            rp = dof_index_plus(g_minus, g_plus, comp, Ip)
            for r in (rm, rp)
                P[r, :] .= 0.0; P[r, rm] = 0.5; P[r, rp] = 0.5
            end
        end
        dropzeros!(P)
    end
    tt["H"] = @elapsed begin
        Hm = sparse(inner_product(g_minus, stencil_set)); Hp = sparse(inner_product(g_plus, stencil_set))
        H = blockdiag(blockdiag(fill(Hm, D)...), blockdiag(fill(Hp, D)...))
    end
    tt["HP_DSAT = H*P*DSAT"] = @elapsed HP_DSAT = H * P * DSAT
    tt["A = -HP_DSAT*P"] = @elapsed A = -HP_DSAT * P
    return A, HP_DSAT, P, Dmat, SATmat, H, tt
end

println("\n--- assembly profile")
ta = @elapsed (A, HP_DSAT, P, Dmat, SATmat, H, tt) = profiled_split_node_system(g_minus, g_plus, λ, μ, set)
for (k, v) in sort(collect(tt); by=last, rev=true)
    @printf("  %-30s %8.1f s  (%4.1f%%)\n", k, v, 100v / ta)
end
@printf("  %-30s %8.1f s\n", "total", ta)
@printf("  nnz(A) = %.3e (%.1f per row), nnz(SAT) = %.2e, nnz(P) = %.2e\n", nnz(A), nnz(A) / Ntot, nnz(SATmat), nnz(P))

# ---------------------------------------------------------------- 2. matrix-free operator
# 1D operators along each axis of each side (as small CSR-able sparse matrices), and
# a check that Diffinitive's 3D operators really are their Kronecker products.

function axes1d(g)
    # TensorGrid of 3 EquidistantGrids
    return g.grids
end
struct Ops1D
    d1::Vector{SparseMatrixCSC{Float64,Int}}
    d2::Vector{SparseMatrixCSC{Float64,Int}}
    h::Vector{Vector{Float64}}
end
function Ops1D(g, set)
    gs = axes1d(g)
    d1 = [sparse(first_derivative(gs[d], set)) for d in 1:3]
    d2 = [sparse(second_derivative(gs[d], set)) for d in 1:3]
    h = [diag(sparse(inner_product(gs[d], set))) for d in 1:3]
    return Ops1D(d1, d2, h)
end
ops_m = Ops1D(g_minus, set); ops_p = Ops1D(g_plus, set)

# Kronecker check on the small operator
let
    D1_3 = sparse(first_derivative(g_minus, set, 2))
    K = kron(sparse(1.0I, n23, n23), kron(ops_m.d1[2], sparse(1.0I, n1, n1)))   # column-major: axis1 fastest
    @printf("Kronecker check D1 axis 2: max|diff| = %.2e\n", maximum(abs, D1_3 - K))
    D2_3 = sparse(second_derivative(g_minus, set, 1))
    K2 = kron(sparse(1.0I, n23 * n23, n23 * n23), ops_m.d2[1])
    @printf("Kronecker check D2 axis 1: max|diff| = %.2e\n", maximum(abs, D2_3 - K2))
end

# --- CPU reference of the axis-pass application (Array), for correctness
# apply 1D op M along axis d of 3D array u, accumulate α*result into out
function axpass!(out, M::SparseMatrixCSC, u::Array{Float64,3}, d, α)
    n = size(u)
    Mt = sparse(transpose(M))  # rows of M as columns of Mt for CSC access
    rows = rowvals(Mt); vals = nonzeros(Mt)
    if d == 1
        @inbounds for k in 1:n[3], j in 1:n[2], i in 1:n[1]
            s = 0.0
            for idx in nzrange(Mt, i); s += vals[idx] * u[rows[idx], j, k]; end
            out[i, j, k] += α * s
        end
    elseif d == 2
        @inbounds for k in 1:n[3], j in 1:n[2]
            for idx in nzrange(Mt, j)
                c = vals[idx] * α; jj = rows[idx]
                for i in 1:n[1]; out[i, j, k] += c * u[i, jj, k]; end
            end
        end
    else
        @inbounds for k in 1:n[3]
            for idx in nzrange(Mt, k)
                c = vals[idx] * α; kk = rows[idx]
                for j in 1:n[2], i in 1:n[1]; out[i, j, k] += c * u[i, j, kk]; end
            end
        end
    end
    return out
end

# Navier operator on one side: out_j = μ Lap u_j + μ D2_j u_j + (λ+μ) D1_j div − μ D1_j w_j,  w_k = D1_k u_k
function navier!(out::NTuple{3}, u::NTuple{3}, ops::Ops1D, λ, μ, axpass!)
    w = ntuple(k -> axpass!(zero(u[k]), ops.d1[k], u[k], k, 1.0), 3)
    div = w[1] .+ w[2] .+ w[3]
    for j in 1:3
        fill!(out[j], 0.0)
        for i in 1:3; axpass!(out[j], ops.d2[i], u[j], i, μ); end
        axpass!(out[j], ops.d2[j], u[j], j, μ)
        tmp = (λ + μ) .* div .- μ .* w[j]
        axpass!(out[j], ops.d1[j], tmp, j, 1.0)
    end
    return out
end

sz_m = size(g_minus); sz_p = size(g_plus)
function split_fields(v, sz_m, sz_p)
    Nm = prod(sz_m); Np = prod(sz_p)
    um = ntuple(c -> reshape(view(v, (c-1)*Nm+1:c*Nm), sz_m), 3)
    up = ntuple(c -> reshape(view(v, 3Nm + (c-1)*Np+1:3Nm + c*Np), sz_p), 3)
    return um, up
end

# Dmat check (CPU)
Random.seed!(1)
v = randn(Ntot)
ref = Dmat * v
um, up = split_fields(v, sz_m, sz_p)
om = ntuple(_ -> zeros(sz_m), 3); op = ntuple(_ -> zeros(sz_p), 3)
navier!(om, map(collect, um), ops_m, λ, μ, axpass!)
navier!(op, map(collect, up), ops_p, λ, μ, axpass!)
mf = vcat(vec.(om)..., vec.(op)...)
@printf("\nmatrix-free D vs assembled Dmat: rel err = %.2e\n", norm(mf - ref) / norm(ref))

# Full A: -H P (D + SAT) P v
Hd = diag(H)
function A_free(v)
    pv = P * v
    um, up = split_fields(pv, sz_m, sz_p)
    om = ntuple(_ -> zeros(sz_m), 3); op = ntuple(_ -> zeros(sz_p), 3)
    navier!(om, map(collect, um), ops_m, λ, μ, axpass!)
    navier!(op, map(collect, up), ops_p, λ, μ, axpass!)
    dsat = vcat(vec.(om)..., vec.(op)...) .+ SATmat * pv
    return -(Hd .* (P * dsat))
end
@printf("matrix-free A vs assembled A:    rel err = %.2e\n", norm(A_free(v) - A * v) / norm(A * v))

# ---------------------------------------------------------------- 3. GPU
if use_gpu
    println("\n--- GPU: ", CUDA.name(CUDA.device()))

    # 1D op as CSR arrays on device
    struct Op1Dgpu
        rowptr::CuVector{Int32}; colind::CuVector{Int32}; val::CuVector{Float64}
    end
    function Op1Dgpu(M::SparseMatrixCSC)
        Mt = sparse(transpose(M))
        Op1Dgpu(CuVector{Int32}(Mt.colptr), CuVector{Int32}(Mt.rowval), CuVector{Float64}(Mt.nzval))
    end

    function axpass_kernel!(out, u, rowptr, colind, val, α, ::Val{d}) where {d}
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
        k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
        n1, n2, n3 = size(u)
        if i <= n1 && j <= n2 && k <= n3
            r = d == 1 ? i : (d == 2 ? j : k)
            s = 0.0
            @inbounds for idx in rowptr[r]:(rowptr[r+1]-1)
                c = colind[idx]
                s += val[idx] * (d == 1 ? u[c, j, k] : (d == 2 ? u[i, c, k] : u[i, j, c]))
            end
            @inbounds out[i, j, k] += α * s
        end
        return nothing
    end
    function axpass_gpu!(out::CuArray{Float64,3}, M::Op1Dgpu, u::CuArray{Float64,3}, d, α)
        n = size(u)
        threads = (32, 4, 2)
        blocks = cld.(n, threads)
        @cuda threads=threads blocks=blocks axpass_kernel!(out, u, M.rowptr, M.colind, M.val, α, Val(d))
        return out
    end
    struct OpsGPU
        d1::Vector{Op1Dgpu}; d2::Vector{Op1Dgpu}
    end
    OpsGPU(o::Ops1D) = OpsGPU(Op1Dgpu.(o.d1), Op1Dgpu.(o.d2))
    gops_m = OpsGPU(ops_m); gops_p = OpsGPU(ops_p)

    # preallocated workspace for one side
    struct SideWS
        w::NTuple{3,CuArray{Float64,3}}; div::CuArray{Float64,3}; tmp::CuArray{Float64,3}
    end
    SideWS(sz) = SideWS(ntuple(_ -> CUDA.zeros(Float64, sz), 3), CUDA.zeros(Float64, sz), CUDA.zeros(Float64, sz))
    function navier_gpu!(out::NTuple{3}, u::NTuple{3}, ops::OpsGPU, ws::SideWS, λ, μ)
        for k in 1:3
            fill!(ws.w[k], 0.0); axpass_gpu!(ws.w[k], ops.d1[k], u[k], k, 1.0)
        end
        ws.div .= ws.w[1] .+ ws.w[2] .+ ws.w[3]
        for j in 1:3
            fill!(out[j], 0.0)
            for i in 1:3; axpass_gpu!(out[j], ops.d2[i], u[j], i, μ); end
            axpass_gpu!(out[j], ops.d2[j], u[j], j, μ)
            ws.tmp .= (λ + μ) .* ws.div .- μ .* ws.w[j]
            axpass_gpu!(out[j], ops.d1[j], ws.tmp, j, 1.0)
        end
        return out
    end

    # matrix-free A on GPU
    struct AFree
        P::CuSparseMatrixCSR{Float64}
        SAT::CuSparseMatrixCSR{Float64}
        Hd::CuVector{Float64}
        gops_m::OpsGPU; gops_p::OpsGPU
        wsm::SideWS; wsp::SideWS
        pv::CuVector{Float64}; dsat::CuVector{Float64}
        sz_m::NTuple{3,Int}; sz_p::NTuple{3,Int}
        λ::Float64; μ::Float64
    end
    function gpu_fields(v::CuVector, sz_m, sz_p)
        Nm = prod(sz_m); Np = prod(sz_p)
        um = ntuple(c -> reshape(view(v, (c-1)*Nm+1:c*Nm), sz_m), 3)
        up = ntuple(c -> reshape(view(v, 3Nm + (c-1)*Np+1:3Nm + c*Np), sz_p), 3)
        return um, up
    end
    function LinearAlgebra.mul!(y::CuVector{Float64}, Af::AFree, v::CuVector{Float64})
        mul!(Af.pv, Af.P, v)
        um, up = gpu_fields(Af.pv, Af.sz_m, Af.sz_p)
        om, op = gpu_fields(Af.dsat, Af.sz_m, Af.sz_p)
        navier_gpu!(om, um, Af.gops_m, Af.wsm, Af.λ, Af.μ)
        navier_gpu!(op, up, Af.gops_p, Af.wsp, Af.λ, Af.μ)
        mul!(Af.dsat, Af.SAT, Af.pv, 1.0, 1.0)
        mul!(y, Af.P, Af.dsat)
        y .= .-Af.Hd .* y
        return y
    end
    Base.size(Af::AFree) = (length(Af.Hd), length(Af.Hd))
    Base.size(Af::AFree, i) = length(Af.Hd)
    Base.eltype(::AFree) = Float64
    LinearAlgebra.issymmetric(::AFree) = true

    to_csr32(M) = Base.get_extension(EarthquakeDiffinitive, :EarthquakeDiffinitiveCUDAExt).to_csr(M)
    Af = AFree(to_csr32(P), to_csr32(SATmat), CuVector(Hd), gops_m, gops_p, SideWS(sz_m), SideWS(sz_p),
               CUDA.zeros(Float64, Ntot), CUDA.zeros(Float64, Ntot), sz_m, sz_p, λ, μ)
    vg = CuVector(v); yg = similar(vg)
    mul!(yg, Af, vg)
    @printf("GPU matrix-free A vs assembled A: rel err = %.2e\n", norm(Array(yg) - A * v) / norm(A * v))

    A_gpu = to_csr32(A)
    yg2 = similar(vg)
    mul!(yg2, A_gpu, vg)
    @printf("GPU CSR A vs assembled A:         rel err = %.2e\n", norm(Array(yg2) - A * v) / norm(A * v))

    # timing
    CUDA.@sync mul!(yg, Af, vg); CUDA.@sync mul!(yg2, A_gpu, vg)
    nrep = 20
    t_mf = CUDA.@elapsed for _ in 1:nrep; mul!(yg, Af, vg); end
    t_csr = CUDA.@elapsed for _ in 1:nrep; mul!(yg2, A_gpu, vg); end
    t_mf /= nrep; t_csr /= nrep
    bytes_csr = nnz(A) * 12 + Ntot * 8 * 2
    @printf("\nper mat-vec:  CSR SpMV %.2f ms (%.0f GB/s effective on %.2f GB of matrix)\n",
            1e3t_csr, bytes_csr / t_csr / 1e9, nnz(A) * 12 / 1e9)
    @printf("              matrix-free %.2f ms   -> %.2fx\n", 1e3t_mf, t_csr / t_mf)
    @printf("device memory: CSR A = %.2f GB;  matrix-free (P + SAT + vectors) = %.3f GB\n",
            nnz(A) * 12 / 1e9, (nnz(P) * 12 + nnz(SATmat) * 12 + 8Ntot * 2 + 8 * 5 * (prod(sz_m) + prod(sz_p))) / 1e9)

    # ---- fused kernel variant
    include(joinpath(@__DIR__, "matrix_free_prototype_fused.jl"))
    Afu = AFused(Af.P, Af.SAT, Af.Hd, FusedSide(ops_m), FusedSide(ops_p),
                 CUDA.zeros(Float64, Ntot), CUDA.zeros(Float64, Ntot), sz_m, sz_p, λ, μ)
    yg3 = similar(vg)
    CUDA.@sync mul!(yg3, Afu, vg)
    @printf("GPU FUSED matrix-free A vs assembled A: rel err = %.2e\n", norm(Array(yg3) - A * v) / norm(A * v))
    t_fu = CUDA.@elapsed for _ in 1:nrep; mul!(yg3, Afu, vg); end
    t_fu /= nrep
    @printf("              fused matrix-free %.2f ms   -> %.2fx vs CSR, %.2fx vs axis-pass\n", 1e3t_fu, t_csr / t_fu, t_mf / t_fu)
    # how much of the fused mat-vec is the P / SAT / H bookkeeping vs the Navier kernels?
    t_nav = CUDA.@elapsed for _ in 1:nrep
        um_, up_ = gpu_fields(Afu.pv, sz_m, sz_p); om_, op_ = gpu_fields(Afu.dsat, sz_m, sz_p)
        navier_fused!(om_, um_, Afu.fm, λ, μ); navier_fused!(op_, up_, Afu.fp, λ, μ)
    end
    t_nav /= nrep
    t_P = CUDA.@elapsed for _ in 1:nrep; mul!(Afu.pv, Afu.P, vg); end
    t_P /= nrep
    @printf("              (fused Navier kernels alone %.2f ms; one P SpMV %.2f ms; SAT+P+H rest %.2f ms)\n",
            1e3t_nav, 1e3t_P, 1e3(t_fu - t_nav))

    # a CG solve on a real RHS with both operators
    fe_like_chi = zeros(Ntot)
    pairs = fault_node_pairs(g_minus, g_plus)
    Im, Ip = pairs[length(pairs) ÷ 2]
    fe_like_chi[dof_index_plus(g_minus, g_plus, 2, Ip)] = 0.5
    fe_like_chi[dof_index_minus(g_minus, 2, Im)] = -0.5
    rhs = CuVector(HP_DSAT * fe_like_chi)
    ws1 = CgWorkspace(Ntot, Ntot, CuVector{Float64}); ws2 = CgWorkspace(Ntot, Ntot, CuVector{Float64}); ws3 = CgWorkspace(Ntot, Ntot, CuVector{Float64})
    CUDA.@sync cg!(ws1, A_gpu, rhs; rtol=1e-8, itmax=20000)
    CUDA.@sync cg!(ws2, Af, rhs; rtol=1e-8, itmax=20000)
    CUDA.@sync cg!(ws3, Afu, rhs; rtol=1e-8, itmax=20000)
    t1 = CUDA.@elapsed cg!(ws1, A_gpu, rhs; rtol=1e-8, itmax=20000)
    t2 = CUDA.@elapsed cg!(ws2, Af, rhs; rtol=1e-8, itmax=20000)
    t3 = CUDA.@elapsed cg!(ws3, Afu, rhs; rtol=1e-8, itmax=20000)
    @printf("\nCG (rtol 1e-8): CSR %d iters %.2f s | axis-pass %d iters %.2f s | fused %d iters %.2f s | rel diff of solutions %.2e / %.2e\n",
            ws1.stats.niter, t1, ws2.stats.niter, t2, ws3.stats.niter, t3, norm(ws1.x - ws2.x) / norm(ws1.x), norm(ws1.x - ws3.x) / norm(ws1.x))
end

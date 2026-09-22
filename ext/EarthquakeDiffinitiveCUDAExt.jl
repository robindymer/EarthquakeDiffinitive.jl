module EarthquakeDiffinitiveCUDAExt

# Adds `EarthquakeDiffinitive.FaultResponse.fault_stiffness_gpu` — see that
# function's docstring for the design and the measured production timings.

using EarthquakeDiffinitive
using EarthquakeDiffinitive.FaultResponse: FaultElasticity, frictional_node_count,
                                           d4_setup, build_chi!
using EarthquakeDiffinitive.ElasticitySplitNode: SplitNodeOperator, AssembledSplitNode,
                                                 Rows1D, SideOps, apply_P!, hp_dsat!
import EarthquakeDiffinitive.ElasticitySplitNode: axpass!, scale_H!
using CUDA
using CUDA.CUSPARSE: CuSparseMatrixCSR
using Krylov: CgWorkspace, cg!
using SparseArrays
using LinearAlgebra: mul!

"""
    is_symmetric(M; rtol=1e-12) -> Bool

Whether `M` equals its own transpose in pattern *and* value, to a relative
tolerance, in `O(nnz)` time and `O(n)` extra memory.

Not `M == transpose(M)`: materialising the transpose costs another full copy,
tens of GB of host RAM at production size — the thing [`to_csr`](@ref) exists
to avoid.

**Values are checked, not just the pattern**, since structural symmetry alone
does not license `to_csr`'s reinterpretation. `P` is structurally symmetric, so
a pattern-only test would wave it through and a numerically asymmetric `P`
would reach the device as `Pᵀ`.

`rtol` is scaled by the largest magnitude in `M`, putting the test on the same
footing as the `‖A-Aᵀ‖/‖A‖ ≈ 6e-17` quoted for `A` — exact equality would
reject `A` itself, whose triangles differ in the last bits from assembly
summation order.

Relies on CSC `rowval` being sorted within a column, which `SparseMatrixCSC`
guarantees: columns are visited in increasing `j`, so one monotonic cursor per
column suffices.

**Every** off-diagonal entry is checked, not only `i > j`. Trusting the mirror
pair to check itself fails in exactly the case being tested for — if `(i, j)`
is present and `(j, i)` absent, the absent entry is never a loop iteration.
"""
function is_symmetric(M::SparseMatrixCSC; rtol=1e-12)
    n = size(M, 2)
    size(M, 1) == n || return false
    cp, rv, nz = SparseArrays.getcolptr(M), rowvals(M), nonzeros(M)
    cursor = cp[1:n]
    scale = isempty(nz) ? 0.0 : maximum(abs, nz)
    tol = rtol * scale
    @inbounds for j in 1:n, k in cp[j]:(cp[j+1]-1)
        i = rv[k]
        i == j && continue
        p = cursor[i]
        stop = cp[i+1]
        while p < stop && rv[p] < j
            p += 1
        end
        cursor[i] = p
        (p < stop && rv[p] == j) || return false
        abs(nz[k] - nz[p]) <= tol || return false
    end
    return true
end

"""
    to_csr(M) -> CuSparseMatrixCSR

Upload a host CSC matrix to the device as CSR without holding two copies in
VRAM. Only used by `representation=:assembled`; the default matrix-free path
uploads no `A` at all.

**Not just `CuSparseMatrixCSR(M)`**, which is
`CuSparseMatrixCSR(CuSparseMatrixCSC{T}(M))`: it uploads the CSC and converts
on the device, so both representations plus cuSPARSE scratch are resident at
once and peak VRAM is over twice the final matrix. At Δz = 10 m on
(1600, 1200) that is ~122 GB against a ~57 GB matrix — the difference between
fitting a 94 GB H100 NVL and not.

For a **structurally symmetric** matrix no conversion is needed: reading CSC
`(colptr, rowval, nzval)` as CSR `(rowPtr, colVal, nzVal)` gives
`transpose(M)`, which for `M == transpose(M)` is `M`. `A = -HP(D+SAT)P` is
symmetric by construction (`SYMMETRIC_SAT.md`, measured 6e-17), which CG
already assumes.

Symmetry is **checked**, not assumed, by [`is_symmetric`](@ref). `A` and `P`
pass; `T2`/`T3` are not square and take the generic converting path, at a few
percent of `A`'s size.

**Index width comes from `nnz`, not a fixed `Int32`.** `Int32` saves 4 bytes
per nonzero (~14 GB on the converged Δz = 10 m `A`), but the CSR row pointer
must hold `nnz`, and past `typemax(Int32)` ≈ 2.15e9 the 32-bit form silently
overflows. The two production domains sit on opposite sides of that boundary
(~1.99e9 and ~3.55e9), so it is checked rather than assumed.
"""
function to_csr(M::SparseMatrixCSC{Tv}) where {Tv}
    Ti = nnz(M) <= typemax(Int32) - 1 ? Int32 : Int64
    if is_symmetric(M)
        return CuSparseMatrixCSR{Tv,Ti}(CuVector{Ti}(SparseArrays.getcolptr(M)),
                                        CuVector{Ti}(rowvals(M)),
                                        CuVector{Tv}(nonzeros(M)), size(M))
    end
    return CuSparseMatrixCSR(M)
end


# The matrix-free operator on the device.
#
# `SplitNodeOperator`'s `mul!`/`hp_dsat!`/`apply_P!` are written against flat
# vectors with offsets and plain broadcasting, so the same code runs on
# `CuVector` once the two hot loops have device methods: `axpass!` (one 1D
# operator along one axis) and `scale_H!` (the Kronecker weights), below.
# Everything else — `P` as a masked broadcast plus gather/scatter, the SAT
# SpMV, the element-wise combinations — uses CUDA.jl's generic paths.
#
# Thread mapping is (i, j, k) with the contiguous axis `i` on `threadIdx().x`,
# so reads are coalesced whichever axis the pass runs along. Indices are
# `Int32`.

to_device(r::Rows1D) = Rows1D(r.n, CuVector{Int32}(r.rowptr), CuVector{Int32}(r.colind),
                              CuVector{Float64}(r.val))
to_device(s::SideOps) = SideOps(s.n, map(to_device, s.d1), map(to_device, s.d2),
                                map(CuVector{Float64}, s.h))

"""
    to_device(op::SplitNodeOperator) -> SplitNodeOperator

The operator with every array on the GPU: 1D operators as `Int32` CSR, `P`
data, the SAT block via [`to_csr`](@ref), and fresh device scratch. Device
memory is the scratch (~9 field- or system-length vectors) plus SAT — ~9 GB at
Δz = 10 m on (1600, 1600), where the assembled `A` alone would be ~85 GB.
"""
function to_device(op::SplitNodeOperator)
    N = length(op.div)
    return SplitNodeOperator(to_device(op.minus), to_device(op.plus), op.λ, op.μ,
                             CuVector{Float64}(op.mask), CuVector{Int32}(op.pm), CuVector{Int32}(op.pp),
                             to_csr(op.SAT), op.Ntot,
                             CUDA.zeros(Float64, op.Ntot), CUDA.zeros(Float64, op.Ntot),
                             CUDA.zeros(Float64, length(op.pm)),
                             ntuple(_ -> CUDA.zeros(Float64, N), 3),
                             CUDA.zeros(Float64, N), CUDA.zeros(Float64, N))
end

function axpass_kernel!(out, off_out, u, off_u, rp, ci, val, n1, n2, n3, α, ::Val{d}) where {d}
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if i <= n1 && j <= n2 && k <= n3
        r = d == 1 ? i : (d == 2 ? j : k)
        s = 0.0
        @inbounds for idx in rp[r]:(rp[r+1]-1)
            c = ci[idx]
            p = d == 1 ? (off_u + c + (j-1)*n1 + (k-1)*n1*n2) :
                d == 2 ? (off_u + i + (c-1)*n1 + (k-1)*n1*n2) :
                         (off_u + i + (j-1)*n1 + (c-1)*n1*n2)
            s += val[idx] * u[p]
        end
        @inbounds out[off_out + i + (j-1)*n1 + (k-1)*n1*n2] += α * s
    end
    return nothing
end

const AXPASS_THREADS = (32, 4, 2)

function axpass!(out::CuVector{Float64}, off_out::Int, M::Rows1D{<:CuVector}, u::CuVector{Float64},
                 off_u::Int, n::NTuple{3,Int}, d::Int, α::Float64)
    n1, n2, n3 = n
    blocks = cld.(n, AXPASS_THREADS)
    vd = d == 1 ? Val(1) : (d == 2 ? Val(2) : Val(3))
    @cuda threads=AXPASS_THREADS blocks=blocks axpass_kernel!(
        out, off_out, u, off_u, M.rowptr, M.colind, M.val, n1, n2, n3, α, vd)
    return out
end

function scaleH_kernel!(y, off, h1, h2, h3, n1, n2, n3, α)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if i <= n1 && j <= n2 && k <= n3
        N = n1 * n2 * n3
        @inbounds begin
            c = α * h1[i] * h2[j] * h3[k]
            q = off + i + (j-1)*n1 + (k-1)*n1*n2
            y[q] *= c
            y[q + N] *= c
            y[q + 2N] *= c
        end
    end
    return nothing
end

function scale_H!(y::CuVector{Float64}, off::Int, side::SideOps{<:Any,<:CuVector}, α::Float64)
    n1, n2, n3 = side.n
    blocks = cld.(side.n, AXPASS_THREADS)
    @cuda threads=AXPASS_THREADS blocks=blocks scaleH_kernel!(
        y, off, side.h[1], side.h[2], side.h[3], n1, n2, n3, α)
    return y
end

# The three things `fault_stiffness_gpu` needs from either representation:
# the operator CG runs on, and the two auxiliary applications.
gpu_system(op::SplitNodeOperator) = to_device(op)
function gpu_system(w::AssembledSplitNode)
    # `HP_DSAT` stays on the host: the RHS is formed there (one SpMV per
    # representative, negligible next to the solve) so device memory is `A` + `P`,
    # which is what `build_stiffness_cache_gpu.jl`'s estimate assumes.
    return AssembledSplitNode(to_csr(w.A), w.HP_DSAT, to_csr(w.P))
end
gpu_rhs!(b::CuVector, op::SplitNodeOperator, χ_gpu::CuVector, χ::Vector) = hp_dsat!(b, op, χ_gpu)
gpu_rhs!(b::CuVector, w::AssembledSplitNode, χ_gpu::CuVector, χ::Vector) = copyto!(b, w.HP_DSAT * χ)

function EarthquakeDiffinitive.FaultResponse.fault_stiffness_gpu(fe::FaultElasticity; verbose=false,
                                                                 shard=nothing, nshards=nothing)
    rs = fe.rs
    (shard === nothing) == (nshards === nothing) ||
        error("fault_stiffness_gpu: pass both `shard` and `nshards`, or neither")
    sharded = shard !== nothing
    sharded && (1 <= shard <= nshards ||
        error("fault_stiffness_gpu: shard must be in 1:nshards, got shard=$shard nshards=$nshards"))
    rs.precond === :none ||
        error("fault_stiffness_gpu: only precond=:none is supported (the CPU " *
              "preconditioners are untested on GPU-resident arrays); rebuild " *
              "`fe` without a `precond` keyword")

    nf = frictional_node_count(fe)
    perms, Qs, reps, targets = d4_setup(fe, "fault_stiffness_gpu")
    ncols = 2nf
    Ntot = fe.Ntot

    verbose && @info "fault_stiffness_gpu: moving the elastic system, T2, T3 to the GPU" device = CUDA.name(CUDA.device()) representation = typeof(fe.op).name.name
    A_gpu = gpu_system(fe.op)
    T2_gpu = to_csr(fe.T2)
    T3_gpu = to_csr(fe.T3)

    # Sharding splits the D4 orbit representatives exactly as
    # `fault_stiffness_d4_shard` does and returns the same `(cols, Kshard)`.
    # Unsharded, `mycols` is the identity and the full `K` comes back.
    positions = sharded ? collect(shard:nshards:length(reps)) : collect(1:length(reps))
    mycols = sharded ? [tcol for pos in positions for (_, tcol) in targets[pos]] : collect(1:ncols)
    colpos = sharded ? Dict(c => i for (i, c) in enumerate(mycols)) : nothing
    K = zeros(ncols, length(mycols))
    ws = CgWorkspace(Ntot, Ntot, CuVector{Float64})
    s2, s3 = zeros(nf), zeros(nf)
    χ = zeros(Ntot)
    χ_gpu = CUDA.zeros(Float64, Ntot)
    rhs_gpu = CUDA.zeros(Float64, Ntot)
    U_gpu = CUDA.zeros(Float64, Ntot)
    t0 = time()

    for (done, pos) in enumerate(positions)
        col = reps[pos]
        node = col <= nf ? col : col - nf
        comp = col <= nf ? 1 : 2
        fill!(s2, 0.0)
        fill!(s3, 0.0)
        comp == 1 ? (s2[node] = 1.0) : (s3[node] = 1.0)
        build_chi!(χ, fe, s2, s3)
        copyto!(χ_gpu, χ)
        gpu_rhs!(rhs_gpu, A_gpu, χ_gpu, χ)

        cg!(ws, A_gpu, rhs_gpu; M=rs.M, ldiv=rs.ldiv, rtol=rs.rtol, atol=rs.atol, itmax=rs.itmax)
        st = ws.stats
        rs.stats.solves += 1
        rs.stats.iterations += st.niter
        rs.stats.max_iterations = max(rs.stats.max_iterations, st.niter)
        if !st.solved
            rs.stats.unconverged += 1
            rs.stats.unconverged == 1 && @warn """
                CG did not converge on a GPU split-node solve (status "$(st.status)") \
                after $(st.niter) iterations. The returned displacement is not a \
                solution, and fault_stiffness_gpu would fold it into K silently. \
                Raise itmax or loosen rtol.""" rtol = rs.rtol itmax = rs.itmax
        end

        apply_P!(U_gpu, A_gpu, ws.x)
        U_gpu .+= χ_gpu
        τ2 = Array(T2_gpu * U_gpu)[fe.omega]
        τ3 = Array(T3_gpu * U_gpu)[fe.omega]

        @inbounds for (g, tcol) in targets[pos]
            Q = Qs[g]
            perm = perms[g]
            σ = Q[1, comp] != 0 ? Q[1, comp] : Q[2, comp]
            out = sharded ? colpos[tcol] : tcol
            for i in 1:nf
                v1 = Q[1, 1] * τ2[i] + Q[1, 2] * τ3[i]
                v2 = Q[2, 1] * τ2[i] + Q[2, 2] * τ3[i]
                ti = perm[i]
                K[ti, out] = σ * v1
                K[nf+ti, out] = σ * v2
            end
        end

        if verbose && (done % 20 == 0 || done == length(positions))
            el = time() - t0
            @info "fault_stiffness_gpu: representative $done/$(length(positions))" elapsed = round(el, digits=1) eta = round(el * (length(positions) - done) / done, digits=1)
        end
    end

    verbose && @info "fault_stiffness_gpu: done" seconds = round(time() - t0, digits=1) representatives = length(positions) columns = length(mycols) shard nshards
    return sharded ? (mycols, K) : K
end

end # module EarthquakeDiffinitiveCUDAExt

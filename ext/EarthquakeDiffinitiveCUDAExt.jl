module EarthquakeDiffinitiveCUDAExt

# Adds `EarthquakeDiffinitive.FaultResponse.fault_stiffness_gpu` — see that
# function's docstring for the design (D4 symmetry, no sharding, sequential
# solves) and its validation status (measured only at small scale on
# consumer hardware; not yet measured at production scale or on
# Hopper/Ada-class GPUs).

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

Whether `M` equals its own transpose — in pattern *and* in value, to a
relative tolerance — in `O(nnz)` time and `O(n)` extra memory.

Deliberately **not** `M == transpose(M)` or a comparison against
`sparse(transpose(M))`: materialising the transpose costs another full copy of
the matrix, which at production size is tens of gigabytes of host RAM — the
very thing [`to_csr`](@ref) exists to avoid spending on the device.

**Values are checked, not just the pattern.** Structural symmetry alone is not
enough to license `to_csr`'s reinterpretation, and this is not hypothetical
here: `P` is structurally symmetric, so a pattern-only test would wave it
through, and had it been numerically asymmetric the device would have silently
received `Pᵀ`. It happens to be exactly symmetric (measured 0.0), but that is a
fact about `P`, not a property the check may assume.

`rtol` is scaled by the largest magnitude in `M`, so the test is on the same
footing as the `‖A-Aᵀ‖/‖A‖ ≈ 6e-17` figure quoted for `A`: exact equality would
reject `A` itself, whose two triangles differ in the last bit or two from
floating-point summation order during assembly.

Relies on CSC `rowval` being sorted within each column, which
`SparseMatrixCSC` guarantees. Columns are visited in increasing `j`, so for a
fixed partner column `i` the row being sought only ever increases, and one
monotonically advancing cursor per column suffices.

**Every** off-diagonal entry is checked, not just those with `i > j`. Halving
the work by trusting the mirror pair to check itself is wrong precisely in the
case being tested for: if `(i, j)` is present and `(j, i)` is absent, the
absent entry never comes up as a loop iteration, so the check has to be driven
from the present one whichever side of the diagonal it lies on.
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

Upload a host CSC matrix to the device as CSR, without ever holding two
copies of it in VRAM.

**Why this is not just `CuSparseMatrixCSR(M)`.** That constructor is
`CuSparseMatrixCSR(CuSparseMatrixCSC{T}(M))` — it uploads the CSC, then runs
`cusparseCsr2cscEx2` on the device, so both representations *and* cuSPARSE's
scratch buffer are resident at once. Peak VRAM is a bit over twice the final
matrix, and `A` dominates everything else here: at Δz = 10 m on the converged
(1600, 1200) domain it is ~57 GB, so the naive path peaks at ~122 GB and
cannot fit the 94 GB H100 NVL, while the converted matrix leaves ~29 GB spare.
This is the difference between that configuration running on one card and not
running at all.

For a **structurally symmetric** matrix no conversion is needed: reading CSC
arrays `(colptr, rowval, nzval)` as CSR `(rowPtr, colVal, nzVal)` gives
`transpose(M)` in CSR, and for `M == transpose(M)` that is `M` itself. `A`
here is `-HP(D+SAT)P`, symmetric by construction (`SYMMETRIC_SAT.md`;
measured at `‖A-Aᵀ‖/‖A‖ ≈ 6e-17`) — and CG already assumes exactly that, so
this introduces no assumption the solve was not already making.

The symmetry is **checked, not assumed**, by [`is_symmetric`](@ref) — pattern
and values both. `A` and `P` pass; `T2`/`T3` are not even square and take the
generic converting path. They are a few percent of `A`'s size, so doubling
their peak costs nothing that matters.

**Index width is chosen from `nnz`, not fixed at `Int32`.** cuSPARSE supports
both (`CUSPARSE_INDEX_32I`/`64I`) and CUDA.jl selects on `Ti`, so `Int32`
saves 4 bytes per nonzero — ~14 GB on the converged Δz = 10 m `A`. But that
`A` has ~3.55e9 nonzeros, and the CSR row pointer must be able to *hold*
`nnz`: past `typemax(Int32)` ≈ 2.15e9 the 32-bit form silently overflows. The
relaxed (1200, 1200) domain sits at ~1.99e9 — under, but by only ~8% — so the
two production domains land on opposite sides of this boundary and the margin
on the near side is thin. Hence a check rather than a constant.
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


# ==============================================================================
# The matrix-free operator on the device.
#
# `SplitNodeOperator`'s `mul!`/`hp_dsat!`/`apply_P!` are written against flat
# vectors with offsets and plain broadcasting, so the same code runs on
# `CuVector` once the two hot loops — `axpass!` (one 1D operator along one
# axis) and `scale_H!` (the Kronecker inner-product weights) — have device
# methods. Those are the two kernels below; everything else (`P` as a masked
# broadcast with a gather/scatter of the fault pairs, the SAT SpMV, the
# element-wise field combinations) goes through CUDA.jl's generic paths.
#
# Thread mapping is (i, j, k) with `i` — the contiguous axis — across
# `threadIdx().x`, so every read `u[base + i]` or `u[base + ci[idx]]` is
# coalesced whichever axis the pass runs along. Index vectors are `Int32`.
# ==============================================================================

to_device(r::Rows1D) = Rows1D(r.n, CuVector{Int32}(r.rowptr), CuVector{Int32}(r.colind),
                              CuVector{Float64}(r.val))
to_device(s::SideOps) = SideOps(s.n, map(to_device, s.d1), map(to_device, s.d2),
                                map(CuVector{Float64}, s.h))

"""
    to_device(op::SplitNodeOperator) -> SplitNodeOperator

A copy of the operator with every array resident on the GPU: 1D operators as
`Int32` CSR, `P` data, the SAT block as `CuSparseMatrixCSR` (via `to_csr`),
and fresh device scratch. Total device memory is the scratch — ~9 vectors of
field or system length — plus SAT: about 9 GB at Δz = 10 m on (1600, 1600),
where the assembled `A` alone would be ~85 GB.
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
    # `HP_DSAT` is not uploaded: the RHS is formed on the host from it (one SpMV
    # per representative, negligible next to the solve) to keep device memory
    # to `A` + `P`, which is what the pre-flight estimate in
    # `build_stiffness_cache_gpu.jl` assumed for this representation.
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
    # `fault_stiffness_d4_shard` does, and returns the same `(cols, Kshard)` so
    # `merge_stiffness_shards` needs no format change. Unsharded, `mycols` is
    # the identity and the full `K` comes back.
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

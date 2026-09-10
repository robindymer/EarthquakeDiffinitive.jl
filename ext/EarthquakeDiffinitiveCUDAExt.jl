module EarthquakeDiffinitiveCUDAExt

# Adds `EarthquakeDiffinitive.FaultResponse.fault_stiffness_gpu` — see that
# function's docstring for the design (D4 symmetry, no sharding, sequential
# solves) and its validation status (measured only at small scale on
# consumer hardware; not yet measured at production scale or on
# Hopper/Ada-class GPUs).

using EarthquakeDiffinitive
using EarthquakeDiffinitive.FaultResponse: FaultElasticity, frictional_node_count,
                                           d4_setup, build_chi!
using CUDA
using CUDA.CUSPARSE: CuSparseMatrixCSR
using Krylov: CgWorkspace, cg!
using SparseArrays

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

function EarthquakeDiffinitive.FaultResponse.fault_stiffness_gpu(fe::FaultElasticity; verbose=false)
    rs = fe.rs
    rs.precond === :none ||
        error("fault_stiffness_gpu: only precond=:none is supported (the CPU " *
              "preconditioners are untested on GPU-resident arrays); rebuild " *
              "`fe` without a `precond` keyword")

    nf = frictional_node_count(fe)
    perms, Qs, reps, targets = d4_setup(fe, "fault_stiffness_gpu")
    ncols = 2nf
    Ntot = fe.Ntot

    verbose && @info "fault_stiffness_gpu: moving A, P, T2, T3 to the GPU" device = CUDA.name(CUDA.device())
    A_gpu = to_csr(rs.A)
    P_gpu = to_csr(fe.P)
    T2_gpu = to_csr(fe.T2)
    T3_gpu = to_csr(fe.T3)

    K = zeros(ncols, ncols)
    ws = CgWorkspace(Ntot, Ntot, CuVector{Float64})
    s2, s3 = zeros(nf), zeros(nf)
    χ = zeros(Ntot)
    t0 = time()

    for (pos, col) in enumerate(reps)
        node = col <= nf ? col : col - nf
        comp = col <= nf ? 1 : 2
        fill!(s2, 0.0)
        fill!(s3, 0.0)
        comp == 1 ? (s2[node] = 1.0) : (s3[node] = 1.0)
        build_chi!(χ, fe, s2, s3)
        rhs_gpu = CuVector(fe.HP_DSAT * χ)

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

        U_gpu = P_gpu * ws.x .+ CuVector(χ)
        τ2 = Array(T2_gpu * U_gpu)[fe.omega]
        τ3 = Array(T3_gpu * U_gpu)[fe.omega]

        @inbounds for (g, tcol) in targets[pos]
            Q = Qs[g]
            perm = perms[g]
            σ = Q[1, comp] != 0 ? Q[1, comp] : Q[2, comp]
            for i in 1:nf
                v1 = Q[1, 1] * τ2[i] + Q[1, 2] * τ3[i]
                v2 = Q[2, 1] * τ2[i] + Q[2, 2] * τ3[i]
                ti = perm[i]
                K[ti, tcol] = σ * v1
                K[nf+ti, tcol] = σ * v2
            end
        end

        if verbose && (pos % 20 == 0 || pos == length(reps))
            el = time() - t0
            @info "fault_stiffness_gpu: representative $pos/$(length(reps))" elapsed = round(el, digits=1) eta = round(el * (length(reps) - pos) / pos, digits=1)
        end
    end

    verbose && @info "fault_stiffness_gpu: done" seconds = round(time() - t0, digits=1) representatives = length(reps) columns = ncols
    return K
end

end # module EarthquakeDiffinitiveCUDAExt

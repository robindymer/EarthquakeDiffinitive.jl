# Fused matrix-free Navier kernel: one kernel per (side, output component); every thread
# evaluates its full stencil from the three input component fields, no intermediate fields.
# Included by matrix_free_prototype.jl when run with "gpu". Depends on names defined there.

# per-axis 1D CSR tables (rows of the 1D operator)
struct Axis1D{V<:AbstractVector{Int32},W<:AbstractVector{Float64}}
    rp::V; ci::V; v::W
end
Axis1D(M::SparseMatrixCSC) = (Mt = sparse(transpose(M));
    Axis1D(CuVector{Int32}(Mt.colptr), CuVector{Int32}(Mt.rowval), CuVector{Float64}(Mt.nzval)))
const Adapt = Base.loaded_modules[Base.PkgId(Base.UUID("79e6a3ab-5dfb-504d-930d-738a2a938a0e"), "Adapt")]; const adapt = Adapt.adapt
Adapt.adapt_structure(to, x::Axis1D) = Axis1D(adapt(to, x.rp), adapt(to, x.ci), adapt(to, x.v))

@inline function getax(u, a, b, c, ::Val{ax}, p) where {ax}
    ax == 1 ? u[p, b, c] : (ax == 2 ? u[a, p, c] : u[a, b, p])
end
@inline axidx(a, b, c, ::Val{ax}) where {ax} = ax == 1 ? a : (ax == 2 ? b : c)

# Σ_p M[row,p] u[... p along ax ...]
@inline function apply1(u, a, b, c, ax::Val, rp, ci, v)
    r = axidx(a, b, c, ax)
    s = 0.0
    @inbounds for idx in rp[r]:(rp[r+1]-1)
        s += v[idx] * getax(u, a, b, c, ax, ci[idx])
    end
    return s
end
# Σ_p Σ_q Mj[rj,p] Mk[rk,q] u[.. p along axj, q along axk ..]   (axj != axk)
@inline function apply2(u, a, b, c, axj::Val{J}, axk::Val{K}, rpj, cij, vj, rpk, cik, vk) where {J,K}
    rj = axidx(a, b, c, axj)
    s = 0.0
    @inbounds for idx in rpj[rj]:(rpj[rj+1]-1)
        p = cij[idx]
        a2 = J == 1 ? p : a; b2 = J == 2 ? p : b; c2 = J == 3 ? p : c
        s += vj[idx] * apply1(u, a2, b2, c2, axk, rpk, cik, vk)
    end
    return s
end

function fused_kernel!(out, u1, u2, u3, d1, d2, λ, μ, ::Val{J}) where {J}
    a = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    b = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    c = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    n1, n2, n3 = size(out)
    if a <= n1 && b <= n2 && c <= n3
        uj = J == 1 ? u1 : (J == 2 ? u2 : u3)
        # μ Lap u_j + μ D2_j u_j
        s = μ * (apply1(uj, a, b, c, Val(1), d2[1].rp, d2[1].ci, d2[1].v) +
                 apply1(uj, a, b, c, Val(2), d2[2].rp, d2[2].ci, d2[2].v) +
                 apply1(uj, a, b, c, Val(3), d2[3].rp, d2[3].ci, d2[3].v))
        s += μ * apply1(uj, a, b, c, Val(J), d2[J].rp, d2[J].ci, d2[J].v)
        # λ D1_j D1_j u_j  (same axis: nested apply1 along J twice)
        vj = Val(J)
        rj = axidx(a, b, c, vj)
        t = 0.0
        @inbounds for idx in d1[J].rp[rj]:(d1[J].rp[rj+1]-1)
            p = d1[J].ci[idx]
            a2 = J == 1 ? p : a; b2 = J == 2 ? p : b; c2 = J == 3 ? p : c
            t += d1[J].v[idx] * apply1(uj, a2, b2, c2, vj, d1[J].rp, d1[J].ci, d1[J].v)
        end
        s += λ * t
        # (λ+μ) Σ_{k≠j} D1_j D1_k u_k
        if J != 1
            s += (λ + μ) * apply2(u1, a, b, c, vj, Val(1), d1[J].rp, d1[J].ci, d1[J].v, d1[1].rp, d1[1].ci, d1[1].v)
        end
        if J != 2
            s += (λ + μ) * apply2(u2, a, b, c, vj, Val(2), d1[J].rp, d1[J].ci, d1[J].v, d1[2].rp, d1[2].ci, d1[2].v)
        end
        if J != 3
            s += (λ + μ) * apply2(u3, a, b, c, vj, Val(3), d1[J].rp, d1[J].ci, d1[J].v, d1[3].rp, d1[3].ci, d1[3].v)
        end
        @inbounds out[a, b, c] = s
    end
    return nothing
end

struct FusedSide
    d1::NTuple{3,Axis1D}; d2::NTuple{3,Axis1D}
end
FusedSide(o::Ops1D) = FusedSide(ntuple(d -> Axis1D(o.d1[d]), 3), ntuple(d -> Axis1D(o.d2[d]), 3))

function navier_fused!(out::NTuple{3}, u::NTuple{3}, fs::FusedSide, λ, μ)
    n = size(out[1])
    threads = (32, 4, 2)
    blocks = cld.(n, threads)
    for J in 1:3
        @cuda threads=threads blocks=blocks fused_kernel!(out[J], u[1], u[2], u[3], fs.d1, fs.d2, λ, μ, Val(J))
    end
    return out
end

struct AFused
    P::CuSparseMatrixCSR{Float64}
    SAT::CuSparseMatrixCSR{Float64}
    Hd::CuVector{Float64}
    fm::FusedSide; fp::FusedSide
    pv::CuVector{Float64}; dsat::CuVector{Float64}
    sz_m::NTuple{3,Int}; sz_p::NTuple{3,Int}
    λ::Float64; μ::Float64
end
function LinearAlgebra.mul!(y::CuVector{Float64}, Af::AFused, v::CuVector{Float64})
    mul!(Af.pv, Af.P, v)
    um, up = gpu_fields(Af.pv, Af.sz_m, Af.sz_p)
    om, op = gpu_fields(Af.dsat, Af.sz_m, Af.sz_p)
    navier_fused!(om, um, Af.fm, Af.λ, Af.μ)
    navier_fused!(op, up, Af.fp, Af.λ, Af.μ)
    mul!(Af.dsat, Af.SAT, Af.pv, 1.0, 1.0)
    mul!(y, Af.P, Af.dsat)
    y .= .-Af.Hd .* y
    return y
end
Base.size(Af::AFused) = (length(Af.Hd), length(Af.Hd))
Base.size(Af::AFused, i) = length(Af.Hd)
Base.eltype(::AFused) = Float64
LinearAlgebra.issymmetric(::AFused) = true

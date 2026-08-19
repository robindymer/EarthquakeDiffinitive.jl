module FaultResponse

using Diffinitive
using Diffinitive.Grids
using Diffinitive.SbpOperators
using Diffinitive.LazyTensors
using SparseArrays
using Tokens
using StaticArrays
using ..Elasticity: traction_blocks
using ..ElasticitySplitNode: split_node_system, CGSolver,
                             split_node_solve, solver_report,
                             duplicate, merge_stats!

export FaultElasticity, fault_grid_axes, frictional_node_count,
       shear_traction, shear_traction!, fault_stiffness, fault_stiffness_toeplitz,
       elastic_solver_report

# ==============================================================================
# The slip → shear-traction map on the fault.
#
# `ElasticitySplitNode` solves `-HP(D+SAT)P u = HP(D+SAT)χ(s)` for a given
# slip distribution, by CG on the assembled `A` directly (no factorization —
# see `ElasticitySplitNode.CGSolver`). Since λ and μ are constant `A` never
# changes as slip evolves, so this module assembles it once and then either
#
#   * solves per evaluation (`shear_traction`), or
#   * precomputes the dense fault stiffness `K : slip ↦ Δτ` once
#     (`fault_stiffness`) so the time loop is a single dense mat-vec.
#
# The latter is what makes a 30-day integration tractable: an adaptive
# integrator evaluates the right-hand side thousands of times, and one CG
# solve per evaluation on a 3D system dominates everything else. Its cost is
# `2*N_Ωf` CG solves up front, which only pays off because slip is confined to
# `Ω_f` (BP8 eq. 13) — a small subset of the fault plane, and the only place
# tractions are needed. Those `2*N_Ωf` solves are independent right-hand sides
# against the same `A`, so `fault_stiffness` threads the build across them —
# see `duplicate`/`merge_stats!`.
#
# SIGN. `traction_blocks` returns σ_i1 on the fixed +x₁ axis, which is exactly
# BP8's τ_i (eq. 6 makes it side-independent; both sides agree here to machine
# precision and are averaged). No sign flip: a positive slip patch produces
# Δσ21 < 0 at its centre, i.e. slip relieves the driving shear stress.
# ==============================================================================

"""
    boundary_selection(g, stencil_set, bid)

Linear indices into `g` of the boundary nodes of `bid`, ordered to match both
`boundary_grid(g, bid)`'s own linear ordering and the row ordering of the
operators returned by `traction_blocks`.

Note this is deliberately *not* `boundary_indices`, whose iteration order does
not agree with either. It is read straight off `boundary_restriction`, which
is the operator the traction blocks are actually built from.
"""
function boundary_selection(g, stencil_set, bid)
    e = sparse(boundary_restriction(g, stencil_set, bid))
    nnz(e) == size(e, 1) && all(≈(1.0), nonzeros(e)) ||
        error("boundary_restriction is not a pure selection for this stencil set; " *
              "the fault node ↔ DOF map below assumes it is")
    sel = zeros(Int, size(e, 1))
    rows = rowvals(e)
    for col in 1:size(e, 2), idx in nzrange(e, col)
        sel[rows[idx]] = col
    end
    return sel
end

"""
    FaultElasticity(g_minus, g_plus, λ, μ, stencil_set; l_f)

Assembles and factorizes the split-node system for the two grids, and works
out the fault-node ↔ DOF bookkeeping. `l_f` is the half-length of the
frictional domain `Ω_f = (-l_f,l_f)²`; the grids must have nodes exactly on
`±l_f` in both fault directions.

Slip and traction vectors are indexed over the `Ω_f` nodes only, in
column-major order (x2 fastest) over an `(n2f, n3f)` grid — the same ordering
`PorePressure`'s grid uses, so the two couple entry-wise.
"""
struct FaultElasticity
    P::SparseMatrixCSC{Float64,Int}
    HP_DSAT::SparseMatrixCSC{Float64,Int}
    rs::CGSolver
    T2::SparseMatrixCSC{Float64,Int}   # Nb × Ntot, fault-averaged σ21 extraction
    T3::SparseMatrixCSC{Float64,Int}
    chi_rows::Matrix{Int}              # [Ω_f node, component 2:3] → (minus,plus) DOFs
    chi_rows_plus::Matrix{Int}
    omega::Vector{Int}                 # Ω_f nodes as indices into the fault boundary
    x2f::Vector{Float64}
    x3f::Vector{Float64}
    Ntot::Int
end

function FaultElasticity(g_minus, g_plus, λ, μ, stencil_set; l_f, solver_kwargs...)
    bid_minus = CartesianBoundary{1,UpperBoundary}()
    bid_plus = CartesianBoundary{1,LowerBoundary}()
    Nm, Np = length(g_minus), length(g_plus)
    Ntot = 3 * (Nm + Np)

    sel_minus = boundary_selection(g_minus, stencil_set, bid_minus)
    sel_plus = boundary_selection(g_plus, stencil_set, bid_plus)

    bg = boundary_grid(g_minus, bid_minus)
    n2, n3 = size(bg)
    bci = CartesianIndices((n2, n3))
    coords = [bg[I] for I in bci]
    x2_all = [coords[bci[i, 1]][2] for i in 1:n2]
    x3_all = [coords[bci[1, j]][3] for j in 1:n3]

    # `Ω_f` nodes, column-major (x2 fastest) so the ordering matches the
    # pore-pressure grid's `vec`.
    tol = 1e-9 * max(l_f, 1.0)
    in_omega(x) = abs(x) <= l_f + tol
    i2 = findall(in_omega, x2_all)
    i3 = findall(in_omega, x3_all)
    isempty(i2) && error("no fault nodes inside Ω_f")
    for (name, ax, idx) in (("x2", x2_all, i2), ("x3", x3_all, i3))
        (isapprox(ax[idx[1]], -l_f; atol=tol) && isapprox(ax[idx[end]], l_f; atol=tol)) ||
            error("the elastic grid has no nodes exactly on $name = ±l_f = ±$l_f " *
                  "(nearest are $(ax[idx[1]]) and $(ax[idx[end]])); pick a spacing " *
                  "that divides l_f")
    end
    lin = LinearIndices((n2, n3))
    omega = [lin[a, b] for b in i3 for a in i2]

    A, HP_DSAT, P = split_node_system(g_minus, g_plus, λ, μ, stencil_set)

    Tm = traction_blocks(g_minus, λ, μ, stencil_set, bid_minus)
    Tp = traction_blocks(g_plus, λ, μ, stencil_set, bid_plus)
    extract(j) = 0.5 .* hcat(reduce(hcat, Tm[j, :]), reduce(hcat, Tp[j, :]))

    nf = length(omega)
    chi_rows = Matrix{Int}(undef, nf, 2)       # minus side, components 2 and 3
    chi_rows_plus = Matrix{Int}(undef, nf, 2)
    for (k, b) in enumerate(omega), (ci, comp) in enumerate(2:3)
        chi_rows[k, ci] = (comp - 1) * Nm + sel_minus[b]
        chi_rows_plus[k, ci] = 3Nm + (comp - 1) * Np + sel_plus[b]
    end

    return FaultElasticity(P, HP_DSAT, CGSolver(A; solver_kwargs...),
                           extract(2), extract(3),
                           chi_rows, chi_rows_plus, omega,
                           collect(x2_all[i2]), collect(x3_all[i3]), Ntot)
end

"""
    elastic_solver_report(fe) -> NamedTuple

What solving the elastic system has cost so far (CG iteration counts). See
`ElasticitySplitNode.solver_report`.
"""
elastic_solver_report(fe::FaultElasticity) = solver_report(fe.rs)

"""
    fault_grid_axes(fe) -> (x2, x3)

The `Ω_f` node coordinates along strike and depth.
"""
fault_grid_axes(fe::FaultElasticity) = (fe.x2f, fe.x3f)

"""
    frictional_node_count(fe)

Number of `Ω_f` nodes, i.e. the length of the slip/state/pressure vectors.
"""
frictional_node_count(fe::FaultElasticity) = length(fe.omega)

"""
    build_chi!(χ, fe, s2, s3)

In-place `χ(s)`: `±s_j/2` on the two sides at the `Ω_f` fault nodes, zero
elsewhere (BP8 eq. 13 gives zero slip outside `Ω_f`).
"""
function build_chi!(χ, fe::FaultElasticity, s2, s3)
    fill!(χ, 0.0)
    @inbounds for k in eachindex(s2)
        χ[fe.chi_rows[k, 1]] = -s2[k] / 2
        χ[fe.chi_rows_plus[k, 1]] = s2[k] / 2
        χ[fe.chi_rows[k, 2]] = -s3[k] / 2
        χ[fe.chi_rows_plus[k, 2]] = s3[k] / 2
    end
    return χ
end

"""
    shear_traction(fe, s2, s3) -> (Δτ2, Δτ3)

Shear traction change at the `Ω_f` nodes produced by the slip distribution
`(s2, s3)`, via one solve of the split-node system. For repeated evaluation
build [`fault_stiffness`](@ref) instead.
"""
function shear_traction(fe::FaultElasticity, s2, s3)
    Δτ2 = similar(s2)
    Δτ3 = similar(s3)
    shear_traction!(Δτ2, Δτ3, fe, s2, s3, zeros(fe.Ntot))
    return Δτ2, Δτ3
end

function shear_traction!(Δτ2, Δτ3, fe::FaultElasticity, s2, s3, χ, solver=fe.rs)
    build_chi!(χ, fe, s2, s3)
    U = fe.P * split_node_solve(solver, fe.HP_DSAT * χ) .+ χ
    τ2 = fe.T2 * U
    τ3 = fe.T3 * U
    @inbounds for (k, b) in enumerate(fe.omega)
        Δτ2[k] = τ2[b]
        Δτ3[k] = τ3[b]
    end
    return Δτ2, Δτ3
end

"""
    fault_stiffness(fe; verbose=false) -> K

The dense fault stiffness `K` mapping stacked slip `[s2; s3]` on `Ω_f` to
stacked traction change `[Δτ2; Δτ3]`, built one column at a time from unit
slip at each `Ω_f` degree of freedom. Costs `2*N_Ωf` CG solves against the
already-assembled system; afterwards each right-hand-side evaluation in the
time loop is a single dense mat-vec.

`K` is negative-definite in the physically meaningful sense that slip relieves
the stress driving it — `K[i,i] < 0`.

The columns are independent, so this build is **embarrassingly parallel** and
threads across `Threads.nthreads()` by default (start Julia with `-t auto`).
Pass `threaded=false` to force the serial path.
"""
function fault_stiffness(fe::FaultElasticity; verbose=false,
                         threaded=Threads.nthreads() > 1)
    nf = frictional_node_count(fe)
    K = Matrix{Float64}(undef, 2nf, 2nf)
    ncols = 2nf
    t0 = time()

    # Fills `cols` of K using `solver`, with buffers private to this call.
    #
    # This MUST be a function rather than a `begin` block inside the spawn:
    # `if`/`else` and `begin` do not introduce scope in Julia, so buffers
    # assigned there would be locals of `fault_stiffness` and every task would
    # share the same `χ`/`s2`/`Δτ` arrays. A function body is a real scope, so
    # each invocation gets its own. (Learned the hard way — the threaded `K`
    # came out 2.35 relative off before this was a function.)
    function run_columns!(cols, solver; progress=false)
        s2, s3 = zeros(nf), zeros(nf)
        Δτ2, Δτ3 = zeros(nf), zeros(nf)
        χ = zeros(fe.Ntot)
        done = 0
        for col in cols
            fill!(s2, 0.0)
            fill!(s3, 0.0)
            col <= nf ? (s2[col] = 1.0) : (s3[col-nf] = 1.0)
            shear_traction!(Δτ2, Δτ3, fe, s2, s3, χ, solver)
            K[1:nf, col] .= Δτ2
            K[nf+1:2nf, col] .= Δτ3
            done += 1
            if progress && (done % 50 == 0 || done == length(cols))
                el = time() - t0
                @info "fault_stiffness: column $done/$(length(cols))" elapsed = round(el, digits=1) eta = round(el * (length(cols) - done) / done, digits=1)
            end
        end
        return solver
    end

    if threaded
        nt = min(Threads.nthreads(), ncols)
        verbose && @info "fault_stiffness: threaded build" columns = ncols threads = nt
        # Strided partition: iteration counts vary a little between columns, so
        # interleaving balances the chunks better than contiguous blocks.
        tasks = [Threads.@spawn run_columns!(t:nt:ncols, duplicate(fe.rs)) for t in 1:nt]
        # Fold each worker's counters back so `solver_report` totals the build.
        for task in tasks
            merge_stats!(fe.rs, fetch(task))
        end
    else
        run_columns!(1:ncols, fe.rs; progress=verbose)
    end
    verbose && @info "fault_stiffness: done" seconds = round(time() - t0, digits=1)
    return K
end

"""
    fault_stiffness_toeplitz(fe; verbose=false) -> K

`K` built from **5 sources** — 10 CG solves instead of `2·N_Ωf` — by exploiting
the translation invariance of the whole-space kernel.

In a homogeneous whole-space the traction at node `i` from unit slip at node `j`
depends only on the separation `x_i − x_j`, so `K` is block-Toeplitz with
Toeplitz blocks and one source column determines the rest. The far-field `u=0`
truncation breaks that exactly — nodes near the boundary see a different medium
— but the departure is small and, crucially, concentrated where the kernel is
already negligible.

## Priority, not averaging — this distinction is the whole design

Sources are consulted in order and **the first one to supply a separation wins**.
Averaging them together instead is a ~500× regression: corner and edge sources
sit against the truncation boundary and the locked `Ω_f` ring, so their kernels
are contaminated, and averaging lets that contamination into the near field that
drives the solution.

Measured end-to-end against the full build at Δz = 50 m (`PERFORMANCE.md` §4b),
worst relative error in `V_max(t)` over the run:

| sources | solves | 100 h | 30 days | **30 d, converged domain** |
|---|---|---|---|---|
| centre only | 2 | 0.030% | 138% | **96.9%** |
| **centre + 4 corners (priority)** | **10** | 0.005% | 5.8% | **0.41%** |
| 4 corners, *averaged* | 8 | 16.5% | 162% | 213% |

**Why the corners are needed despite being contaminated.** The centre alone
cannot reach separations larger than half the grid (pairs on opposite edges,
~44% of entries). Through the injection phase those are genuinely negligible —
centre-only scores 0.03%. After injection stops at `t_off`, `V_max` falls an
order of magnitude and the relaxation phase is far more sensitive to them:
centre-only degrades to 97%. Adding the corners *for those separations only*
fixes it while leaving the near field untouched.

Hence ordered, not averaged, and the ordering must not be changed to put a
boundary source first. Note also that the last column is the only one describing
the configuration that will actually be run — a change validated at 100 h alone,
or at the small domain alone, is not validated.

## This is an approximation

It introduces ~0.03% in `V_max`, against a ~53% domain-truncation bias
(`PROGRESS.md` "Results") and a larger resolution error — so it is far from the
accuracy-limiting step. But it *is* opt-in for that reason: [`fault_stiffness`](@ref)
remains the exact build and the default.
"""
function fault_stiffness_toeplitz(fe::FaultElasticity; verbose=false)
    n2, n3 = length(fe.x2f), length(fe.x3f)
    nf = frictional_node_count(fe)
    nf == n2 * n3 || error("Ω_f is $n2×$n3 = $(n2*n3) but nf = $nf; the Toeplitz " *
                           "expansion assumes the nodes form that full grid")
    t0 = time()

    # Sources in PRIORITY order: the centre first, then the corners. The centre
    # is furthest from every boundary and sits where the injection is, so its
    # kernel is the clean one and it wins wherever it reaches. The corners exist
    # only to supply separations the centre cannot reach — pairs on opposite
    # edges, |da| > (n2−1)/2 — which the centre leaves at zero.
    ac, bc = (n2 + 1) ÷ 2, (n3 + 1) ÷ 2
    idx(a, b) = (b - 1) * n2 + a
    sources = [idx(ac, bc), idx(1, 1), idx(n2, 1), idx(1, n3), idx(n2, n3)]

    s2, s3 = zeros(nf), zeros(nf)
    Δτ2, Δτ3 = zeros(nf), zeros(nf)
    χ = zeros(fe.Ntot)

    # kernel[(da,db)] = (τ2 from s2, τ3 from s2, τ2 from s3, τ3 from s3)
    kernel = Dict{Tuple{Int,Int},NTuple{4,Float64}}()
    for src in sources
        asrc, bsrc = mod1(src, n2), (src - 1) ÷ n2 + 1
        fill!(s2, 0.0); fill!(s3, 0.0)
        s2[src] = 1.0
        shear_traction!(Δτ2, Δτ3, fe, s2, s3, χ)
        c22, c32 = copy(Δτ2), copy(Δτ3)
        fill!(s2, 0.0)
        s3[src] = 1.0
        shear_traction!(Δτ2, Δτ3, fe, s2, s3, χ)
        for i in 1:nf
            ai, bi = mod1(i, n2), (i - 1) ÷ n2 + 1
            sep = (ai - asrc, bi - bsrc)
            haskey(kernel, sep) && continue          # earlier source wins
            kernel[sep] = (c22[i], c32[i], Δτ2[i], Δτ3[i])
        end
    end

    K = zeros(2nf, 2nf)
    @inbounds for j in 1:nf
        aj, bj = mod1(j, n2), (j - 1) ÷ n2 + 1
        for i in 1:nf
            ai, bi = mod1(i, n2), (i - 1) ÷ n2 + 1
            v = get(kernel, (ai - aj, bi - bj), nothing)
            v === nothing && continue                # genuinely unreachable → 0
            K[i, j]       = v[1]
            K[nf+i, j]    = v[2]
            K[i, nf+j]    = v[3]
            K[nf+i, nf+j] = v[4]
        end
    end
    verbose && @info "fault_stiffness_toeplitz: done" seconds = round(time() - t0, digits=1) solves = 2length(sources) instead_of = 2nf
    return K
end

end # module FaultResponse

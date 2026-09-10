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
       fault_stiffness_d4_shard, elastic_solver_report

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
    square_symmetry_group(n) -> (perms, Qs)

The 8 elements of `D4`, the symmetry group of a square, acting on the flat
column-major index `a + (b-1)*n` of an `n × n` grid, paired with the 2×2
signed permutation each element induces on the fault-parallel vector
components `(v2, v3)` — e.g. reflecting `x2 → −x2` fixes `b`, reverses `a`,
and flips the sign of `v2` alone. `perms[k][i]` is where element `k` sends
node `i`; `Qs[k]` is its component action. See `PERFORMANCE.md` §5 item 0b.
"""
function square_symmetry_group(n)
    idx(a, b) = a + (b - 1) * n
    r(x) = n + 1 - x
    maps = ((a, b) -> (a, b), (a, b) -> (r(a), b), (a, b) -> (a, r(b)),
            (a, b) -> (r(a), r(b)), (a, b) -> (b, a), (a, b) -> (r(b), r(a)),
            (a, b) -> (r(b), a), (a, b) -> (b, r(a)))
    Qs = (SA[1.0 0.0; 0.0 1.0], SA[-1.0 0.0; 0.0 1.0], SA[1.0 0.0; 0.0 -1.0],
          SA[-1.0 0.0; 0.0 -1.0], SA[0.0 1.0; 1.0 0.0], SA[0.0 -1.0; -1.0 0.0],
          SA[0.0 -1.0; 1.0 0.0], SA[0.0 1.0; -1.0 0.0])
    perms = map(maps) do m
        p = Vector{Int}(undef, n * n)
        for b in 1:n, a in 1:n
            a2, b2 = m(a, b)
            p[idx(a, b)] = idx(a2, b2)
        end
        p
    end
    return perms, Qs
end

# One representative source column per orbit of the `2n²` source columns
# (node × {s2,s3}) under `square_symmetry_group(n)`, plus, for each
# representative, the `(group index, target column)` pairs its single solve
# determines. Purely combinatorial — no CG solve here — so it can run once,
# up front, before any threading decision.
function column_orbits(n, perms, Qs)
    nf = n * n
    ncols = 2nf
    visited = falses(ncols)
    reps = Int[]
    orbit_targets = Vector{Vector{Tuple{Int,Int}}}()
    for col in 1:ncols
        visited[col] && continue
        node = col <= nf ? col : col - nf
        comp = col <= nf ? 1 : 2
        push!(reps, col)
        targets = Tuple{Int,Int}[]
        for g in eachindex(perms)
            c2 = Qs[g][1, comp] != 0 ? 1 : 2
            tnode = perms[g][node]
            tcol = c2 == 1 ? tnode : nf + tnode
            visited[tcol] && continue
            visited[tcol] = true
            push!(targets, (g, tcol))
        end
        push!(orbit_targets, targets)
    end
    return reps, orbit_targets
end

"""
    fault_stiffness(fe; verbose=false, cols=nothing, symmetry=false) -> K

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

`cols` restricts the build to a subset of the `1:2*N_Ωf` column indices —
`nothing` (default) builds all of them and returns the usual `2N_Ωf × 2N_Ωf`
matrix. Passing a range or vector instead returns a `2N_Ωf × length(cols)`
matrix holding just those columns, in the order given — this is what makes
the build shardable across independent processes (`build_stiffness_cache.jl`
`[shard] [nshards]`) at a resolution where a single node's worth of solves is
not tractable: each shard rebuilds `fe` (cheap — minutes, PERFORMANCE.md §4)
and computes only its slice of columns, since the columns need no
communication with each other.

## `symmetry=true`: exploit `K`'s `D4` symmetry instead of sharding

`PERFORMANCE.md` §5 item 0b. When `Ω_f` and the surrounding elastic grid are
square and centred in the two fault-parallel directions (as `BP8.jl` always
builds them: same `L_fault`, same node count, on both axes), the whole
discretization is invariant under the 8 symmetries of the square acting
jointly on node position and on `(s2,s3)`/`(τ2,τ3)`. That gives an **exact**
discrete identity — not an approximation like [`fault_stiffness_toeplitz`](@ref) —
`K[g·i, g·j] = Q·K[i,j]·Qᵀ`, verified against a full build to 1e-16
(Frobenius, relative). One CG solve pair therefore determines up to 8 column
pairs instead of 1, cutting the number of solves needed by **6.5–7.8×**
(growing with resolution, since fewer nodes sit on the symmetry axes/diagonal
as a fraction of the total). Still threads across `Threads.nthreads()`
exactly like the plain build, over the *orbit representatives* rather than
the raw columns — the two are compatible for the same reason plain threading
is: distinct orbits fill disjoint columns of `K`.

Requires a square, centred `Ω_f` (checked; throws otherwise) and is not
compatible with `cols` (it builds the whole matrix by construction — combine
with the `EQD_STIFFNESS_CACHE` mechanism in `PERFORMANCE.md` §4c instead of
sharding if a single node still isn't enough).
"""
function fault_stiffness(fe::FaultElasticity; verbose=false,
                         threaded=Threads.nthreads() > 1, cols=nothing, symmetry=false)
    if symmetry
        cols === nothing ||
            error("fault_stiffness: symmetry=true builds the whole matrix and is not " *
                  "compatible with `cols`")
        return fault_stiffness_d4(fe; verbose, threaded)
    end
    nf = frictional_node_count(fe)
    cols = cols === nothing ? (1:2nf) : cols
    ncols = length(cols)
    K = Matrix{Float64}(undef, 2nf, ncols)
    t0 = time()

    # Fills the assigned (position, column) pairs of K using `solver`, with
    # buffers private to this call.
    #
    # This MUST be a function rather than a `begin` block inside the spawn:
    # `if`/`else` and `begin` do not introduce scope in Julia, so buffers
    # assigned there would be locals of `fault_stiffness` and every task would
    # share the same `χ`/`s2`/`Δτ` arrays. A function body is a real scope, so
    # each invocation gets its own. (Learned the hard way — the threaded `K`
    # came out 2.35 relative off before this was a function.)
    function run_columns!(items, solver; progress=false)
        s2, s3 = zeros(nf), zeros(nf)
        Δτ2, Δτ3 = zeros(nf), zeros(nf)
        χ = zeros(fe.Ntot)
        done = 0
        for (pos, col) in items
            fill!(s2, 0.0)
            fill!(s3, 0.0)
            col <= nf ? (s2[col] = 1.0) : (s3[col-nf] = 1.0)
            shear_traction!(Δτ2, Δτ3, fe, s2, s3, χ, solver)
            K[1:nf, pos] .= Δτ2
            K[nf+1:2nf, pos] .= Δτ3
            done += 1
            if progress && (done % 50 == 0 || done == length(items))
                el = time() - t0
                @info "fault_stiffness: column $done/$(length(items))" elapsed = round(el, digits=1) eta = round(el * (length(items) - done) / done, digits=1)
            end
        end
        return solver
    end

    indexed = collect(enumerate(cols))
    if threaded
        nt = min(Threads.nthreads(), ncols)
        verbose && @info "fault_stiffness: threaded build" columns = ncols threads = nt
        # Strided partition: iteration counts vary a little between columns, so
        # interleaving balances the chunks better than contiguous blocks.
        # Only task 1 reports, or 12 threads interleave their progress lines.
        # Its ETA is representative of the whole build even though it sees just
        # its own slice: the slices are equal-sized and run concurrently, so the
        # time for one to finish IS the time for all of them. Without this the
        # progress reporting only worked on the serial path, which is the one
        # nobody runs at production size — a 2.1 h build with no visible progress.
        tasks = [Threads.@spawn run_columns!(indexed[t:nt:end], duplicate(fe.rs);
                                             progress=(verbose && t == 1)) for t in 1:nt]
        # Fold each worker's counters back so `solver_report` totals the build.
        for task in tasks
            merge_stats!(fe.rs, fetch(task))
        end
    else
        run_columns!(indexed, fe.rs; progress=verbose)
    end
    verbose && @info "fault_stiffness: done" seconds = round(time() - t0, digits=1)
    return K
end

# Common precondition check + orbit setup for both `symmetry=true` entry
# points (`fault_stiffness_d4`, `fault_stiffness_d4_shard`): a square, centred
# `Ω_f`, and the D4 group/orbit tables for its side length. `caller` names the
# public function in error messages.
function d4_setup(fe::FaultElasticity, caller)
    n2, n3 = length(fe.x2f), length(fe.x3f)
    n2 == n3 ||
        error("$caller: symmetry=true needs a square Ω_f (n2 == n3); " *
              "got $n2 × $n3 — see PERFORMANCE.md §5 item 0b")
    n = n2
    atol = 1e-9 * max(maximum(abs, fe.x2f), 1.0)
    isapprox(fe.x2f, fe.x3f; atol) ||
        error("$caller: symmetry=true needs identical x2/x3 grids on Ω_f " *
              "(the elastic domain must be square in both fault-parallel directions)")
    all(a -> isapprox(fe.x2f[a], -fe.x2f[n+1-a]; atol), 1:n) ||
        error("$caller: symmetry=true needs Ω_f centred about 0 on both axes")

    perms, Qs = square_symmetry_group(n)
    reps, targets = column_orbits(n, perms, Qs)
    return perms, Qs, reps, targets
end

# The `symmetry=true` path of `fault_stiffness`: one CG solve pair per D4
# orbit representative, propagated to the rest of the orbit by symmetry
# instead of solved for. See that docstring and PERFORMANCE.md §5 item 0b for
# the identity this implements and its preconditions.
function fault_stiffness_d4(fe::FaultElasticity; verbose=false,
                            threaded=Threads.nthreads() > 1)
    nf = frictional_node_count(fe)
    perms, Qs, reps, targets = d4_setup(fe, "fault_stiffness")
    ncols = 2nf
    K = zeros(ncols, ncols)
    t0 = time()

    # Solves the assigned representatives and, for each, fills every column
    # its orbit determines. Distinct representatives' orbits fill disjoint
    # columns of K (column_orbits partitions 1:ncols), so this is safe to run
    # concurrently across tasks exactly like `run_columns!` above — and for
    # the same reason must be a function, not a `begin` block, so each task's
    # buffers are private (see the comment on `run_columns!`).
    function run_reps!(items, solver; progress=false)
        s2, s3 = zeros(nf), zeros(nf)
        Δτ2, Δτ3 = zeros(nf), zeros(nf)
        χ = zeros(fe.Ntot)
        done = 0
        for (pos, col) in items
            node = col <= nf ? col : col - nf
            comp = col <= nf ? 1 : 2
            fill!(s2, 0.0)
            fill!(s3, 0.0)
            comp == 1 ? (s2[node] = 1.0) : (s3[node] = 1.0)
            shear_traction!(Δτ2, Δτ3, fe, s2, s3, χ, solver)
            @inbounds for (g, tcol) in targets[pos]
                Q = Qs[g]
                perm = perms[g]
                σ = Q[1, comp] != 0 ? Q[1, comp] : Q[2, comp]
                for i in 1:nf
                    v1 = Q[1, 1] * Δτ2[i] + Q[1, 2] * Δτ3[i]
                    v2 = Q[2, 1] * Δτ2[i] + Q[2, 2] * Δτ3[i]
                    ti = perm[i]
                    K[ti, tcol] = σ * v1
                    K[nf+ti, tcol] = σ * v2
                end
            end
            done += 1
            if progress && (done % 20 == 0 || done == length(items))
                el = time() - t0
                @info "fault_stiffness (D4 symmetry): representative $done/$(length(items))" elapsed = round(el, digits=1) eta = round(el * (length(items) - done) / done, digits=1)
            end
        end
        return solver
    end

    indexed = collect(enumerate(reps))
    if threaded
        nt = min(Threads.nthreads(), length(reps))
        verbose && @info "fault_stiffness (D4 symmetry): threaded build" representatives = length(reps) columns = ncols threads = nt
        tasks = [Threads.@spawn run_reps!(indexed[t:nt:end], duplicate(fe.rs);
                                          progress=(verbose && t == 1)) for t in 1:nt]
        for task in tasks
            merge_stats!(fe.rs, fetch(task))
        end
    else
        run_reps!(indexed, fe.rs; progress=verbose)
    end
    verbose && @info "fault_stiffness (D4 symmetry): done" seconds = round(time() - t0, digits=1) representatives = length(reps) columns = ncols
    return K
end

"""
    fault_stiffness_d4_shard(fe, shard, nshards; verbose=false,
                             threaded=Threads.nthreads() > 1) -> (cols, Kshard)

D4 symmetry (`fault_stiffness(fe; symmetry=true)`, PERFORMANCE.md §5 item 0b)
combined with cross-node sharding, for a resolution where even the
symmetry-reduced solve count does not fit one node's wall-clock budget (e.g.
Δz = 10 m: `:exact` needs `2·N_Ωf` solves, D4 cuts that ~6.5-7.8×, and that
can still be node-days). Splits the D4 orbit **representatives** — not the
raw columns — `shard:nshards:end` across `nshards` independent processes;
`shard` is 1-based.

Returns `(cols, Kshard)`: the *global* column indices (into `1:2·N_Ωf`) this
shard's representatives determine, and the corresponding `2·N_Ωf ×
length(cols)` slice. This is deliberately the same shape
[`save_stiffness_shard`](@ref) takes for the plain (non-symmetric) shard
build, so a shard from this function drops straight into the existing
`merge_stiffness_shards` with no format change — each representative's single
solve fixes up to 8 columns of its symmetry orbit, so unlike a raw-column
shard, which columns a shard covers isn't known until after the orbit tables
are built, which is why it's returned rather than passed in.

`column_orbits` partitions the *whole* `1:2·N_Ωf` range across representatives
regardless of how those representatives are split across shards, so the union
of `cols` across all `nshards` shards is still exactly `1:2·N_Ωf` with no gaps
or duplicates — `merge_stiffness_shards`'s coverage check is unaffected by
switching to this builder.

Same square, centred `Ω_f` precondition as `fault_stiffness(fe; symmetry=true)`
(checked; throws otherwise).
"""
function fault_stiffness_d4_shard(fe::FaultElasticity, shard::Integer, nshards::Integer;
                                  verbose=false, threaded=Threads.nthreads() > 1)
    1 <= shard <= nshards ||
        error("fault_stiffness_d4_shard: shard must be in 1:nshards, got " *
              "shard=$shard nshards=$nshards")
    nf = frictional_node_count(fe)
    perms, Qs, reps, targets = d4_setup(fe, "fault_stiffness_d4_shard")
    myreps = collect(shard:nshards:length(reps))

    mycols = Int[]
    for pos in myreps, (_, tcol) in targets[pos]
        push!(mycols, tcol)
    end
    colpos = Dict(c => i for (i, c) in enumerate(mycols))
    Kshard = zeros(2nf, length(mycols))
    t0 = time()

    # Same private-buffers-via-function-scope requirement as `run_columns!`/
    # `run_reps!` above (see the comment on `run_columns!`): `if`/`else` and
    # `begin` are not scopes in Julia.
    function run_reps!(items, solver; progress=false)
        s2, s3 = zeros(nf), zeros(nf)
        Δτ2, Δτ3 = zeros(nf), zeros(nf)
        χ = zeros(fe.Ntot)
        done = 0
        for pos in items
            col = reps[pos]
            node = col <= nf ? col : col - nf
            comp = col <= nf ? 1 : 2
            fill!(s2, 0.0)
            fill!(s3, 0.0)
            comp == 1 ? (s2[node] = 1.0) : (s3[node] = 1.0)
            shear_traction!(Δτ2, Δτ3, fe, s2, s3, χ, solver)
            @inbounds for (g, tcol) in targets[pos]
                Q = Qs[g]
                perm = perms[g]
                σ = Q[1, comp] != 0 ? Q[1, comp] : Q[2, comp]
                out = colpos[tcol]
                for i in 1:nf
                    v1 = Q[1, 1] * Δτ2[i] + Q[1, 2] * Δτ3[i]
                    v2 = Q[2, 1] * Δτ2[i] + Q[2, 2] * Δτ3[i]
                    ti = perm[i]
                    Kshard[ti, out] = σ * v1
                    Kshard[nf+ti, out] = σ * v2
                end
            end
            done += 1
            if progress && (done % 20 == 0 || done == length(items))
                el = time() - t0
                @info "fault_stiffness_d4_shard: representative $done/$(length(items))" elapsed = round(el, digits=1) eta = round(el * (length(items) - done) / done, digits=1)
            end
        end
        return solver
    end

    if threaded
        nt = min(Threads.nthreads(), length(myreps))
        verbose && @info "fault_stiffness_d4_shard: threaded build" shard nshards representatives = length(myreps) columns = length(mycols) threads = nt
        tasks = [Threads.@spawn run_reps!(myreps[t:nt:end], duplicate(fe.rs);
                                          progress=(verbose && t == 1)) for t in 1:nt]
        for task in tasks
            merge_stats!(fe.rs, fetch(task))
        end
    else
        run_reps!(myreps, fe.rs; progress=verbose)
    end
    verbose && @info "fault_stiffness_d4_shard: done" seconds = round(time() - t0, digits=1) shard nshards representatives = length(myreps) columns = length(mycols)
    return mycols, Kshard
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

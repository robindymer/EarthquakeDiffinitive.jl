module FaultResponse

using Diffinitive
using Diffinitive.Grids
using Diffinitive.SbpOperators
using Diffinitive.LazyTensors
using SparseArrays
using Tokens
using StaticArrays
using ..Elasticity: traction_blocks
using ..ElasticitySplitNode: split_node_system, split_node_operator,
                             AssembledSplitNode, SplitNodeOperator,
                             apply_P!, hp_dsat!, CGSolver,
                             split_node_solve, solver_report,
                             duplicate, merge_stats!

export FaultElasticity, fault_grid_axes, frictional_node_count,
       shear_traction, shear_traction!, fault_stiffness, fault_stiffness_toeplitz,
       fault_stiffness_d4_shard, fault_stiffness_gpu, elastic_solver_report

# The slip → shear-traction map on the fault.
#
# `ElasticitySplitNode` solves `-HP(D+SAT)P u = HP(D+SAT)χ(s)` by CG on `A`
# directly. `A` defaults to the matrix-free `SplitNodeOperator`;
# `representation=:assembled` builds the explicit matrices for comparison.
# λ and μ are constant, so `A` never changes as slip evolves: build it once,
# then either
#
#   * solve per evaluation (`shear_traction`), or
#   * precompute the dense fault stiffness `K : slip ↦ Δτ` (`fault_stiffness`)
#     so the time loop is one dense mat-vec.
#
# The second is what makes a 30-day integration tractable — an adaptive
# integrator evaluates the RHS thousands of times, and a 3D CG solve per
# evaluation dominates everything else. It costs `2*N_Ωf` solves up front,
# which only pays because slip is confined to `Ω_f` (BP8 eq. 13). Those solves
# are independent right-hand sides against one `A`, so the build threads across
# them (`duplicate`/`merge_stats!`).
#
# SIGN. `traction_blocks` returns σ_i1 on the fixed +x₁ axis, which is BP8's
# τ_i (eq. 6 makes it side-independent; both sides agree to machine precision
# and are averaged). No sign flip: a positive slip patch gives Δσ21 < 0 at its
# centre, i.e. slip relieves the driving shear stress.

"""
    boundary_selection(g, stencil_set, bid)

Linear indices into `g` of `bid`'s boundary nodes, ordered to match both
`boundary_grid(g, bid)` and `traction_blocks`' row ordering.

Deliberately not `boundary_indices`, whose order agrees with neither. Read
straight off `boundary_restriction`, the operator the traction blocks are
built from.
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

Builds the split-node system for the two grids and the fault-node ↔ DOF
bookkeeping. `l_f` is the half-length of `Ω_f = (-l_f,l_f)²`; the grids must
have nodes exactly on `±l_f` in both fault directions.

Slip and traction vectors span the `Ω_f` nodes only, column-major (x2 fastest)
over an `(n2f, n3f)` grid — the ordering `PorePressure`'s grid uses, so the two
couple entry-wise.
"""
struct FaultElasticity{TOp}
    op::TOp                            # SplitNodeOperator or AssembledSplitNode; also rs.A
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

"""
    FaultElasticity(g_minus, g_plus, λ, μ, stencil_set; l_f,
                    representation=:kronecker, solver_kwargs...)

The elastic system behind the slip → traction map, ready to solve.
`representation=:kronecker` (default) applies `A` matrix-free through
[`SplitNodeOperator`](@ref): seconds to build, a handful of vectors to hold.
`:assembled` forms [`split_node_system`](@ref)'s explicit matrices — hours and
tens of GB at production size — and is kept for validation and for
`precond=:jacobi`, which needs `diag(A)`. Both give the same `K` to CG
tolerance and share one cache key.
"""
function FaultElasticity(g_minus, g_plus, λ, μ, stencil_set; l_f,
                         representation::Symbol=:kronecker, solver_kwargs...)
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

    op = if representation === :kronecker
        split_node_operator(g_minus, g_plus, λ, μ, stencil_set)
    elseif representation === :assembled
        AssembledSplitNode(split_node_system(g_minus, g_plus, λ, μ, stencil_set)...)
    else
        error("representation must be :kronecker or :assembled, got $representation")
    end

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

    return FaultElasticity(op, CGSolver(op; solver_kwargs...),
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

In-place `χ(s)`: `±s_j/2` on the two sides at the `Ω_f` nodes, zero elsewhere
(BP8 eq. 13 gives zero slip outside `Ω_f`).
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
    # `solver.A`, not `fe.op`: a duplicated solver owns its operator scratch,
    # which is what makes the threaded build race-free.
    op = solver.A
    b = hp_dsat!(similar(χ), op, χ)
    U = apply_P!(similar(χ), op, split_node_solve(solver, b)) .+ χ
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

The 8 elements of `D4` acting on the flat column-major index `a + (b-1)*n` of
an `n × n` grid, each paired with the 2×2 signed permutation it induces on
`(v2, v3)` — reflecting `x2 → −x2`, say, reverses `a` and flips `v2` alone.
`perms[k][i]` is where element `k` sends node `i`, `Qs[k]` its component
action. `PERFORMANCE.md` §5 item 0b.
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
# (node × {s2,s3}) under `square_symmetry_group(n)`, plus the
# `(group index, target column)` pairs each representative's solve determines.
# Purely combinatorial, so it runs once up front.
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

The dense fault stiffness `K`, mapping stacked slip `[s2; s3]` on `Ω_f` to
stacked traction change `[Δτ2; Δτ3]`, one column per unit slip DOF. Costs
`2*N_Ωf` CG solves up front; afterwards each RHS evaluation in the time loop
is one dense mat-vec. `K[i,i] < 0` — slip relieves the stress driving it.

Columns are independent, so the build threads across `Threads.nthreads()` by
default (`julia -t auto`); `threaded=false` forces serial.

`cols` restricts the build to a subset of `1:2*N_Ωf`, returning a
`2N_Ωf × length(cols)` slice in the order given, which is what makes the build
shardable across processes (`build_stiffness_cache.jl [shard] [nshards]`).
Each shard rebuilds `fe` (minutes) and computes only its columns.

## `symmetry=true`: `K`'s `D4` symmetry instead of sharding

When `Ω_f` and the elastic grid are square and centred in both fault-parallel
directions — as `BP8.jl` always builds them — the discretization is invariant
under the 8 symmetries of the square acting jointly on node position and on
`(s2,s3)`/`(τ2,τ3)`. That is an **exact** discrete identity, unlike
[`fault_stiffness_toeplitz`](@ref): `K[g·i, g·j] = Q·K[i,j]·Qᵀ`, verified
against a full build to 1e-16. One solve pair then fixes up to 8 column pairs,
cutting solves by **6.5-7.8×** (more at higher resolution). Threads over the
orbit representatives, safe for the same reason the plain build is — distinct
orbits fill disjoint columns.

Needs a square, centred `Ω_f` (checked) and is incompatible with `cols`, since
it builds the whole matrix. `PERFORMANCE.md` §5 item 0b.
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

    # Fills the assigned (position, column) pairs of K, with private buffers.
    #
    # MUST be a function, not a `begin` block in the spawn: `if`/`else` and
    # `begin` are not scopes in Julia, so the buffers would be locals of
    # `fault_stiffness` and every task would share them. The threaded `K` came
    # out 2.35 relative off before this was a function.
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
        # Strided partition: iteration counts vary between columns, so
        # interleaving balances better than contiguous blocks. Only task 1
        # reports, or the threads interleave their progress lines; its ETA still
        # covers the whole build, since the slices are equal and concurrent.
        tasks = [Threads.@spawn run_columns!(indexed[t:nt:end], duplicate(fe.rs);
                                             progress=(verbose && t == 1)) for t in 1:nt]
        # Fold the workers' counters back so `solver_report` totals the build.
        for task in tasks
            merge_stats!(fe.rs, fetch(task))
        end
    else
        run_columns!(indexed, fe.rs; progress=verbose)
    end
    verbose && @info "fault_stiffness: done" seconds = round(time() - t0, digits=1)
    return K
end

# Precondition check + orbit setup shared by `fault_stiffness_d4` and
# `fault_stiffness_d4_shard`: a square, centred `Ω_f` and the D4 group/orbit
# tables. `caller` names the public function in error messages.
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

# The `symmetry=true` path of `fault_stiffness`: one CG solve pair per D4 orbit
# representative, propagated to the rest of the orbit by symmetry.
function fault_stiffness_d4(fe::FaultElasticity; verbose=false,
                            threaded=Threads.nthreads() > 1)
    nf = frictional_node_count(fe)
    perms, Qs, reps, targets = d4_setup(fe, "fault_stiffness")
    ncols = 2nf
    K = zeros(ncols, ncols)
    t0 = time()

    # Solves the assigned representatives and fills every column their orbits
    # determine. `column_orbits` partitions `1:ncols`, so distinct
    # representatives fill disjoint columns and this is concurrency-safe. Must
    # be a function for private buffers — see `run_columns!`.
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

D4 symmetry plus cross-node sharding, for a resolution where even the
symmetry-reduced solve count exceeds one node's walltime. Splits the D4 orbit
**representatives** — not the raw columns — `shard:nshards:end` across
`nshards` processes; `shard` is 1-based.

Returns `(cols, Kshard)`: the global column indices (into `1:2·N_Ωf`) this
shard's representatives determine, and the matching `2·N_Ωf × length(cols)`
slice. That is the same shape [`save_stiffness_shard`](@ref) takes for the
plain shard build, so these shards merge unchanged. `cols` is returned rather
than passed in because which columns a shard covers is only known once the
orbit tables exist.

`column_orbits` partitions all of `1:2·N_Ωf` across representatives however
they are split, so the union of `cols` over all shards is exactly `1:2·N_Ωf` —
`merge_stiffness_shards`' coverage check still holds.

Same square, centred `Ω_f` precondition as `symmetry=true`.
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

    # Must be a function for private buffers — see `run_columns!`.
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

`K` from **5 sources** — 10 CG solves instead of `2·N_Ωf` — using the
translation invariance of the whole-space kernel. An **approximation**, kept as
a cheap fallback; D4 symmetry plus the GPU build (see
[`fault_stiffness_gpu`](@ref)) made the exact build affordable and is the
production route.

In a homogeneous whole-space the traction at `i` from unit slip at `j` depends
only on `x_i − x_j`, so `K` is block-Toeplitz and one source column determines
the rest. The far-field `u=0` truncation breaks that, but only where the kernel
is already negligible.

## Priority, not averaging

Sources are consulted in order and **the first to supply a separation wins**.
Averaging instead is a ~500× regression: corner and edge sources sit against
the truncation boundary and the locked `Ω_f` ring, so averaging lets their
contaminated kernels into the near field.

The corners are needed despite that contamination because the centre cannot
reach separations beyond half the grid (~44% of entries). Those are negligible
during injection (centre-only: 0.03% in `V_max`) but not during the relaxation
after `t_off`, where centre-only degrades to 97%. The corners supply exactly
those separations and leave the near field untouched — so the ordering must not
be changed to put a boundary source first.

Worst relative error in `V_max(t)` against the full build, converged domain:
**0.41%** for centre + 4 corners, against 96.9% centre-only and 213% averaged.
`PERFORMANCE.md` §4b has the full table and the Δz dependence.
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

"""
    fault_stiffness_gpu(fe; verbose=false) -> K
    fault_stiffness_gpu(fe; shard, nshards, verbose=false) -> (cols, Kshard)

The production `K` build. Same D4-orbit reduction as
[`fault_stiffness`](@ref)`(fe; symmetry=true)`, but each representative's CG
solve runs on the GPU against an elastic system resident there for the whole
build, rather than CPU-threaded.

With the default matrix-free representation "resident" means the 1D operators,
`P`'s index data, the SAT block and ~9 system-length vectors — ~9 GB at
Δz = 10 m on (1600, 1600), where the assembled `A` alone would be ~85 GB — so
any datacenter card holds any BP8 configuration. `representation=:assembled`
uploads the CSR `A` and `P` instead and is limited by their size (~23 GB at
Δz = 10 m on (1150, 1150)).

`shard`/`nshards` split the representatives exactly as
[`fault_stiffness_d4_shard`](@ref) does and return its `(cols, Kshard)`, so
`merge_stiffness_cache.jl` reassembles GPU shards unchanged. Use it to spread a
build over cards or to fit a walltime limit.

**Sequential, not threaded like the CPU path.** One GPU has one bandwidth
budget, and this path exists because the bandwidth-bound mat-vec is faster
there. Concurrent solves on one device would contend for that bandwidth rather
than add to it, unlike CPU threads with their own caches.

Measured at the production point: Δz = 10 m on (1600, 1600), 99.5 M DOF, 8
shards of 211 representatives on one L40S each, 6.01 h per shard — 48.1 h of
card time, mean 1599 CG iterations, none unconverged. Correctness against the
CPU build is covered by `test/fault_response_gpu_test.jl` (gated behind
`CUDA.functional()`, up to 56k DOF).

Requires `using CUDA` first (loads `EarthquakeDiffinitiveCUDAExt`). Without it
this is a `MethodError` rather than a silent CPU fallback, which would hide a
missing `using CUDA` the caller wants to know about.
"""
function fault_stiffness_gpu end

end # module FaultResponse

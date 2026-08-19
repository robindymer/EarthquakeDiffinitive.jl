# Is the fault stiffness `K` (near-)block-Toeplitz?
#
# WHY THIS MATTERS. `fault_stiffness` costs `2·N_Ωf` CG solves — 13,122 at the
# benchmark's Δz = 10 m, and that count is ~98% of the whole run (PERFORMANCE.md
# §5). In a homogeneous whole-space the slip→traction kernel is translation
# invariant: the traction at node i from unit slip at node j should depend only
# on the separation x_i − x_j. If that holds discretely, K is block-Toeplitz
# with Toeplitz blocks (BTTB) and ONE column determines almost all of it —
# collapsing thousands of solves to a handful, and changing the SCALING rather
# than its constant.
#
# WHAT BREAKS IT. The far-field truncation. `u = 0` at a finite distance is not
# translation invariant, so nodes near the boundary see a different medium than
# nodes in the middle. The question is therefore quantitative, not yes/no: how
# large is the departure, and does it shrink when boundary-adjacent nodes are
# excluded? If the interior collapses tightly, a hybrid is possible — Toeplitz
# for the interior, explicit solves for a boundary ring.
#
# Run:  julia --project=. -t auto scripts/k_toeplitz_structure.jl [Δz] [L_fault] [L_normal]
using EarthquakeDiffinitive
using EarthquakeDiffinitive.BP8
using Printf, Statistics

Δz       = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 50.0
L_fault  = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 800.0
L_normal = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 400.0

@info "building model" Δz L_fault L_normal
t0 = time()
m = build_model(; Δz, L_fault, L_normal)
@info "built" seconds = round(time() - t0, digits=1) nf = m.nf

const n2, n3 = length(m.x2), length(m.x3)
const nf = m.nf
@assert nf == n2 * n3

# Ω_f node k ↔ (a,b), a fastest (column-major, matching `omega`'s construction).
node_ab(k) = (mod1(k, n2), (k - 1) ÷ n2 + 1)

"""
Collapse one nf×nf block of K onto separation vectors. Returns, for each
separation with at least `minrep` representatives, the relative spread
(std/|mean|) of the entries that a Toeplitz structure says must be equal.
`ring` excludes nodes within that many cells of the Ω_f edge.
"""
function separation_spread(B; ring::Int=0, minrep::Int=4)
    groups = Dict{Tuple{Int,Int},Vector{Float64}}()
    keep(a, b) = a > ring && a <= n2 - ring && b > ring && b <= n3 - ring
    for j in 1:nf
        aj, bj = node_ab(j)
        keep(aj, bj) || continue
        for i in 1:nf
            ai, bi = node_ab(i)
            keep(ai, bi) || continue
            push!(get!(groups, (ai - aj, bi - bj), Float64[]), B[i, j])
        end
    end
    seps = Tuple{Tuple{Int,Int},Int,Float64,Float64}[]
    for (sep, vals) in groups
        length(vals) >= minrep || continue
        μ, σ = mean(vals), std(vals)
        push!(seps, (sep, length(vals), μ, abs(μ) < 1e-30 ? 0.0 : σ / abs(μ)))
    end
    return seps
end

# Weight the summary by |mean|: separations with tiny kernel values are
# numerically noisy and physically irrelevant, so an unweighted mean of
# relative spreads would be dominated by the far tail.
function summarize(seps, label)
    isempty(seps) && (@printf("%-28s (no separations with enough reps)\n", label); return)
    w = [abs(s[3]) for s in seps]
    rel = [s[4] for s in seps]
    wmean = sum(w .* rel) / sum(w)
    # the self-term and nearest neighbours carry most of K's action
    big = sortperm(w; rev=true)[1:min(10, length(w))]
    @printf("%-28s seps=%4d  weighted spread=%7.3f%%  median=%7.3f%%  top-10-by-weight=%7.3f%%\n",
            label, length(seps), 100wmean, 100median(rel), 100mean(rel[big]))
end

println("\n", "="^100)
@printf("K block-Toeplitz structure:  Δz=%.0f  L_fault=%.0f  L_normal=%.0f  Ω_f=%d×%d\n",
        Δz, L_fault, L_normal, n2, n3)
println("="^100)
println("A perfectly translation-invariant K would give 0% spread. Large spread means")
println("truncation (or genuine inhomogeneity) breaks the structure.\n")

blocks = (("K22 (s2→τ2)", m.K[1:nf, 1:nf]),
          ("K33 (s3→τ3)", m.K[nf+1:2nf, nf+1:2nf]),
          ("K23 (s3→τ2)", m.K[1:nf, nf+1:2nf]))

for (name, B) in blocks
    for ring in (0, 2, 4)
        summarize(separation_spread(B; ring), "$name ring=$ring")
    end
    println()
end

# How much of K would a single column actually reproduce? Build the BTTB
# approximation from the separation means and compare to the true K.
function bttb_error(B; ring::Int=0)
    seps = Dict{Tuple{Int,Int},Float64}()
    counts = Dict{Tuple{Int,Int},Int}()
    keep(a, b) = a > ring && a <= n2 - ring && b > ring && b <= n3 - ring
    for j in 1:nf, i in 1:nf
        ai, bi = node_ab(i); aj, bj = node_ab(j)
        (keep(ai, bi) && keep(aj, bj)) || continue
        k = (ai - aj, bi - bj)
        seps[k] = get(seps, k, 0.0) + B[i, j]
        counts[k] = get(counts, k, 0) + 1
    end
    num = 0.0; den = 0.0
    for j in 1:nf, i in 1:nf
        ai, bi = node_ab(i); aj, bj = node_ab(j)
        (keep(ai, bi) && keep(aj, bj)) || continue
        k = (ai - aj, bi - bj)
        approx = seps[k] / counts[k]
        num += (B[i, j] - approx)^2
        den += B[i, j]^2
    end
    return sqrt(num / den)
end

println("Relative Frobenius error of the best BTTB approximation:")
for (name, B) in blocks, ring in (0, 2, 4)
    @printf("  %-14s ring=%d  %8.3f%%\n", name, ring, 100bttb_error(B; ring))
end

# ---------------------------------------------------------------------------
# The operational test: reconstruct K from only a handful of columns.
#
# This is what actually matters. The spread statistics above say entries with
# equal separation agree; this says whether the separations a few sources
# *cover* are enough to fill the whole matrix. A source at node j supplies the
# kernel at every separation (x_i − x_j), so a corner source covers one
# quadrant of separation space and four corners cover all of it.
# ---------------------------------------------------------------------------
"""
    reconstruct_from_columns(B, sources) -> (relative Frobenius error, uncovered fraction)

Builds the separation kernel using ONLY the given source columns — exactly the
information `2·length(sources)` CG solves would provide — then predicts all of
`B` from it. Uncovered separations are charged as full error, so the number is
honest about gaps rather than quietly skipping them.
"""
function reconstruct_from_columns(B, sources)
    acc = Dict{Tuple{Int,Int},Vector{Float64}}()
    for j in sources
        aj, bj = node_ab(j)
        for i in 1:nf
            ai, bi = node_ab(i)
            push!(get!(acc, (ai - aj, bi - bj), Float64[]), B[i, j])
        end
    end
    kmean = Dict(k => mean(v) for (k, v) in acc)
    num = 0.0; den = 0.0; uncovered = 0
    for j in 1:nf, i in 1:nf
        ai, bi = node_ab(i); aj, bj = node_ab(j)
        k = (ai - aj, bi - bj)
        if haskey(kmean, k)
            num += (B[i, j] - kmean[k])^2
        else
            uncovered += 1
            num += B[i, j]^2
        end
        den += B[i, j]^2
    end
    return sqrt(num / den), uncovered / nf^2
end

corner(a, b) = (b - 1) * n2 + a
const CORNERS = [corner(1, 1), corner(n2, 1), corner(1, n3), corner(n2, n3)]
const CENTRE = [corner((n2 + 1) ÷ 2, (n3 + 1) ÷ 2)]

println("\nReconstruction from a few columns (what a Toeplitz-exploiting build would cost):")
@printf("  %-26s %8s  %10s  %10s\n", "source set", "solves", "rel.error", "uncovered")
for (label, srcs) in (("centre only", CENTRE),
                      ("1 corner", CORNERS[1:1]),
                      ("2 corners", CORNERS[1:2]),
                      ("4 corners", CORNERS),
                      ("4 corners + centre", vcat(CORNERS, CENTRE)))
    for (name, B) in blocks[1:1]   # K22 is representative of the diagonal blocks
        err, unc = reconstruct_from_columns(B, srcs)
        @printf("  %-26s %8d  %9.4f%%  %9.2f%%\n", label, 2length(srcs), 100err, 100unc)
    end
end
@printf("\n  (for reference, the full build costs %d solves)\n", 2nf)

println("""

READING THIS. Under ~1% for the interior (ring=4) would make a hybrid viable:
one solve populates the Toeplitz interior, explicit solves handle a boundary
ring, and the K build stops scaling with N_Ωf. Tens of percent means truncation
dominates the kernel at this domain size and the idea is dead as stated —
though it would then be worth re-testing at the converged domain, since a
larger domain is exactly what restores translation invariance.""")

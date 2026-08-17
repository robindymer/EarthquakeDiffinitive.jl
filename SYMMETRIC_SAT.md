# Making `-HP(D+SAT)P` symmetric

Diagnosis and a working prototype. **Not yet in `src/`** — two verification gates
remain, listed at the end. Reproduce everything here with
`julia --project=. scripts/symmetry_decomposition.jl 9`.

## The one-paragraph version

Diffinitive provides two operators that both approximate `∂u/∂n` at a boundary
with different coefficients: `first_derivative` restricted to the boundary row
(`e·D₁`), and `normal_derivative` (`D̂`), a special-purpose boundary stencil that
comes with the narrow `second_derivative`. Measured gap `‖D̂ - e·D₁‖ ≈ 160` at
order 4. Discrete integration by parts gives `H·D₂ = (symmetric part) +
(boundary term)`, and **the narrow operator's boundary term is written in `D̂`
while the wide operator's is written in `D₁`**. The SAT's job is to cancel that
boundary term; ours is built from `traction_blocks`, which uses `D₁`. So against
the narrow μ terms it cancels the wrong operator, and the leftover `D̂ - D₁` is
the asymmetry. The fix is not to change the SAT but to correct `D₂` at its
boundary rows so its boundary term is expressed in `D₁` — Almquist & Dunham's
"adapted fully compatible" operator.

## Why it is the narrow μ terms specifically

`elastic_blocks` mixes the two schemes:

```
M[j][j] = μ·Σᵢ D₂ᵢ + μ·D₂ⱼ + λ·(D₁ⱼ∘D₁ⱼ)      ← narrow μ, wide λ
M[j][k] = (λ+μ)·(D₁ⱼ∘D₁ₖ)          k ≠ j       ← wide only
```

λ enters only through wide sandwiches, whose boundary term genuinely *is*
`e∘D₁` — correctly paired with `traction_blocks`. μ's diagonal uses narrow
`second_derivative` and gets the same `D₁`-based flux, which is the wrong one.

Measured on one grid, free surface, no interface, no projection:

| | asymmetry |
|---|---|
| `H*D`, interior rows/cols only | 6.96e-17 |
| `H*(D + traction SAT)`, λ only (μ=0) | 1.67e-16 |
| `H*(D + traction SAT)`, μ only (λ=0) | 3.12e-01 |
| `H*(D + traction SAT)`, full | 2.49e-01 |

And on the scalar Laplacian, where no elasticity can confuse matters — **one `H`
throughout**, all four pairings:

| | SAT from `normal_derivative` | SAT from `e∘first_derivative` |
|---|---|---|
| narrow `D₂` | **1.01e-16** | 3.00e-01 |
| wide `D₁∘D₁` | 5.39e-01 | **9.36e-17** |

Perfectly diagonal: each scheme is exact with its own boundary operator and
broken with the other's. μ's 3.12e-01 matches narrow/`D₁`'s 3.00e-01.

## What is *not* the problem

Ruled out by direct measurement, so no effort should go here:

- **`P`.** Exactly symmetric (0.00e+00) and exactly idempotent (0.00e+00).
- **Operator ordering.** `H*P = P*H` exactly, so `-H*P*(D+SAT)*P` and
  `-P*(H*(D+SAT))*P` are the same operator.
- **The interface, the projection, the two-grid coupling.** The defect reproduces
  on a single grid with a plain free surface.
- **`elastic_blocks`.** The bulk is symmetric to 7e-17.
- **The quadrature `H`.** `inner_product` reads one diagonal `H` per stencil set,
  and Mattsson's identity — which contains *both* schemes, `H·D₂` and `D₁ᵀHD₁` —
  holds against that single `H`. Solving it for `R` gives `asym(R) = 8e-17`,
  `min eig(R) = -6e-16`, and `‖R·x‖ = 4e-15`, i.e. Mattsson's `R_xx` reproduced
  to machine precision. A separate narrow-scheme norm is neither present nor
  needed.

## The fix: correct `traction_blocks` (this project's own notebook already does)

**`context/notebooks/elastic_clean.jl`'s `IsotropicTractionOperator` already
makes this distinction, deliberately and with comments.**
`src/Elasticity.jl`'s `traction_blocks` dropped it — which is the whole bug.
(`IsotropicTractionOperator` itself is used nowhere in `src/`, `test/` or
`scripts/`; the package went its own way with `traction_blocks`.)

From [elastic_clean.jl:413-447](context/notebooks/elastic_clean.jl#L413-L447):

```julia
# λnᵢdⱼ
# Note: The diagonal terms Dᵢᵢ(λ) should still use the wide stencil
# second derivative Dᵢ∘λ∘Dᵢ to avoid dispersion errors ...
Dⱼ = first_derivative(g, stencil_set, j)        # λ: first_derivative, ALL i,j

# μnᵢdⱼ
# Note: Here we use narrow-stencil second derivatives for the diagonal terms.
if i == j
    dᵢ = normal_derivative(g, stencil_set, b)   # μ DIAGONAL: boundary derivative
    s = Grids._boundary_sign(component_type(g), b)
    return s * μ ∘ Nᵢ ∘ dᵢ
else
    Dⱼ = first_derivative(g, stencil_set, j)    # μ off-diagonal: first_derivative
end
```

Its TODO even names the subtlety: *"We should really use the boundary derivative
here, rather than the normal derivative, since the operator nᵢdⱼ is something
else than the normal derivative for i == j"* — hence the `s *` undoing
Diffinitive's outward sign.

`traction_blocks` uses `e ∘ first_derivative` for every term
([Elasticity.jl:201-210](src/Elasticity.jl#L201-L210)). Restoring the split —
`traction_blocks_nb` in the diagnostic script — is the fix:

```julia
n = grid_id(bid)
d̂ = outward_sign(bid) .* normal_derivative(g, set, bid)   # boundary derivative
T[n, n] = λ*(e*∂[n]) + 2μ*d̂          # λ wide → ∂;  2μ narrow → d̂
T[n, k] = λ*(e*∂[k])                 # k ≠ n: wide λ
T[i, n] = μ*(e*∂[i])                 # i ≠ n: tangential, no boundary term
T[i, i] = μ*d̂                        # i ≠ n: narrow → d̂
```

**Nothing else changes.** `elastic_blocks` untouched, `P` untouched, `H`
untouched, the SAT's structure and signs untouched. And **no accuracy is given
up anywhere** — the boundary derivative is the operator that ships with the
narrow scheme precisely to be accurate at the boundary.

## Alternative fix: adapt the operator instead (Almquist & Dunham eq 67)

Achieves the same symmetry from the other side — leave the traction operator
alone and change `D₂` so its boundary term is expressed in `D₁`. Documented
here because it corroborates the diagnosis independently, but **it is the worse
option**: it costs one order of accuracy at exactly one grid point per boundary,
and in this problem the fault *is* a boundary — the very nodes where
`FaultResponse` extracts `Δτ`.

Almquist & Dunham, [arXiv:2003.12811](https://arxiv.org/abs/2003.12811), eq (67).
Their §4 names our exact situation: **Mattsson's operators (what
`standard_diagonal.toml` implements) are *compatible* but not *fully
compatible***, i.e. `e₀,Nᵀ D₁ ≠ e₀,Nᵀ D̂`. Their eq (147) states that without full
compatibility the discrete traction operator becomes `T = n·C·(D + ΔD)`, and
they note that proving stability in that case **remains an open problem**.

Their remedy is the *adapted fully compatible* operator — correct `D₂` at its
boundary rows so `D̂` is replaced by `D₁`:

```
D₂_FC = D₂ + H⁻¹e₀e₀ᵀ(D̂ - D₁) - H⁻¹e_N e_Nᵀ(D̂ - D₁)
```

Implemented as `adapted_narrow` in `scripts/symmetry_decomposition.jl`, using
`ElasticitySplitNode._prolongation` (= `-H⁻¹∘e'∘Hᵧ`) so the signs fold in:

```julia
function adapted_narrow(g, set, dir)
    D2 = sparse(second_derivative(g, set, dir))
    D1 = sparse(first_derivative(g, set, dir))
    for bid in boundary_identifiers(g)
        grid_id(bid) == dir || continue          # only boundaries ⊥ dir
        e = sparse(boundary_restriction(g, set, bid))
        d = sparse(normal_derivative(g, set, bid))
        D2 = D2 + ElasticitySplitNode._prolongation(g, set, bid) *
                  (d - outward_sign(bid) .* (e * D1))
    end
    return D2
end
```

**Nothing else changes.** The SAT is untouched. `traction_blocks` stays the
physical traction operator. `P` and `H` are untouched.

## Prototype results

n = 9, order 4. Two sanity checks pass first: the rebuilt operator matches
`elastic_blocks` bit-for-bit (0.00e+00), and the unadapted assembly reproduces
the source `A` (1.26e-17).

| | current | notebook traction | eq (67) |
|---|---|---|---|
| what changes | — | `traction_blocks` | `second_derivative` |
| accuracy cost | — | **none** | one order at boundary point |
| single grid `H*(D+SAT)` | 2.49e-01 | **1.15e-16** | 1.07e-16 |
| two grids `A = -HP(D+SAT)P` | 1.44e-01 | **7.60e-17** | 7.87e-17 |

Both reach machine precision; the notebook's costs nothing. Downstream
properties, measured on the eq (67) variant but structural to symmetry itself:

| | before | after |
|---|---|---|
| Galerkin `SᵀAS` asymmetry | 1.98e-01 | **8.90e-17** |
| negative eigenvalues of `SᵀAS` | — | **0 / 2205** |
| condition number κ | 1.99e+02 | **1.18e+02** |
| CG | stalls: 5000 its, res 3.7e-02 | **73 its, res 9.0e-11** |

So `A` becomes symmetric positive definite and CG converges in 73 iterations,
with the residual recomputed from scratch (CG's internal recursive residual is
only trustworthy once the matrix really is symmetric, which it now is).

`H*(D_adapted+SAT)` *before* projection reads 8.68e-01, worse than unadapted.
That is expected: the far-field faces carry no SAT — they are Dirichlet through
`P` — so their boundary terms stay uncancelled until `P` zeroes those rows. Only
the fault boundary needs consistency, and after adaptation it has it.

## Second, independent change: the reduction

Even with symmetric `A`, `reduced_solve` cannot feed CG. `factorize_reduced`
builds `E·A·S` — columns summed over merge pairs, rows merely *selected* — which
is not a congruence transform and stays asymmetric even for symmetric `A`. The
Galerkin form `SᵀAS` is required. Low-risk: the prolongation `S` is already
written (`prolongation` in `scripts/split_node_spd.jl`), and solving `SᵀAS`
directly agrees with the current LU path to 2.6e-15.

## Verification gates

Gate 1 **passed** — see below. Gate 2 not done.

### Gate 1: the interface physics — PASSED

`scripts/verify_notebook_traction.jl` re-runs
`test/elasticity_split_node_test.jl`'s interface checks against both traction
operators, using each one *both* in the SAT and for extracting σ_i1:

| | n=11 current | n=11 notebook | n=15 current | n=15 notebook |
|---|---|---|---|---|
| eq 3 no opening | 0 | 0 | 0 | 0 |
| eq 4 `[u₂]` − slip | 0 | 0 | 0 | 0 |
| eq 4 `[u₃]` | 0 | 0 | 0 | 0 |
| eq 6b σ₂₁ continuity | 4.5e-15 | 2.9e-15 | 1.0e-14 | 2.8e-14 |
| eq 6c σ₃₁ continuity | 7.1e-15 | 1.1e-14 | 1.5e-14 | 2.2e-14 |
| antisym split | 0 | 0 | 0 | 5.6e-16 |
| σ₁₁/σ₂₁ | 6.47e-02 | 6.29e-02 | 3.63e-02 | 3.62e-02 |
| `max diag K` (<0) | -4.51 | -5.07 | -6.31 | -7.09 |
| `K` asymmetry | 2.85e-02 | **3.29e-02** | 2.42e-02 | **2.79e-02** |

eq 3/4 and the antisymmetric split are exactly zero in both, as they must be —
those are imposed by `P` and `χ`, which this fix does not touch. Traction
continuity stays at machine precision. σ₁₁/σ₂₁ halves from n=11 to n=15 in both,
so the σ₁₁ → 0 convergence is preserved. `diag(K) < 0` holds.

**One metric does not improve: `K`'s asymmetry is consistently ~0.4 points
worse** at both resolutions. Worth understanding rather than dismissing:
reciprocity requires the *forcing* operator `χ` and the *extraction* operator
`T` to be mutually adjoint, which is a separate condition from `A` being
symmetric — so changing `T` alone shifts that balance. Both values are also
~15× worse than the 0.19% PROGRESS.md records at production size, because
`L=1, n=11..15` is a severely truncated domain, so truncation dominates here.
The takeaway: **the payoff of this fix is `A`'s symmetry (hence CG), not better
tractions.** Do not expect `K` to become symmetric.
2. **`Δτ` changes, because `traction_blocks` is also what `FaultResponse` uses
   to extract physical traction.** The notebook fix alters the μ-diagonal terms
   of the traction operator, so `K` and hence peak `V` will shift. Both `d̂` and
   `e∘∂ₙ` are consistent approximations of `∂ₙ`; the boundary derivative is
   typically *more* accurate at the boundary (order q+1 vs q in Mattsson's
   construction), so this may well be an improvement — but it needs measuring
   against the runs already in `output/`, not assuming. Note this gate applies
   to the notebook fix and *not* to eq (67), which leaves `traction_blocks`
   alone; that is eq (67)'s one advantage.

Then: `diag(K) < 0` and `K` symmetric to far better than the current 0.19%
(`test/fault_response_test.jl`, `test/bp8_test.jl`), and a production run whose
peak `V` and final slip move less than the documented resolution spread.

## Note on the reference

Almquist & Dunham impose *everything* by SAT — Robin (including traction),
displacement, and interface conditions — and explicitly contrast this with
Duru & Virta (SAT for traction, strong injection for displacement) and
Petersson & Sjögreen (ghost points). This project's `-HP(D+SAT)P` is a
*projection* method: `P` for continuity, SAT for traction. So the paper is not
a drop-in template, but its §4 diagnosis and eq (67) transfer directly, and the
two-term SAT it needs in §5.2 is for *displacement* conditions — not our case,
since `P` handles continuity and the SAT carries traction only.

## Corrections to an earlier draft of this file

For the record, two claims in the first version were wrong:

1. It derived a replacement flux operator `F` by peeling the left factor of
   every wide sandwich uniformly, concluded `F` differed from the physical
   traction at O(1), and recommended *against* implementing the fix on that
   basis. The arrangement was wrong; a correct one exists, and in any case
   eq (67) achieves symmetry without changing the flux operator at all.
2. It claimed the λ-only test cleared the extra-μ discrepancy in the wide cross
   terms. It cannot — that difference vanishes at μ=0, so the test is blind
   to it.

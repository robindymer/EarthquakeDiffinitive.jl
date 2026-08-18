## Mixed - what happens for boundary traction terms
On the continious level, there is only one normal derivative.
But how it is calculated on the discrete level differs for the narrow
and wide scheme. The wide FD scheme simply calculates e*D1, while the D1_narrow
(normal_derivative in Diffinitiv) is a different entitity with its own stencil, that
is derived from the SBP property of narrow D2 SBP.
So the narrow scheme uses normal_derivative, while the wide scheme uses e*D1. Both want to compute the same thing, but does it in different ways, for SBP to be fulfilled.

## The idea behind -HP(D+SAT)P \mathbf{u} = HP(D+SAT)\boldsymbol{\chi}(\vec{s})
The χ(s) exists because we need to enforce the +/- s/2 jump condition. This isn't quite Dirichlet: only the antisymmetric (jump) part of the fault DOFs is prescribed data — the symmetric (average) part stays a free unknown, solved for by the physics. P alone can't supply this, since it forces both sides to the same averaged value, i.e. Pu can only ever represent zero jump. Therefore we modify U with this. We then do the SAT for Neumann conditions, D for the discrete part, and P on everything to make sure we're in the correct space once again.

Regarding why we don't do plain injection: it breaks symmetry, because it overwrites a row without touching the matching column. P(D+SAT)P avoids this by transforming both sides with the same (symmetric) P — a congruence transform, PᵀMP, which preserves symmetry when M does. Same trick reused later in SᵀAS.

## The SAT sign convention
Generic SAT: -H^-1 e' H (t_out - data)
t_out = d typically, ourward-normal traction, and data is what t_out should equal
At g_-: fault = t_out- = tau_-
At g_+: fault = t_out+ = -tau_+ (minus since the outward normal is at -x1 instead of +x1)
Then remember, NIII, that t_out- + t_out+ = 0 => data- = -t_out+, data+ = -t_out-

SAT_- = -H^-1 e' H (tau_- - tau_+) = -H^-1 e' H (-tau_+ + tau_-)
SAT_+ = -H^-1 e' H (-tau_+ + tau_-)

If we mess up the sign of e.g. SAT_+ we get
SAT_+ = -H^-1 e' H (tau_+ + tau_-), and that drives the system toward tau_- = -tau_+
=> no good

So recap: SAT for traction continuity and P for u- = u+ condition directly

## Why P averages u1, and the fault-edge ring
Traction (SAT) only constrains stress, i.e. derivatives of u — not u itself.
A uniform shift of u1 on one side changes no derivatives, so it changes no
traction: traction continuity alone cannot stop the fault from opening a gap
or interpenetrating. So u1 needs its own value-based (Dirichlet-like)
condition, which P's averaging supplies.

u1 uses the exact same P+χ mechanism as the slip components (u2,u3), just
with χ=0 instead of ±s/2. U's jump at a fault pair equals χ's jump,
regardless of avg(u):
- j=2,3: χ=±s/2 → jump = s (prescribed slip)
- j=1:   χ=0    → jump = 0 (no-opening, BP8 eq 3)

Ring nodes (where the fault plane meets the domain's truncation boundary in
x2/x3) are excluded from fault pairing. They carry the far-field u=0
Dirichlet instead (P row all-zero). If pairing ran on them too, it would
overwrite that all-zero row with an averaging row — turning a pinned
boundary DOF into a free-floating one, silently wrong, no error raised.

## Why CG works despite A being singular
A = -HP(D+SAT)P is singular by construction: the right-hand P annihilates
null(P) (far-field DOFs, and the antisymmetric/jump half of every fault
pair), so null(P) ⊆ null(A) trivially.

Normally a singular A rules out CG. Three things make it fine here:
1. The system is consistent: b ⊥ null(A). For symmetric A this means
   b ∈ range(A). Concretely, for v ∈ null(P): vᵀb = vᵀHP(D+SAT)χ =
   (HPv)ᵀ(D+SAT)χ = 0, using HP=PH and P=Pᵀ, since Pv=0.
2. Starting from x₀=0, every CG iterate lives in span{b, Ab, A²b, ...} ⊆
   range(A) — the null space is never excited in the first place, no
   special handling needed.
3. On range(A), A is genuinely positive definite, so CG converges there at
   the normal rate. Even if something did leak into null(A), it wouldn't
   matter: the only use of the solve is U = Pu + χ, and P annihilates
   null(P) ⊇ anything that leaked — so null-space content is not just
   small, it's irrelevant to the answer.

Same idea as the P(D+SAT)P congruence trick: consistency + starting inside
range(A) is what replaces the "eliminate the null space by hand" step that
factorize_reduced does explicitly for the direct solver.
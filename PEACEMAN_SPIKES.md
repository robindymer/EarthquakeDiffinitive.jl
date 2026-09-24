# Slip-rate spikes at the BP8-PW well node

Run: `run_bp8.jl pw 10 1150 1150` (Δz = 10 m, cached exact `K`, QNDF, `saveat = 200`),
compared against the GS run of the same configuration. Equation numbers refer to
`context/SEAS_BP8_Benchmark_Description.pdf`.

**TLDR.** The spike in log₁₀V at `fltst_strk+000dp+000` (peak −3.99 at 73.2 h, which
is also the global `max_slip_rate`) is **not in the slip**. The slip rate taken from
differences of saved slip is flat at ~9e-8 m/s throughout. The spike exists only in
the V *derived* at the save points, at the one node where σ̄ sits on the `σ̄_min`
floor, where V(τ, θ) is extremely ill-conditioned.

## 1. Pressure: log(1/r), not 1/r

Both variants share the prefactor `P₀ = q₀/(4παβφ)` = 0.003/(4π·0.05·1e-8·0.1) = **4.78 MPa**.

- **GS, eq. 21 at the origin:** `p(0,t) = P₀·ln((2L_g² + 4αt)/(2L_g²))`. This stays
  finite at r = 0 because the source is spread over L_g = 50 m. At t = 100 h it gives
  4.78·ln(77000/5000) = **13.1 MPa**, and the run shows 13.0.
- **PW, eq. 25:** `p = P₀·E₁(r²/4αt)`. For small arguments E₁(x) ≈ −γ − ln x, so
  p ≈ P₀·[ln(4αt/r²) − γ]. That is **log(1/r)**. The 1/r belongs to the gradient: the
  Darcy flux of a 2D point source is Q/(2πr).
- **PW well node:** the Peaceman cell (eq. 22–23) reports eq. 25 evaluated at
  r_e = 0.268·Δz = 2.68 m (`PorePressure.SBP4_RE_FACTOR`). That gives
  4.78·(ln(72000/7.18) − 0.577) = **41.2 MPa**, and the run peaks at 41.21 MPa. The
  pressure is exactly as specified.
- **Where p equals σ:** p = σ = 25 MPa at r ≈ 14.7 m. The node at (0, 0) crosses it
  after ~4 h, and the node at (10, 0) after ~45 h (measured: 25.0 MPa at 46 h). From
  then on σ̄ = σ − p (eq. 7) is floored at `σ̄_min` = 1 kPa at those nodes.

## 2. Friction at a floored node (eq. 10, 12)

Force balance `τ − ηV = σ̄·f(V,θ)` with the regularised law of eq. 12 inverts to

    V = 2V₀ · e^(−ψ/a) · sinh((τ − ηV) / (a·σ̄)),    ψ = f₀ + b·ln(V₀θ/D_RS)

- **GS:** a·σ̄ ≈ 0.016 × 12 MPa ≈ 190 kPa, so V responds gently to τ.
- **Floored PW node:** a·σ̄ = 0.016 × 1 kPa = **16 Pa**. V then changes by one decade
  for every 37 Pa change in τ, or for every 1.6-decade change in θ (b/a = 0.625).
- **Physically:** the floored node has no strength. τ falls to ~0.6 kPa and the node
  slips at whatever rate its neighbours impose through `K`. This matches the hourly
  profiles, where V at x₂ = 0, 10, 20 and 30 m agrees to within 0.05 decades from
  20 h to 99 h.

## 3. What the output shows

Station (0, 0) around the peak (every third 200 s row):

| t (h) | reported log₁₀V₂ | log₁₀(Δslip/Δt) | τ₂ (Pa) | log₁₀θ |
|---|---|---|---|---|
| 72.61 | −5.49 | −7.03 | 609 | 2.27 |
| 73.00 | −4.66 | −7.03 | 619 | 1.37 |
| 73.22 | **−3.99** | −7.04 | 628 | 0.70 |
| 73.28 | −7.04 | −7.03 | 547 | 2.04 |

- The actual slip rate is flat. The reported V climbs three decades and then drops
  back in a single row, while θ drifts down and snaps back, and τ drifts by +80 Pa
  and snaps back.
- Those drifts account for the size of the spike: 80 Pa / 16 Pa ≈ 5 e-folds
  (2.2 decades), plus 1.6 decades of θ × 0.625 (1.0 decade), gives about 3 decades.
- GS has no spike: σ̄ never approaches the floor, so a·σ̄ is ~10⁴ times larger and
  the same drifts do not matter.

## 4. Source of the drifts — confirmed: the `saveat` interpolant

The `saveat` points between integrator steps are filled in by the integrator's
interpolant. With 3,369 QNDF steps over 720 h, a step often spans several save
points. The pattern fits a step boundary: a smooth drift inside the step, then a
snap back at 73.28 h.

**Test** (`run_bp8(...; land_on_saveat)`, which puts the save times in `tstops`).
Δz = 10 m, (400, 400), `toeplitz`, 0–100 h, `saveat = 200`:

| `land_on_saveat` | steps | well max log₁₀V derived | well max log₁₀(Δslip/Δt) | global `V_max` |
|---|---|---|---|---|
| false | 2,711 | −5.62 | −6.73 | 8.4e-6 at 63.9 h, node (0, −10) |
| true | 4,776 | **−6.73** | −6.73 | 1.8e-6 at 0.5 h, well (pre-floor) |

The spikes are gone, and so is the spurious global `V_max`. `land_on_saveat` now
defaults to on for PW. Cost is 1.8× the steps, not the 4× guessed.

Caveat: the row-by-row max |derived − Δslip| (t > 10 h) fell from 14.7 to 4.2
decades but is not zero. That metric compares |V| against the x₂ component only,
and takes the log of Δslip over near-stationary rows, so it is loose. It is not
evidence of a remaining spike.

## 5. Why the floor is needed at all

**Measured:** PW at Δz = 50 m, `:toeplitz` stiffness, `cache=:off`, 0–120 h,
`saveat = 600`:

| `σ̄_min` | result | steps | wall time |
|---|---|---|---|
| 1 kPa | Success | 721 | 7 s |
| none (−Inf) | **retcode Unstable at t = 83.8 h** | 504 | 49 s |

The two runs are identical up to 83.8 h, which is when σ̄ at the well node first goes
negative (lowest value −523 Pa; PROGRESS.md has the zero crossing at 82–84 h).

**Why it fails:** for σ̄ < 0, `g(V) = ηV + σ̄·a·asinh(u) − T` is no longer monotone,
and `solve_slip_rate` assumes it is. Its log-space Newton diverges and returns NaN for
every σ̄ < 0 tried (−523 Pa to −16 MPa), and the integrator shrinks its step until it
gives up.

**What the equations would give with a working solver:** the root still exists and
is unique for T > 0, since g(0) = −T < 0 and g → +∞. But the friction term has
flipped sign, so friction now pushes the node forward and only radiation damping
brakes it: **V ≈ (T + |σ̄|·f)/η**. Found by bisection with θ = 100 s,
η = 4.62 MPa·s/m:

| σ̄ | T | V |
|---|---|---|
| −523 Pa | 600 Pa | 2e-4 m/s |
| −1 MPa | 600 Pa | 0.17 m/s |
| −16 MPa (Δz = 10 m well node) | 600 Pa | **2.8 m/s** |

That is seismic slip in a quasi-static, velocity-strengthening benchmark. Reasoned,
not measured: the node's own stiffness would push T through zero within milliseconds,
the slip direction would flip, and the same speed would repeat in reverse, so there is
no steady state.

**So the floor is the physics.** When p > σ the fault would open (the case eq. 3
excludes), and an open crack carries no shear traction. `max(σ̄, 0)` is that limit,
and the run reproduces it: τ ≈ 0.6 kPa at the well. `σ̄_min` = 1 kPa is only the ε
that keeps the friction law defined. The §3 spike is a cost of evaluating V near
that limit, not a reason to remove the floor.

## Consequences

- PW's reported global `max_slip_rate` (peak −3.99) and V at station (0, 0) are, at
  their peaks, this artefact and not physics. Every other PW output checked (slip,
  pressure, the off-well stations) is unaffected.
- `PROGRESS.md` "σ̄_min as a regularization" found, at Δz = 50 m and 25 m, that global
  `V_max` never came from the floored node. At Δz = 10 m the reported `V_max` does,
  but only through this derived-V spike. Update that entry once the `tstops` test has
  settled the mechanism.

## TODO

In this order. The σ̄_min sweep comes last: a larger floor would hide the spike
(a·σ̄ grows, so V becomes less sensitive to τ and θ) without fixing it, and
PROGRESS.md "σ̄_min as a regularization" already argues against changing the physics
this way.

- [x] **`tstops` test.** Done on (400, 400), see §4: the spikes disappear, and
      `land_on_saveat` is now the PW default. **Remaining:** regenerate the PW
      outputs (`pw 10 1150 1150`, `pw 10 1600 1600`) with the new default.
      Original plan: Add an option to `run_bp8.jl` that passes `tstops` equal to
      the `saveat` times, then rerun `pw 10 1150 1150`. Compare the reported log₁₀V₂
      against log₁₀(Δslip/Δt) at station (0, 0) around 72–74 h (§3), and compare the
      global `max_slip_rate`. Expect ~4× more steps (≈ 1 h).
  - If the spikes disappear: make this the default for PW runs and regenerate the PW
    outputs.
  - If they remain: the drift is in the step solutions themselves; look at the
    nonlinear-solve tolerance at floored nodes.
- [x] **Update `PROGRESS.md`** "σ̄_min as a regularization" and "Known limitations" 3
      with the Δz = 10 m finding and the mechanism, once settled.
- [ ] **σ̄_min sensitivity (only after the above).** Rerun with 1 kPa, 10 kPa and
      100 kPa and check that slip, moment rate and the off-well stations do not depend
      on the ε. σ̄_min can now be set with `BP8_SIGMA_MIN=<Pa>` for `run_bp8.jl`;
      output goes to `..._smin<N>`, and the `K` cache is reused.

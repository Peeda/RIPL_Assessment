## What the video and the description actually say

The pair starts with almost no clear space between their nearest **faces**. The wrist is frozen, so the fingers cannot square up to the slot; the hand descends beside cubeA, a finger or the hand body catches cubeB on the way down or on the way up, and cubeB is shoved. After that the stack either lands off-target or has nothing to land on. Settling is fine; the defect is entirely in **approach, descent and lift-out**.

So the behaviour to shape is *clearance of the hand from cubeB while the hand is low and cubeA is not yet over cubeB*. Not "cubeB stayed still" — that outcome term is maximised by never approaching (hacking.md #6).

## The clearance term, and why it is expressed relative to cubeA

`intrusion = height_gate * side * not_over`, each factor in [0,1]:

* `side` = where the tcp sits along the horizontal **cubeA→cubeB axis, measured from cubeA's centre**. `p_side > 0` means the hand is in the slot between the cubes; `p_side ≤ -8 mm` means it is on the far side of cubeA and the penalty is fully relieved. 8 mm is chosen because it is the offset a bounded residual can accumulate, and because cubeA (±20 mm) is still inside the Panda's ±40 mm finger span at that offset — so the offset the term rewards is one that can still grasp.
  I deliberately did **not** use "horizontal distance from tcp to cubeB", which is the obvious formulation, because that quantity is relieved by *knocking cubeB away* — the reward would then pay for the failure. Measuring relative to cubeA means shoving cubeB only rotates the axis; it does not reduce the penalty.
* `height_gate` = 1 when the tcp is at or below cubeB's top face, fading to 0 at 60 mm above it (Panda fingers are ~50 mm long above the tcp, so at 60 mm nothing on the hand can reach cubeB). This makes "lift clear before translating" the cheap way out, which is exactly the fix for the catch-on-the-way-up case.
* `not_over` = 0 when cubeA is within 25 mm of cubeB horizontally, 1 beyond 45 mm. Without this, the *intended* final descent — hand directly over cubeB, low — would be punished. With it, the penalty is only ever active when the hand is low near cubeB while cubeA is somewhere else, which is precisely the intrusion.

The term enters **multiplicatively** on each stage's shaping (`clear = 1 - 0.35*intrusion`), never as a standing bonus. Hovering far from everything therefore earns ~0, not a clearance payment (hacking.md #3), and the penalty is largest exactly where it matters (close in, low down).

Local check of the trade-off at grasp height, tight config: reward ≈ `2·(1−10δ)·(0.825+21.9δ)` for a lateral offset δ away from cubeB; this is maximised at δ ≈ 8 mm (1.84 vs 1.65 at δ=0) and falls again by 16 mm. So the shaping asks for a few-millimetre far-side bias, not a refusal to approach — and the grasp rung (+3) still dominates any offset that would spoil the grasp.

## The ladder, with bands

| stage | condition | value | band |
|---|---|---|---|
| A approach | not grasped, not stacked | `2 · reach · clear` | **[0.00, 2.00]** |
| B carry | grasped, not stacked | `3 + 2 · place · clear` | **[3.00, 5.00]** |
| C stacked | `is_cubeA_on_cubeB`, not success | `6 + (ungrasp + static)/2` | **[6.00, 7.00]** |
| D success | `success` | `8` | **8.00** |

Bands are disjoint with a 1.0 margin between each rung, and `clear ∈ [0.65, 1]` cannot lift a rung. Grasp-and-hold tops out at 5.0; a completed stack is 8.0 and a stack still held in the gripper is at most 6.75 (`ungrasp ≈ 0.5` with fingers closed on a 40 mm cube) — so letting go is worth +1.25 and is never optional (hacking.md #1, #2).

`reach` uses tanh at both 5 and 15, `place` at both 5 and 20, so there is still gradient at the millimetre scale a residual operates on; the built-in's single `tanh(5·d)` is nearly flat inside 10 mm, which is where the off-target landings in this failure mode live. `place` is the full 3-D distance to `cubeB + [0,0,0.04]` (hacking.md #5), and the goal is unambiguously *cubeA above cubeB* (#4). Stage C is deliberately the built-in's own formulation: the description says settling is already fine, so there is nothing to improve and nothing new to exploit there.

No mutation: `pose.p` is a view into the sim buffer, so the goal is formed arithmetically (`dz = pA_z - pB_z - 2·hs`) rather than by cloning-and-editing.

## Sampler: the axis, its nominal distribution, and how thin the region is

Derived from `_initialize_episode`, not guessed:

* Both cubes get one shared offset `xy ~ U[-0.1,0.1]²`, which cancels out of the relative pose, plus independent placements in `[-0.1,0.1]×[-0.2,0.2]`. So Δ = cubeA−cubeB has a triangular x (width ±0.2) and trapezoid-ish y (width ±0.4); density at Δ=0 is `5 × 2.5 = 12.5 m⁻²`, so `P(|Δ| < r) ≈ 12.5πr²` for small r.
* The rejection radius is `‖(0.02,0.02)‖+0.001 = 0.029284`, so `d ≥ 0.058568` and 13.5% of raw draws are rejected.
* Face gap along the centre line: `gap = d − s_A − s_B`, `s = 0.02(|cos ψ|+|sin ψ|) ∈ [0.020, 0.028284]`. Nominal `E[s] = 0.02546`, `E[|Δ|] ≈ 0.18`, so **the nominal gap is ~130 mm** — the policy sees a comfortable slot almost always.
* Tightest physically possible gap: `0.05857 − 2(0.028284) = 0.0020 m`, and it requires *both* cubes corner-first. That is the extreme, and my sampler reaches it: `gap` is drawn from 0.0005 upward and then floored by the separation constraint, so the tightest rows land at ~2 mm. No safety margin left inside the region.
* Nominal frequency: `P(gap ≤ 12 mm)` ≈ 1.8–2%; restricted further to both ψ within 18° of corner-first (my region) ≈ **0.7% of nominal draws**. Including my 12% easier tail (gap up to 24 mm) the whole region is ≈ **2% of nominal draws**. That is a thin corner, consistent with a base policy that is competent everywhere else.

Variety, so the residual transfers to held-out draws from the same corner: the pair direction α is uniform on [0,2π); each cube's yaw is corner-first ±18° *and* randomised over the four 90° symmetry branches, so absolute yaw covers the range the base policy was trained on; the pair midpoint is drawn from the nominal single-cube marginals (triangular in x, trapezoid in y) over the whole table; gap is continuous over [0.002, 0.024] with 88% of mass in the tight core. Nothing is fixed except "corner-first and nearly touching".

Support: bounded `for` (48 iterations) rejection on box (`|x|≤0.2`, `|y|≤0.3`) and reach (`≤0.868` from the base at (−0.615,0)) — the box's far corner is 0.8685, so the reach test does bite and is applied rather than avoided; the initialisation draw is valid by construction and is the fallback, then a final clamp guarantees no invalid row ever leaves the function. Separation is `≥ 0.05860` by construction on every row.

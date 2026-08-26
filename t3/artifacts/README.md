# T-III generated artifacts

Everything the LLM produced, per failure mode, per generation. **This is a
deliverable** — the prompts, the responses and the generated code are what the
assignment asks to be documented, and T-IV trains against the code in here.

Previously these lived under `checkpoints/`, where `.gitignore`'s
`checkpoints/*` silently swallowed them: the entire T-III output was untracked
and one `rm` from gone. `t3/artifacts/` is not ignored and is the path
`t3/run.sh` already defaults to (`GEN=2` -> `t3/artifacts/<mode>_gen2`).

| | |
|---|---|
| `gap/`, `farb/` | **generation 1** |
| `gap_gen2/`, `farb_gen2/` | **generation 2 — what T-IV trains against** |

Each directory holds `reward.py`, `sampler.py`, `rationale.md`,
`uncertainties.md`, the assembled `prompt.txt` and `system.txt` the model
actually saw, `frames/` (its video input), the raw `response*.json`, and
`manifest.json` / `prompt_manifest.json` / `request.json` for provenance.

```bash
T3_RUN=t3/artifacts/gap_gen2 MODE=gap bash t3/run.sh check
```

## Why gen2

Both generations are kept because the difference between them **is** the
"manual effort to elicit desired results" deliverable. gen2 was produced after
`t3/prompts/calibration.md` was added and `spec.REACH_MAX` was corrected from a
hardcoded 0.8 m (which forbade the model from sampling a fifth of the `farb`
region it was being asked to target).

Sampler hit rate against `t2/geometry.MODES[tag]`, 4096 draws under
`torch.manual_seed(0)`, base rate from the committed 25,000-row
`t2/results/seeds.csv`:

| run | mode | hit rate | base | enrichment | sd(x)/sd(y) |
|---|---|--:|--:|--:|---|
| `gap` | gap | 0.969 | 0.0468 | 20.7x | 0.064 / 0.089 |
| **`gap_gen2`** | gap | **0.897** | 0.0468 | **19.2x** | **0.081 / 0.130** |
| `farb` | farb | **0.133** | 0.0337 | 4.0x | 0.071 / 0.149 |
| **`farb_gen2`** | farb | **0.996** | 0.0337 | **29.5x** | 0.078 / 0.171 |

**gen1 `farb` is arithmetically incapable of hitting its own region.** Its
sampler draws `rB ~ U(0.695, 0.775)` while the mode's clause is
`dist_B >= 0.76`, so `P = 0.015/0.080 = 0.1875` is a hard ceiling — and the
measured region has `dist_B` median 0.783, max 0.820, i.e. the sampler's entire
range sits below the region's median. gen2 sets `D_B_MIN = 0.78`, above the
threshold, and lands at 0.996.

**gen1 `gap` hits more often but is worse.** 0.969 against gen2's 0.897, yet
gen2 is the one to train on: its xy spread is ~40% wider on both axes, so it
covers the region rather than one corner of it. A sampler that returns a single
configuration would score a hit rate of 1.000 and teach a residual one initial
state — spread is the check that catches that, and it is why
`spec.SAMPLER_SD_MIN_XY` exists.

All four load clean through the static checker and are deterministic under a
fixed torch seed:

```bash
for d in gap farb gap_gen2 farb_gen2; do python3 t3/loader.py t3/artifacts/$d; done
```

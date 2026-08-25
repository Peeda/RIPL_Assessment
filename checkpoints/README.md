# The frozen base policy

`stackcube_rgb_spatial_800demos.pt` — 33 MB, committed on purpose.

This is **the T-I deliverable and T-IV's frozen base**. Every T-II number in
`notes/t2-failure-modes.md` came from these weights, and every T-IV before/after
must come from them too. `t2/run.sh` defaults to it, so the harness runs on a
fresh pod with no environment variable and no rsync from a pod that may no
longer exist.

| | |
|---|---|
| env | `StackCube-v1`, `pd_ee_delta_pos`, `physx_cpu` |
| obs mode | `rgb` (128×128 base camera + wrist camera, `panda_wristcam`) |
| encoder | **spatial** — `pool_feature_map=False`, `patches/0001` |
| demos | 800 motionplanning, replayed with `--use-first-env-state` |
| schedule | 100k iters, cosine with 500 warmup, peak lr 1e-4 |
| `success_once` | **0.737** (SD 0.078, 3 × 100, held-out seeds 6000–6299) |
| sha256[:16] | `b6195b5a7d72c396` |

## It contains `ema_agent` only

`train_rgbd.py` saves two full copies of the weights, `agent` and `ema_agent`,
which is why the original file is 67 MB. `ema_agent` is what `train.py`
evaluates with, so it is what produced every reported number, and
`harness.load_weights` reads `sd.get("ema_agent") or sd.get("agent")` — the raw
`agent` weights are never used. Dropping them halves the file and changes
nothing.

The two are genuinely different models (checked, not assumed), so this is a
deliberate loss of the non-EMA arm rather than deduplication. If you ever need
it, retrain — do not add it back here.

## Why this is in git at all

`CLAUDE.md` says nothing large goes into git, and that rule stands: datasets,
training runs and wandb output stay out, and `.gitignore` still blocks `*.pt`
everywhere including this directory. The negation is **by filename**, not
`!checkpoints/*.pt`, so a training run writing `best_eval_success_at_end.pt`
beside this file will not be committed by accident.

The exception is justified by what this file is: 33 MB, written once, never
changing, and the single artifact that makes every downstream number
reproducible. The alternative — an env var pointing into a pod's filesystem —
means the evidence base cannot be re-derived once that pod is terminated, and
RunPod pods are terminated routinely.

**It is still a one-way door.** `git rm` does not remove a blob from history;
undoing this means rewriting history again. Do not extend the exception without
that in mind.

## Checking it is the right file

`harness.inspect_ckpt` infers the encoder variant from the weights rather than
from the path, and prints it. The line to look for is:

```
visual encoder:  spatial (8x8 map kept)  [fc in=8192]
```

`fc in=128` means the pooled encoder — the wrong arm, the one that plateaus at
0.43. See the RGB-encoder section of `CLAUDE.md`.

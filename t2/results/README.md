# T-II evidence base

Every number in `notes/t2-failure-modes.md` comes from these files. Committed so
the analysis reproduces from the repo alone, with no ManiSkill install and no
pod. `t2/geometry.py` and `t2/verify.py` need nothing at all; `t2/report.py`
needs numpy and matplotlib:

```bash
python3 t2/test_geometry.py                      # checks the geometry itself
nix-shell -p "python3.withPackages(ps: [ps.numpy ps.matplotlib])" \
  --run "python3 t2/report.py --discovery t2/results/nominal.csv"
```

**These are the DISCOVERY pass.** They are what identified the two failure
modes; they are not the per-mode deliverable. That is produced fresh by
`t2/eval_modes.py` on seeds above 10,000 — disjoint from everything here,
because re-measuring a region on the episodes that found it measures noise.
See `t2/README.md`.

The `region_near` / `region_far` files below are **superseded**: they are
separation-band passes from before the axes were corrected to `face_gap` and
distance-from-base. Kept as the record, not quoted as results.

| file | episodes | seeds | what it is |
|---|---|---|---|
| `nominal.csv` | 1200 | 1000–2199 | The discovery pass. **The only one that estimates P(success)** — 0.713. |
| `region_near.csv` | 400 | ≥2212, sep < 80 mm | **Superseded.** The separation-band pass for mode 1, before `face_gap` replaced the axis. |
| `region_far.csv` | 400 | ≥2203, sep > 260 mm | **Superseded.** Mostly a reach effect seen through a correlated variable. |
| `t1_seed{1,2,3}.csv` | 3 × 100 | 6000–6299 | **T-I: the assignment's 100 × 3.** Held out. |
| `t1_trainseed{1,2,3}.csv` | 3 × 100 | 0–299 | The *contaminated* pass — these are demonstration seeds. Kept because it is the memorisation measurement, **not** because it is a T-I number. |
| `seeds.csv` | — | 0–7999 | The policy-free seed index. Reset-only, no rollouts. Rebuild to 25,000 before running `eval_modes.py`, which draws above seed 10,000. |
| `region_*_seeds.csv` | — | — | The seed lists each targeted pass drew, for audit. |
| `*_manifest.json` | — | — | GPU, git sha, ManiSkill sha, gymnasium version, ckpt path, wall clock. |
| `run.log` | — | — | Pod stdout for the whole finishing pass. |

Two things that are easy to get wrong when reading these:

- **A targeted pass is not a sample of the nominal distribution.** It
  over-samples its region by construction, so `region_*.csv` gives a rate
  *conditional* on that region. Only `nominal.csv` gives the unconditional one.
  Pooling them is valid **within a separation bin** and nowhere else.
- **The targeted passes repeat each seed twice**, so 400 episodes carry the
  information of 200 states. Wilson assumes independent draws; the honest n for
  a region claim is the distinct-seed count. `analyze_rollouts.py` says so in
  its output and sizes the figure's error bars by seeds rather than episodes.

The `.npz` sidecars (per-episode traces, initial/terminal sim states) stayed on
the pod — see `notes/t2-failure-modes.md` for what they hold and
`setup/transfer.sh` for pulling them.

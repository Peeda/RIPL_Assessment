# T-II evidence base

Every number in `notes/t2-failure-modes.md` comes from these files. Committed so
the analysis reproduces from the repo alone, with no ManiSkill install and no
pod — `t2/analyze_rollouts.py` needs only `numpy` and `matplotlib`:

```bash
uv venv ~/.venv-ripl && uv pip install --python ~/.venv-ripl/bin/python numpy matplotlib
~/.venv-ripl/bin/python t2/analyze_rollouts.py \
    t2/results/nominal.csv t2/results/region_near.csv t2/results/region_far.csv \
    --figdir figures
```

| file | episodes | seeds | what it is |
|---|---|---|---|
| `nominal.csv` | 1200 | 1000–2199 | The discovery pass. **The only one that estimates P(success)** — 0.713. |
| `region_near.csv` | 400 | ≥2212, sep < 80 mm | Mode 1, pre-registered. 200 seeds × 2 repeats. |
| `region_far.csv` | 400 | ≥2203, sep > 260 mm | Mode 2, exploratory. 200 seeds × 2 repeats. |
| `t1_seed{1,2,3}.csv` | 3 × 100 | 6000–6299 | **T-I: the assignment's 100 × 3.** Held out. |
| `t1_trainseed{1,2,3}.csv` | 3 × 100 | 0–299 | The *contaminated* pass — these are demonstration seeds. Kept because it is the memorisation measurement, **not** because it is a T-I number. |
| `seeds.csv` | — | 0–7999 | The policy-free seed index. Reset-only, no rollouts. |
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

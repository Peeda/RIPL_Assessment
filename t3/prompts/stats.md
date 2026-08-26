## Measured behaviour of the base policy

*Included only when the pipeline is run with `--with-stats`. The default run
deliberately withholds this, so that the reward the model writes is a response
to the video and the qualitative description rather than to a table. Both arms
are reported.*

Held-out evaluation of the frozen base policy, 1,200 nominal episodes plus
per-region confirmation passes. `grasped` / `placed` are the fractions of
episodes in which the cube was ever grasped / ever on the target; `held|placed`
is the fraction of placed episodes that ended in success.

|          | success | grasped | placed | held\|placed |
|----------|--------:|--------:|-------:|-------------:|
| nominal  |   0.713 |   0.984 |  0.884 |        0.807 |
| `gap`    |   0.523 |   0.955 |  0.659 |        0.793 |
| `farb`   |   0.561 |   1.000 |  0.854 |        0.657 |

In `gap` the placement rate collapses and holding is normal. In `farb` grasping
is perfect, placement is near baseline, and holding collapses.

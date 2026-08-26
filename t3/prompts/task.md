## What to produce

Two self-contained Python modules, returned through the `emit_artifacts` tool.

### (a) A dense reward function targeting this failure mode

It must incentivise **recovery from, or avoidance of, the failure described
above**, and it will be used to train a small bounded residual correction on top
of the frozen policy — the base policy's actions are not being replaced, only
nudged by a few millimetres per step. So the reward should reward *the precision
that the base policy is missing in this region*, not re-teach the task from
scratch. The base policy already reaches, grasps and transports competently;
what it gets wrong is specific and is described above.

The reward is also compared, on identical episodes, against the environment's
own eight-stage dense reward. Beating it at the stage this mode fails at is the
bar. Reproducing it exactly is not useful; ignoring it entirely is usually worse
than building on its structure.

### (b) An episode-configuration sampler biased toward this failure regime

It must draw initial cube configurations that land **inside the failure region**
far more often than the environment's own sampler does, so that reinforcement
learning spends its episodes on the states that matter instead of on the ~95% of
the distribution the policy already handles.

Two requirements that pull against each other, and getting the balance right is
the substance of this half:

- **Concentrated enough to be worth training on.** A sampler that reproduces the
  nominal distribution has done nothing.
- **Varied enough to generalise.** The residual will be *evaluated* on a fixed,
  held-out set of episodes drawn from this same region that it has never seen.
  A sampler that collapses onto one configuration, or onto one corner of the
  region, produces a residual that overfits and does not transfer — and that is
  measured, not assumed.

Stay inside the environment's own support. The biased distribution must be a
**subset** of what the environment can produce, never a superset: the frozen
base policy has never seen anything outside it, and training on states it has
never seen measures a distribution shift rather than a failure mode.

## The interface

{{CONTRACT}}

## One call

You get **one** tool call. There is no follow-up turn, no review round, and no
chance to fill anything in later: what you return is written to disk as
`reward.py` and `sampler.py` and executed against the simulator as-is. A field
containing the word `placeholder` becomes a file containing the word
`placeholder`.

So return all four fields complete in the same call, and finish the reward
before you start the sampler — the reward is the harder of the two and it is
the one T-IV actually trains against.

## Also return

- `rationale`: why each term is there, what mechanism from the video and the
  description it addresses, and — for the reward — the explicit stage ladder
  with the numeric band each stage occupies.
- `uncertainties`: what you are least sure of, which term you would expect an
  optimiser to exploit first, and what you predict will happen when the
  validation described above is run. Be specific; this is read.

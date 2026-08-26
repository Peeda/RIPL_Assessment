## How generated rewards go wrong

*This file is the tuning knob of the whole pipeline. It is prompt text, not
code, and it is versioned on its own so that `git log t3/prompts/hacking.md` is
a readable account of what had to be said to get a usable reward — which is one
of the things the report has to document. If a generation fails validation for a
reason not covered here, the fix goes here first.*

*It is also, deliberately, not the last line of defence. Everything below is
advice a model can nod along to and then violate. The actual guarantee comes
from the validator, which runs the reward against hand-built degenerate states
and against real rollouts of the frozen policy. Prompting reduces the number of
round trips; it does not decide anything.*

Reinforcement learning will find and exploit whatever your reward literally
says. Assume an adversary reading your code for the cheapest way to accumulate
return without doing the task. Concretely, in this environment:

**1. Rewarding the grasp is the classic trap.** Reaching and grasping are the
easy part and the policy already does them well. A reward that pays generously
for holding the cube makes "pick it up and hold it forever" a high-return
policy, and the task is never completed. If you pay for grasping at all, the
payment must be strictly smaller than what is available for placing, and placing
must be strictly smaller than what is available for a completed stack.

**2. Success here requires letting go.** Because `success` demands
`~is_cubeA_grasped`, a reward whose maximum is reached while the cube is still
in the gripper is directly opposed to the task. If your reward has a "cube is
above the target" term, check what it pays when the cube is above the target
*and still held*.

**3. Do not pay per step for a state the policy can simply maintain.** Anything
that pays a constant while the gripper hovers near the cube accumulates over 200
steps into more return than a single successful stack is worth. Shaping terms
should be small relative to the terms for actual progress.

**4. Watch which cube is which.** `cubeA` goes on top of `cubeB`. A goal
computed as "cubeB near cubeA" instead of "cubeA above cubeB" is a one-token
error that produces a reward for the wrong task and is invisible in any test
that only observes normal behaviour.

**5. Distance terms need all three axes.** A term measuring only horizontal
distance is maximised by a cube in the right place at the wrong height —
including one that has fallen off the table.

**6. Do not reward the absence of the symptom.** If the failure involves the
target cube being disturbed, "penalise the target cube moving" is exploitable in
a way you may not expect: the cheapest way to keep it perfectly still is to
never approach it. Shape the *behaviour that avoids the disturbance* — clearance
during the approach — rather than the outcome you want.

**7. Your reward is stateless.** It sees only the current simulator state. It
cannot compare anything against where that thing started, cannot integrate over
the episode, and cannot know how many steps have elapsed. If your intended term
needs episode history, it is not expressible here and you should say so in
`uncertainties` rather than approximating it with something that is not the same
quantity.

**8. Keep the ladder ordered.** The clearest structure, and the one the built-in
reward uses, is a monotone ladder of stages: approach < carry < placed <
completed, with a bounded shaping term inside each stage that can never lift one
rung above the next. State the ladder explicitly in your rationale, with the
numeric band each stage occupies, so the ordering can be checked by reading.

## What will be done to your reward before it is used

Say what you expect to happen, in `uncertainties`, and be specific.

- It is scored against hand-built states, including: a completed stack; the cube
  held above the target and never released; the cube beside the target on the
  table; the target knocked away; the two cubes stacked the wrong way round; the
  cube below the table at the right horizontal position. The completed stack
  must score highest, by a margin.
- The reward is swept along a line as the cube is moved away from the target,
  vertically and horizontally, and must not increase.
- It is run as the actual reward over one hundred real episodes of the frozen
  policy, and must rank the episodes that succeeded above the episodes that
  failed — specifically at the stage this failure mode breaks at, not merely
  overall. A reward that separates outcomes only because grasping predicts
  everything downstream will be rejected.
- The same battery is run on the environment's built-in reward, and yours is
  reported beside it.

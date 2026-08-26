## The video

The {{N_FRAMES}} images above are frames from a single episode of the trained
policy attempting this task, in time order, sampled across the episode. The
episode ran from initial-state seed {{SEED}} and its outcome was: **{{OUTCOME}}**.

The red cube is `cubeA` — the one that must end up on top. The green cube is
`cubeB` — the base of the stack. The arm is a Franka Panda.

Two things about the frames that are easy to misread:

- The episode always runs the full 200 steps. Nothing stops early, so a long
  static tail at the end is normal and is not itself the failure.
- The camera is fixed. Apparent motion is the scene moving, never the view.

What you are looking for is *when* and *how* the attempt breaks down, and what
about the starting configuration made it break down.

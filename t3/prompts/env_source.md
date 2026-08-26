## The environment, in full

This is the complete source of the task you are writing a reward for. The
success criterion, the existing dense reward, and the initial-state sampler you
are being asked to bias are all in here — read them rather than assuming.

```python
{{ENV_SOURCE}}
```

Three things in that source that matter more than they look:

1. **Success requires the robot to LET GO.** `evaluate()` computes
   `success = is_cubeA_on_cubeB & is_cubeA_static & ~is_cubeA_grasped`. A cube
   held perfectly in place on top of the other is a *failure*.
2. **`compute_dense_reward` already exists** and is an eight-stage ladder. Your
   reward is compared against it on the same episodes. You are not filling a
   vacuum; you are trying to beat a competent baseline at one specific failure.
3. **`_initialize_episode` is the sampler you are biasing.** Note the single
   shared `xy` offset applied to *both* cubes — it cancels out of their relative
   pose, so where the pair sits on the table is statistically independent of how
   the two cubes are arranged relative to each other.

One fact that is **not** visible in this file and that changes the problem: the
policy acts through a `pd_ee_delta_pos` controller, which is 4-dimensional —
three translation components and a gripper command. The end effector's
**orientation is frozen at whatever it was on reset, for all 200 steps**. The
wrist cannot rotate. The cubes, however, are spawned at a uniformly random yaw.
So the gripper meets each cube at an arbitrary misalignment and cannot square up
to it.

"""ManiSkill's own 8-stage StackCube reward, rewritten to the T-III contract.

Not a fixture of a MISTAKE - this is the calibration arm. It is a faithful
transcription of StackCubeEnv.compute_dense_reward (stack_cube.py:145-181),
changed only where the contract requires: `self` becomes `env`, and
`env.evaluate()`'s flags are read from `info` instead of recomputed.

Running the whole validation battery against it is what turns every AUC in the
T-III report into a comparison rather than a bare number. Without this column
"the generated reward separates success at AUC 0.82" is uninterpretable; with
it, "0.82 against the stock reward's 0.79 on the same 100 episodes, at the
stage the mode actually fails" is a result - and so is the other outcome.

CLAUDE.md: "A dense reward already exists ... T-III's LLM-generated reward has a
real baseline to be compared against rather than invented in a vacuum."
"""
import torch

REWARD_MAX = 8.0


def compute_reward(env, obs, action, info):
    tcp_pos = env.agent.tcp.pose.p
    cubeA_pos = env.cubeA.pose.p
    cubeB_pos = env.cubeB.pose.p

    # stage 1: reach cubeA
    cubeA_to_tcp_dist = torch.linalg.norm(tcp_pos - cubeA_pos, axis=1)
    reward = 2 * (1 - torch.tanh(5 * cubeA_to_tcp_dist))

    # stage 2: carry cubeA to the point directly above cubeB
    goal_xyz = torch.hstack(
        [cubeB_pos[:, 0:2], (cubeB_pos[:, 2] + env.cube_half_size[2] * 2)[:, None]]
    )
    cubeA_to_goal_dist = torch.linalg.norm(goal_xyz - cubeA_pos, axis=1)
    place_reward = 1 - torch.tanh(5.0 * cubeA_to_goal_dist)
    grasped = info["is_cubeA_grasped"]
    reward[grasped] = (4 + place_reward)[grasped]

    # stage 3: let go, and let it settle
    gripper_width = env.agent.robot.get_qlimits()[0, -1, 1] * 2
    ungrasp_reward = torch.sum(env.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
    ungrasp_reward[~grasped] = 1.0
    v = torch.linalg.norm(env.cubeA.linear_velocity, axis=1)
    av = torch.linalg.norm(env.cubeA.angular_velocity, axis=1)
    static_reward = 1 - torch.tanh(v * 10 + av)
    on_b = info["is_cubeA_on_cubeB"]
    reward[on_b] = (6 + (ungrasp_reward + static_reward) / 2.0)[on_b]

    reward[info["success"]] = 8.0
    return reward

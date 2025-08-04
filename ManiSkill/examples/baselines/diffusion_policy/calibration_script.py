"""
Quick calibration script to compute pixel-per-meter factors for DDPM policy video overlay.

Usage:
    python calibrate_dp_pixels.py \
        --env_id PickCube-v1 \
        --control_mode pd_ee_delta_pos \
        --move_distance 0.1 \
        --num_trials 5

This will command the robot's EE to move +move_distance in X and Y directions,
measure the corresponding pixel shifts in the rendered image, and compute
average px_per_m_x and px_per_m_y values.
Ensure your training script module is importable as `train_checkp.py`.
"""
import numpy as np
import torch
import os
import tyro
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from diffusion_policy.make_env import make_eval_envs
from train_checkp import Agent, Args

def project_pixel(raw_obs, env, px_per_m_x=None, px_per_m_y=None):
    # approximate projection using state XY and center-based scaling
    frame = env.envs[0].render()
    h, w, _ = frame.shape
    state = raw_obs['state'][-1]
    # if px_per_m provided, use that; else assume center+state*1m->full width
    if px_per_m_x and px_per_m_y:
        u = w/2 + state[0] * px_per_m_x
        v = h/2 - state[1] * px_per_m_y
    else:
        u = w/2 + state[0] * (w/1.0)
        v = h/2 - state[1] * (h/1.0)
    return u, v

def calibrate(env, axis, move_dist, num_trials):
    shifts = []
    for _ in range(num_trials):
        # reset and get initial
        raw_obs_batch, _ = env.reset()
        raw_obs = {k: raw_obs_batch[k][0] for k in raw_obs_batch}
        u0, v0 = project_pixel(raw_obs, env)
        # prepare action
        action = np.zeros(env.single_action_space.shape)
        if axis == 'x':
            action[0] = move_dist
        else:
            action[1] = move_dist
        # step
        raw_next, _, _, _, _ = env.step(action[None])
        raw_obs2 = {k: raw_next[k][0] for k in raw_next}
        u1, v1 = project_pixel(raw_obs2, env)
        shifts.append((u1 - u0, v1 - v0))
    shifts = np.array(shifts)
    # compute px per meter
    avg = shifts.mean(axis=0)
    px_per_m = avg / move_dist
    return px_per_m[0], -px_per_m[1]

def main(
    env_id: str = "PickCube-v1",
    control_mode: str = "pd_ee_delta_pos",
    move_distance: float = 0.1,
    num_trials: int = 5
):
    # build env
    args = Args()
    args.env_id = env_id
    args.control_mode = control_mode
    args.obs_mode = 'rgb'
    args.obs_horizon = 2
    args.act_horizon = 8
    args.pred_horizon = 16
    args.sim_backend = 'physx_cpu'
    args.max_episode_steps = 1
    args.cuda = False
    env = make_eval_envs(
        args.env_id, num_envs=1, sim_backend=args.sim_backend,
        env_kwargs={
            'control_mode': args.control_mode,
            'reward_mode': 'sparse',
            'obs_mode': args.obs_mode,
            'render_mode': 'rgb_array',
            'human_render_camera_configs': {'shader_pack': 'default'},
            'max_episode_steps': args.max_episode_steps
        },
        other_kwargs={'obs_horizon': args.obs_horizon},
        wrappers=[FlattenRGBDObservationWrapper]
    )
    # x axis
    px_x, py_x = calibrate(env, 'x', move_distance, num_trials)
    # y axis
    px_y, py_y = calibrate(env, 'y', move_distance, num_trials)
    print(f"Pixel-per-meter X: {px_x:.2f}  (v shift factor {py_x:.2f})")
    print(f"Pixel-per-meter Y: {px_y:.2f}  (v shift factor {py_y:.2f})")
    env.close()

if __name__ == '__main__':
    tyro.cli(main)

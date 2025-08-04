#!/usr/bin/env python
# visualize_diffusion_loop.py

import os
import cv2
import torch
import numpy as np
import tyro
from diffusion_policy.make_env import make_eval_envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from train_checkp import Agent, Args
from gymnasium.error import ResetNeeded

def project_from_ee(traj_m, ee_uv, pxx, pxy):
    """
    Convert a sequence of (dx,dy) in meters into pixel coordinates,
    starting from ee_uv.  pxx, pxy are pixels-per-meter calibration.
    """
    u, v = ee_uv
    pts = []
    for dx, dy in traj_m:
        u -= dx * pxx
        v += dy * pxy
        pts.append((int(u), int(v)))
    return pts

def draw_dots(img, pts, color=(0,0,255)):
    for u, v in pts:
        cv2.circle(img, (u, v), radius=3, color=color, thickness=-1, lineType=cv2.LINE_AA)
    return img

def draw_line(img, pts, color=(0,255,0)):
    for i in range(len(pts)-1):
        cv2.line(img, pts[i], pts[i+1], color, thickness=2, lineType=cv2.LINE_AA)
    return img

def safe_render(env):
    """
    Call env.render(), handle ResetNeeded, and unwrap [0] if returned list.
    """
    try:
        out = env.render()
    except ResetNeeded:
        env.reset()
        out = env.render()
    if isinstance(out, (list, tuple)) and len(out) > 0:
        return out[0]
    return out

def main(
    checkpoint: str,
    env_id: str = "PickCube-v1",
    control_mode: str = "pd_ee_delta_pos",
    obs_horizon: int = 2,
    act_horizon: int = 8,
    pred_horizon: int = 16,
    diffusion_steps: int = 100,
    max_steps: int = 64,
    px_per_m_x: float = 50.94,
    px_per_m_y: float = 33.21,
    output: str = "diffusion_loop.mp4",
    sim_backend: str = "physx_cpu",
):
    # 1) Build args & env
    args = Args()
    args.env_id = env_id
    args.obs_mode = "rgb"
    args.obs_horizon = obs_horizon
    args.act_horizon = act_horizon
    args.pred_horizon = pred_horizon
    args.control_mode = control_mode
    args.sim_backend = sim_backend
    args.max_episode_steps = max_steps
    args.cuda = False
    device = torch.device("cpu")

    env = make_eval_envs(
        env_id,
        num_envs=1,
        sim_backend=sim_backend,
        env_kwargs=dict(
            control_mode=control_mode,
            reward_mode="sparse",
            obs_mode="rgb",
            render_mode="rgb_array",
            human_render_camera_configs={"shader_pack": "default"},
            max_episode_steps=max_steps,
        ),
        other_kwargs={"obs_horizon": obs_horizon},
        wrappers=[FlattenRGBDObservationWrapper],
    )

    # 2) Load agent
    agent = Agent(env, args).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()

    # 3) Prepare video writer
    first_frame = safe_render(env.envs[0])
    H, W = first_frame.shape[:2]
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))

    # 4) Reset once to get initial obs
    obs_buf, _ = env.reset()

    step = 0
    while step < max_steps:
        # --- Build conditioning obs from the stacked horizon in obs_buf ---
        # obs_buf["rgb"]: (1, obs_horizon, H, W, 3)
        rgb_np = obs_buf["rgb"]                   # numpy array
        state_np = obs_buf["state"]               # numpy array

        # to torch and permute to (1, obs_h, C, H, W)
        rgb_t = torch.from_numpy(rgb_np).permute(0, 1, 4, 2, 3).to(device)
        st_t  = torch.from_numpy(state_np).to(device)

        obs_cond = agent.encode_obs({"rgb": rgb_t, "state": st_t}, eval_mode=True)

        # Compute EE pixel start from last state:
        ee_xyz = state_np[0, -1, :3]
        u0 = W/2 + ee_xyz[0]*px_per_m_x
        v0 = H/2 - ee_xyz[1]*px_per_m_y
        ee_uv = (u0, v0)

        # 5) Diffusion process visualization (red dots)
        sample = torch.randn((1, pred_horizon, agent.act_dim), device=device)
        scheduler = agent.noise_scheduler
        timesteps = scheduler.timesteps

        for t in range(min(diffusion_steps, len(timesteps))):
            ts = timesteps[t]
            with torch.no_grad():
                noise_pred = agent.noise_pred_net(sample, ts, global_cond=obs_cond)
                out = scheduler.step(model_output=noise_pred, timestep=ts, sample=sample)
                sample = out.prev_sample

            traj = sample[0, :pred_horizon, :2].cpu().numpy()
            pts = project_from_ee(traj, ee_uv, px_per_m_x, px_per_m_y)
            vis = safe_render(env.envs[0]).copy()
            vis = draw_dots(vis, pts, color=(0,0,255))
            cv2.circle(vis, (int(u0), int(v0)), 5, (0,255,0), 2)
            writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        # 6) Execute the next act_horizon steps, drawing green path
        uv_path = [ee_uv]
        act_seq = sample[0, :act_horizon].cpu().numpy()
        for a in act_seq:
            obs_buf, _, term, trunc, _ = env.step(a[None,...])
            step += 1

            new_xyz = obs_buf["state"][0, -1, :3]
            u = W/2 + new_xyz[0]*px_per_m_x
            v = H/2 - new_xyz[1]*px_per_m_y
            uv_path.append((u, v))

            vis = safe_render(env.envs[0]).copy()
            line_pts = [(int(x),int(y)) for x,y in uv_path]
            vis = draw_line(vis, line_pts, color=(0,255,0))
            writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            if term or trunc or step >= max_steps:
                break

        # write one “pause” frame
        vis = safe_render(env.envs[0]).copy()
        writer.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    writer.release()
    env.close()

if __name__ == "__main__":
    tyro.cli(main)

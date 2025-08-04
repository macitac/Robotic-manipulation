# record_dp_video.py
"""
Script to load a pretrained Diffusion Policy (DP) agent and record demo videos
with the executed trajectory overlaid on each frame, using RGB-only input.

Usage:
    python record_dp_video.py \
        --checkpoint runs/YourRun/checkpoints/best_eval_success_at_end.pt \
        --env_id PickCube-v1 \
        --control_mode pd_ee_delta_pos \
        --output_dir recorded_dp_videos \
        --num_episodes 5 \
        --max_steps 100

Ensure your training script module is importable as `train_dp.py`.
"""
import os
import cv2
import torch
import numpy as np
from tqdm import trange
import tyro
import gymnasium as gym

# Import your training definitions
from train_checkp import Agent, Args


def draw_trajectory_on_frame(frame: np.ndarray, traj: list, color=(0, 255, 0)) -> np.ndarray:
    """
    Overlay a 2D trajectory as connected lines onto an RGB frame.
    traj: list of (u, v) pixel coordinates.
    """
    for i in range(len(traj) - 1):
        pt1 = tuple(map(int, traj[i]))
        pt2 = tuple(map(int, traj[i + 1]))
        cv2.line(frame, pt1, pt2, color, thickness=2)
    return frame


def main(
    checkpoint: str,
    env_id: str = "PickCube-v1",
    control_mode: str = "pd_ee_delta_pos",
    output_dir: str = "recorded_dp_videos",
    num_episodes: int = 5,
    max_steps: int = 100,
    obs_horizon: int = 2,
    act_horizon: int = 8,
    pred_horizon: int = 16
):
    # Setup args
    args = Args()
    args.env_id = env_id
    args.obs_mode = "rgb"
    args.obs_horizon = obs_horizon
    args.act_horizon = act_horizon
    args.pred_horizon = pred_horizon
    args.control_mode = control_mode
    args.max_episode_steps = max_steps
    args.cuda = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

            # Build vectorized environment (1 instance) with the same wrappers as training
    env = make_eval_envs(
        args.env_id,
        num_envs=1,
        sim_backend=args.sim_backend,
        env_kwargs=dict(
            control_mode=args.control_mode,
            reward_mode="sparse",
            obs_mode=args.obs_mode,
            render_mode="rgb_array",
            human_render_camera_configs={"shader_pack": "default"},
            max_episode_steps=args.max_episode_steps
        ),
        other_kwargs={},  # no built-in frame stacking
        wrappers=[FlattenRGBDObservationWrapper]
    )

    # Load pretrained agent
    agent = Agent(env, args).to(device)(env, args).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()

    # Build preprocessing function
    from diffusion_policy.utils import convert_obs, build_state_obs_extractor
    obs_process = partial(
        convert_obs,
        concat_fn=lambda *arrays: np.concatenate(arrays, axis=-1),
        transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),
        state_obs_extractor=build_state_obs_extractor(args.env_id),
        depth=False
    )

    os.makedirs(output_dir, exist_ok=True)(output_dir, exist_ok=True)

    for ep in range(num_episodes):
        # Reset and get first observation
        obs, _ = env.reset()
        obs_seq = [obs]
        traj_pixels = []
        frames = []

        for _ in trange(args.max_episode_steps, desc=f"Episode {ep}"):
            # Stack last obs_horizon observations
            history = obs_seq[-args.obs_horizon:]
            rgb_stack = np.stack([h['rgb'] for h in history], axis=0)      # (obs_horizon, H, W, 3)
            state_stack = np.stack([h['state'] for h in history], axis=0)  # (obs_horizon, state_dim)

            # Prepare batch tensors
            rgb_t = torch.from_numpy(rgb_stack).unsqueeze(0).to(device)    # (1, obs_horizon, H, W, 3)
            state_t = torch.from_numpy(state_stack).unsqueeze(0).to(device)
            obs_batch = {'rgb': rgb_t, 'state': state_t}

            # Compute action sequence
            with torch.no_grad():
                act_seq = agent.get_action(obs_batch)  # shape (1, act_horizon, act_dim)
                action = act_seq[0, 0].cpu().numpy()

            # Step environment
            obs, (_, term, trunc, _) = env.step(action)
            obs_seq.append(obs)

            # Render frame
            frame = env.render()[0]  # (H, W, 3)
            h, w, _ = frame.shape
            # Project action[0:2] to pixel coords
            u = (action[0] + 1) / 2 * w
            v = (action[1] + 1) / 2 * h
            traj_pixels.append((u, v))

            # Overlay trajectory
            overlay = draw_trajectory_on_frame(frame.copy(), traj_pixels)
            frames.append(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            if term or trunc:
                break

        # Write video
        video_path = os.path.join(output_dir, f"dp_episode_{ep}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        H, W, _ = frames[0].shape
        writer = cv2.VideoWriter(video_path, fourcc, 30, (W, H))
        for f in frames:
            writer.write(f)
        writer.release()
        print(f"Saved video: {video_path}")

    env.close()


if __name__ == "__main__":
    tyro.cli(main)

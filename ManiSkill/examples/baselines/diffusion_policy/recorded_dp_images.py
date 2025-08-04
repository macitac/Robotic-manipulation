import os
import cv2
import torch
import numpy as np
from tqdm import trange
import tyro

from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from diffusion_policy.make_env import make_eval_envs
from train_checkp import Agent, Args

def draw_trajectory_on_frame(frame: np.ndarray, traj: list, color=(0, 255, 0)) -> np.ndarray:
    for i in range(len(traj) - 1):
        pt1 = tuple(map(int, traj[i])); pt2 = tuple(map(int, traj[i+1]))
        cv2.line(frame, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)
    return frame


def main(
    checkpoint: str,
    env_id: str = "PickCube-v1",
    control_mode: str = "pd_ee_delta_pos",
    output_dir: str = "recorded_dp_images",
    num_episodes: int = 3,
    max_steps: int = 100,
    obs_horizon: int = 2,
    act_horizon: int = 8,
    pred_horizon: int = 16,
    px_per_m_x: float = 50.94,
    px_per_m_y: float = 33.21
):
    # Setup args
    args = Args()
    args.env_id = env_id
    args.obs_mode = "rgb"
    args.obs_horizon = obs_horizon
    args.act_horizon = act_horizon
    args.pred_horizon = pred_horizon
    args.control_mode = control_mode
    args.sim_backend = "physx_cpu"
    args.max_episode_steps = max_steps
    args.cuda = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Build env
    env = make_eval_envs(
        args.env_id, num_envs=1, sim_backend=args.sim_backend,
        env_kwargs=dict(
            control_mode=args.control_mode,
            reward_mode="sparse",
            obs_mode=args.obs_mode,
            render_mode="rgb_array",
            human_render_camera_configs={"shader_pack": "default"},
            max_episode_steps=args.max_episode_steps
        ),
        other_kwargs={"obs_horizon": args.obs_horizon},
        wrappers=[FlattenRGBDObservationWrapper]
    )

    # Load agent
    agent = Agent(env, args).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()

    # Create output root
    os.makedirs(output_dir, exist_ok=True)

    for ep in range(num_episodes):
        # Episode directory
        ep_dir = os.path.join(output_dir, f"episode_{ep}")
        os.makedirs(ep_dir, exist_ok=True)

        # Reset
        raw_obs_batch, _ = env.reset()
        raw_obs = {k: raw_obs_batch[k][0] for k in raw_obs_batch}

        traj_pixels = []
        for t in trange(max_steps, desc=f"Episode {ep}"):
            # Prepare model input
            rgb_arr = raw_obs['rgb']
            state_arr = raw_obs['state']
            rgb_t = torch.from_numpy(rgb_arr).unsqueeze(0).to(device)
            state_t = torch.from_numpy(state_arr).unsqueeze(0).to(device)
            obs_batch = {'rgb': rgb_t, 'state': state_t}

            # Predict
            with torch.no_grad():
                act_seq = agent.get_action(obs_batch)
                action = act_seq[0, 0].cpu().numpy()

            # Step
            raw_next, _, term_b, trunc_b, _ = env.step(action[None])
            raw_obs = {k: raw_next[k][0] for k in raw_next}
            term, trunc = term_b[0], trunc_b[0]

            # Render underlying single env
            frame = env.envs[0].render()
            h, w, _ = frame.shape

            # Actual EE projection
            ee = raw_obs['state'][-1, :3]
            u_act = w/2 + ee[0] * px_per_m_x
            v_act = h/2 - ee[1] * px_per_m_y
            traj_pixels.append((u_act, v_act))
            overlay = draw_trajectory_on_frame(frame.copy(), traj_pixels, color=(0, 255, 0))

            # Planned future (optional)
            u_p, v_p = u_act, v_act
            future = act_seq[0].cpu().numpy()[:pred_horizon]
            for dx, dy in future[:, :2]:
                u_p += dx * px_per_m_x
                v_p -= dy * px_per_m_y
                cv2.circle(overlay, (int(u_p), int(v_p)), 3, (0, 0, 255), -1)

            # Save frame image
            img_path = os.path.join(ep_dir, f"frame_{t:04d}.png")
            cv2.imwrite(img_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            if term or trunc:
                break

        print(f"Saved images for episode {ep} at {ep_dir}")

    env.close()

if __name__ == "__main__":
    tyro.cli(main)

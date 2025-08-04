#!/usr/bin/env python
import os
import sys
import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Import wrappers and environment maker from ManiSkill / your diff policy package.
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.evaluate import evaluate

# Import model components for the agent.
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# -----------------------------------------------------------------------------
# Define the Agent class (same as in training)
# -----------------------------------------------------------------------------
class Agent(nn.Module):
    def __init__(self, env, args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        obs_state_dim = env.single_observation_space["state"].shape[1]
        total_visual_channels = 0
        self.include_rgb = "rgb" in env.single_observation_space.keys()
        self.include_depth = "depth" in env.single_observation_space.keys()
        if self.include_rgb:
            total_visual_channels += env.single_observation_space["rgb"].shape[-1]
        if self.include_depth:
            total_visual_channels += env.single_observation_space["depth"].shape[-1]

        visual_feature_dim = 256
        self.visual_encoder = PlainConv(
            in_channels=total_visual_channels,
            out_dim=visual_feature_dim,
            pool_feature_map=True
        )
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=env.single_action_space.shape[0],
            global_cond_dim=args.obs_horizon * (visual_feature_dim + obs_state_dim),
            diffusion_step_embed_dim=args.diffusion_step_embed_dim,
            down_dims=args.unet_dims,
            n_groups=args.n_groups,
        )
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.act_dim = env.single_action_space.shape[0]

    def encode_obs(self, obs_seq, eval_mode):
        if self.include_rgb:
            rgb = obs_seq["rgb"].float() / 255.0  # (B, obs_horizon, 3, H, W)
            img_seq = rgb
        if self.include_depth:
            depth = obs_seq["depth"].float() / 1024.0  # (B, obs_horizon, 1, H, W)
            img_seq = depth
        if self.include_rgb and self.include_depth:
            img_seq = torch.cat([rgb, depth], dim=2)
        batch_size = img_seq.shape[0]
        img_seq = img_seq.flatten(end_dim=1)
        visual_feature = self.visual_encoder(img_seq)
        visual_feature = visual_feature.reshape(batch_size, self.obs_horizon, visual_feature.shape[1])
        feature = torch.cat((visual_feature, obs_seq["state"]), dim=-1)
        return feature.flatten(start_dim=1)

    def get_action(self, obs_seq):
        B = obs_seq["state"].shape[0]
        with torch.no_grad():
            # Permute visual inputs if available.
            if self.include_rgb:
                obs_seq["rgb"] = obs_seq["rgb"].permute(0, 1, 4, 2, 3)
            if self.include_depth:
                obs_seq["depth"] = obs_seq["depth"].permute(0, 1, 4, 2, 3)
            obs_cond = self.encode_obs(obs_seq, eval_mode=True)
            noisy_action_seq = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq["state"].device)
            for k in self.noise_scheduler.timesteps:
                noise_pred = self.noise_pred_net(noisy_action_seq, k, global_cond=obs_cond)
                noisy_action_seq = self.noise_scheduler.step(
                    model_output=noise_pred, timestep=k, sample=noisy_action_seq
                ).prev_sample
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]

# -----------------------------------------------------------------------------
# Argument parser for evaluation
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BC_Diffusion_rgbd_UNet Policy")
    parser.add_argument("--env_id", type=str, default="PegInsertionSide-v1", help="Environment ID")
    parser.add_argument("--policy_checkpoint", type=str, required=True, help="Path to the policy checkpoint (e.g., best.pt)")
    parser.add_argument("--num_eval_episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--num_eval_envs", type=int, default=10, help="Number of parallel evaluation environments")
    parser.add_argument("--sim_backend", type=str, default="physx_gpu", help="Simulation backend (e.g., physx_gpu or physx_cpu)")
    parser.add_argument("--obs_mode", type=str, default="rgb+depth", help="Observation mode")
    parser.add_argument("--control_mode", type=str, default="pd_joint_delta_pos", help="Control mode")
    parser.add_argument("--max_episode_steps", type=int, required=True, help="Maximum episode steps")
    parser.add_argument("--capture_video", action="store_true", help="Capture evaluation videos")
    parser.add_argument("--obs_horizon", type=int, default=2, help="Observation horizon")
    parser.add_argument("--act_horizon", type=int, default=8, help="Action horizon")
    parser.add_argument("--pred_horizon", type=int, default=16, help="Prediction horizon")
    parser.add_argument("--diffusion_step_embed_dim", type=int, default=64, help="Diffusion step embed dimension")
    parser.add_argument("--unet_dims", type=int, nargs='+', default=[64, 128, 256], help="UNet dimensions")
    parser.add_argument("--n_groups", type=int, default=8, help="Number of groups in the UNet")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA")
    return parser.parse_args()

# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build evaluation environment
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default")
    )
    env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    video_dir = None
    if args.capture_video:
        video_dir = os.path.join("eval_videos", args.env_id)
        os.makedirs(video_dir, exist_ok=True)

    eval_envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=video_dir,
        wrappers=[FlattenRGBDObservationWrapper],
    )

    # Instantiate the agent using the same architecture as during training.
    agent = Agent(eval_envs, args).to(device)

    # Load checkpoint. Prefer the EMA weights if available.
    checkpoint = torch.load(args.policy_checkpoint, map_location=device)
    if "ema_agent" in checkpoint:
        agent.load_state_dict(checkpoint["ema_agent"])
    elif "agent" in checkpoint:
        agent.load_state_dict(checkpoint["agent"])
    else:
        print("Checkpoint does not contain valid agent state dict!")
        sys.exit(1)
    agent.eval()

    # Run evaluation.
    print("Starting evaluation ...")
    eval_metrics = evaluate(args.num_eval_episodes, agent, eval_envs, device, args.sim_backend)
    print("Evaluation Metrics:")
    for key, value in eval_metrics.items():
        print(f"  {key} mean: {np.mean(value):.4f}")

    eval_envs.close()

if __name__ == "__main__":
    main()

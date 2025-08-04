#!/usr/bin/env python

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.utils import build_state_obs_extractor, convert_obs


@dataclass
class Args:
    resume_path: str
    """Path to the checkpoint saved during training (e.g. runs/your_run/checkpoints/last.pt)"""
    
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = True

    env_id: str = "StackCube-v1"
    demo_path: str = "~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"
    max_episode_steps: int = 300
    obs_mode: str = "rgb"
    control_mode: str = "pd_joint_delta_pos"
    num_eval_envs: int = 10
    num_eval_episodes: int = 100
    sim_backend: str = "physx_cpu"

    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_groups: int = 8


def load_ckpt(checkpoint_path, agent, ema_agent, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(checkpoint["agent"])
    ema_agent.load_state_dict(checkpoint["ema_agent"])
    start_iter = checkpoint["iteration"] + 1
    return start_iter


class Agent(nn.Module):
    def __init__(self, env, args: Args):
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
            in_channels=total_visual_channels, out_dim=visual_feature_dim, pool_feature_map=True
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
            rgb = obs_seq["rgb"].float() / 255.0
            img_seq = rgb
        if self.include_depth:
            depth = obs_seq["depth"].float() / 1024.0
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


def main():
    args = tyro.cli(Args)

    # Set seeds and determinism.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create evaluation environments (using the same configuration as training).
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default"),
    )
    assert args.max_episode_steps is not None, "max_episode_steps must be specified"
    env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    run_name = args.exp_name or "eval_run"
    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
        wrappers=[FlattenRGBDObservationWrapper],
    )

    # Initialize the agent and EMA agent (exactly as in training).
    agent = Agent(envs, args).to(device)
    ema_agent = Agent(envs, args).to(device)

    # Load the checkpoint (this must match the training configuration).
    _ = load_ckpt(args.resume_path, agent, ema_agent, device)
    print(f"Loaded checkpoint from {args.resume_path}")

    # Switch EMA agent to evaluation mode.
    ema_agent.eval()

    # Run evaluation.
    eval_metrics = evaluate(args.num_eval_episodes, ema_agent, envs, device, args.sim_backend)
    print("Evaluation Results:")
    for metric, values in eval_metrics.items():
        print(f"{metric}: {np.mean(values):.4f}")

    envs.close()


if __name__ == "__main__":
    main()

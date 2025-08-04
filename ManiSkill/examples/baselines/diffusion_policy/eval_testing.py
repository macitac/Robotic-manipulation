ALGO_NAME = "BC_Diffusion_rgbd_UNet"

import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import tyro
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.utils import (IterationBasedBatchSampler,
                                    build_state_obs_extractor, convert_obs,
                                    worker_init_fn)

@dataclass
class Args:
    exp_name: Optional[str] = "default_exp"
    seed: int = 1
    cuda: bool = True

    env_id: str = "StackCube-v1"
    demo_path: str = "demos/PegInsertionSide-v1/trajectory.state.pd_ee_delta_pos.physx_cpu.h5"
    total_iters: int = 500000
    batch_size: int = 64

    lr: float = 1e-4
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_groups: int = 8

    obs_mode: str = "rgb"
    max_episode_steps: int = 300
    num_eval_episodes: int = 100
    num_eval_envs: int = 10
    sim_backend: str = "physx_cpu"
    control_mode: str = "pd_ee_delta_pos"

    mode: str = "evaluate"  # choose between "train" and "evaluate"
    checkpoint_path: Optional[str] = None

class Agent(nn.Module):
    def __init__(self, env: VectorEnv, args: Args):
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
        img_seq = []
        if self.include_rgb:
            rgb = obs_seq["rgb"].float() / 255.0  # (B, obs_horizon, H, W, 3)
            rgb = rgb.permute(0, 1, 4, 2, 3)      # Now (B, obs_horizon, 3, H, W)
            img_seq.append(rgb)
        if self.include_depth:
            depth = obs_seq["depth"].float() / 1024.0  # (B, obs_horizon, H, W, 1)
            depth = depth.permute(0, 1, 4, 2, 3)       # Now (B, obs_horizon, 1, H, W)
            img_seq.append(depth)

        img_seq = torch.cat(img_seq, dim=2)  # Concatenate RGB and Depth channels
        batch_size = img_seq.shape[0]
        img_seq = img_seq.flatten(end_dim=1)  # Merge batch and obs_horizon dimensions
        visual_feature = self.visual_encoder(img_seq)
        visual_feature = visual_feature.reshape(batch_size, self.obs_horizon, visual_feature.shape[1])
        feature = torch.cat((visual_feature, obs_seq["state"]), dim=-1)
        return feature.flatten(start_dim=1)



    def compute_loss(self, obs_seq, action_seq):
        B = obs_seq["state"].shape[0]
        obs_cond = self.encode_obs(obs_seq, eval_mode=False)
        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq["state"].device)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=obs_seq["state"].device).long()
        noisy_action_seq = self.noise_scheduler.add_noise(action_seq, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy_action_seq, timesteps, global_cond=obs_cond)
        return F.mse_loss(noise_pred, noise)

    def get_action(self, obs_seq):
        B = obs_seq["state"].shape[0]
        with torch.no_grad():
            obs_cond = self.encode_obs(obs_seq, eval_mode=True)
            noisy_action_seq = torch.randn((B, self.pred_horizon, self.act_dim), device=obs_seq["state"].device)
            for k in self.noise_scheduler.timesteps:
                noise_pred = self.noise_pred_net(noisy_action_seq, k, global_cond=obs_cond)
                noisy_action_seq = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=noisy_action_seq).prev_sample
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]
    


def main():
    args = tyro.cli(Args)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        max_episode_steps=args.max_episode_steps,
        human_render_camera_configs=dict(shader_pack="default")
    )
    other_kwargs = dict(obs_horizon=args.obs_horizon)

    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=f"runs/{args.exp_name}/videos" if args.mode == "evaluate" else None,
        wrappers=[FlattenRGBDObservationWrapper],
    )

    agent = Agent(envs, args).to(device)
    ema_agent = Agent(envs, args).to(device)
    ema = EMAModel(parameters=agent.parameters(), power=0.75)

    if args.mode == "train":
        dataset = SmallDemoDataset_DiffusionPolicy(
            data_path=args.demo_path,
            obs_process_fn=obs_process_fn,
            obs_space=envs.single_observation_space,
            include_rgb="rgb" in envs.single_observation_space,
            include_depth="depth" in envs.single_observation_space,
            num_traj=None,
        )
        
        sampler = RandomSampler(dataset, replacement=False)
        batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
        batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
        train_dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

        optimizer = torch.optim.AdamW(agent.parameters(), lr=args.lr)
        lr_scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=500, num_training_steps=args.total_iters)

        agent.train()
        for iteration, batch in enumerate(train_dataloader):
            batch = move_to_device(batch, device)
            loss = agent.compute_loss(batch["observations"], batch["actions"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            ema.step(agent.parameters())

            if iteration % 5000 == 0:
                print(f"Iteration {iteration}, Loss: {loss.item():.4f}")

        torch.save({"agent": agent.state_dict(), "ema_agent": ema_agent.state_dict()}, f"runs/{args.exp_name}/model_final.pt")

    elif args.mode == "evaluate":
        assert args.checkpoint_path, "Please provide checkpoint_path for evaluation mode"
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        agent.load_state_dict(checkpoint["agent"])
        ema_agent.load_state_dict(checkpoint["ema_agent"])
        ema.copy_to(ema_agent.parameters())

        ema_agent.eval()

        eval_metrics = evaluate(
            args.num_eval_episodes,
            ema_agent,
            envs,
            device,
            args.sim_backend
        )

        print("Evaluation Metrics:")
        for k, v in eval_metrics.items():
            print(f"{k}: {np.mean(v):.4f}")

    envs.close()


if __name__ == "__main__":
    main()

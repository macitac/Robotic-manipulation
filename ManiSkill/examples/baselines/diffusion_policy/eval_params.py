import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Optional, List

import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from tqdm import tqdm

from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.utils import (
    IterationBasedBatchSampler,
    build_state_obs_extractor,
    convert_obs,
    load_demo_dataset,
    worker_init_fn,
)
import pick_cube_new

# -----------------------------------------------------------------------------
# 1) Parameter counting utility
# -----------------------------------------------------------------------------
def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# -----------------------------------------------------------------------------
@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ManiSkill"
    wandb_entity: Optional[str] = None
    capture_video: bool = True

    env_id: str = "PegInsertionSide-v1"
    demo_path: str = "demos/PegInsertionSide-v1/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
    num_demos: Optional[int] = None
    total_iters: int = 1_000_000
    batch_size: int = 64
    lr: float = 1e-4

    obs_horizon: int = 1
    act_horizon: int = 1
    pred_horizon: int = 1

    diffusion_step_embed_dim: int = 64
    unet_dims: List[int] = None
    n_groups: int = 8

    obs_mode: str = "rgb"
    max_episode_steps: Optional[int] = None

    log_freq: int = 1000
    eval_freq: int = 5000
    save_freq: Optional[int] = None

    num_eval_episodes: int = 100
    num_eval_envs: int = 10
    sim_backend: str = "physx_cpu"
    num_dataload_workers: int = 16

    control_mode: str = "pd_ee_delta_pos"
    demo_type: Optional[str] = None

    resume: bool = False
    resume_path: Optional[str] = None

# Utility to reorder keys in obs dict
def reorder_keys(d, ref_dict):
    out = {}
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Dataset class (unchanged from your code)
# -----------------------------------------------------------------------------
class SmallDemoDataset_DiffusionPolicy(Dataset):
    def __init__(self, data_path, obs_process_fn, obs_space, include_rgb, include_depth, num_traj):
        self.include_rgb = include_rgb
        self.include_depth = include_depth

        trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
        obs_traj_list = []
        for obs in trajectories["observations"]:
            o = reorder_keys(obs, obs_space)
            o = obs_process_fn(o)
            if include_depth:
                o["depth"] = torch.tensor(o["depth"].astype(np.float32), dtype=torch.float16)
            if include_rgb:
                o["rgb"] = torch.from_numpy(o["rgb"])
            o["state"] = torch.from_numpy(o["state"])
            obs_traj_list.append(o)
        trajectories["observations"] = obs_traj_list
        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.tensor(trajectories["actions"][i])

        # pad logic as before…
        self.pad_action_arm = torch.zeros((trajectories["actions"][0].shape[1] - 1,))
        self.obs_horizon = args.obs_horizon
        self.pred_horizon = args.pred_horizon
        self.slices = []
        total_trans = 0
        for idx in range(len(trajectories["actions"])):
            L = trajectories["actions"][idx].shape[0]
            total_trans += L
            pad_b = self.obs_horizon - 1
            pad_a = self.pred_horizon - self.obs_horizon
            self.slices += [
                (idx, s, s + self.pred_horizon)
                for s in range(-pad_b, L - self.pred_horizon + pad_a)
            ]
        self.trajectories = trajectories

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, _ = self.trajectories["actions"][traj_idx].shape
        obs = self.trajectories["observations"][traj_idx]
        # slicing with padding logic… (as in your code)
        # return {"observations": obs_seq, "actions": act_seq}
        # For brevity, please copy your original __getitem__ here
        raise NotImplementedError("Insert your full __getitem__ here")

# -----------------------------------------------------------------------------
# BC Actor (unchanged from your code)
# -----------------------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, env: VectorEnv, obs_horizon, pred_horizon):
        super().__init__()
        self.include_rgb = "rgb" in env.single_observation_space.keys()
        self.include_depth = "depth" in env.single_observation_space.keys()
        tot_vis = 0
        if self.include_rgb:
            tot_vis += env.single_observation_space["rgb"].shape[-1]
        if self.include_depth:
            tot_vis += env.single_observation_space["depth"].shape[-1]

        self.encoder = PlainConv(in_channels=tot_vis, out_dim=256, pool_feature_map=True)
        self.final_mlp = nn.Sequential(
            nn.Linear(obs_horizon*(256+state_dim), 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, pred_horizon*action_dim),
        )
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

    def forward(self, obs):
        # copy your full forward implementation here
        raise NotImplementedError("Insert your full Actor.forward here")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    args = tyro.cli(Args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # --- Build environments & dataset & dataloader (as in your script) ---
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default"),
        max_episode_steps=args.max_episode_steps,
    )
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    envs = make_eval_envs(
        args.env_id, args.num_eval_envs, args.sim_backend,
        env_kwargs, other_kwargs,
        video_dir=None,
        wrappers=[FlattenRGBDObservationWrapper],
    )

    tmp = gym.make(args.env_id, **env_kwargs)
    obs_space = tmp.observation_space
    include_rgb = "rgb" in obs_space.spaces
    include_depth = "depth" in obs_space.spaces
    tmp.close()

    obs_process_fn = partial(
        convert_obs,
        concat_fn=partial(np.concatenate, axis=-1),
        transpose_fn=partial(np.transpose, axes=(0,3,1,2)),
        state_obs_extractor=build_state_obs_extractor(args.env_id),
        depth="rgbd" in args.demo_path,
    )

    dataset = SmallDemoDataset_DiffusionPolicy(
        args.demo_path, obs_process_fn, obs_space,
        include_rgb, include_depth, num_traj=args.num_demos
    )
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = IterationBasedBatchSampler(BatchSampler(sampler, batch_size=args.batch_size, drop_last=True), args.total_iters)
    dataloader = DataLoader(
        dataset, batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers, pin_memory=True,
        worker_init_fn=lambda wid: worker_init_fn(wid, base_seed=args.seed),
        persistent_workers=(args.num_dataload_workers>0),
    )

    # Determine state dimension
    sample = dataset[0]["observations"]["state"]
    state_dim = sample.shape[1]

    # Instantiate BC Actor
    actor = Actor(
        state_dim,
        envs.single_action_space.shape[0],
        env=envs,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
    ).to(device)

    # --- Count & print parameters ---
    total_p, trainable_p = count_parameters(actor)
    print(f"Actor total parameters:     {total_p:,}")
    print(f"Actor trainable parameters: {trainable_p:,}")
    # ------------------------------------

    # Exit immediately after printing counts
    exit(0)

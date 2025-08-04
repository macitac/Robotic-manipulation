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
import torch.nn.functional as F
import torch.optim as optim
import tyro
from diffusers.optimization import get_scheduler
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.utils import IterationBasedBatchSampler, build_state_obs_extractor, convert_obs, worker_init_fn
from diffusion_policy.utils import load_demo_dataset
import pick_cube_new

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
    obs_horizon: int = 1       # Number of frames in the observation sequence.
    act_horizon: int = 1
    pred_horizon: int = 1     # Number of actions in the target sequence.
    diffusion_step_embed_dim: int = 64
    unet_dims: List[int] = None  # Not used in BC, kept for compatibility.
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

# Utility function to reorder keys (used during observation pre-processing)
def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset: same pre-processing as in diffusion policy
class SmallDemoDataset_DiffusionPolicy(Dataset):
    def __init__(self, data_path, obs_process_fn, obs_space, include_rgb, include_depth, num_traj):
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        
        trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
        print("Raw trajectory loaded, beginning observation pre-processing...")
        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict = reorder_keys(obs_traj_dict, obs_space)
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            if self.include_depth:
                _obs_traj_dict["depth"] = torch.tensor(_obs_traj_dict["depth"].astype(np.float32),
                                                         dtype=torch.float16)
            if self.include_rgb:
                _obs_traj_dict["rgb"] = torch.from_numpy(_obs_traj_dict["rgb"])
            _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"])
            obs_traj_dict_list.append(_obs_traj_dict)
        self.obs_keys = list(_obs_traj_dict.keys())
        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.tensor(trajectories["actions"][i])
        print("Obs/action pre-processing is done, start to pre-compute the slice indices...")
        if ("delta_pos" in args.control_mode or args.control_mode == "base_pd_joint_vel_arm_pd_joint_vel"):
            print("Detected a delta controller type, padding with a zero action.")
            self.pad_action_arm = torch.zeros((trajectories["actions"][0].shape[1] - 1,))
        else:
            raise NotImplementedError(f"Control Mode {args.control_mode} not supported")
        self.obs_horizon, self.pred_horizon = args.obs_horizon, args.pred_horizon
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == L + 1
            total_transitions += L
            pad_before = args.obs_horizon - 1
            pad_after = args.pred_horizon - args.obs_horizon
            self.slices += [
                (traj_idx, start, start + args.pred_horizon)
                for start in range(-pad_before, L - args.pred_horizon + pad_after)
            ]
        print(f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}")
        self.trajectories = trajectories

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape
        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            slice_v = v[max(0, start): start + args.obs_horizon]
            if not isinstance(slice_v, torch.Tensor):
                slice_v = torch.tensor(slice_v)
            if start < 0:
                pad_obs_seq = torch.stack([slice_v[0]] * abs(start), dim=0)
                slice_v = torch.cat((pad_obs_seq, slice_v), dim=0)
            obs_seq[k] = slice_v
        act_seq = self.trajectories["actions"][traj_idx][max(0, start): end]
        if start < 0:
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:
            gripper_action = act_seq[-1, -1]
            pad_action = torch.cat((self.pad_action_arm, gripper_action[None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
        assert obs_seq["state"].shape[0] == args.obs_horizon and act_seq.shape[0] == args.pred_horizon
        return {"observations": obs_seq, "actions": act_seq}

    def __len__(self):
        return len(self.slices)

def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    else:
        return data

# -----------------------------------------------------------------------------
# BC Agent (Actor) with sequence output.
# -----------------------------------------------------------------------------

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, env: VectorEnv, obs_horizon, pred_horizon):
        super().__init__()
        self.include_rgb = "rgb" in env.single_observation_space.keys()
        self.include_depth = "depth" in env.single_observation_space.keys()
        total_visual_channels = 0
        if self.include_rgb:
            total_visual_channels += env.single_observation_space["rgb"].shape[-1]
        if self.include_depth:
            total_visual_channels += env.single_observation_space["depth"].shape[-1]
        # Build visual encoder
        self.encoder = PlainConv(in_channels=total_visual_channels, out_dim=256, pool_feature_map=True)
        # Final MLP input dimension: obs_horizon * (256 + state_dim)
        # Final MLP output dimension: pred_horizon * action_dim (flattened sequence)
        self.final_mlp = make_mlp(obs_horizon * (256 + state_dim), [512, 256, pred_horizon * action_dim], last_act=False)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.get_eval_action = self.get_action = self.forward

    def forward(self, obs):
        # Process visual inputs:
        if self.include_rgb:
            rgb = obs["rgb"].float() / 255.0
            if rgb.dim() == 4:
                rgb = rgb.unsqueeze(1)
            elif rgb.dim() == 5 and rgb.shape[2] > 10:
                rgb = rgb.permute(0, 1, 4, 2, 3)
        if self.include_depth:
            depth = obs["depth"].float() / 1024.0
            if depth.dim() == 4:
                depth = depth.unsqueeze(1)
            elif depth.dim() == 5 and depth.shape[2] > 10:
                depth = depth.permute(0, 1, 4, 2, 3)
        if self.include_rgb and self.include_depth:
            img_seq = torch.cat([rgb, depth], dim=2)
        elif self.include_rgb:
            img_seq = rgb
        elif self.include_depth:
            img_seq = depth
        else:
            raise ValueError("No visual input provided.")
        B, T, C, H, W = img_seq.shape
        img_seq_flat = img_seq.reshape(B * T, C, H, W)
        visual_feature = self.encoder(img_seq_flat)  # (B*T, 256)
        feature_dim = visual_feature.shape[1]  # should be 256
        visual_feature = visual_feature.reshape(B, T, feature_dim)  # (B, T, 256)
        
        # Process state input:
        state = obs["state"].float()  # target shape should be (B, T, state_dim)
        if state.dim() == 2:
            # If state is (B, state_dim), unsqueeze and repeat along time dimension.
            state = state.unsqueeze(1).repeat(1, self.obs_horizon, 1)
        elif state.dim() == 3 and state.shape[1] != self.obs_horizon:
            # If temporal dimension exists but doesn't match, force it.
            state = state.unsqueeze(1).repeat(1, self.obs_horizon, 1)
        
        # Now both visual_feature and state should have shape (B, T, *)
        feature = torch.cat([visual_feature, state], dim=-1)  # (B, T, 256 + state_dim)
        feature = feature.flatten(start_dim=1)  # (B, T*(256 + state_dim))
        out = self.final_mlp(feature)  # (B, pred_horizon * action_dim)
        out = out.reshape(B, self.pred_horizon, -1)  # (B, pred_horizon, action_dim)
        return out

def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    layers = []
    for idx, c_out in enumerate(mlp_channels):
        layers.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            layers.append(act_builder())
        c_in = c_out
    return nn.Sequential(*layers)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[:-len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    if args.demo_path.endswith(".h5"):
        import json
        json_file = args.demo_path[:-2] + "json"
        with open(json_file, "r") as f:
            demo_info = json.load(f)
            if "control_mode" in demo_info["env_info"]["env_kwargs"]:
                control_mode = demo_info["env_info"]["env_kwargs"]["control_mode"]
            elif "control_mode" in demo_info["episodes"][0]:
                control_mode = demo_info["episodes"][0]["control_mode"]
            else:
                raise Exception("Control mode not found in json")
            assert control_mode == args.control_mode, (
                f"Control mode mismatched. Dataset has control mode {control_mode}, but args has {args.control_mode}"
            )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    control_mode = os.path.split(args.demo_path)[1].split(".")[2]

    env_kwargs = {
        "control_mode": args.control_mode,
        "reward_mode": "sparse",
        "obs_mode": args.obs_mode,
        "render_mode": "rgb_array",
        "human_render_camera_configs": {"shader_pack": "default"}
    }
    assert args.max_episode_steps is not None, "max_episode_steps must be specified."
    env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = {"obs_horizon": args.obs_horizon}
    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
        wrappers=[FlattenRGBDObservationWrapper],
    )

    # Create the actor model
    sample_state = torch.randn(1, envs.single_observation_space["state"].shape[0])  # Dummy state for dimension
    state_dim = sample_state.shape[1]
    action_dim = envs.single_action_space.shape[0]
    actor = Actor(state_dim, action_dim, env=envs, obs_horizon=args.obs_horizon, pred_horizon=args.pred_horizon).to(device=device)

    # Count and print the number of parameters
    num_params = count_parameters(actor)
    print(f"\nTotal trainable parameters in the model: {num_params:,}")
    print("\nParameter breakdown by layer:")
    total = 0
    for name, param in actor.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            print(f"{name}: {param_count:,}")
            total += param_count
    print(f"\nTotal verified: {total:,}")

    # Clean up
    envs.close()
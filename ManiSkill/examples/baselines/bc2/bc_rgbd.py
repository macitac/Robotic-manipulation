ALGO_NAME = "BC_Diffusion_rgbd_UNet"

import os
import random
import time
import json
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
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import tyro

# Diffusion modules
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

# Environment wrappers
from gymnasium import spaces
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

# Diffusion-policy modules
from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.make_env import make_eval_envs as diffusion_make_eval_envs
from diffusion_policy.plain_conv import PlainConv as DiffusionPlainConv
from diffusion_policy.utils import (
    IterationBasedBatchSampler,
    build_state_obs_extractor,
    convert_obs,
    worker_init_fn,
)

# Behavior cloning modules (for evaluation and env creation)
from behavior_cloning.make_env import make_eval_envs as bc_make_eval_envs
from behavior_cloning.evaluate import evaluate as bc_evaluate
# Use the diffusion evaluate function for diffusion algo
from diffusion_policy.evaluate import evaluate as diffusion_evaluate

# -----------------------------------------------------------------------------
# Unified arguments
# -----------------------------------------------------------------------------
@dataclass
class Args:
    algo: str = "diffusion"  # "diffusion" or "bc"
    exp_name: Optional[str] = None
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ManiSkill"
    wandb_entity: Optional[str] = None
    capture_video: bool = True

    # Environment and data paths
    env_id: str = "PegInsertionSide-v1"
    demo_path: str = "demos/PegInsertionSide-v1/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
    num_demos: Optional[int] = None
    max_episode_steps: Optional[int] = None  # must be set (e.g. via CLI)
    
    # Training hyperparameters (most apply to both but can be overridden)
    total_iters: int = 1_000_000
    batch_size: int = 64  # diffusion default; for bc you might want to override this
    lr: float = 1e-4     # diffusion default; for bc you can override

    # Diffusion-specific arguments
    obs_horizon: int = 2
    act_horizon: int = 8
    pred_horizon: int = 16
    diffusion_step_embed_dim: int = 64
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    n_groups: int = 8

    # Behavior cloning specific arguments
    normalize_states: bool = False  # for BC if you want to normalize state features

    # Logging, evaluation, saving
    log_freq: int = 1000
    eval_freq: int = 5000   # diffusion default; for bc you may wish to change this
    save_freq: Optional[int] = None
    num_eval_episodes: int = 100
    num_eval_envs: int = 10

    # Simulation / control
    sim_backend: str = "physx_gpu"  # diffusion default; for bc, e.g., "cpu" might be used
    control_mode: str = "pd_joint_delta_pos"
    obs_mode: str = "rgb+depth"
    num_dataload_workers: int = 16
    demo_type: Optional[str] = None

# -----------------------------------------------------------------------------
# Diffusion dataset (unchanged from diffusion code)
# -----------------------------------------------------------------------------
class SmallDemoDataset_DiffusionPolicy(Dataset):
    def __init__(self, data_path, obs_process_fn, obs_space, include_rgb, include_depth, num_traj):
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        from diffusion_policy.utils import load_demo_dataset
        trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
        print("Raw trajectory loaded, beginning observation pre-processing...")

        obs_traj_dict_list = []
        for obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict = reorder_keys(obs_traj_dict, obs_space)
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            if self.include_depth:
                _obs_traj_dict["depth"] = torch.tensor(
                    _obs_traj_dict["depth"].astype(np.float32), dtype=torch.float16
                )
            if self.include_rgb:
                _obs_traj_dict["rgb"] = torch.from_numpy(_obs_traj_dict["rgb"])
            _obs_traj_dict["state"] = torch.from_numpy(_obs_traj_dict["state"])
            obs_traj_dict_list.append(_obs_traj_dict)
        trajectories["observations"] = obs_traj_dict_list
        self.obs_keys = list(_obs_traj_dict.keys())
        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.tensor(trajectories["actions"][i])
        print("Obs/action pre-processing is done, start to pre-compute the slice indices...")

        if ("delta_pos" in args.control_mode or args.control_mode == "base_pd_joint_vel_arm_pd_joint_vel"):
            print("Detected a delta controller type, padding with a zero action to ensure the arm stays still after solving tasks.")
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

# A simple utility to reorder keys (used in the dataset)
def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out

# Utility to move mini-batches to device
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
# Diffusion Policy Agent (same as in diffusion code)
# -----------------------------------------------------------------------------
class DiffusionAgent(nn.Module):
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
        self.visual_encoder = DiffusionPlainConv(
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
        if hasattr(self, "aug") and not eval_mode:
            img_seq = self.aug(img_seq)
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
# Behavior Cloning Actor (adapted from the BC code)
# -----------------------------------------------------------------------------
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, in_channels=4):
        super().__init__()
        self.encoder = DiffusionPlainConv(in_channels=in_channels, out_dim=256, pool_feature_map=False)
        self.final_mlp = nn.Sequential(
            nn.Linear(256 + state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, rgbd, state):
        # Assume input rgbd is (B, H, W, C); permute to (B, C, H, W)
        if rgbd.ndim == 4 and rgbd.shape[-1] in [3, 4]:
            img = rgbd.permute(0, 3, 1, 2)
        else:
            img = rgbd
        feature = self.encoder(img)
        x = torch.cat([feature, state], dim=1)
        return self.final_mlp(x)

# -----------------------------------------------------------------------------
# A helper to convert a diffusion sample to a BC sample
# -----------------------------------------------------------------------------
def diffusion_to_bc_sample(sample):
    obs = sample["observations"]
    # Use the last observation in the sequence for each key
    rgb = obs["rgb"][-1] if "rgb" in obs else None
    depth = obs["depth"][-1] if "depth" in obs else None
    state = obs["state"][-1]
    if rgb is not None and depth is not None:
        rgbd = torch.cat([rgb, depth], dim=0)  # channel-first (e.g. 4 channels)
    elif rgb is not None:
        rgbd = rgb
    elif depth is not None:
        rgbd = depth
    else:
        raise ValueError("No visual observation found")
    # For action, take the first action from the action sequence
    action = sample["actions"][0]
    return {"rgbd": rgbd, "state": state, "action": action}

# -----------------------------------------------------------------------------
# Checkpoint helpers for diffusion and BC
# -----------------------------------------------------------------------------
def save_ckpt_diffusion(run_name, tag, iteration, agent, ema_agent, optimizer, lr_scheduler, ema):
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save({
        "iteration": iteration,
        "agent": agent.state_dict(),
        "ema_agent": ema_agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }, f"runs/{run_name}/checkpoints/{tag}.pt")

def load_ckpt_diffusion(checkpoint_path, agent, ema_agent, optimizer, lr_scheduler, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(checkpoint["agent"])
    ema_agent.load_state_dict(checkpoint["ema_agent"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    start_iter = checkpoint["iteration"] + 1
    return start_iter

def save_ckpt_bc(run_name, tag, actor):
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    torch.save({"actor": actor.state_dict()}, f"runs/{run_name}/checkpoints/{tag}.pt")

# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[:-3]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    # Load demo info from json (if present)
    if args.demo_path.endswith(".h5"):
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
                f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"
            )
    # Basic sanity check on horizons
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create evaluation environments.
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default"),
    )
    assert args.max_episode_steps is not None, "max_episode_steps must be specified"
    env_kwargs["max_episode_steps"] = args.max_episode_steps

    if args.algo == "diffusion":
        envs = diffusion_make_eval_envs(
            args.env_id,
            args.num_eval_envs,
            args.sim_backend,
            env_kwargs,
            other_kwargs={"obs_horizon": args.obs_horizon},
            video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
            wrappers=[FlattenRGBDObservationWrapper],
        )
    else:
        envs = bc_make_eval_envs(
            args.env_id,
            args.num_eval_envs,
            args.sim_backend,
            env_kwargs,
            video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
            wrappers=[FlattenRGBDObservationWrapper],
        )

    if args.track:
        import wandb
        config = vars(args)
        if args.algo == "diffusion":
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, env_horizon=args.max_episode_steps)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=config,
                name=run_name,
                save_code=True,
                group="DiffusionPolicy",
                tags=["diffusion_policy"],
            )
        else:
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=config,
                name=run_name,
                save_code=True,
                group="BehaviorCloning",
                tags=["behavior_cloning"],
            )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Data loading: use the diffusion dataset loader.
    tmp_env = gym.make(args.env_id, **env_kwargs)
    orignal_obs_space = tmp_env.observation_space
    include_rgb = tmp_env.unwrapped.obs_mode_struct.visual.rgb
    include_depth = tmp_env.unwrapped.obs_mode_struct.visual.depth
    tmp_env.close()

    dataset = SmallDemoDataset_DiffusionPolicy(
        data_path=args.demo_path,
        obs_process_fn=partial(
            convert_obs,
            concat_fn=partial(np.concatenate, axis=-1),
            transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),
            state_obs_extractor=build_state_obs_extractor(args.env_id),
            depth="rgbd" in args.demo_path,
        ),
        obs_space=orignal_obs_space,
        include_rgb=include_rgb,
        include_depth=include_depth,
        num_traj=args.num_demos,
    )
    sampler = RandomSampler(dataset, replacement=False)
    batch_sampler = BatchSampler(sampler, batch_size=args.batch_size, drop_last=True)
    batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
    train_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.num_dataload_workers,
        pin_memory=True,
        worker_init_fn=lambda worker_id: worker_init_fn(worker_id, base_seed=args.seed),
        persistent_workers=(args.num_dataload_workers > 0),
    )

    # -------------------------------------------------------------------------
    # Training loops
    # -------------------------------------------------------------------------
    if args.algo == "diffusion":
        # ----- Diffusion Policy Training -----
        agent = DiffusionAgent(envs, args).to(device)
        optimizer = optim.AdamW(agent.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)
        lr_scheduler = get_scheduler(
            name="cosine", optimizer=optimizer, num_warmup_steps=500, num_training_steps=args.total_iters
        )
        ema = EMAModel(parameters=agent.parameters(), power=0.75)
        ema_agent = DiffusionAgent(envs, args).to(device)

        best_eval_metrics = defaultdict(float)
        timings = defaultdict(float)

        def evaluate_and_save_best(iteration):
            if iteration % args.eval_freq == 0:
                last_tick = time.time()
                ema.copy_to(ema_agent.parameters())
                eval_metrics = diffusion_evaluate(args.num_eval_episodes, ema_agent, envs, device, args.sim_backend)
                timings["eval"] += time.time() - last_tick
                print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
                for k in eval_metrics.keys():
                    eval_metrics[k] = np.mean(eval_metrics[k])
                    writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                    print(f"{k}: {eval_metrics[k]:.4f}")
                save_on_best_metrics = ["success_once", "success_at_end"]
                for k in save_on_best_metrics:
                    if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                        best_eval_metrics[k] = eval_metrics[k]
                        save_ckpt_diffusion(run_name, f"best_eval_{k}", iteration, agent, ema_agent, optimizer, lr_scheduler, ema)
                        print(f"New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint.")

        def log_metrics(iteration, total_loss):
            if iteration % args.log_freq == 0:
                writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], iteration)
                writer.add_scalar("losses/total_loss", total_loss.item(), iteration)
                for k, v in timings.items():
                    writer.add_scalar(f"time/{k}", v, iteration)

        start_iter = 0
        if hasattr(args, "resume") and getattr(args, "resume", False) and hasattr(args, "resume_path") and args.resume_path is not None and os.path.exists(args.resume_path):
            start_iter = load_ckpt_diffusion(args.resume_path, agent, ema_agent, optimizer, lr_scheduler, device)
            print(f"Resumed training from iteration {start_iter}")

        agent.train()
        pbar = tqdm(total=args.total_iters - start_iter, initial=start_iter)
        iteration = start_iter
        dataloader_iter = iter(train_dataloader)
        last_tick = time.time()

        while iteration < args.total_iters:
            try:
                data_batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(train_dataloader)
                data_batch = next(dataloader_iter)

            data_batch = move_to_device(data_batch, device)
            timings["data_loading"] += time.time() - last_tick
            last_tick = time.time()

            total_loss = agent.compute_loss(
                obs_seq=data_batch["observations"],
                action_seq=data_batch["actions"],
            )
            timings["forward"] += time.time() - last_tick
            last_tick = time.time()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            timings["backward"] += time.time() - last_tick
            last_tick = time.time()

            ema.step(agent.parameters())
            timings["ema"] += time.time() - last_tick

            evaluate_and_save_best(iteration)
            log_metrics(iteration, total_loss)

            if args.save_freq is not None and iteration % args.save_freq == 0:
                save_ckpt_diffusion(run_name, str(iteration), iteration, agent, ema_agent, optimizer, lr_scheduler, ema)
            pbar.update(1)
            pbar.set_postfix({"loss": total_loss.item()})
            iteration += 1
            last_tick = time.time()

        evaluate_and_save_best(args.total_iters)
        log_metrics(args.total_iters, total_loss)

    else:
        # ----- Behavior Cloning Training -----
        # Here we re-use the diffusion dataset but convert each sample to a BC sample.
        sample0 = diffusion_to_bc_sample(dataset[0])
        state_dim = sample0["state"].shape[0]
        action_dim = envs.single_action_space.shape[0]
        actor = Actor(state_dim, action_dim).to(device)
        optimizer = optim.Adam(actor.parameters(), lr=args.lr)
        best_eval_metrics = defaultdict(float)

        for iteration, batch in enumerate(train_dataloader):
            # Convert each sample in the batch to BC format
            bc_batch = {"rgbd": [], "state": [], "action": []}
            for sample in batch:
                converted = diffusion_to_bc_sample(sample)
                bc_batch["rgbd"].append(converted["rgbd"])
                bc_batch["state"].append(converted["state"])
                bc_batch["action"].append(converted["action"])
            bc_batch["rgbd"] = torch.stack(bc_batch["rgbd"], dim=0)
            bc_batch["state"] = torch.stack(bc_batch["state"], dim=0)
            bc_batch["action"] = torch.stack(bc_batch["action"], dim=0)
            # Normalize rgbd (assume first three channels are rgb and fourth is depth)
            norm_tensor = torch.tensor([255.0, 255.0, 255.0, 1024.0]).to(device)
            bc_batch["rgbd"] = bc_batch["rgbd"].float() / norm_tensor.view(4, 1, 1)

            optimizer.zero_grad()
            preds = actor(bc_batch["rgbd"], bc_batch["state"])
            loss = F.mse_loss(preds, bc_batch["action"])
            loss.backward()
            optimizer.step()

            if iteration % args.log_freq == 0:
                print(f"Iteration {iteration}, loss: {loss.item()}")
                writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], iteration)
                writer.add_scalar("losses/total_loss", loss.item(), iteration)

            if iteration % args.eval_freq == 0:
                actor.eval()
                norm_tensor = torch.tensor([255.0, 255.0, 255.0, 1024.0]).to(device)

                def sample_fn(obs):
                    if isinstance(obs["rgbd"], np.ndarray):
                        for k, v in obs.items():
                            obs[k] = torch.from_numpy(v).float().to(device)
                    obs["rgbd"] = obs["rgbd"] / norm_tensor.view(4, 1, 1)
                    action = actor(obs["rgbd"], obs["state"])
                    if args.sim_backend == "cpu":
                        action = action.cpu().numpy()
                    return action

                with torch.no_grad():
                    eval_metrics = bc_evaluate(args.num_eval_episodes, sample_fn, envs)
                actor.train()
                print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
                for k in eval_metrics.keys():
                    eval_metrics[k] = np.mean(eval_metrics[k])
                    writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                    print(f"{k}: {eval_metrics[k]:.4f}")

                save_on_best_metrics = ["success_once", "success_at_end"]
                for k in save_on_best_metrics:
                    if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                        best_eval_metrics[k] = eval_metrics[k]
                        save_ckpt_bc(run_name, f"best_eval_{k}", actor)
                        print(f"New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint.")

            if args.save_freq is not None and iteration % args.save_freq == 0:
                save_ckpt_bc(run_name, str(iteration), actor)

    envs.close()
    writer.close()
    if args.track:
        import wandb
        wandb.finish()


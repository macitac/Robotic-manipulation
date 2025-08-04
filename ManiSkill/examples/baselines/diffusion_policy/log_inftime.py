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
import pick_cube_new

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "PegInsertionSide-v1"
    """the id of the environment"""
    demo_path: str = (
        "demos/PegInsertionSide-v1/trajectory.state.pd_ee_delta_pose.physx_cpu.h5"
    )
    """the path of demo dataset, it is expected to be a ManiSkill dataset h5py format file"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 1
    """the batch size of sample from the replay memory"""

    # Diffusion Policy specific arguments
    lr: float = 1e-4
    """the learning rate of the diffusion policy"""
    obs_horizon: int = 2  # 1,2,4 work well
    act_horizon: int = 8  # 4,8,15 work well
    pred_horizon: int = 16
    """for example, 16->8 leads to worse performance"""
    diffusion_step_embed_dim: int = 64
    """not very important"""
    unet_dims: List[int] = field(default_factory=lambda: [64, 128, 256])
    """default setting is about ~4.5M params"""
    n_groups: int = 8

    # Environment/experiment specific arguments
    obs_mode: str = "rgb+depth"
    """The observation mode to use for the environment. Can be "rgb", "depth", or "rgb+depth"."""
    max_episode_steps: Optional[int] = None
    """Max episode steps for the environment."""
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 5000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints"""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_eval_envs: int = 1
    """the number of parallel environments to evaluate the agent on"""
    sim_backend: str = "physx_gpu"
    """the simulation backend to use for evaluation environments"""
    num_dataload_workers: int = 16
    """the number of workers to use for loading training data"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for evaluation environments. Must match the dataset."""
    demo_type: Optional[str] = None

    # New arguments for resuming training
    resume: bool = False
    """if toggled, resume training from a checkpoint"""
    resume_path: Optional[str] = None
    """path to the checkpoint to resume from (e.g. runs/your_run/checkpoints/last.pt)"""

def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict) or isinstance(v, spaces.Dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out

class SmallDemoDataset_DiffusionPolicy(Dataset):
    def __init__(self, data_path, obs_process_fn, obs_space, include_rgb, include_depth, num_traj):
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        from diffusion_policy.utils import load_demo_dataset
        trajectories = load_demo_dataset(data_path, num_traj=num_traj, concat=False)
        print("Raw trajectory loaded, beginning observation pre-processing...")

        # Pre-process the observations; keep data on CPU
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
        trajectories["observations"] = obs_traj_dict_list
        self.obs_keys = list(_obs_traj_dict.keys())
        # Pre-process the actions; keep on CPU
        for i in range(len(trajectories["actions"])):
            trajectories["actions"][i] = torch.tensor(trajectories["actions"][i])
        print("Obs/action pre-processing is done, start to pre-compute the slice indices...")

        # Pre-compute slice indices specific to Diffusion Policy
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
            # Leave data on CPU; transfer happens later.
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

def benchmark_agent_inference(agent, env, device, obs_horizon, num_runs=100, batch_size=1):
    """
    Benchmark the inference speed of the diffusion policy agent using real data from the environment.
    The observation from env.reset() is repeated along the time dimension (obs_horizon) to form the input sequence.
    """
    print(f"\nRunning agent inference benchmark with batch_size={batch_size} using real data...")
    
    # Get a real observation from the vectorized evaluation environment.
    # Note: env.reset() returns (obs, info). Here, obs is expected to be a dict.
    obs, _ = env.reset()
    
    # Build an observation sequence dictionary (obs_seq) that matches the agent's expected input.
    # For visual inputs (e.g. "rgb" or "depth"), if the reset observation is a single frame (shape: [B, H, W, C]),
    # we add a time dimension and repeat it to match obs_horizon.
    obs_seq = {}
    for key, value in obs.items():
        # Select the first batch_size elements.
        value = value[:batch_size]
        # Process visual inputs.
        if key in ["rgb", "depth"]:
            # If the observation is a single frame, add a time dimension.
            if len(value.shape) == 3:  # (H, W, C)
                value = np.expand_dims(value, axis=0)  # now (1, H, W, C)
            if len(value.shape) == 4:  # (B, H, W, C)
                value = np.expand_dims(value, axis=1)  # now (B, 1, H, W, C)
                # Repeat along the time dimension.
                value = np.repeat(value, obs_horizon, axis=1)
            # Convert to torch tensor.
            obs_seq[key] = torch.tensor(value).to(device)
        else:
            # For "state" (or other keys), if the shape is (B, state_dim), add a time dimension.
            if len(value.shape) == 1:  # (state_dim,)
                value = np.expand_dims(value, axis=0)  # (1, state_dim)
            if len(value.shape) == 2:  # (B, state_dim)
                value = np.expand_dims(value, axis=1)  # (B, 1, state_dim)
                value = np.repeat(value, obs_horizon, axis=1)
            obs_seq[key] = torch.tensor(value).to(device)

    # Warm up the agent (to avoid one-time overhead).
    print("Running warmup iterations...")
    for _ in range(10):
        torch.cuda.synchronize()
        with torch.no_grad():
            _ = agent.get_action(obs_seq)
        torch.cuda.synchronize()
    
    # Benchmark inference over num_runs iterations.
    total_time = 0.0
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = agent.get_action(obs_seq)
        torch.cuda.synchronize()
        total_time += time.time() - start
    
    avg_time = total_time / num_runs
    fps = batch_size / avg_time
    
    print("\nAgent Inference Benchmark Results:")
    print(f"Batch size: {batch_size}")
    print(f"Number of runs: {num_runs}")
    print(f"Average inference time per batch: {avg_time * 1000:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    return avg_time, fps


class Agent(nn.Module):
    def __init__(self, env: VectorEnv, args: Args):
        super().__init__()
        self.obs_horizon = args.obs_horizon
        self.act_horizon = args.act_horizon
        self.pred_horizon = args.pred_horizon
        self.env = env  # Store env reference for benchmarking
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
        self.inference_times = []
        self.batch_sizes = []
        
    

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
        torch.cuda.synchronize() 
        start_time = time.time()
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
                noisy_action_seq = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=noisy_action_seq).prev_sample
            
            # End timing and record
            torch.cuda.synchronize()
            end_time = time.time()
            inference_time = end_time - start_time
            self.inference_times.append(inference_time)
            self.batch_sizes.append(B)
            
            # Print stats periodically
            if len(self.inference_times) % 100 == 0:
                avg_time = np.mean(self.inference_times[-100:])
                avg_batch = np.mean(self.batch_sizes[-100:])
                fps = avg_batch / avg_time
                print(f"\nInference stats (last 100): {avg_time*1000:.2f}ms per batch (batch_size={avg_batch:.1f}), {fps:.2f} FPS")
        
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        return noisy_action_seq[:, start:end]

    def benchmark_inference_speed(self, num_runs=100, batch_size=1):
        """Benchmark the inference speed of the agent using real data from the environment"""
        print(f"\nRunning inference benchmark with batch_size={batch_size} using real data...")

        # Reset the environment to get a real observation
        obs, _ = self.env.reset()
        # If the environment returns a dict, select only the first batch_size elements
        if isinstance(obs, dict):
            obs = {k: v[:batch_size] for k, v in obs.items()}
        else:
            obs = obs[:batch_size]

        # Convert observations to torch tensors and create a sequence by repeating each observation
        obs_seq = {}
        for key, value in obs.items():
            # Convert to tensor and send to the proper device
            value_tensor = torch.tensor(value).to(value.dtype).to(next(self.parameters()).device)
            # Add a time dimension and repeat it to form a sequence of length self.obs_horizon
            obs_seq[key] = value_tensor.unsqueeze(1).repeat(1, self.obs_horizon, *([1] * (value_tensor.ndim - 1)))

        # Warm up
        print("Running warmup...")
        for _ in range(10):
            _ = self.get_action(obs_seq)

        # Benchmark
        print(f"Running benchmark with {num_runs} iterations...")
        start_time = time.time()
        for _ in range(num_runs):
            _ = self.get_action(obs_seq)
        total_time = time.time() - start_time

        # Calculate stats
        avg_time = total_time / num_runs
        fps = batch_size / avg_time

        print("\nInference Benchmark Results:")
        print(f"Batch size: {batch_size}")
        print(f"Number of runs: {num_runs}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per inference: {avg_time*1000:.2f}ms")
        print(f"Throughput: {fps:.2f} FPS")

        return avg_time, fps


def save_ckpt(run_name, tag, iteration):
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    ema.copy_to(ema_agent.parameters())
    torch.save({
        "iteration": iteration,
        "agent": agent.state_dict(),
        "ema_agent": ema_agent.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }, f"runs/{run_name}/checkpoints/{tag}.pt")

def load_ckpt(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(checkpoint["agent"])
    ema_agent.load_state_dict(checkpoint["ema_agent"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    start_iter = checkpoint["iteration"] + 1
    return start_iter

def evaluate_and_save_best(iteration):
    
    if iteration % args.eval_freq == 0 or (args.resume and iteration == start_iter):
        original_num_envs = args.num_eval_envs
        args.num_eval_envs = 1  # <-- TEMPORARY CHANGE
        last_tick = time.time()
        ema.copy_to(ema_agent.parameters())
        eval_metrics = evaluate(args.num_eval_episodes, ema_agent, envs, device, args.sim_backend)
        args.num_eval_envs = original_num_envs
        # Benchmark inference speed periodically
        if iteration == 0 or iteration % (args.eval_freq * 5) == 0:
            ema_agent.benchmark_inference_speed(num_runs=100, batch_size=args.num_eval_envs)
            
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
                save_ckpt(run_name, f"best_eval_{k}", iteration)
                print(f"New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint.")

def log_metrics(iteration, total_loss):
    if iteration % args.log_freq == 0:
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], iteration)
        writer.add_scalar("losses/total_loss", total_loss.item(), iteration)
        for k, v in timings.items():
            writer.add_scalar(f"time/{k}", v, iteration)

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
                f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"
            )
    assert args.obs_horizon + args.act_horizon - 1 <= args.pred_horizon
    assert args.obs_horizon >= 1 and args.act_horizon >= 1 and args.pred_horizon >= 1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Create evaluation environment
    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        human_render_camera_configs=dict(shader_pack="default")
    )
    assert args.max_episode_steps is not None, "max_episode_steps must be specified as it impacts task solve speed"
    env_kwargs["max_episode_steps"] = args.max_episode_steps
    other_kwargs = dict(obs_horizon=args.obs_horizon)
    envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        other_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
        wrappers=[FlattenRGBDObservationWrapper],
    )

    if args.track:
        import wandb
        config = vars(args)
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
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n%s" %
                    ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    obs_process_fn = partial(
        convert_obs,
        concat_fn=partial(np.concatenate, axis=-1),
        transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),
        state_obs_extractor=build_state_obs_extractor(args.env_id),
        depth="rgbd" in args.demo_path
    )
    tmp_env = gym.make(args.env_id, **env_kwargs)
    orignal_obs_space = tmp_env.observation_space
    include_rgb = tmp_env.unwrapped.obs_mode_struct.visual.rgb
    include_depth = tmp_env.unwrapped.obs_mode_struct.visual.depth
    tmp_env.close()

    dataset = SmallDemoDataset_DiffusionPolicy(
        data_path=args.demo_path,
        obs_process_fn=obs_process_fn,
        obs_space=orignal_obs_space,
        include_rgb=include_rgb,
        include_depth=include_depth,
        num_traj=args.num_demos
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

    agent = Agent(envs, args).to(device)
    optimizer = optim.AdamW(params=agent.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_iters,
    )
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = Agent(envs, args).to(device)

    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    start_iter = 0
    if args.resume and args.resume_path is not None and os.path.exists(args.resume_path):
        start_iter = load_ckpt(args.resume_path)
        print(f"Resumed training from iteration {start_iter}")
        
        # Optionally, perform some debug checks.
        print("Evaluating the loaded policy...")
        ema.copy_to(ema_agent.parameters())
        eval_metrics = evaluate(args.num_eval_episodes, ema_agent, envs, device, args.sim_backend)
        print("Evaluation results on resumed checkpoint:")
        for k in eval_metrics.keys():
            print(f"{k}: {np.mean(eval_metrics[k]):.4f}")
        
        # Now call the inference benchmark.
        avg_time, fps = benchmark_agent_inference(agent, envs, device, obs_horizon=args.obs_horizon, num_runs=100, batch_size=1)
        print(f"Inference benchmark: {avg_time * 1000:.2f} ms per batch, {fps:.2f} FPS")
        
        envs.close()
        writer.close()
        exit(0)


    agent.train()
    pbar = tqdm(total=args.total_iters - start_iter, initial=start_iter)
    dataloader_iter = iter(train_dataloader)
    last_tick = time.time()
    iteration = start_iter
    while iteration < args.total_iters:
        try:
            data_batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_dataloader)
            data_batch = next(dataloader_iter)

        # Now, move the mini-batch to GPU in the main process.
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
            save_ckpt(run_name, str(iteration), iteration)
        pbar.update(1)
        pbar.set_postfix({"loss": total_loss.item()})
        iteration += 1
        last_tick = time.time()

    # Final evaluation and benchmark
    evaluate_and_save_best(args.total_iters)
    log_metrics(args.total_iters, total_loss)
    
    print("\nRunning final inference benchmarks...")
    ema_agent.benchmark_inference_speed(num_runs=100, batch_size=1)
    #ema_agent.benchmark_inference_speed(num_runs=100, batch_size=args.num_eval_envs)
    if args.num_eval_envs > 1:
        ema_agent.benchmark_inference_speed(num_runs=100, batch_size=args.num_eval_envs*2)

    envs.close()
    writer.close()
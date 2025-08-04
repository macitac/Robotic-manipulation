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

import torchvision.models as models  # Added missing import

from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.evaluate import evaluate
from diffusion_policy.make_env import make_eval_envs
from diffusion_policy.plain_conv import PlainConv
from diffusion_policy.utils import IterationBasedBatchSampler, build_state_obs_extractor, convert_obs, worker_init_fn
from diffusion_policy.utils import load_demo_dataset

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
    obs_horizon: int = 10       # Number of frames in the observation sequence.
    act_horizon: int = 1
    pred_horizon: int = 1     # Number of actions in the target sequence.
    diffusion_step_embed_dim: int = 64
    unet_dims: List[int] = None  # Not used in BC, kept for compatibility.
    n_groups: int = 8
    obs_mode: str = "rgb+depth"
    max_episode_steps: Optional[int] = None
    log_freq: int = 1000
    eval_freq: int = 5000
    save_freq: Optional[int] = None
    num_eval_episodes: int = 100
    num_eval_envs: int = 10
    sim_backend: str = "physx_gpu"
    num_dataload_workers: int = 16
    control_mode: str = "pd_joint_delta_pos"
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
        trajectories["observations"] = obs_traj_dict_list
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
class ActorRNNVisualGMM(nn.Module):
    def __init__(self, state_dim, action_dim, obs_horizon, pred_horizon, hidden_dim=256, num_modes=5):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Visual encoder (ResNet18 backbone, pretrained)
        backbone = models.resnet18(pretrained=True)
        self.visual_encoder = nn.Sequential(*list(backbone.children())[:-1])  # remove last FC
        self.visual_out_dim = 512  # ResNet18 output

        # LSTM to process sequence of [visual + state]
        self.lstm = nn.LSTM(
            input_size=self.visual_out_dim + state_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # GMM head
        self.mean = nn.Linear(hidden_dim, num_modes * action_dim)
        self.log_std = nn.Linear(hidden_dim, num_modes * action_dim)
        self.logits = nn.Linear(hidden_dim, num_modes)

    def forward(self, obs):
        rgb_seq = obs["rgb"].float() / 255.0  # (B, T, C, H, W)
        B, T, C, H, W = rgb_seq.shape
        rgb_seq = rgb_seq.reshape(B * T, C, H, W)
        with torch.no_grad():
            visual_feat = self.visual_encoder(rgb_seq).squeeze(-1).squeeze(-1)  # (B*T, 512)
        visual_feat = visual_feat.view(B, T, self.visual_out_dim)

        state_seq = obs["state"].float()  # (B, T, state_dim)
        input_seq = torch.cat([visual_feat, state_seq], dim=-1)  # (B, T, state+visual)

        lstm_out, _ = self.lstm(input_seq)
        last_hidden = lstm_out[:, -1, :]  # (B, hidden_dim)

        means = self.mean(last_hidden).view(B, self.num_modes, self.action_dim)
        log_stds = self.log_std(last_hidden).view(B, self.num_modes, self.action_dim)
        stds = torch.exp(log_stds.clamp(min=-20, max=2))
        logits = self.logits(last_hidden)

        return {
            "means": means,
            "stds": stds,
            "logits": logits
        }

    def sample_action(self, gmm_params):
        logits = gmm_params["logits"]
        mix_dist = torch.distributions.Categorical(logits=logits)
        chosen = mix_dist.sample()
        B = logits.shape[0]
        idx = chosen.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.action_dim)
        means = torch.gather(gmm_params["means"], 1, idx).squeeze(1)
        stds = torch.gather(gmm_params["stds"], 1, idx).squeeze(1)
        normal = torch.distributions.Normal(means, stds)
        return normal.sample()
    def get_action(self, obs):
        self.eval()
        with torch.no_grad():
            # Ensure 'rgb' is channels-first.
            rgb = obs["rgb"]
            # Handle single frame: (B, H, W, C) -> (B, C, H, W)
            if rgb.dim() == 4 and rgb.shape[-1] == 3:
                rgb = rgb.permute(0, 3, 1, 2)
            # Handle sequence: (B, T, H, W, C) -> (B, T, C, H, W)
            elif rgb.dim() == 5 and rgb.shape[-1] == 3:
                rgb = rgb.permute(0, 1, 4, 2, 3)
            obs["rgb"] = rgb
            # If there's no time dimension, add one.
            if obs["rgb"].dim() == 4:
                obs["rgb"] = obs["rgb"].unsqueeze(1)
            
            # Forward pass to get GMM parameters and sample an action.
            gmm_params = self.forward(obs)
            action = self.sample_action(gmm_params)  # Expected shape: (B, action_dim)
            
            # Expand the single action to a sequence of actions over the prediction horizon.
            # This makes the output shape: (B, pred_horizon, action_dim)
            action_seq = action.unsqueeze(1).repeat(1, self.pred_horizon, 1)
        return action_seq




def make_mlp(in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True):
    c_in = in_channels
    layers = []
    for idx, c_out in enumerate(mlp_channels):
        layers.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            layers.append(act_builder())
        c_in = c_out
    return nn.Sequential(*layers)

def save_ckpt(run_name, tag):
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    torch.save({"actor": actor.state_dict()}, f"runs/{run_name}/checkpoints/{tag}.pt")

def gmm_nll_loss(gmm_params, target_actions):
    B, num_modes, act_dim = gmm_params["means"].shape
    target = target_actions[:, -1, :].unsqueeze(1).expand(-1, num_modes, -1)

    dist = torch.distributions.Normal(gmm_params["means"], gmm_params["stds"])
    log_probs = dist.log_prob(target).sum(dim=-1)
    log_probs += F.log_softmax(gmm_params["logits"], dim=-1)

    nll = -torch.logsumexp(log_probs, dim=-1).mean()
    return nll

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

    if args.track:
        import wandb
        config = vars(args)
        config["eval_env_cfg"] = {**env_kwargs, "num_envs": args.num_eval_envs, "env_id": args.env_id, "env_horizon": args.max_episode_steps}
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
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n" +
                    "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]))

    obs_process_fn = partial(
        convert_obs,
        concat_fn=partial(np.concatenate, axis=-1),
        transpose_fn=partial(np.transpose, axes=(0, 3, 1, 2)),
        state_obs_extractor=build_state_obs_extractor(args.env_id),
        depth="rgbd" in args.demo_path
    )
    tmp_env = gym.make(args.env_id, **env_kwargs)
    original_obs_space = tmp_env.observation_space
    include_rgb = "rgb" in original_obs_space.keys()
    include_depth = "depth" in original_obs_space.keys()
    tmp_env.close()

    dataset = SmallDemoDataset_DiffusionPolicy(
        data_path=args.demo_path,
        obs_process_fn=obs_process_fn,
        obs_space=original_obs_space,
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

    sample_state = dataset[0]["observations"]["state"]
    state_dim = sample_state.shape[1]
    actor = ActorRNNVisualGMM(
            state_dim=state_dim,
            action_dim=envs.single_action_space.shape[0],
            obs_horizon=args.obs_horizon,
            pred_horizon=args.pred_horizon,
        ).to(device=device)

    optimizer = optim.AdamW(actor.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)
    best_eval_metrics = defaultdict(float)
    timings = defaultdict(float)

    def evaluate_and_save_best(iteration):
        if iteration % args.eval_freq == 0:
            eval_metrics = evaluate(args.num_eval_episodes, actor, envs, device, args.sim_backend)
            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                print(f"{k}: {eval_metrics[k]:.4f}")
            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics.get(k, -1):
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(run_name, f"best_eval_{k}")
                    print(f"New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint.")

    def log_metrics(iteration, total_loss):
        if iteration % args.log_freq == 0:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], iteration)
            writer.add_scalar("losses/total_loss", total_loss.item(), iteration)
            for k, v in timings.items():
                writer.add_scalar(f"time/{k}", v, iteration)

    start_iter = 0
    if args.resume and args.resume_path is not None and os.path.exists(args.resume_path):
        # TODO: load the checkpoint if necessary
        pass

    actor.train()
    # Removed the premature test forward pass using data_batch

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

        preds = actor(data_batch["observations"])
        total_loss = gmm_nll_loss(preds, data_batch["actions"])  # Use custom GMM NLL loss
        timings["forward"] += time.time() - last_tick
        last_tick = time.time()

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        timings["backward"] += time.time() - last_tick
        last_tick = time.time()

        log_metrics(iteration, total_loss)
        evaluate_and_save_best(iteration)

        pbar.update(1)
        pbar.set_postfix({"loss": total_loss.item()})
        iteration += 1
        last_tick = time.time()

    evaluate_and_save_best(args.total_iters)
    log_metrics(args.total_iters, total_loss)
    
    def measure_inference_time(actor, obs, device, runs=100):
        import time
        actor.eval()
        obs = move_to_device(obs, device)

        # Warm-up
        with torch.no_grad():
            for _ in range(10):
                _ = actor(obs)

        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(runs):
                _ = actor(obs)
        torch.cuda.synchronize()
        end = time.time()

        avg_ms = (end - start) / runs * 1000
        return avg_ms
    
    sample_obs = dataset[0]["observations"]
    sample_obs = {k: v.unsqueeze(0).to(device) for k, v in sample_obs.items()}  # batch dim
    inference_time_ms = measure_inference_time(actor, sample_obs, device)
    print(f"Average inference time: {inference_time_ms:.2f} ms")
    writer.add_scalar("eval/inference_time_ms", inference_time_ms, iteration)

    envs.close()
    writer.close()

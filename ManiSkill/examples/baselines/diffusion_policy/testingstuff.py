import torch
import torchlens as tl
import numpy as np
from diffusion_policy.make_env import make_eval_envs
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from diffusion_policy.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.plain_conv import PlainConv
from torch import nn
from bc_rgb_cpoint import Actor
# === Minimal Args class ===
class Args:
    def __init__(self):
        self.obs_horizon = 1
        self.pred_horizon = 1
        self.act_horizon = 1
        self.unet_dims = [64, 128, 256]
        self.diffusion_step_embed_dim = 64
        self.env_id = "PickCube-v1"
        self.num_eval_envs = 1
        self.obs_mode = "rgb"
        self.max_episode_steps = 100
        self.sim_backend = "physx_cpu"
        self.control_mode = "pd_joint_delta_pos"
        self.batch_size = 64
        self.n_groups = 8



# === Environment creation ===
args = Args()
env_kwargs = {
    "control_mode": args.control_mode,
    "reward_mode": "sparse",
    "obs_mode": args.obs_mode,
    "render_mode": "rgb_array",
    "max_episode_steps": args.max_episode_steps,
    "human_render_camera_configs": {"shader_pack": "default"},
}
envs = make_eval_envs(
    args.env_id,
    args.num_eval_envs,
    args.sim_backend,
    env_kwargs,
    {"obs_horizon": args.obs_horizon},
    wrappers=[FlattenRGBDObservationWrapper],
)

agent = Actor(envs, args)
agent.eval()

# === Create dummy inputs ===
B = 2
act_dim = args.act_horizon
state_dim = envs.single_observation_space["state"].shape[1]
dummy_actions = torch.randn((B, args.pred_horizon, act_dim))
dummy_timesteps = torch.randint(0, agent.num_diffusion_iters, (B,), dtype=torch.long)
dummy_obs_cond = torch.randn((B, args.obs_horizon * (256 + state_dim)))

# === TorchLens trace ===
model_history = tl.log_forward_pass(
    agent.noise_pred_net,
    (dummy_actions, dummy_timesteps),
    {"global_cond": dummy_obs_cond},
    layers_to_save="all",
    vis_opt="unrolled",
)



print(model_history)
model_history.visualize(save_path="unet_architecture.svg")

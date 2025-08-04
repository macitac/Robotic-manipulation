#!/usr/bin/env python
import argparse
import os
import sys
import gymnasium as gym
from mani_skill.utils.wrappers import CPUGymWrapper
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from mani_skill.utils import common

# Import the DiffusionPolicy module provided by ManiSkill.
# Make sure ManiSkill is installed (pip install -e .) so this import works.
try:
    import DiffusionPolicy
except ModuleNotFoundError:
    sys.exit("Could not find the DiffusionPolicy module. "
             "Please install ManiSkill in editable mode (pip install -e .) from the ManiSkill repo.")

def cpu_make_env(env_id, env_kwargs=dict()):
    def thunk():
        env = gym.make(env_id, reconfiguration_freq=1, **env_kwargs)
        env = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
        return env
    return thunk

def evaluate(n: int, agent, eval_envs, device, sim_backend: str, progress_bar: bool = True):
    """
    Evaluate the given agent over n episodes on the provided vectorized environments.
    """
    agent.eval()
    if progress_bar:
        pbar = tqdm(total=n)
    with torch.no_grad():
        eval_metrics = defaultdict(list)
        obs, info = eval_envs.reset()
        eps_count = 0
        while eps_count < n:
            obs_tensor = common.to_tensor(obs, device)
            action_seq = agent.get_action(obs_tensor)
            if sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()
            # Execute the sequence of actions.
            for i in range(action_seq.shape[1]):
                obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                if truncated.any():
                    break

            if truncated.any():
                # All environments are expected to terminate simultaneously for fair evaluation.
                assert truncated.all() == truncated.any(), (
                    "All episodes should truncate at the same time for fair evaluation with other algorithms"
                )
                # Collect metrics.
                if isinstance(info["final_info"], dict):
                    for k, v in info["final_info"]["episode"].items():
                        eval_metrics[k].append(v.float().cpu().numpy())
                else:
                    for final_info in info["final_info"]:
                        for k, v in final_info["episode"].items():
                            eval_metrics[k].append(v)
                eps_count += eval_envs.num_envs
                if progress_bar:
                    pbar.update(eval_envs.num_envs)
    agent.train()
    # Stack metrics for reporting.
    for k in eval_metrics.keys():
        eval_metrics[k] = np.stack(eval_metrics[k])
    return eval_metrics

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up evaluation environments.
    env_kwargs = dict(obs_mode=args.obs_mode)
    # Using AsyncVectorEnv if more than one environment is requested.
    if args.num_eval_envs == 1:
        vector_cls = gym.vector.SyncVectorEnv
    else:
        vector_cls = lambda env_fns: gym.vector.AsyncVectorEnv(env_fns, context="forkserver")
    eval_envs = vector_cls([cpu_make_env(args.env, env_kwargs) for _ in range(args.num_eval_envs)])

    # Load the pretrained policy using DiffusionPolicy.from_pretrained.
    policy = DiffusionPolicy.from_pretrained(args.policy_checkpoint, map_location=device)

    # Evaluate the policy.
    metrics = evaluate(args.num_episodes, policy, eval_envs, device, args.sim_backend)

    # Print averaged evaluation metrics.
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key} mean: {np.mean(value)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Policy on ManiSkill environment.")
    parser.add_argument("--env", type=str, default="StackCube-v1",
                        help="The ManiSkill environment id to use (default: StackCube-v1)")
    parser.add_argument("--policy-checkpoint", type=str, required=True,
                        help="Path to the pretrained policy checkpoint")
    parser.add_argument("--sim_backend", type=str, default="physx_cpu",
                        help="Simulation backend to use (default: physx_cpu)")
    parser.add_argument("--num_eval_envs", type=int, default=8,
                        help="Number of parallel evaluation environments (default: 8)")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Total number of episodes to evaluate (default: 100)")
    parser.add_argument("--obs_mode", type=str, default="rgb",
                        help="Observation mode to use in the environment (default: rgb)")
    args = parser.parse_args()
    main(args)

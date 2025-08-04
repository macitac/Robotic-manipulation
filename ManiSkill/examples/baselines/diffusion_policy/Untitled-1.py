#!/usr/bin/env python
import os
import sys
import argparse
import torch
import gym
from tqdm import tqdm

# ManiSkill evaluation utilities
from mani_skill.evaluation.evaluator import BaseEvaluator
from mani_skill.utils.io_utils import dump_json, load_json, write_txt
from mani_skill.utils.wrappers import RecordEpisode

# Import wrappers and environment creator used during training
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper
from diffusion_policy.make_env import make_eval_envs

# Import your diffusion policy Agent.
# Adjust the import path to match your project structure.
from diffusion_policy.train_script import Agent

# -----------------------------------------------------------------------------
# Evaluator class adapted for local evaluation
# -----------------------------------------------------------------------------
class Evaluator(BaseEvaluator):
    """Local evaluation."""
    def __init__(self, output_dir: str, record_dir=None):
        if os.path.exists(output_dir):
            print(f"{output_dir} already exists.")
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.record_dir = record_dir

    def setup(self, env_id, policy, render_mode, env_kwargs):
        # Initialize the environment using the parent class's setup.
        super().setup(env_id, policy, render_mode=render_mode, env_kwargs=env_kwargs)
        if self.record_dir is not None:
            self.env = RecordEpisode(self.env, self.record_dir, clean_on_close=False)

    def submit(self):
        # Save per-episode results.
        json_path = os.path.join(self.output_dir, "episode_results.json")
        dump_json(json_path, self.result)
        print("The per-episode evaluation result is saved to {}.".format(json_path))

        # Save averaged metrics.
        json_path = os.path.join(self.output_dir, "average_metrics.json")
        merged_metrics = self.merge_result()
        self.merged_metrics = merged_metrics
        dump_json(json_path, merged_metrics)
        print("The averaged evaluation result is saved to {}.".format(json_path))

    def error(self, *args):
        write_txt(os.path.join(self.output_dir, "error.log"), args)

# -----------------------------------------------------------------------------
# Progress callback for evaluation episodes
# -----------------------------------------------------------------------------
class TqdmCallback:
    def __init__(self, n: int):
        self.n = n
        self.pbar = tqdm(total=n)

    def __call__(self, i, metrics):
        self.pbar.update()

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--env-id", type=str, required=True, help="Environment ID"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="~/Thesis-results/ManiSkill/examples/baselines/diffusion_policy/runs/bc-StackCube-v1-rgb-400_motionplanning_demos-1/checkpoints/best_eval_succes_once.pt",
        help="Path to the saved policy checkpoint.",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to the config file. If None, a dummy config is used.",
    )
    parser.add_argument(
        "-n", "--num-episodes", type=int, help="Number of episodes to evaluate."
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        help="Directory to record videos and trajectories. Use '@' to use the output directory.",
    )
    args = parser.parse_args()
    return args

# -----------------------------------------------------------------------------
# Main evaluation routine
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.record_dir == "@":
        args.record_dir = args.output_dir

    evaluator = Evaluator(args.output_dir, record_dir=args.record_dir)

    # ---------------------------------------------------------------------------- #
    # Load evaluation configuration
    # ---------------------------------------------------------------------------- #
    try:
        if args.config_file is not None:
            config = load_json(args.config_file)
            config_env_id = config["env_info"]["env_id"]
            assert config_env_id == args.env_id, (config_env_id, args.env_id)
        else:
            # If no config file is provided, use the evaluator's dummy config.
            config = evaluator.generate_dummy_config(args.env_id, args.num_episodes)
    except Exception as e:
        exc_info = sys.exc_info()
        print("Failed to load evaluation configuration.", exc_info)
        evaluator.error("Failed to load evaluation configuration.", str(e))
        sys.exit(1)

    # ---------------------------------------------------------------------------- #
    # Load Diffusion Policy checkpoint and instantiate the policy
    # ---------------------------------------------------------------------------- #
    try:
        # Get environment kwargs from the configuration.
        env_kwargs = config["env_info"].get("env_kwargs", {})
        # Determine sim_backend (fallback to 'physx_gpu' if not provided).
        sim_backend = env_kwargs.get("sim_backend", "physx_gpu")
        # Prepare a dummy config (SimpleNamespace) for the Agent.
        # Ensure these values match those used during training.
        from types import SimpleNamespace
        dummy_args = SimpleNamespace(
            obs_horizon=2,
            act_horizon=8,
            pred_horizon=16,
            diffusion_step_embed_dim=64,
            unet_dims=[64, 128, 256],
            n_groups=8,
        )
        # Create a vectorized evaluation environment with 1 instance.
        temp_envs = make_eval_envs(
            args.env_id,
            num_eval_envs=1,
            sim_backend=sim_backend,
            env_kwargs=env_kwargs,
            other_kwargs={'obs_horizon': dummy_args.obs_horizon},
            video_dir=None,
            wrappers=[FlattenRGBDObservationWrapper],
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize the Agent using the vectorized environment.
        policy = Agent(temp_envs, dummy_args).to(device)
        # Do not close the environment here if the Agent requires it.
        # Load the saved checkpoint.
        checkpoint_path = os.path.expanduser(args.checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(checkpoint["agent"])
        policy.eval()
    except Exception as e:
        exc_info = sys.exc_info()
        print("Failed to load Diffusion Policy.", exc_info)
        evaluator.error("Failed to load Diffusion Policy.", str(e))
        sys.exit(2)

    # ---------------------------------------------------------------------------- #
    # Set up evaluator with the loaded policy
    # ---------------------------------------------------------------------------- #
    try:
        env_kwargs = config["env_info"].get("env_kwargs", {})
        evaluator.setup(args.env_id, policy, render_mode="cameras", env_kwargs=env_kwargs)
    except Exception as e:
        exc_info = sys.exc_info()
        print("Failed during evaluator setup.", exc_info)
        evaluator.error("Evaluator setup failed.", str(e))
        sys.exit(3)

    # ---------------------------------------------------------------------------- #
    # Evaluate episodes
    # ---------------------------------------------------------------------------- #
    try:
        episodes = config["episodes"]
        if args.num_episodes is not None:
            episodes = episodes[: args.num_episodes]
        cb = TqdmCallback(len(episodes))
        evaluator.evaluate_episodes(episodes, callback=cb)
    except Exception as e:
        exc_info = sys.exc_info()
        print("Evaluation failed.", exc_info)
        evaluator.error("Evaluation failed.", str(e))
        sys.exit(4)

    evaluator.submit()
    evaluator.close()

if __name__ == "__main__":
    main()

import sapien.core as sapien
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.agents.robots.unitree_g1.g1 import UnitreeG1
from typing import Union

@register_env("EmptyG1Env-v0", max_episode_steps=100)
class EmptyG1Env(BaseEnv):
    SUPPORTED_ROBOTS = ["unitree_g1"]
    agent: UnitreeG1

    def __init__(self, *args, robot_uids="unitree_g1", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))

    def _load_scene(self, options: dict):
        pass

    def _initialize_episode(self, env_idx, options: dict):
        pass

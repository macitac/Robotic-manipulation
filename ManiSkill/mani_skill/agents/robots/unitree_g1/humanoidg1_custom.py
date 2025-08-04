import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig, PassiveControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor

@register_agent()
class HumanoidG1_Custom(BaseAgent):
    """
    A custom version of the Unitree G1 robot that uses the same URDF and physical settings 
    as the simplified upper-body version but with a custom sensor configuration that includes 
    four cameras:
      - Two head-mounted cameras ("cam_left_high" and "cam_right_high")
      - Two wrist-mounted cameras ("cam_left_wrist" and "cam_right_wrist")
    """
    uid = "humanoidg1_custom"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/g1_humanoid/g1_simplified_upper_body.urdf"
    urdf_config = dict(
        _materials=dict(
            finger=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link={
            **{
                f"left_{k}_link": dict(
                    material="finger", patch_radius=0.1, min_patch_radius=0.1
                )
                for k in ["one", "two", "three", "four", "five", "six"]
            },
            **{
                f"right_{k}_link": dict(
                    material="finger", patch_radius=0.1, min_patch_radius=0.1
                )
                for k in ["one", "two", "three", "four", "five", "six"]
            },
            "left_palm_link": dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
            "right_palm_link": dict(
                material="finger", patch_radius=0.1, min_patch_radius=0.1
            ),
        },
    )
    fix_root_link = True
    load_multiple_collisions = False

    keyframes = dict(
        standing=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0.755]),
            qpos=np.array([0.0] * 25),
        )
    )

    body_joints = [
        "torso_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "right_elbow_pitch_joint",
        "left_elbow_roll_joint",
        "right_elbow_roll_joint",
        "left_zero_joint",
        "left_three_joint",
        "left_five_joint",
        "right_zero_joint",
        "right_three_joint",
        "right_five_joint",
        "left_one_joint",
        "left_four_joint",
        "left_six_joint",
        "right_one_joint",
        "right_four_joint",
        "right_six_joint",
        "left_two_joint",
        "right_two_joint",
    ]
    body_stiffness = 1e3
    body_damping = 1e2
    body_force_limit = 100

    @property
    def _controller_configs(self):
        body_pd_joint_pos = PDJointPosControllerConfig(
            self.body_joints,
            lower=None,
            upper=None,
            stiffness=self.body_stiffness,
            damping=self.body_damping,
            force_limit=self.body_force_limit,
            normalize_action=False,
        )
        body_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.body_joints,
            lower=[-0.2] * 11 + [-0.5] * 14,
            upper=[0.2] * 11 + [0.5] * 14,
            stiffness=self.body_stiffness,
            damping=self.body_damping,
            force_limit=self.body_force_limit,
            use_delta=True,
        )
        return dict(
            pd_joint_delta_pos=dict(
                body=body_pd_joint_delta_pos, balance_passive_force=True
            ),
            pd_joint_pos=dict(body=body_pd_joint_pos, balance_passive_force=True),
        )

    @property
    def _sensor_configs(self):
        # Four camera configurations:
        return [
            CameraConfig(
                uid="cam_left_high",
                pose=sapien.Pose(p=[0.05, 0.1, 1.2], q=euler2quat(0, 0, 0)),
                width=640,
                height=480,
                near=0.01,
                far=100,
                fov=np.pi / 2,
                mount=self.robot.links_map["head_link"],
            ),
            CameraConfig(
                uid="cam_right_high",
                pose=sapien.Pose(p=[0.05, -0.1, 1.2], q=euler2quat(0, 0, 0)),
                width=640,
                height=480,
                near=0.01,
                far=100,
                fov=np.pi / 2,
                mount=self.robot.links_map["head_link"],
            ),
            CameraConfig(
                uid="cam_left_wrist",
                pose=sapien.Pose(p=[0, 0.05, 0], q=euler2quat(0, 0, 0)),
                width=640,
                height=480,
                near=0.01,
                far=100,
                fov=np.pi / 2,
                mount=self.robot.links_map["left_tcp_link"],
            ),
            CameraConfig(
                uid="cam_right_wrist",
                pose=sapien.Pose(p=[0, -0.05, 0], q=euler2quat(0, 0, 0)),
                width=640,
                height=480,
                near=0.01,
                far=100,
                fov=np.pi / 2,
                mount=self.robot.links_map["right_tcp_link"],
            ),
        ]

    def _after_init(self):
        # Set up links and joints as in the original agent.
        self.right_hand_finger_link_l_1 = self.robot.links_map["right_two_link"]
        self.right_hand_finger_link_r_1 = self.robot.links_map["right_four_link"]
        self.right_hand_finger_link_r_2 = self.robot.links_map["right_six_link"]
        self.right_tcp = self.robot.links_map["right_tcp_link"]
        self.right_finger_joints = [
            "right_one_joint",
            "right_two_joint",
            "right_three_joint",
            "right_four_joint",
            "right_five_joint",
            "right_six_joint",
        ]
        self.right_finger_joint_indexes = [
            self.robot.active_joints_map[joint].active_index[0].item()
            for joint in self.right_finger_joints
        ]

        self.left_hand_finger_link_l_1 = self.robot.links_map["left_two_link"]
        self.left_hand_finger_link_r_1 = self.robot.links_map["left_four_link"]
        self.left_hand_finger_link_r_2 = self.robot.links_map["left_six_link"]
        self.left_tcp = self.robot.links_map["left_tcp_link"]
        self.left_finger_joints = [
            "left_one_joint",
            "left_two_joint",
            "left_three_joint",
            "left_four_joint",
            "left_five_joint",
            "left_six_joint",
        ]
        self.left_finger_joint_indexes = [
            self.robot.active_joints_map[joint].active_index[0].item()
            for joint in self.left_finger_joints
        ]

        # Disable collisions as in the original agent.
        link_names = ["one", "three", "four", "five", "six"]
        for ln in link_names:
            self.robot.links_map[f"left_{ln}_link"].set_collision_group_bit(2, 1, 1)
            self.robot.links_map[f"right_{ln}_link"].set_collision_group_bit(2, 2, 1)
        self.robot.links_map["left_palm_link"].set_collision_group_bit(2, 1, 1)
        self.robot.links_map["right_palm_link"].set_collision_group_bit(2, 2, 1)
        self.robot.links_map["left_elbow_roll_link"].set_collision_group_bit(2, 1, 1)
        self.robot.links_map["right_elbow_roll_link"].set_collision_group_bit(2, 2, 1)

        self.robot.links_map["torso_link"].set_collision_group_bit(2, 3, 1)
        self.robot.links_map["left_shoulder_roll_link"].set_collision_group_bit(2, 3, 1)
        self.robot.links_map["right_shoulder_roll_link"].set_collision_group_bit(2, 3, 1)

        # Print sensor configurations for verification.
        print("Custom HumanoidG1_Custom sensor configurations:")
        for sensor in self._sensor_configs:
            print("  Camera UID:", sensor.uid)

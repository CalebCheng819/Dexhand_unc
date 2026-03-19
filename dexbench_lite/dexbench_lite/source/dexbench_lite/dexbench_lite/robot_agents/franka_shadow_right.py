# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Kuka-lbr-iiwa arm robots and Allegro Hand.

The following configurations are available:

* :obj:`FR3_SHADOW_RIGHT_CFG`: Franka Research 3 with Shadow Hand.

Reference:

* https://www.kuka.com/en-us/products/robotics-systems/industrial-robots/lbr-iiwa
* https://www.wonikrobotics.com/robot-hand

"""

import math
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from pathlib import Path
import numpy as np
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg, JointPositionActionCfg, RelativeJointPositionActionCfg
from isaaclab.utils import configclass
from isaaclab.controllers import DifferentialIKControllerCfg

@configclass
class FR3ShadowRightRelJointPosActionCfg:
    action = RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)


@configclass
class FR3ShadowRightIKPoseRelJointHybridActionsCfg:
    """Hybrid control: IK for arm, joint position for fingers."""

    arm_ik = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint(1|2|3|4|5|6|7)"],
        body_name="panda_link7",  # TODO: Verify this is the correct palm body name
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
        scale=0.01,
    )

    fingers_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["WRJ(1|2)", "FFJ(1|2|3|4)", "MFJ(1|2|3|4)", "RFJ(1|2|3|4)", "LFJ(1|2|3|4|5)", "THJ(1|2|3|4|5)"],
        use_default_offset=True,
        scale=0.1,
        preserve_order=True,
    )

@configclass 
class FloatingShadowRightRelJointPosActionCfg:
    translation_action = RelativeJointPositionActionCfg(asset_name="robot", joint_names=["(x|y|z)_translation_joint"], scale=0.1)
    rotation_action = RelativeJointPositionActionCfg(asset_name="robot", joint_names=["(x|y|z)_rotation_joint"], scale=0.1)
    finger_action = RelativeJointPositionActionCfg(asset_name="robot", joint_names=["WRJ(1|2)", "FFJ(1|2|3|4)", "MFJ(1|2|3|4)", "RFJ(1|2|3|4)", "LFJ(1|2|3|4|5)", "THJ(1|2|3|4|5)"], scale=0.1)

@configclass 
class FloatingShadowRightAbsJointPosActionCfg:
    """Absolute joint position action config for floating Shadow hand.
    
    This config uses absolute joint position control, which is suitable for teleoperation
    where the retargeter outputs absolute joint angles directly.
    """
    translation_action = JointPositionActionCfg(asset_name="robot", joint_names=["(x|y|z)_translation_joint"], scale=1.0)
    # Rotation actions: must be ordered as [z_rotation_joint, y_rotation_joint, x_rotation_joint] to match retargeter output [yaw, pitch, roll]
    # Action 3 → z_rotation_joint (yaw), Action 4 → y_rotation_joint (pitch), Action 5 → x_rotation_joint (roll)
    # Using explicit names with preserve_order=True ensures the config order is used
    rotation_action = JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["z_rotation_joint", "y_rotation_joint", "x_rotation_joint"], 
        scale=1.0, 
        preserve_order=True  # Must preserve order to match retargeter output [yaw, pitch, roll]
    )
    # Finger actions: Shadow Hand has 22 joints total
    # Order: FFJ1-4, MFJ1-4, RFJ1-4, LFJ1-5, THJ1-5
    # Note: The exact order may need to be adjusted based on the actual robot model
    finger_action = JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["FFJ1", "FFJ2", "FFJ3", "FFJ4", "MFJ1", "MFJ2", "MFJ3", "MFJ4", "RFJ1", "RFJ2", "RFJ3", "RFJ4", "LFJ1", "LFJ2", "LFJ3", "LFJ4", "LFJ5", "THJ1", "THJ2", "THJ3", "THJ4", "THJ5"],
        scale=1.0,
        preserve_order=True  # Preserve the explicit order
    )

##
# Configuration
##
ASSET_DIR = Path(__file__).parent.parent.resolve()  / "robot_usds"

FR3_SHADOW_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/Shadow_hand/FR3_ShadowFull_right/FR3_ShadowFull_right.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": 0.0,
            "panda_joint3": 0.0,
            "panda_joint4": math.radians(-130.0),
            "panda_joint5": 0.0,
            "panda_joint6": math.radians(180.0),
            "panda_joint7": 0.0,
            "WRJ(1|2)": 0.0,
            "FFJ(1|2|3|4)": 0.0,
            "MFJ(1|2|3|4)": 0.0,
            "RFJ(1|2|3|4)": 0.0,
            "LFJ(1|2|3|4|5)": 0.0,
            "THJ(1|2|3|4|5)": 0.0,
        },
    ),
    actuators={
        "fr3_shadowhand_right_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "panda_joint(1|2|3|4|5|6|7)",
                "WRJ(1|2)",
                "FFJ(1|2|3|4)",
                "MFJ(1|2|3|4)",
                "RFJ(1|2|3|4)",
                "LFJ(1|2|3|4|5)",
                "THJ(1|2|3|4|5)",
                
            ],
            effort_limit_sim={
                "panda_joint(1|2|3|4|5|6|7)": 300.0,
                "WRJ(1|2)": 10.0,
                "FFJ(1|2|3|4)": 10.0,
                "MFJ(1|2|3|4)": 10.0,
                "RFJ(1|2|3|4)": 10.0,
                "LFJ(1|2|3|4|5)": 10.0,
                "THJ(1|2|3|4|5)": 10.0,
            },
            stiffness={
                "panda_joint(1|2|3|4|5|6|7)": 6000.0,
                "WRJ(1|2)": 20.0,
                "FFJ(1|2|3|4)": 20.0,
                "MFJ(1|2|3|4)": 20.0,
                "RFJ(1|2|3|4)": 20.0,
                "LFJ(1|2|3|4|5)": 20.0,
                "THJ(1|2|3|4|5)": 20.0,
            },
            damping={
                "panda_joint(1|2|3|4|5|6|7)": 600.0,
                "WRJ(1|2)": 0.1,
                "FFJ(1|2|3|4)": 0.1,
                "MFJ(1|2|3|4)": 0.1,
                "RFJ(1|2|3|4)": 0.1,
                "LFJ(1|2|3|4|5)": 0.1,
                "THJ(1|2|3|4|5)": 0.1,
            },
            friction={
                "panda_joint(1|2|3|4|5|6|7)": 1.0,
                "WRJ(1|2)": 0.01,
                "FFJ(1|2|3|4)": 0.01,
                "MFJ(1|2|3|4)": 0.01,
                "RFJ(1|2|3|4)": 0.01,
                "LFJ(1|2|3|4|5)": 0.01,
                "THJ(1|2|3|4|5)": 0.01
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

FLOATING_SHADOW_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/Shadow_hand/Floating_ShadowWoa_right/floating_shadow_hand.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.2, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "y_translation_joint": 0.0,
            "x_translation_joint": 0.5,
            "z_translation_joint": 0.3,
            "x_rotation_joint": 0.0,
            "y_rotation_joint": 0.0,
            "z_rotation_joint": 0.0,
            "FFJ(1|2|3|4)": 0.0,
            "MFJ(1|2|3|4)": 0.0,
            "RFJ(1|2|3|4)": 0.0,
            "LFJ(1|2|3|4|5)": 0.0,
            "THJ(1|2|3|4|5)": 0.0,
        },
    ),
    actuators={
        "floating_shadow_right_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "(x|y|z)_translation_joint",
                "(x|y|z)_rotation_joint",
                "FFJ(1|2|3|4)",
                "MFJ(1|2|3|4)",
                "RFJ(1|2|3|4)",
                "LFJ(1|2|3|4|5)",
                "THJ(1|2|3|4|5)",
            ],
            effort_limit_sim={
                "(x|y|z)_translation_joint": 15.0,
                "(x|y|z)_rotation_joint": 15.0,
                "FFJ(1|2|3|4)": 10.0,
                "MFJ(1|2|3|4)": 10.0,
                "RFJ(1|2|3|4)": 10.0,
                "LFJ(1|2|3|4|5)": 10.0,
                "THJ(1|2|3|4|5)": 10.0
            },
            stiffness={
                "(x|y|z)_translation_joint": 2000.0, 
                "(x|y|z)_rotation_joint": 2000.0, 
                "FFJ(1|2|3|4)": 10.0,
                "MFJ(1|2|3|4)": 10.0,
                "RFJ(1|2|3|4)": 10.0,
                "LFJ(1|2|3|4|5)": 10.0,
                "THJ(1|2|3|4|5)": 10.0
            },
            damping={
                "(x|y|z)_translation_joint": 400.0, 
                "(x|y|z)_rotation_joint": 400.0, 
                "FFJ(1|2|3|4)": 0.1,
                "MFJ(1|2|3|4)": 0.1,
                "RFJ(1|2|3|4)": 0.1,
                "LFJ(1|2|3|4|5)": 0.1,
                "THJ(1|2|3|4|5)": 0.1
            },
            velocity_limit_sim={
                "(x|y|z)_translation_joint": 10.0,
                "(x|y|z)_rotation_joint": 5.0,
                "FFJ(1|2|3|4)": 5.0,
                "MFJ(1|2|3|4)": 5.0,
                "RFJ(1|2|3|4)": 5.0,
                "LFJ(1|2|3|4|5)": 5.0,
                "THJ(1|2|3|4|5)": 5.0
            },
            friction={
                "(x|y|z)_translation_joint": 0.01, 
                "(x|y|z)_rotation_joint": 0.01, 
                "FFJ(1|2|3|4)": 0.01,
                "MFJ(1|2|3|4)": 0.01,
                "RFJ(1|2|3|4)": 0.01,
                "LFJ(1|2|3|4|5)": 0.01,
                "THJ(1|2|3|4|5)": 0.01,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# Robot-specific constants for body names and joint names
# TODO: Fill in the actual body names from the USD file
FR3_SHADOW_RIGHT_PALM_BODY_NAME = "panda_link7"  # TODO: Verify palm body name
FR3_SHADOW_RIGHT_FINGERTIP_BODY_NAMES = ["thdistal", "ffdistal", "mfdistal", "rfdistal", "lfdistal"]  # TODO: Fill in actual fingertip body names (Shadow Hand typically has FF, MF, RF, LF, TH fingertips)
FR3_SHADOW_RIGHT_HAND_TIPS_BODY_NAMES = [FR3_SHADOW_RIGHT_PALM_BODY_NAME] + FR3_SHADOW_RIGHT_FINGERTIP_BODY_NAMES
FR3_SHADOW_RIGHT_WRIST_JOINT_NAME = "panda_joint7"
FR3_SHADOW_RIGHT_ARM_JOINT_NAMES_EXPR = ["panda_joint(1|2|3|4|5|6|7)"]

FLOATING_SHADOW_RIGHT_PALM_BODY_NAME = "palm"  # TODO: Verify palm body name
FLOATING_SHADOW_RIGHT_FINGERTIP_BODY_NAMES = ["thtip", "fftip", "mftip", "rftip", "lftip"]  # TODO: Fill in actual fingertip body names
FLOATING_SHADOW_RIGHT_HAND_TIPS_BODY_NAMES = [FLOATING_SHADOW_RIGHT_PALM_BODY_NAME] + FLOATING_SHADOW_RIGHT_FINGERTIP_BODY_NAMES
FLOATING_SHADOW_RIGHT_WRIST_JOINT_NAME = "(x|y|z)_rotation_joint"
FLOATING_SHADOW_RIGHT_ARM_JOINT_NAMES_EXPR = ["(x|y|z)_translation_joint"]

FR3_SHADOW_RIGHT_BASE_POS = (-0.75, 0.0, 0.5)  
FR3_SHADOW_RIGHT_BASE_ROT = (1.0, 0.0, 0.0, 0.0)
FLOATING_SHADOW_RIGHT_BASE_POS = (-0.75, 0.0, 0.5)
FLOATING_SHADOW_RIGHT_BASE_ROT = (1.0, 0.0, 0.0, 0.0)
FLOATING_SHADOW_RIGHT_WRIST_POSITION_OFFSET = (
    FLOATING_SHADOW_RIGHT_BASE_POS[0] + 0.5,  # base_x + x_translation_joint
    FLOATING_SHADOW_RIGHT_BASE_POS[1] + 0.0,  # base_y + y_translation_joint  
    FLOATING_SHADOW_RIGHT_BASE_POS[2] + 0.3,  # base_z + z_translation_joint
)  # = (-0.25, 0.0, 0.8) - wrist world position at initial state
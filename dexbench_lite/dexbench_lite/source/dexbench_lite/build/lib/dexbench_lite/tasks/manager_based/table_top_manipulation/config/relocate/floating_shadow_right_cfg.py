"""Robot-specific configuration for Floating Shadow Hand in relocate task."""

from dexbench_lite.robot_agents.franka_shadow_right import (
    FLOATING_SHADOW_RIGHT_BASE_POS,
    FLOATING_SHADOW_RIGHT_BASE_ROT,
    FLOATING_SHADOW_RIGHT_ARM_JOINT_NAMES_EXPR,
    FLOATING_SHADOW_RIGHT_CFG,
    FLOATING_SHADOW_RIGHT_FINGERTIP_BODY_NAMES,
    FLOATING_SHADOW_RIGHT_HAND_TIPS_BODY_NAMES,
    FLOATING_SHADOW_RIGHT_PALM_BODY_NAME,
    FLOATING_SHADOW_RIGHT_WRIST_JOINT_NAME,
    FloatingShadowRightAbsJointPosActionCfg,
)

from isaaclab.utils import configclass

from ... import table_top_manipulation_cfg as table_top_manipulation
from . import relocate_cfg


@configclass
class RelocateEnvFloatingShadowRightCfg(relocate_cfg.RelocateEnvCfg):
    """Relocate environment configuration for Floating Shadow Hand."""

    def __post_init__(self):
        # Set robot-specific configuration
        self.robot_config = table_top_manipulation.RobotConfig(
            palm_body_name=FLOATING_SHADOW_RIGHT_PALM_BODY_NAME,
            fingertip_body_names=FLOATING_SHADOW_RIGHT_FINGERTIP_BODY_NAMES,
            hand_tips_body_names=FLOATING_SHADOW_RIGHT_HAND_TIPS_BODY_NAMES,
            wrist_joint_name=FLOATING_SHADOW_RIGHT_WRIST_JOINT_NAME,
            arm_joint_names_expr=FLOATING_SHADOW_RIGHT_ARM_JOINT_NAMES_EXPR,
            setup_contact_sensors=True,
        )

        # Set robot articulation config with base position
        self.scene.robot = FLOATING_SHADOW_RIGHT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=FLOATING_SHADOW_RIGHT_CFG.init_state.replace(
                pos=FLOATING_SHADOW_RIGHT_BASE_POS,
                rot=FLOATING_SHADOW_RIGHT_BASE_ROT,
            ),
        )

        # Set action configuration (floating hand only supports joint position control)
        self.actions = FloatingShadowRightAbsJointPosActionCfg()
        # Override controller_mode since floating hand doesn't support IK
        self.controller_mode = "joint"

        # Call parent __post_init__ to set up observations and rewards
        super().__post_init__()
 
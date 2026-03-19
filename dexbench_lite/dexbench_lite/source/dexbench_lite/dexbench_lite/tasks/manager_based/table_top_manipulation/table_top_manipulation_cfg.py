# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
# DEBUG_YCB_OBJ_DIR is not available in dexbench_lite - using placeholder if needed
# For relocate task, objects are defined directly in relocate_cfg.py
DEBUG_YCB_OBJ_DIR = None

from . import mdp
from .adr_curriculum import CurriculumCfg

# Path to debug articulations directory
DEBUG_ARTICULATIONS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "assets" / "articulations" / "debug_articulations"


def make_obj_cfg_list(usd_parent_dir, object_ids=None):
    import os 
    import glob 
    # find all the usd files recursively in subdirectories
    # Filter out Props/instanceable_meshes.usd files - only get the main object files
    usd_files_all = glob.glob(os.path.join(usd_parent_dir, "**/*.usd"), recursive=True)
    usd_files = []
    if object_ids is not None:
        for ids in object_ids:
            for file in usd_files_all:
                if ids in file:
                    usd_files.append(file)
    else:
        usd_files = usd_files_all
    return [
        sim_utils.UsdFileCfg(
            usd_path=usd_file,
        )
        for usd_file in usd_files if "instanceable_meshes" not in usd_file
    ]


def make_default_object_cfg(
    usd_parent_dir=DEBUG_YCB_OBJ_DIR,
    object_ids: list[str] | None = None,
    init_pos: tuple[float, float, float] = (0.55, 0.0, 0.14),
    mass: float = 0.3,
) -> RigidObjectCfg:
    """Create a default object config for tabletop tasks."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=make_obj_cfg_list(usd_parent_dir=usd_parent_dir, object_ids=object_ids),
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos),
    )


@configclass
class SceneCfg(InteractiveSceneCfg):
    # Robot and task-specific objects should be provided by the task config.
    robot: ArticulationCfg = MISSING
    object: RigidObjectCfg | None = None
    
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.5, 0.5),  # Table dimensions: width=1.0m, depth=1.5m, height=0.5m
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.5, 0.5),  # Dark grey color
                roughness=0.5,
            ),
            visible=True,
        ),
        # Center at z=0.35m so bottom is at z=0.1m (ground) and top is at z=0.6m
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.35), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # plane
    # Use a local procedural cuboid ground instead of GroundPlaneCfg to avoid
    # remote ISAAC_NUCLEUS asset dependency in headless/offline servers.
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
        spawn=sim_utils.CuboidCfg(
            size=(20.0, 20.0, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.5, 0.5),
                roughness=0.8,
            ),
            visible=True,
        ),
        collision_group=-1,
    )

    # lights
    # Avoid remote HDRI texture dependency by using plain dome light color.
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            color=(0.95, 0.95, 0.95),
            texture_file=None,
        ),
    )


@configclass
class RobotConfig:
    """Robot-specific configuration for body names, joint names, etc.
    
    This class can be overridden in derived configs to specify robot-specific names.
    Default values are for Kuka Allegro robot.
    """
    
    # Body names for observations and rewards
    palm_body_name: str = "base"
    """Name of the palm/base body for end-effector control."""
    
    fingertip_body_names: list[str] = ["thumb_fingertip", "fingertip", "fingertip2", "fingertip3"]
    """List of fingertip body names for observations and rewards."""
    
    hand_tips_body_names: list[str] | None = None
    """List of body names for hand_tips_state_b observation. 
    If None, defaults to [palm_body_name] + fingertip_body_names."""
    
    # Joint names for reset functions
    wrist_joint_name: str | None = "fr3_hand_joint"
    """Name of the wrist joint for reset_robot_wrist_joint. If None, this reset is disabled."""
    
    arm_joint_names_expr: str | list[str] | None = ["fr3_joint.*"]
    """Joint name pattern(s) for arm joints used in IK reset functions. If None, IK resets are disabled."""
    
    # Contact sensor configuration
    setup_contact_sensors: bool = False
    """Whether to set up contact sensors for fingertips. Should be True in robot-specific configs."""
    
    fingertip_contact_sensor_prefix: str | None = None
    """Prefix path for fingertip contact sensors. If None, contact sensors won't be set up."""
    
    def __post_init__(self):
        """Set default hand_tips_body_names if not provided."""
        if self.hand_tips_body_names is None:
            self.hand_tips_body_names = [self.palm_body_name] + self.fingertip_body_names


@configclass
class CommandsCfg:
    """Command terms for the MDP.
    
    Base command configuration. Task-specific configs (e.g., TopDownGraspCommandsCfg)
    should extend this to add task-specific commands like object_pose.
    """

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group.
        
        Base policy observations. Task-specific configs should extend this to add
        object-related observations (e.g., object_quat_b, target_object_pose_b).
        """
        # Note: object_quat_b and target_object_pose_b are task-specific and should be added
        # in task configs that manipulate objects (e.g., TopDownGraspObservationsCfg)
        actions = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0  # Disable history stacking (0 = no history, default)

    @configclass
    class ProprioObsCfg(ObsGroup):
        """Observations for proprioception group."""

        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.0, n_max=0.0))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.0, n_max=0.0))
        hand_tips_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            # Yunchao: below heuristics are from dexsuite's environments. 
            # good behaving number for position in m, velocity in m/s, rad/s,
            # and quaternion are unlikely to exceed -2 to 2 range
            clip=(-2.0, 2.0),
            params={
                # body_names will be set in __post_init__ based on robot_config
                "body_asset_cfg": SceneEntityCfg("robot", body_names=["base", "thumb_fingertip", "fingertip", "fingertip2", "fingertip3"]),
                "base_asset_cfg": SceneEntityCfg("robot"),
            },
        )
        contact: ObsTerm = MISSING

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 0  # Disable history stacking (0 = no history, default)

    @configclass
    class PerceptionObsCfg(ObsGroup):
        """Observations for perception group.
        
        Base perception observations. Task-specific configs should extend this to add
        object-related observations (e.g., object_point_cloud).
        """
        # Note: object_point_cloud is task-specific and should be added in task configs
        # that manipulate objects (e.g., TopDownGraspObservationsCfg)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_dim = 0
            # Disable concatenation by default since base config has no terms
            # Task-specific configs should set this to True if they add terms
            self.concatenate_terms = False
            self.flatten_history_dim = True
            self.history_length = 0  # Disable history stacking (0 = no history, default)

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    perception: PerceptionObsCfg = PerceptionObsCfg()


@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- pre-startup
    # Scale randomization disabled for YCB objects (USD-sourced objects may have incompatible transform structures)
    # If using primitive shapes, uncomment the following:
    # randomize_object_scale = EventTerm(
    #     func=mdp.randomize_rigid_body_scale,
    #     mode="prestartup",
    #     params={"scale_range": (1.0, 1.0), "asset_cfg": SceneEntityCfg("object")},
    # )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": [2.0, 2.0],
            "dynamic_friction_range": [2.0, 2.0],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 250,
        },
    )

    # Note: object_physics_material is task-specific and should be added in task configs
    # that manipulate objects (e.g., TopDownGraspEventCfg)

    joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": [1.0, 1.0],
            "damping_distribution_params": [1.0, 1.0],
            "operation": "scale",
        },
    )

    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": [1.0, 1.0],
            "operation": "scale",
        },
    )

    # Note: object_scale_mass is task-specific and should be added in task configs
    # that manipulate objects (e.g., TopDownGraspEventCfg)

    reset_table = EventTerm(
        func=mdp.reset_root_pose_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.00, 0.00], "y": [-0.00, 0.00], "z": [0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("table"),
        },
    )

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "z": [-0.03 , -0.03],
                "roll": [0, 0],
                "pitch": [0, 0],
                "yaw": [-1.8, -1.3],
            },
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "yaw": [-0.0, 0.0]},
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": [0.0, 0.0],
            "velocity_range": [0.0, 0.0],
        },
    )


    reset_robot_wrist_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="fr3_joint7"),
            "position_range": [0.0, 0.0],
            "velocity_range": [0.0, 0.0],
        },
    )

@configclass
class ActionsCfg:
    pass


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.005)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.005)

    position_tracking: RewTerm | None = None
    orientation_tracking: RewTerm | None = None
    success: RewTerm | None = None

    # early_termination = RewTerm(func=mdp.is_terminated_term, weight=-1, params={"term_keys": "abnormal_robot"})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP.
    
    Base termination configuration. Task-specific configs should extend this to add
    object-related terminations (e.g., object_out_of_bound).
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Note: object_out_of_bound is task-specific and should be added in task configs
    # that manipulate objects (e.g., TopDownGraspTerminationsCfg)

    # abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state)



@configclass
class TableTopManipulationEnvCfg(ManagerBasedEnvCfg):
    """Dexsuite reorientation task definition, also the base definition for derivative Lift task and evaluation task"""

    # Scene settings
    viewer: ViewerCfg = ViewerCfg(eye=(-2.0, 0.0, 1.25), lookat=(0.0, 0.0, 0.45), origin_type="env")
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg | None = None#CurriculumCfg()
    # Robot-specific configuration (can be overridden in derived configs)
    robot_config: RobotConfig = RobotConfig()
    # Task capability flags
    supports_object_pose_command: bool = False
    """Whether this task uses the object_pose command (for object target pose tracking)."""
    # Point cloud encoder configuration for RSL-RL
    pc_encoder_type: str | None = None
    """Type of point cloud encoder to use. Options:
    - None: No point cloud encoder (standard wrapper)
    - "pointnet": Use PointNet wrapper (PointNetRslRlVecEnvWrapper)
    - "mlp": Use MLP encoder with ActorCriticWithPCEncoder
    Defaults to None."""
    pc_encoder_num_points: int = 64
    """Number of points in point cloud for MLP encoder. Only used when pc_encoder_type="mlp". Defaults to 64."""

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2  # 50 Hz

        # Apply robot-specific configuration
        # Set command body name (only if object_pose command exists)
        if hasattr(self.commands, "object_pose"):
            self.commands.object_pose.body_name = self.robot_config.palm_body_name
            # *single-goal setup
            self.commands.object_pose.resampling_time_range = (10.0, 10.0)
            self.commands.object_pose.position_only = False
        
        # Note: Observation and reward body names should be overridden in robot-specific mixins 
        # (e.g., KukaAllegroMixinCfg) to keep the base config generic. 
        # The base config provides defaults that work for Kuka Allegro.
        
        # Set event term joint/body names
        if self.robot_config.wrist_joint_name is not None and self.events.reset_robot_wrist_joint is not None:
            self.events.reset_robot_wrist_joint.params["asset_cfg"].joint_names = self.robot_config.wrist_joint_name
        elif self.robot_config.wrist_joint_name is None:
            self.events.reset_robot_wrist_joint = None

        # If robot root is fixed, avoid writing velocities on reset.
        # This prevents PhysX errors for kinematic/fixed-base articulations.
        if self.events.reset_root is not None:
            fix_root = False
            if hasattr(self.scene.robot, "spawn") and self.scene.robot.spawn is not None:
                props = getattr(self.scene.robot.spawn, "articulation_props", None)
                if props is not None:
                    fix_root = bool(getattr(props, "fix_root_link", False))
            if fix_root:
                self.events.reset_root.func = mdp.reset_root_pose_uniform
                if "velocity_range" in self.events.reset_root.params:
                    del self.events.reset_root.params["velocity_range"]
            
        # *single-goal setup
        if self.scene.object is not None:
            if hasattr(self.commands, "object_pose"):
                self.commands.object_pose.resampling_time_range = (10.0, 10.0)
                self.commands.object_pose.position_only = False
                self.commands.object_pose.success_visualizer_cfg.markers["failure"] = self.scene.table.spawn.replace(
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.15, 0.15), roughness=0.25),
                    visible=True,
                )
                self.commands.object_pose.success_visualizer_cfg.markers["success"] = self.scene.table.spawn.replace(
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.25, 0.15), roughness=0.25),
                    visible=True,
                )
        else:
            # Disable object-dependent terms when no object is present.
            if hasattr(self.commands, "object_pose"):
                self.commands.object_pose = None
            self.observations.policy.object_quat_b = None
            self.observations.policy.target_object_pose_b = None
            self.observations.perception.object_point_cloud = None
            self.terminations.object_out_of_bound = None
            self.events.object_physics_material = None
            self.rewards.position_tracking = None
            self.rewards.success = None

        self.episode_length_s = 4.0
        self.is_finite_horizon = True

        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15


class TableTopManipulationLiftEnvCfg(TableTopManipulationEnvCfg):
    """Dexsuite lift task definition"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.orientation_tracking = None  # no orientation reward
        if hasattr(self.commands, "object_pose"):
            self.commands.object_pose.position_only = True

class TableTopManipulationLiftEnvCfg_PLAY(TableTopManipulationLiftEnvCfg):
    """Dexsuite lift task evaluation environment definition"""

    def __post_init__(self):
        super().__post_init__()
        if hasattr(self.commands, "object_pose"):
            self.commands.object_pose.resampling_time_range = (2.0, 3.0)
            self.commands.object_pose.debug_vis = True
            self.commands.object_pose.position_only = True

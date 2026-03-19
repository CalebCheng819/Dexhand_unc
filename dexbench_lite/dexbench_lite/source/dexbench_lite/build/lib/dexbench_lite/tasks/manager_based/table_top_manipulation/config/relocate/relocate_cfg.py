"""Task configuration for relocate task with tabletop manipulation."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from ... import table_top_manipulation_cfg as table_top_manipulation
from ... import mdp


SPHERE_RADIUS = 0.035
TARGET_OPACITY = 0.35
CENTER_SQUARE_SIZE = 0.45
TARGET_RANGE_Z = (0.10, 0.20)
BOUND_Z_MIN = -0.2
BOUND_Z_MAX = 1.5
LINE_THICKNESS = 0.005
LINE_HEIGHT = 0.002
LINE_Z_OFFSET = 0.001

OBJECT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.SphereCfg(
        radius=SPHERE_RADIUS,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.0)),
)

LINE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/RelocateArea_Line",
    spawn=sim_utils.CuboidCfg(
        size=(0.1, LINE_THICKNESS, LINE_HEIGHT),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        visible=True,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
)


@configclass
class RelocateCommandsCfg(table_top_manipulation.CommandsCfg):
    """Command terms for relocate task."""

    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(3.0, 5.0),
        debug_vis=True,
        use_world_frame=True,
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=TARGET_RANGE_Z,
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        success_vis_asset_name="object",
        position_only=True,
    )


@configclass
class RelocateObservationsCfg(table_top_manipulation.ObservationsCfg):
    """Observation specifications for relocate task."""

    @configclass
    class PolicyCfg(table_top_manipulation.ObservationsCfg.PolicyCfg):
        object_pos_b = ObsTerm(func=mdp.root_state_b, noise=Unoise(n_min=-0.0, n_max=0.0), params={"asset_cfg": SceneEntityCfg("object"), "base_asset_cfg": SceneEntityCfg("table")})
        object_state_robot_b = ObsTerm(func=mdp.root_state_b, noise=Unoise(n_min=-0.0, n_max=0.0), params={"asset_cfg": SceneEntityCfg("object"), "base_asset_cfg": SceneEntityCfg("robot")})
        frames_vis = ObsTerm(func=mdp.object_world_frame_vis)
        

    policy: PolicyCfg = PolicyCfg()


@configclass
class RelocateRewardsCfg(table_top_manipulation.RewardsCfg):
    """Reward terms for relocate task."""

    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance,
        params={
            "std": 0.4,
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
        },
        weight=2.0,
    )

    lift_when_grasping = RewTerm(
        func=mdp.lift_when_grasping_reward,
        weight=0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=None),
            "object_cfg": SceneEntityCfg("object"),
            "threshold": 0.08,
        },
    )

    position_tracking = RewTerm(
        func=mdp.position_command_error_exp_from_metrics,
        weight=2.0,
        params={
            "std": 0.15,
            "command_name": "object_pose",
        },
    )

    success = RewTerm(
        func=mdp.success_reward_from_metrics,
        weight=8.0,
        params={
            "pos_std": 0.05,
            "rot_std": None,
            "command_name": "object_pose",
        },
    )


@configclass
class RelocateEventCfg(table_top_manipulation.EventCfg):
    """Event configuration for relocate task."""

    debug_reset = EventTerm(
        func=mdp.debug_reset_reasons,
        mode="reset",
        params={"prefix": "Env reset"},
    )


@configclass
class RelocateTerminationsCfg(table_top_manipulation.TerminationsCfg):
    """Termination terms for relocate task."""

    success = DoneTerm(
        func=mdp.object_at_goal_position,
        params={
            "command_name": "object_pose",
            "threshold": 0.03,
        },
    )


@configclass
class RelocateEnvCfg(table_top_manipulation.TableTopManipulationEnvCfg):
    """Relocate task configuration (base, robot-agnostic)."""

    supports_object_pose_command: bool = True

    commands: RelocateCommandsCfg = RelocateCommandsCfg()
    observations: RelocateObservationsCfg = RelocateObservationsCfg()
    rewards: RelocateRewardsCfg = RelocateRewardsCfg()
    events: RelocateEventCfg = RelocateEventCfg()
    terminations: RelocateTerminationsCfg = RelocateTerminationsCfg()
    scene: table_top_manipulation.SceneCfg = table_top_manipulation.SceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
        object=OBJECT_CFG,
    )

    def __post_init__(self):
        super().__post_init__()

        # Keep a single target per episode (no mid-episode resampling) and position-only tracking.
        self.episode_length_s = 10.0
        self.commands.object_pose.resampling_time_range = (self.episode_length_s + 1.0, self.episode_length_s + 1.0)
        self.commands.object_pose.position_only = True

        # Visual target sphere (no collision) at the commanded goal position.
        self.commands.object_pose.goal_pose_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_pose",
            markers={
                "target": sim_utils.SphereCfg(
                    radius=SPHERE_RADIUS,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.2, 0.6, 0.9),
                        opacity=TARGET_OPACITY,
                    ),
                )
            },
        )

        # Replace success markers with spheres (avoid table geometry markers).
        self.commands.object_pose.success_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/SuccessMarkers",
            markers={
                "failure": sim_utils.SphereCfg(
                    radius=SPHERE_RADIUS,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
                ),
                "success": sim_utils.SphereCfg(
                    radius=SPHERE_RADIUS,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.2)),
                ),
            },
        )

        # Randomize object reset on the tabletop.
        if self.events.reset_object is not None:
            table_size = self.scene.table.spawn.size
            table_pos = self.scene.table.init_state.pos
            object_z = table_pos[2] + table_size[2] * 0.5 + SPHERE_RADIUS
            half_side = CENTER_SQUARE_SIZE * 0.5
            table_top_z = table_pos[2] + table_size[2] * 0.5
            x_min = table_pos[0] - half_side
            x_max = table_pos[0] + half_side
            y_min = table_pos[1] - half_side
            y_max = table_pos[1] + half_side
            # Ensure initial object z sits on the tabletop.
            object_pos = self.scene.object.init_state.pos
            self.scene.object.init_state.pos = (
                object_pos[0],
                object_pos[1],
                object_z,
            )
            self.events.reset_object.params["pose_range"] = {
                "x": [0.0, 0.0],
                "y": [-half_side, half_side],
                "z": [0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [-3.14, 3.14],
            }
            self.commands.object_pose.use_world_frame = True
            self.commands.object_pose.ranges.pos_x = (table_pos[0], table_pos[0])
            self.commands.object_pose.ranges.pos_y = (table_pos[1], table_pos[1])
            self.commands.object_pose.ranges.pos_z = (
                table_top_z + TARGET_RANGE_Z[0],
                table_top_z + TARGET_RANGE_Z[1],
            )

            # Add red outline of the center square on the tabletop.
            z_line = table_pos[2] + table_size[2] * 0.5 + LINE_HEIGHT * 0.5 + LINE_Z_OFFSET
            line_specs = [
                ("line_y_pos", (CENTER_SQUARE_SIZE, LINE_THICKNESS, LINE_HEIGHT), (table_pos[0], y_max, z_line)),
                ("line_y_neg", (CENTER_SQUARE_SIZE, LINE_THICKNESS, LINE_HEIGHT), (table_pos[0], y_min, z_line)),
                ("line_x_pos", (LINE_THICKNESS, CENTER_SQUARE_SIZE, LINE_HEIGHT), (x_max, table_pos[1], z_line)),
                ("line_x_neg", (LINE_THICKNESS, CENTER_SQUARE_SIZE, LINE_HEIGHT), (x_min, table_pos[1], z_line)),
            ]
            for name, size, pos in line_specs:
                line_cfg = LINE_CFG.replace(prim_path=f"{{ENV_REGEX_NS}}/RelocateArea_{name}")
                line_cfg.spawn.size = size
                line_pos = line_cfg.init_state.pos
                line_cfg.init_state.pos = (line_pos[0], line_pos[1], pos[2])
                setattr(self.scene, name, line_cfg)

        # Setup contact sensors if enabled
        if self.robot_config.setup_contact_sensors:
            tip_prim_prefix = "{ENV_REGEX_NS}/Robot/"
            finger_tip_body_list = self.robot_config.fingertip_body_names

            for link_name in finger_tip_body_list:
                sensor_path = f"{tip_prim_prefix}{link_name}"
                setattr(
                    self.scene,
                    f"{link_name}_object_s",
                    ContactSensorCfg(
                        prim_path=sensor_path,
                        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                    ),
                )

            self.observations.proprio.contact = ObsTerm(
                func=mdp.fingers_contact_force_b,
                params={"contact_sensor_names": [f"{link}_object_s" for link in finger_tip_body_list]},
                clip=(-20.0, 20.0),
            )
        else:
            self.observations.proprio.contact = None

        # Override observation body names with robot-specific names
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = self.robot_config.hand_tips_body_names

        # Override reward body names with robot-specific values
        self.rewards.fingers_to_object.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names
        self.rewards.lift_when_grasping.params["asset_cfg"].body_names = self.robot_config.fingertip_body_names

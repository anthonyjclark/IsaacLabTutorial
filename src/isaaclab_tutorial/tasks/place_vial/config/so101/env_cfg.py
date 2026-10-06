"""Manager-based SO-101 vial placement task with physical reset replay."""

# Imports are supplied for the lesson snippets, including unfinished regions.
# ruff: noqa: F401

from __future__ import annotations

import math
from dataclasses import MISSING
from typing import Any

import isaaclab.sim as sim_utils
import newton
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, VideoRecorderCfg
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysicsEvent
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.utils import clone
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_assets.robots.so101 import SO101_CFG
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonManager
from isaaclab_tasks.utils import PresetCfg
from isaaclab_tasks.utils.hydra import preset
from pxr import Gf

from isaaclab_tutorial.assets import MAT_USD, RACK_USD, RESET_DATASET, VIAL_USD
from isaaclab_tutorial.tasks.place_vial import mdp
from isaaclab_tutorial.tasks.place_vial.mdp.actions import (
    SoftLimitRelativeGripperActionCfg,
    SoftLimitRelativeJointPositionActionCfg,
)
from isaaclab_tutorial.tasks.place_vial.reset.curriculum import ALL_PHASES, CANONICAL_START

from .lesson_scaffold import AgentObservationGroupsCfg, LessonEventsCfg, LessonTerminationsCfg

TABLETOP_VIAL_HEADING_RANGE = (-0.35, 0.35)
TABLETOP_VIAL_POSITION = (0.231, -0.017, 0.06)

# Map workshop commands onto the USD's [-10, 100] degree range.
PREGRASP_GRIPPER_POSITION = math.radians(-10.0 + 1.1 * 22.4)
GRASP_GRIPPER_POSITION = math.radians(-10.0 + 1.1 * 1.0)
RELEASE_GRIPPER_POSITION = math.radians(-10.0 + 1.1 * 42.7)

WORKSHOP_INITIAL_JOINT_POSITION = (
    -0.1221070742,
    -0.9066845838,
    0.1900876486,
    1.4797928525,
    -0.8044013083,
    PREGRASP_GRIPPER_POSITION,
)

_CONTACT_STIFFNESS = 1.57e5
_CONTACT_DAMPING = 1.12e3
_FRICTION = 0.7
_ROLLING_FRICTION = 0.05
_TORSIONAL_FRICTION = 0.005
_SOLIMP = (0.7, 0.95, 0.0001, 0.5, 2.0)
_SOLREF = (0.002, 1.5)
_contact_model_registered = False


def _apply_camera_clipping_range(stage: Any, robot_prim_path: str) -> None:
    camera = stage.GetPrimAtPath(f"{robot_prim_path}/gripper/wowrobo_2MP_camera")
    camera.GetAttribute("clippingRange").Set(Gf.Vec2f(0.001, 5.0))


@clone
def _spawn_so101_with_camera_overrides(
    prim_path: str,
    cfg: Any,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    prim = spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    _apply_camera_clipping_range(prim.GetStage(), prim_path)
    return prim


WORKSHOP_SO101_CFG = SO101_CFG.replace(
    spawn=SO101_CFG.spawn.replace(func=_spawn_so101_with_camera_overrides),
)


def _initialize_contacts(_event: PhysicsEvent) -> None:
    """Apply the workshop-validated contact model to every Newton shape."""
    builder = NewtonManager._builder
    if builder is None:
        return

    num_shapes = len(builder.shape_body)
    for shape_index in range(num_shapes):
        builder.shape_material_ke[shape_index] = _CONTACT_STIFFNESS
        builder.shape_material_kd[shape_index] = _CONTACT_DAMPING
        builder.shape_material_mu[shape_index] = _FRICTION
        builder.shape_material_mu_rolling[shape_index] = _ROLLING_FRICTION
        builder.shape_material_mu_torsional[shape_index] = _TORSIONAL_FRICTION

    # Prototype builders register these attributes, but Newton's cloner does
    # not currently carry that registration to the main builder.
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    for name, value in (("mujoco:geom_solimp", _SOLIMP), ("mujoco:geom_solref", _SOLREF)):
        attribute = builder.custom_attributes.get(name)
        if attribute is None:
            continue
        if attribute.values is None:
            attribute.values = {}
        for shape_index in range(num_shapes):
            attribute.values[shape_index] = value


def _register_contact_model() -> None:
    """Register the contact initializer once per process."""
    global _contact_model_registered
    if _contact_model_registered:
        return
    NewtonManager.register_callback(
        _initialize_contacts,
        PhysicsEvent.MODEL_INIT,
        name="so101_workshop_contact_model",
    )
    _contact_model_registered = True


_register_contact_model()

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = JOINTS[:-1]


@configclass
class SO101SceneCfg(InteractiveSceneCfg):
    """One SO-101, one vial, one rack, and a collision mat."""

    # BEGIN lesson-02-robot
    robot: ArticulationCfg = MISSING
    # END lesson-02-robot

    # BEGIN lesson-03-objects
    pass
    # END lesson-03-objects

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1200.0, color=(0.9, 0.9, 0.9)),
    )


@configclass
class ResetJointActionsCfg:
    """Direct joint targets used only by reset generation and diagnostics."""

    joint_delta: SoftLimitRelativeJointPositionActionCfg = SoftLimitRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINTS,
        preserve_order=True,
        scale={
            "shoulder_lift|elbow_flex": 0.04,
            "shoulder_pan|wrist_.*": 0.03,
            "gripper": 1.0,
        },
        gripper_open_position=RELEASE_GRIPPER_POSITION,
        gripper_close_position=GRASP_GRIPPER_POSITION,
    )


# BEGIN lesson-04-actions
@configclass
class ActionsCfg:
    pass
# END lesson-04-actions


# BEGIN lesson-05-policy-observations
@configclass
class PolicyStateGroupCfg(ObsGroup):
    pass
# END lesson-05-policy-observations


# BEGIN lesson-05-critic-observations
@configclass
class CriticStateGroupCfg(PolicyStateGroupCfg):
    pass
# END lesson-05-critic-observations


# BEGIN lesson-05-observation-groups
@configclass
class ObservationsCfg:
    pass
# END lesson-05-observation-groups


# BEGIN lesson-05-dataset-resets
@configclass
class DatasetEventsCfg:
    pass
# END lesson-05-dataset-resets


# BEGIN lesson-05-tabletop-resets
@configclass
class ResetEventsCfg:
    pass
# END lesson-05-tabletop-resets


# BEGIN lesson-05-rewards
@configclass
class RewardsCfg:
    pass
# END lesson-05-rewards


# BEGIN lesson-05-terminations
@configclass
class TerminationsCfg:
    pass
# END lesson-05-terminations


@configclass
class PhysicsCfg(PresetCfg):
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=300,
            nconmax=200,
            cone="elliptic",
            impratio=10.0,
            update_data_interval=2,
            iterations=100,
            ls_iterations=15,
            use_mujoco_contacts=False,
            ccd_iterations=35,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(),
        num_substeps=2,
        debug_mode=False,
    )
    default = newton_mjwarp


# BEGIN lesson-environment
@configclass
class SO101VialEnvCfg(ManagerBasedRLEnvCfg):
    pass
# END lesson-environment


@configclass
class SO101VialGeneratorEnvCfg(SO101VialEnvCfg):
    """Raw task scene used by the standalone reset generator."""

    scene: SO101SceneCfg = SO101SceneCfg(num_envs=256, env_spacing=0.9, replicate_physics=True)
    actions: ResetJointActionsCfg = ResetJointActionsCfg()
    events: ResetEventsCfg = ResetEventsCfg()
    rewards = None
    terminations = None

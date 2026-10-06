"""Wrist-camera observation variant of the SO-101 vial task."""

# Imports are supplied for the lesson snippets, including unfinished regions.
# ruff: noqa: F401

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab_tasks.utils.presets import MultiBackendRendererCfg

from isaaclab_tutorial.tasks.place_vial import mdp
from isaaclab_tutorial.tasks.place_vial.config.so101.env_cfg import (
    CriticStateGroupCfg,
    PolicyStateGroupCfg,
    SO101SceneCfg,
    SO101VialEnvCfg,
)


# BEGIN lesson-06-SO101CameraSceneCfg
@configclass
class SO101CameraSceneCfg(SO101SceneCfg):
    pass
# END lesson-06-SO101CameraSceneCfg


# BEGIN lesson-06-WristImageCfg
@configclass
class WristImageCfg(ObsGroup):
    pass
# END lesson-06-WristImageCfg


# BEGIN lesson-06-ProprioceptionCfg
@configclass
class ProprioceptionCfg(ObsGroup):
    pass
# END lesson-06-ProprioceptionCfg


# BEGIN lesson-06-CameraObservationsCfg
@configclass
class CameraObservationsCfg:
    pass
# END lesson-06-CameraObservationsCfg


# BEGIN lesson-06-DistillationObservationsCfg
@configclass
class DistillationObservationsCfg(CameraObservationsCfg):
    pass
# END lesson-06-DistillationObservationsCfg


# BEGIN lesson-06-SO101VialCameraEnvCfg
@configclass
class SO101VialCameraEnvCfg(SO101VialEnvCfg):
    pass
# END lesson-06-SO101VialCameraEnvCfg


# BEGIN lesson-06-SO101VialCameraDistillationEnvCfg
@configclass
class SO101VialCameraDistillationEnvCfg(SO101VialCameraEnvCfg):
    pass
# END lesson-06-SO101VialCameraDistillationEnvCfg

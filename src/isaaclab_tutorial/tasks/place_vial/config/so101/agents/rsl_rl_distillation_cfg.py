"""State-teacher to wrist-camera-student distillation configuration."""

# Imports are supplied for the lesson snippets, including unfinished regions.
# ruff: noqa: F401

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)

from isaaclab_tutorial.tasks.place_vial.config.so101.agents.rsl_rl_ppo_cfg import (
    WRIST_CAMERA_CNN_CFG,
    BoundedGaussianDistributionCfg,
)


# BEGIN lesson-06-distillation
@configclass
class SO101CameraDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    pass
# END lesson-06-distillation

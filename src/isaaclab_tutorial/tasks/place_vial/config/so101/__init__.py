"""SO-101 vial-placement task registrations."""

# Imports are supplied for the lesson snippets, including unfinished regions.
# ruff: noqa: F401

import gymnasium as gym

from isaaclab_tutorial.tasks.place_vial.config.so101 import agents

_PACKAGE = "isaaclab_tutorial.tasks.place_vial.config.so101"

# BEGIN lesson-01-registration
# Add the task registration here.
# END lesson-01-registration

# BEGIN lesson-06-camera-registration
# Add the camera task registrations here.
# END lesson-06-camera-registration

# Scene-only inspection uses empty actions before the simple-agent lesson.
gym.register(
    id="IsaacTutorial-Inspect-SO101",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{_PACKAGE}.inspection_env_cfg:SO101InspectionEnvCfg"},
)

gym.register(
    id="IsaacTutorial-Place-Vial-SO101-Camera",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{_PACKAGE}.camera_env_cfg:SO101VialCameraEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SO101CameraPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

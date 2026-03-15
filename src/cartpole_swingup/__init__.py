# Copyright (c) 2026, Harbin Institute of Technology, Weihai.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Cartpole task registration."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import cartpole_swingup_env_cfg
from .rl_cfg import cartpole_ppo_runner_cfg

register_mjlab_task(
  task_id="Cartpole-Swingup",
  env_cfg=cartpole_swingup_env_cfg(),
  play_env_cfg=cartpole_swingup_env_cfg(play=True),
  rl_cfg=cartpole_ppo_runner_cfg(),
)
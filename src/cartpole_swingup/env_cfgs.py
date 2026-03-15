# Copyright (c) 2026, Harbin Institute of Technology, Weihai.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.entity import Entity
from mjlab.envs.mdp import (
  reset_root_state_uniform,
  joint_pos_rel,
  joint_vel_rel,
  reset_joints_by_offset,
  time_out,
)
from mjlab.envs.mdp.actions import JointEffortActionCfg
from mjlab.managers.observation_manager import (
  ObservationTermCfg,
  ObservationGroupCfg,
)
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from mjlab.envs import ManagerBasedRlEnv

from cartpole_swingup.cartpole.cartpole_constants import (
  get_cartpole_cfg,
  SLIDER_TO_CART_CFG,
  CART_TO_POLE_CFG,
)

def pole_angle_cos_sin(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = CART_TO_POLE_CFG,
) -> torch.Tensor:
  """Cosine and sine of the pole hinge angle. Shape: [num_envs, 2]."""
  asset: Entity = env.scene[asset_cfg.name]
  angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)

_GAUSSIAN_SCALE = math.sqrt(-2 * math.log(0.1))
_QUADRATIC_SCALE = math.sqrt(1 - 0.1)

def _gaussian_tolerance(x: torch.Tensor, margin: float) -> torch.Tensor:
  """Gaussian sigmoid tolerance: 1 at x=0, value_at_margin=0.1 at |x|=margin."""
  if margin == 0:
    return (x == 0).float()
  scaled = x / margin * _GAUSSIAN_SCALE
  return torch.exp(-0.5 * scaled**2)


def _quadratic_tolerance(x: torch.Tensor, margin: float) -> torch.Tensor:
  """Quadratic sigmoid tolerance: 1 at x=0, 0 at |x|>=margin."""
  if margin == 0:
    return (x == 0).float()
  scaled = x / margin * _QUADRATIC_SCALE
  return torch.clamp(1 - scaled**2, min=0.0)

def cartpole_smooth_reward(
  env: ManagerBasedRlEnv,
  slide_cfg: SceneEntityCfg = SLIDER_TO_CART_CFG,
  hinge_cfg: SceneEntityCfg = CART_TO_POLE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[slide_cfg.name]

  # Pole angle cosine.
  hinge_angle = asset.data.joint_pos[:, hinge_cfg.joint_ids].squeeze(-1)
  pole_cos = torch.cos(hinge_angle)
  upright = (pole_cos + 1) / 2

  # Cart position.
  cart_pos = asset.data.joint_pos[:, slide_cfg.joint_ids].squeeze(-1)
  centered = (1 + _gaussian_tolerance(cart_pos, margin=2.0)) / 2

  # Control effort (raw action from the policy).
  control = env.action_manager.action.squeeze(-1)
  small_control = (4 + _quadratic_tolerance(control, margin=1.0)) / 5

  # Pole angular velocity.
  hinge_vel = asset.data.joint_vel[:, hinge_cfg.joint_ids].squeeze(-1)
  small_velocity = (1 + _gaussian_tolerance(hinge_vel, margin=5.0)) / 2

  return upright * centered * small_control * small_velocity

def _make_env_cfg() -> ManagerBasedRlEnvCfg:
  scene: SceneCfg = SceneCfg(
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={"cartpole": get_cartpole_cfg()},
    num_envs=1,
    env_spacing=4.0,
  )

  viewer: ViewerConfig = ViewerConfig(
    origin_type=ViewerConfig.OriginType.WORLD,
    entity_name="cartpole",
    distance=15.0,
    elevation=-25.0,
    azimuth=180.0,
  )

  sim: SimulationCfg = SimulationCfg(
    mujoco=MujocoCfg(timestep=0.01),
  )

  actor_terms = {
    "cart_pos": ObservationTermCfg(
      func=joint_pos_rel,
      params={"asset_cfg": SLIDER_TO_CART_CFG},
    ),
    "pole_angle": ObservationTermCfg(
      func=pole_angle_cos_sin,
      params={"asset_cfg": CART_TO_POLE_CFG},
    ),
    "cart_vel": ObservationTermCfg(
      func=joint_vel_rel,
      params={"asset_cfg": SLIDER_TO_CART_CFG},
    ),
    "pole_vel": ObservationTermCfg(
      func=joint_vel_rel,
      params={"asset_cfg": CART_TO_POLE_CFG},
    ),
  }

  critic_terms = {
    **actor_terms,
  }

  observations: dict[str, ObservationGroupCfg] = {
    "actor": ObservationGroupCfg(actor_terms, enable_corruption=True),
    "critic": ObservationGroupCfg(critic_terms),
  }

  actions: dict[str, ActionTermCfg] = {
    "effort": JointEffortActionCfg(
      entity_name="cartpole",
      actuator_names=("slider_to_cart",),
      scale=1.0,
    ),
  }

  events: dict[str, EventTermCfg] = {
    # For positioning the base of the robot at env_origins
    "reset_base": EventTermCfg(
      func=reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("cartpole"),
      },
    ),
    "reset_slide": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (-0.01, 0.01),
        "asset_cfg": SLIDER_TO_CART_CFG,
      },
    ),
    "reset_hinge": EventTermCfg(
      func=reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-math.pi, math.pi),
        "velocity_range": (-0.01, 0.01),
        "asset_cfg": CART_TO_POLE_CFG,
      },
    ),
  }

  rewards: dict[str, RewardTermCfg] = {
    "smooth_reward": RewardTermCfg(
      func=cartpole_smooth_reward,
      weight=1.0,
      params={
        "slide_cfg": SLIDER_TO_CART_CFG, 
        "hinge_cfg": CART_TO_POLE_CFG
      },
    ),
  }

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(
      func=time_out,
      time_out=True,
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=scene,
    observations=observations,
    actions=actions,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=viewer,
    sim=sim,
    decimation=5,
    episode_length_s=50.0,
  )

def cartpole_swingup_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = _make_env_cfg()
  if play:
    cfg.episode_length_s = 1e10
    cfg.observations["actor"].enable_corruption = False
  return cfg
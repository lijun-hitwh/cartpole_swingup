# Copyright (c) 2026, Harbin Institute of Technology, Weihai.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import math
import mujoco

from mjlab.actuator.xml_actuator import XmlMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

##
# MJCF and assets.
##

_HERE = Path(__file__).parent

CARTPOLE_XML: Path = (
    _HERE / "xmls" / "cartpole.xml"
)
assert CARTPOLE_XML.exists()

SLIDER_TO_CART_CFG = SceneEntityCfg("cartpole", joint_names=("slider_to_cart",))
CART_TO_POLE_CFG = SceneEntityCfg("cartpole", joint_names=("cart_to_pole",))

def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(CARTPOLE_XML))

CARTPOLE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    XmlMotorActuatorCfg(
      target_names_expr=("slider_to_cart",),
    ),
  ),
  soft_joint_pos_limit_factor=1.0,
)

SWINGUP_INIT = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={"slider_to_cart": 0.0, "cart_to_pole": math.pi,},
  joint_vel={".*": 0.0},
)

def get_cartpole_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=get_spec,
    articulation=CARTPOLE_ARTICULATION,
    init_state=SWINGUP_INIT,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity import Entity

  robot = Entity(get_cartpole_cfg())

  viewer.launch(robot.spec.compile())
"""touch.render_view: a wall between the camera and the cup -> another angle is chosen."""

import mujoco
import numpy as np

from touch import render_view as RV

XML = """<mujoco><worldbody>
<body name="look" pos="0 0 0.05"/>
<camera name="view" mode="targetbody" target="look" pos="0.8 0 0.4" fovy="50"/>
<geom name="floor" type="plane" size="2 2 0.1"/>
<body name="obj0" pos="0 0 0.045"><freejoint/>
  <geom type="cylinder" size="0.031 0.045" rgba="0 0 1 1"/></body>
{wall}
</worldbody></mujoco>"""
WALL = '<geom name="wall" type="box" size="0.02 0.3 0.3" pos="0.35 0 0.3" group="2"/>'


def _states(m, n=5):
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return np.tile(np.r_[d.qpos, d.qvel], (n, 1))


def test_clear_view_kept():
    m = mujoco.MjModel.from_xml_string(XML.format(wall=""))
    cam, info = RV.best_view(m, _states(m), 16 / 9)
    assert info["turned_deg"] == 0 and info["cup_visible"] > 0.85


def test_blocked_view_turned_and_rendered():
    m = mujoco.MjModel.from_xml_string(XML.format(wall=WALL))
    s = _states(m)
    cam, info = RV.best_view(m, s, 16 / 9)
    assert info["cup_visible_original"] < 0.2
    assert info["turned_deg"] != 0 and info["cup_visible"] > 0.85
    frames = list(RV.render_states(m, s, cam, 160, 90, every=2))
    assert [t for t, _ in frames] == [0, 2, 4]
    assert frames[0][1].shape == (90, 160, 3)

import json

import mujoco
import pytest
from fastapi.testclient import TestClient

import main
from builder import MechanismBuilder, repair_ground_clearance


def lowest_non_plane_geom_z(xml: str) -> float:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lows = []
    for i in range(model.ngeom):
        if int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        gtype = int(model.geom_type[i])
        center_z = float(data.geom_xpos[i][2])
        size = model.geom_size[i]
        if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            lows.append(center_z - float(size[0]))
        elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
            lows.append(center_z - float(size[2]))
        else:
            lows.append(center_z - float(model.geom_rbound[i]))
    return min(lows)


def moving_pendulum_code() -> str:
    return """
b = MechanismBuilder()
b.add_body('arm', 'cylinder', (0.02, 0.75), mass=0.5, color=(0.6, 0.4, 0.2), geom_offset=(0, 0, -0.75), euler=(0, 30, 0))
b.add_body('bob', 'sphere', 0.1, mass=2.0, color=(0.8, 0.1, 0.1))
b.attach_to('world', 'arm', 'hinge', axis='y', limits=(-180, 180), damping=0.05)
b.attach_to('arm', 'bob', 'fixed')
b.position_relative('arm', 'world', (0, 0, 2.5))
b.position_relative('bob', 'arm', (0, 0, -1.5))
xml = b.build()
actuator_schedule = []
"""


def detached_bob_code() -> str:
    return """
b = MechanismBuilder()
b.add_body('arm', 'cylinder', (0.02, 0.75), mass=0.5, color=(0.6, 0.4, 0.2), geom_offset=(0, 0, -0.75))
b.add_body('bob', 'sphere', 0.1, mass=2.0, color=(0.8, 0.1, 0.1), free=True)
b.attach_to('world', 'arm', 'fixed')
b.position_relative('arm', 'world', (0, 0, 2.5))
b.position_relative('bob', 'world', (0, 0, 1.0))
xml = b.build()
actuator_schedule = []
"""


def test_builder_euler_renders_and_validates():
    b = MechanismBuilder()
    b.add_body("link", "box", (0.1, 0.1, 0.4), euler=(0, 30, 0))
    b.attach_to("world", "link", "hinge", axis="y")
    b.position_relative("link", "world", (0, 0, 1))
    xml = b.build()

    assert 'euler="0 30 0"' in xml
    mujoco.MjModel.from_xml_string(xml)


def test_builder_normalizes_free_body_with_joint_attachment():
    b = MechanismBuilder()
    b.add_body("bob", "sphere", 0.1, free=True)
    b.attach_to("world", "bob", "hinge", axis="y")
    b.position_relative("bob", "world", (0, 0, 1))
    xml = b.build()

    assert "<freejoint" not in xml
    assert 'type="hinge"' in xml


def test_builder_infers_link_geom_offset_to_child_joint():
    b = MechanismBuilder()
    b.add_body("arm", "cylinder", (0.02, 0.75), euler=(0, 30, 0))
    b.add_body("bob", "sphere", 0.1)
    b.attach_to("world", "arm", "hinge", axis="y")
    b.attach_to("arm", "bob", "fixed")
    b.position_relative("arm", "world", (0, 0, 2.5))
    b.position_relative("bob", "arm", (0, 0, -1.5))
    xml = b.build()

    assert 'pos="0 0 -0.75"' in xml
    assert 'size="0.02 0.75"' in xml


def test_builder_accepts_position_relative_euler():
    b = MechanismBuilder()
    b.add_body("link", "box", (0.1, 0.1, 0.4))
    b.attach_to("world", "link", "hinge", axis="y")
    b.position_relative("link", "world", (0, 0, 1), euler=(0, 20, 0))
    xml = b.build()

    assert 'euler="0 20 0"' in xml


def test_builder_defaults_hinged_link_away_from_equilibrium():
    b = MechanismBuilder()
    b.add_body("arm", "cylinder", (0.02, 0.75))
    b.add_body("bob", "sphere", 0.1)
    b.attach_to("world", "arm", "hinge", axis="y")
    b.attach_to("arm", "bob", "fixed")
    b.position_relative("arm", "world", (0, 0, 2.5))
    b.position_relative("bob", "arm", (0, 0, -1.5))
    xml = b.build()

    assert 'euler="0 25 0"' in xml
    main._ensure_simulation_moves(xml, [])


def test_builder_skips_actuator_for_body_without_joint():
    b = MechanismBuilder()
    b.add_body("platform", "box", (0.5, 0.05, 0.05))
    b.position_relative("platform", "world", (0, 0, 0.2))
    b.add_actuator("platform", torque=10)
    xml = b.build()

    assert "<actuator>" not in xml


def test_ground_clearance_lifts_free_sphere_above_ground():
    b = MechanismBuilder()
    b.add_body("ball", "sphere", 0.2, free=True)
    b.position_relative("ball", "world", (0, 0, -1.0))
    xml = b.build()

    assert lowest_non_plane_geom_z(xml) >= 0.009


def test_ground_clearance_lifts_box_with_origin_at_ground():
    b = MechanismBuilder()
    b.add_body("box", "box", (0.1, 0.1, 0.2))
    b.position_relative("box", "world", (0, 0, 0.0))
    xml = b.build()

    assert lowest_non_plane_geom_z(xml) >= 0.009


def test_ground_clearance_lifts_nested_assembly_as_one_unit():
    b = MechanismBuilder()
    b.add_body("parent", "box", (0.1, 0.1, 0.1))
    b.add_body("child", "sphere", 0.2)
    b.attach_to("parent", "child", "fixed")
    b.position_relative("parent", "world", (0, 0, 0.1))
    b.position_relative("child", "parent", (0, 0, -0.4))
    xml = b.build()

    assert lowest_non_plane_geom_z(xml) >= 0.009
    assert 'name="child" pos="0 0 -0.4"' in xml


def test_ground_clearance_keeps_ground_plane_at_zero():
    b = MechanismBuilder()
    b.add_body("box", "box", (0.1, 0.1, 0.1))
    b.position_relative("box", "world", (0, 0, 1.0))
    xml = b.build()
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    plane_id = next(i for i in range(model.ngeom) if int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_PLANE))
    assert float(data.geom_xpos[plane_id][2]) == pytest.approx(0.0)


def test_ground_clearance_leaves_valid_scene_unchanged():
    xml = """<mujoco>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="10 10 0.1"/>
    <body name="box" pos="0 0 1">
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>"""

    assert repair_ground_clearance(xml) == xml


def test_exec_builder_code_repairs_negative_z_xml():
    code = """
b = MechanismBuilder()
b.add_body('ball', 'sphere', 0.2, free=True)
b.position_relative('ball', 'world', (0, 0, -1.0))
xml = b.build()
actuator_schedule = []
"""
    xml, _ = main._exec_builder_code(code)

    assert lowest_non_plane_geom_z(xml) >= 0.009


def test_topology_validator_accepts_connected_mechanism():
    xml, schedule = main._exec_builder_code(moving_pendulum_code())

    main._ensure_prompt_semantics(xml, "a swinging hinged linkage with a ball at the end")
    main._ensure_simulation_moves(xml, schedule)


def test_topology_validator_rejects_detached_free_body_for_connected_prompt():
    xml, _ = main._exec_builder_code(detached_bob_code())

    with pytest.raises(ValueError, match="connected mechanism"):
        main._ensure_prompt_semantics(xml, "a swinging hinged linkage with a ball at the end")


def test_topology_validator_allows_explicit_free_body_prompt():
    b = MechanismBuilder()
    b.add_body("cube", "box", (0.1, 0.1, 0.1), free=True)
    b.position_relative("cube", "world", (0, 0, 2))
    xml = b.build()

    main._ensure_prompt_semantics(xml, "a falling cube")


def test_motion_validator_rejects_static_valid_xml():
    b = MechanismBuilder()
    b.add_body("cube", "box", (0.1, 0.1, 0.1))
    b.position_relative("cube", "world", (0, 0, 2))
    xml = b.build()

    with pytest.raises(ValueError, match="valid but static"):
        main._ensure_simulation_moves(xml, [])


def test_clarify_builds_assumptions_without_unneeded_questions():
    result = main.build_clarification("a pendulum")

    assert result["assumptions"]
    assert "connected body/joint graph" in result["clarified_description"]
    assert result["questions"] == []


def test_generation_loop_corrects_bad_topology(monkeypatch):
    responses = iter([detached_bob_code(), moving_pendulum_code()])

    def fake_chat(*, model, messages):
        return {"message": {"content": next(responses)}}

    monkeypatch.setattr(main.ollama, "chat", fake_chat)

    result = main.asyncio.run(main._run_generation_loop("a hinged linkage with a ball at the end"))

    assert "<freejoint" not in result["xml"]
    assert result["actuator_schedule"] == []


def test_generate_endpoint_with_mocked_llm(monkeypatch):
    client = TestClient(main.app)
    responses = iter([detached_bob_code(), moving_pendulum_code()])

    async def fake_images(description):
        return []

    def fake_chat(*, model, messages):
        return {"message": {"content": next(responses)}}

    monkeypatch.setattr(main, "fetch_reference_images", fake_images)
    monkeypatch.setattr(main.ollama, "chat", fake_chat)
    monkeypatch.setattr(main, "_render_preview", lambda xml, schedule: True)

    res = client.post("/generate", json={"description": "a hinged linkage with a ball at the end"})

    assert res.status_code == 200
    data = res.json()
    assert "xml" in data
    assert data["actuator_schedule"] == []
    mujoco.MjModel.from_xml_string(data["xml"])


def test_generation_loop_falls_back_instead_of_error(monkeypatch):
    def fake_chat(*, model, messages):
        return {"message": {"content": "b = MechanismBuilder()\nxml = b.build()\nactuator_schedule = []"}}

    monkeypatch.setattr(main.ollama, "chat", fake_chat)

    result = main.asyncio.run(main._run_generation_loop("a pendulum"))

    assert result["fallback"] is True
    mujoco.MjModel.from_xml_string(result["xml"])
    main._ensure_simulation_moves(result["xml"], result["actuator_schedule"])


def test_clarify_endpoint():
    client = TestClient(main.app)

    res = client.post("/clarify", json={"description": "a robot"})

    assert res.status_code == 200
    data = res.json()
    assert data["assumptions"]
    assert any(q["key"] == "robot_form" for q in data["questions"])
    assert "Generation assumptions" in data["clarified_description"]


def test_simulate_websocket_streams_frames():
    client = TestClient(main.app)
    xml, schedule = main._exec_builder_code(moving_pendulum_code())

    with client.websocket_connect("/ws/simulate") as ws:
        ws.send_text(json.dumps({"xml": xml, "duration": 0.25, "actuator_schedule": schedule}))
        counts = {"init": 0, "frame": 0, "done": 0}
        while True:
            msg = ws.receive_json()
            counts[msg["type"]] = counts.get(msg["type"], 0) + 1
            if msg["type"] == "done":
                break

    assert counts["init"] == 1
    assert counts["frame"] > 2
    assert counts["done"] == 1

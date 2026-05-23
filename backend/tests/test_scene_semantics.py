import xml.etree.ElementTree as ET

import mujoco
import pytest
from fastapi.testclient import TestClient

import main
from scene_semantics import (
    compile_scene_spec,
    heuristic_scene_spec,
    normalize_scene_spec,
    validate_scene_spec,
)


def lowest_non_plane_geom_z(xml: str) -> float:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    lows = []
    for i in range(model.ngeom):
        gtype = int(model.geom_type[i])
        if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        center_z = float(data.geom_xpos[i][2])
        size = model.geom_size[i]
        mat = data.geom_xmat[i].reshape(3, 3)
        zrow = [float(mat[2, 0]), float(mat[2, 1]), float(mat[2, 2])]
        if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            extent = float(size[0])
        elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
            extent = sum(abs(zrow[j]) * float(size[j]) for j in range(3))
        elif gtype in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
            radius = float(size[0])
            half_length = float(size[1])
            axial = abs(zrow[2]) * half_length
            radial = radius * max(0.0, 1.0 - zrow[2] * zrow[2]) ** 0.5
            extent = axial + radial
        else:
            extent = float(model.geom_rbound[i])
        lows.append(center_z - extent)
    return min(lows)


def body_names(xml: str) -> set[str]:
    root = ET.fromstring(xml)
    return {body.get("name") for body in root.iter("body") if body.get("name")}


def test_scene_spec_assigns_roles_and_preserves_falling_relation():
    spec = heuristic_scene_spec("a pendulum falling on a house")

    roles = {entity["id"]: entity["role"] for entity in spec["entities"]}
    assert roles["pendulum"] == "free"
    assert roles["house"] == "static"
    assert {"type": "falling_onto", "source": "pendulum", "target": "house"} in spec["relationships"]
    assert validate_scene_spec(spec) == []


def test_scene_spec_normalizes_weak_llm_output_with_roles_and_parts():
    spec = normalize_scene_spec({"entities": [{"name": "door", "kind": "door"}]}, "a hinged door swinging open")

    assert validate_scene_spec(spec) == []
    assert spec["entities"][0]["role"] in {"passive", "driven", "jointed", "free", "static"}
    assert spec["entities"][0]["parts"]


def test_compiler_builds_recognizable_static_building_above_ground():
    spec = {
        "description": "a house",
        "entities": [
            {
                "id": "house",
                "name": "house",
                "kind": "building",
                "role": "static",
                "parts": [{"id": "base"}, {"id": "roof_left"}, {"id": "roof_right"}],
            }
        ],
        "relationships": [],
        "simulation_intent": {"moving_entities": ["house"], "motion_source": "none", "description": "static building"},
    }
    xml, schedule = compile_scene_spec(spec)

    names = body_names(xml)
    assert "house_body" in names
    assert "house_roof_left" in names
    assert "house_roof_right" in names
    assert lowest_non_plane_geom_z(xml) >= 0.0
    assert schedule == []


def test_compiler_builds_free_compound_falling_pendulum_above_house():
    spec = heuristic_scene_spec("a pendulum falling on a house")
    xml, schedule = compile_scene_spec(spec)

    root = ET.fromstring(xml)
    pendulum_rod = next(body for body in root.iter("body") if body.get("name") == "pendulum_rod")
    assert pendulum_rod.find("freejoint") is not None
    assert pendulum_rod.find("joint") is None
    assert "pendulum_bob" in body_names(xml)
    assert "house_body" in body_names(xml)
    assert lowest_non_plane_geom_z(xml) >= 0.0
    assert schedule == []


def test_compiler_builds_hinged_mechanism():
    spec = heuristic_scene_spec("a pendulum")
    xml, _ = compile_scene_spec(spec)

    root = ET.fromstring(xml)
    rod = next(body for body in root.iter("body") if body.get("name") == "pendulum_rod")
    joint = rod.find("joint")
    assert joint is not None
    assert joint.get("type") == "hinge"
    assert rod.find("freejoint") is None


def test_compiler_builds_vehicle_with_attached_wheels_and_actuators():
    spec = heuristic_scene_spec("a robot with four wheels")
    xml, schedule = compile_scene_spec(spec)

    names = body_names(xml)
    assert "robot_chassis" in names
    assert {"robot_front_left_wheel", "robot_front_right_wheel", "robot_rear_left_wheel", "robot_rear_right_wheel"} <= names
    assert len(schedule) == 8
    mujoco.MjModel.from_xml_string(xml)


@pytest.mark.parametrize(
    "prompt",
    [
        "a pendulum falling on a house",
        "a robot with four wheels",
        "a ball bouncing on a trampoline",
        "a hinged door swinging open",
        "a two-link robotic arm",
        "a box sliding down a ramp",
        "a hammer hitting a glass pane",
        "a crane lifting a load",
    ],
)
def test_semantic_generation_prompt_suite_without_llm(monkeypatch, prompt):
    async def no_llm(description):
        return None

    monkeypatch.setattr(main, "_scene_spec_from_llm", no_llm)

    result = main.asyncio.run(main._run_semantic_generation(prompt))

    assert "xml" in result
    assert validate_scene_spec(result["scene_spec"]) == []
    assert lowest_non_plane_geom_z(result["xml"]) >= 0.0
    mujoco.MjModel.from_xml_string(result["xml"])


def test_generate_endpoint_returns_scene_spec_and_valid_mjcf(monkeypatch):
    client = TestClient(main.app)

    def fake_chat(*, model, messages):
        return {
            "message": {
                "content": """
                {
                  "description": "a pendulum falling on a house",
                  "entities": [
                    {"id": "pendulum", "name": "pendulum", "kind": "pendulum", "role": "free", "parts": [{"id": "rod"}, {"id": "bob"}]},
                    {"id": "house", "name": "house", "kind": "building", "role": "static", "parts": [{"id": "base"}, {"id": "roof"}]}
                  ],
                  "relationships": [{"type": "falling_onto", "source": "pendulum", "target": "house"}],
                  "simulation_intent": {"moving_entities": ["pendulum"], "motion_source": "gravity", "description": "pendulum falls onto house"}
                }
                """
            }
        }

    monkeypatch.setattr(main.ollama, "chat", fake_chat)
    monkeypatch.setattr(main, "_render_preview", lambda xml, schedule: True)

    res = client.post("/generate", json={"description": "a pendulum falling on a house"})

    assert res.status_code == 200
    data = res.json()
    assert data["scene_spec"]["relationships"][0]["type"] == "falling_onto"
    assert "pendulum_rod" in body_names(data["xml"])
    assert "house_body" in body_names(data["xml"])
    assert lowest_non_plane_geom_z(data["xml"]) >= 0.0

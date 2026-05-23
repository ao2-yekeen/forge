"""
Test primitives + assembler with three canonical systems.
All must compile with MuJoCo and run 100 steps without error.
"""
import sys
sys.path.insert(0, "/home/basit/Documents/forge/backend")

import mujoco
from primitives import (
    gravity, ground, rigid_body, sphere_geom, revolute, cylinder_geom,
    capsule_geom, box_geom, actuator, cylindrical, planar, prismatic,
    spherical, spring, damper, contact_pair, plane_geom, screw, fixed,
    mesh_geom,
)
from assembler import assemble

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

errors = []


def run_test(name: str, prim_list: list, ctrl: dict = None):
    try:
        xml = assemble(prim_list)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        if ctrl:
            # Build actuator name→id map
            for i in range(model.nu):
                aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                if aname and aname in ctrl:
                    data.ctrl[i] = ctrl[aname]
        for _ in range(100):
            mujoco.mj_step(model, data)
        print(f"{PASS} {name}: 100 steps OK  (nbody={model.nbody}, ngeom={model.ngeom}, nu={model.nu})")
    except Exception as e:
        print(f"{FAIL} {name}: {e}")
        errors.append((name, e))


# ===========================================================================
# Test 1 — Pendulum
# ===========================================================================
run_test("Pendulum", [
    gravity(),
    ground(),
    rigid_body("pivot", pos="0 0 2"),
    sphere_geom(0.04, rgba="0.4 0.4 0.4 1", density=8000, body="pivot"),
    rigid_body("arm", pos="0 0 0", parent="pivot", euler="0 45 0"),
    revolute("hinge", axis="0 1 0", range="-180 180", damping=0.5, body="arm"),
    cylinder_geom(0.015, 1.0, pos="0 0 -0.5", rgba="0.6 0.4 0.2 1", density=500, body="arm"),
    rigid_body("bob", pos="0 0 -1", parent="arm"),
    sphere_geom(0.12, rgba="0.8 0.1 0.1 1", density=2000, body="bob"),
])

# ===========================================================================
# Test 2 — Robot arm (3-joint, motorized)
# ===========================================================================
run_test("Robot arm", [
    gravity(),
    ground(),
    rigid_body("base", pos="0 0 0.15"),
    cylinder_geom(0.06, 0.3, rgba="0.3 0.3 0.35 1", density=2000, body="base"),
    rigid_body("link1", pos="0 0 0.3", parent="base"),
    revolute("j1", axis="0 1 0", range="-120 120", damping=1.0, body="link1"),
    capsule_geom(0.03, 0.4, pos="0 0 0.2", rgba="0.2 0.5 0.8 1", density=400, body="link1"),
    rigid_body("link2", pos="0 0 0.4", parent="link1"),
    revolute("j2", axis="0 1 0", range="-120 120", damping=1.0, body="link2"),
    capsule_geom(0.025, 0.35, pos="0 0 0.175", rgba="0.2 0.7 0.6 1", density=400, body="link2"),
    rigid_body("link3", pos="0 0 0.35", parent="link2"),
    revolute("j3", axis="0 1 0", range="-120 120", damping=0.5, body="link3"),
    capsule_geom(0.02, 0.25, pos="0 0 0.125", rgba="0.8 0.5 0.1 1", density=400, body="link3"),
    actuator("m1", "j1", gear=5.0),
    actuator("m2", "j2", gear=4.0),
    actuator("m3", "j3", gear=2.0),
], ctrl={"m1": 2.0, "m2": 1.5, "m3": 1.0})

# ===========================================================================
# Test 3 — Hopper (trapdoor + falling ball)
# ===========================================================================
run_test("Hopper", [
    gravity(),
    ground(),
    # Static walls
    rigid_body("wall_front", pos="0 0.0665 0.1"),
    box_geom([0.065, 0.0015, 0.1], rgba="0.7 0.7 0.85 0.7", density=800, body="wall_front"),
    rigid_body("wall_back", pos="0 -0.0665 0.1"),
    box_geom([0.065, 0.0015, 0.1], rgba="0.7 0.7 0.85 0.7", density=800, body="wall_back"),
    rigid_body("wall_left", pos="-0.0665 0 0.1"),
    box_geom([0.0015, 0.065, 0.1], rgba="0.7 0.7 0.85 0.7", density=800, body="wall_left"),
    rigid_body("wall_right", pos="0.0665 0 0.1"),
    box_geom([0.0015, 0.065, 0.1], rgba="0.7 0.7 0.85 0.7", density=800, body="wall_right"),
    # Trapdoor
    rigid_body("trapdoor", pos="0 0 0.003"),
    revolute("trapdoor_hinge", axis="1 0 0", range="0 90", damping=0.1, body="trapdoor"),
    box_geom([0.015, 0.065, 0.003], rgba="0.6 0.3 0.1 1", density=1200, body="trapdoor"),
    # Falling material ball
    rigid_body("ball", pos="0 0 0.25", free=True),
    sphere_geom(0.018, rgba="0.9 0.7 0.1 1", density=500, body="ball"),
    # Motor
    actuator("motor", "trapdoor_hinge", gear=1.0, ctrllimited=True, ctrlrange="0 2"),
], ctrl={"motor": 2.0})

# ===========================================================================
# Test 4 — Additional primitive coverage (spring, damper, spherical joint)
# ===========================================================================
run_test("Spring + spherical", [
    gravity(),
    ground(),
    rigid_body("post", pos="0 0 1"),
    cylinder_geom(0.04, 0.8, rgba="0.4 0.4 0.4 1", density=2000, body="post"),
    rigid_body("pendulum", pos="0 0 0", parent="post"),
    spherical("ball_joint", body="pendulum"),
    sphere_geom(0.08, rgba="0.9 0.3 0.1 1", density=1500, body="pendulum"),
    spring("s1", "ball_joint", stiffness=50.0),
    damper("d1", "ball_joint", damping=2.0),
])

# ===========================================================================
# Test 5 — Cylindrical joint
# ===========================================================================
run_test("Cylindrical joint", [
    gravity(),
    ground(),
    rigid_body("column", pos="0 0 0.5"),
    cylinder_geom(0.03, 1.0, rgba="0.5 0.5 0.6 1", density=1000, body="column"),
    rigid_body("slider", pos="0 0 0.5", parent="column"),
    *cylindrical("cyl", axis="0 0 1", body="slider"),
    sphere_geom(0.06, rgba="0.8 0.2 0.2 1", density=800, body="slider"),
])

# ===========================================================================
# Summary
# ===========================================================================
print()
if errors:
    print(f"{FAIL} {len(errors)} test(s) failed: {[n for n, _ in errors]}")
    sys.exit(1)
else:
    print(f"{PASS} All tests passed.")

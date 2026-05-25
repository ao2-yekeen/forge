from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any
import pyvista as pv
import ollama
from build123d import *
import tempfile

# ======================================================================
# CONFIG
# ======================================================================

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:14b")

client = ollama.Client(host=OLLAMA_HOST)


# ======================================================================
# SYSTEM PROMPT
# ======================================================================

# Paste the JSON SYSTEM_PROMPT from the previous message here.
# It must instruct the model to output:
# {
#   "name": "...",
#   "units": "mm",
#   "operations": [...]
# }
SYSTEM_PROMPT = json.dumps(
    {
        "role": "CAD operation planner",
        "task": (
            "Convert a natural language CAD request into a JSON operation sequence. "
            "Do not output Python code. Do not output markdown. "
            "Only output valid JSON."
        ),
        "output_schema": {
            "name": "string",
            "units": "mm",
            "operations": [
                {
                    "id": "optional unique id",
                    "type": "operation type",
                    "parameters": "operation-specific fields"
                }
            ]
        },
        "strict_rules": [
            "Output only valid JSON.",
            "Never output Python code.",
            "Never output markdown.",
            "All dimensions are in millimetres.",
            "Use only the allowed operation vocabulary.",
            "Never invent new operation types.",
            "For custom geometry, always start with a Sketch operation.",
            "A Sketch operation must always come before Extrude, Revolve, Sweep, Loft, Rib, or Thicken.",
            "Feature operations come after 3D generation operations.",
            "Transform and Boolean operations come after solids exist.",
            "Standard part operations may appear directly because they are plugin-generated parts."
        ],
        "allowed_operations": {
            "sketch_operations": [
                "Rectangle",
                "Circle",
                "Ellipse",
                "RegularPolygon",
                "Slot",
                "Arc",
                "CustomProfile",
                "IBeam",
                "CChannel",
                "HollowRect"
            ],
            "three_d_operations": [
                "Extrude",
                "Revolve",
                "Sweep",
                "Loft",
                "Shell",
                "Thicken"
            ],
            "feature_operations": [
                "Hole",
                "CounterBore",
                "Countersink",
                "Thread",
                "Fillet",
                "Chamfer",
                "Draft",
                "Rib",
                "Knurl"
            ],
            "boolean_operations": [
                "Union",
                "Subtract",
                "Intersect"
            ],
            "transform_operations": [
                "Move",
                "Rotate",
                "Mirror",
                "Scale",
                "LinearArray",
                "CircularArray",
                "BoltCircle"
            ],
            "standard_part_operations": [
                "HexBolt",
                "HexNut",
                "Washer",
                "SocketHeadBolt",
                "SpurGear",
                "BallBearing",
                "CompressionSpring",
                "VBelt"
            ]
        },
        "examples": [
            {
                "input": "a rectangular plate 100mm x 60mm 5mm thick with 4 M4 holes in corners",
                "output": {
                    "name": "mounting_plate",
                    "units": "mm",
                    "operations": [
                        {
                            "id": "base_sketch",
                            "type": "Sketch",
                            "plane": "XY",
                            "profiles": [
                                {
                                    "op": "Rectangle",
                                    "width": 100,
                                    "height": 60,
                                    "origin": [0, 0]
                                }
                            ]
                        },
                        {
                            "id": "base_solid",
                            "type": "Extrude",
                            "sketch": "base_sketch",
                            "amount": 5,
                            "direction": [0, 0, 1],
                            "mode": "ADD"
                        },
                        {
                            "type": "Hole",
                            "solid": "base_solid",
                            "diameter": 4.2,
                            "depth": 5,
                            "face": "top",
                            "pattern": {
                                "type": "grid",
                                "x_spacing": 80,
                                "y_spacing": 40,
                                "x_count": 2,
                                "y_count": 2
                            }
                        },
                        {
                            "type": "Fillet",
                            "solid": "base_solid",
                            "edges": "vertical",
                            "radius": 2
                        }
                    ]
                }
            },
            {
                "input": "a sphere 100mm diameter",
                "output": {
                    "name": "sphere",
                    "units": "mm",
                    "operations": [
                        {
                            "id": "sphere_profile",
                            "type": "Sketch",
                            "plane": "XZ",
                            "profiles": [
                                {
                                    "op": "CustomProfile",
                                    "points": [
                                        [0, -50],
                                        [35.36, -35.36],
                                        [50, 0],
                                        [35.36, 35.36],
                                        [0, 50]
                                    ]
                                }
                            ]
                        },
                        {
                            "id": "sphere_solid",
                            "type": "Revolve",
                            "sketch": "sphere_profile",
                            "axis": "Z",
                            "angle": 360
                        }
                    ]
                }
            },
            {
                "input": "an M10x50 hex bolt with matching nut and washer",
                "output": {
                    "name": "m10_bolt_nut_washer",
                    "units": "mm",
                    "operations": [
                        {
                            "id": "bolt",
                            "type": "HexBolt",
                            "diameter": "M10",
                            "length": 50,
                            "grade": "8.8"
                        },
                        {
                            "id": "nut",
                            "type": "HexNut",
                            "diameter": "M10",
                            "grade": "8"
                        },
                        {
                            "id": "washer",
                            "type": "Washer",
                            "inner_d": 10.5,
                            "outer_d": 20,
                            "thick": 2
                        },
                        {
                            "type": "Move",
                            "solid": "nut",
                            "x": 70,
                            "y": 0,
                            "z": 0
                        },
                        {
                            "type": "Move",
                            "solid": "washer",
                            "x": 95,
                            "y": 0,
                            "z": 0
                        }
                    ]
                }
            }
        ]
    },
    indent=2,
)



# ======================================================================
# HARDCODED TEST PROMPTS
# ======================================================================

TEST_PROMPTS = [
    "a rectangular plate 100mm x 60mm 5mm thick with 4 M4 holes in the corners",
    "a cylindrical shaft 150mm long 20mm diameter",
    "a sphere 100mm diameter",
    "a hollow cylinder 50mm outer diameter 40mm inner diameter 80mm tall",
    "a shaft with a keyway",
    "an L-bracket",
    "a rectangular enclosure 100x80x60mm 3mm walls open top",
    "a pulley 50mm diameter 20mm wide with central 10mm bore and belt groove",
    "a motor mounting plate with holes for a NEMA17 motor",
    "an M10x50 hex bolt with matching nut and washer",
]


# ======================================================================
# ALLOWED VOCABULARY
# ======================================================================

SKETCH_PROFILE_OPS = {
    "Rectangle",
    "Circle",
    "Ellipse",
    "RegularPolygon",
    "Slot",
    "Arc",
    "CustomProfile",
    "IBeam",
    "CChannel",
    "HollowRect",
}

OP_TYPES = {
    "Sketch",
    "Extrude",
    "Revolve",
    "Sweep",
    "Loft",
    "Shell",
    "Thicken",
    "Hole",
    "CounterBore",
    "Countersink",
    "Thread",
    "Fillet",
    "Chamfer",
    "Draft",
    "Rib",
    "Knurl",
    "Union",
    "Subtract",
    "Intersect",
    "Move",
    "Rotate",
    "Mirror",
    "Scale",
    "LinearArray",
    "CircularArray",
    "BoltCircle",
    "HexBolt",
    "HexNut",
    "Washer",
    "SocketHeadBolt",
    "SpurGear",
    "BallBearing",
    "CompressionSpring",
    "VBelt",
}


# ======================================================================
# LLM
# ======================================================================

def generate_operation_spec(description: str) -> dict[str, Any]:
    response = client.chat(
        model=OLLAMA_MODEL,
        options={
            "temperature": 0.1,
            "top_p": 0.9,
            "num_predict": 2000,
        },
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ],
    )

    text = response["message"]["content"].strip()
    spec = extract_json(text)
    return validate_spec(spec)


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        text = text[start:end + 1]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON:\n{text}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON must be an object")

    return parsed


def validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if "operations" not in spec or not isinstance(spec["operations"], list):
        raise ValueError("Spec must contain operations list")

    clean_ops = []

    for i, op in enumerate(spec["operations"]):
        if not isinstance(op, dict):
            raise ValueError(f"Operation {i} must be an object")

        op_type = op.get("type")

        if op_type not in OP_TYPES:
            raise ValueError(f"Unsupported operation type: {op_type}")

        if op_type == "Sketch":
            for profile in op.get("profiles", []):
                if profile.get("op") not in SKETCH_PROFILE_OPS:
                    raise ValueError(f"Unsupported sketch profile: {profile.get('op')}")

        clean_ops.append(op)

    return {
        "name": spec.get("name", "generated_part"),
        "units": "mm",
        "operations": clean_ops,
    }


# ======================================================================
# EXECUTOR
# ======================================================================

def execute_operation_spec(spec: dict[str, Any]):
    operations = spec["operations"]

    sketches: dict[str, dict[str, Any]] = {}
    move_offsets = collect_move_offsets(operations)

    with BuildPart() as result:
        for index, op in enumerate(operations):
            op_type = op["type"]

            try:
                if op_type == "Sketch":
                    sketch_id = op.get("id")
                    if not sketch_id:
                        raise ValueError("Sketch operation needs an id")
                    sketches[sketch_id] = op

                elif op_type == "Extrude":
                    execute_extrude(op, sketches)

                elif op_type == "Revolve":
                    execute_revolve(op, sketches)

                elif op_type == "Hole":
                    execute_hole(op)

                elif op_type == "CounterBore":
                    execute_counterbore(op)

                elif op_type == "Countersink":
                    execute_countersink(op)

                elif op_type == "Thread":
                    execute_thread_approx(op)

                elif op_type == "Fillet":
                    execute_fillet(op, result)

                elif op_type == "Chamfer":
                    execute_chamfer(op, result)

                elif op_type == "Shell":
                    execute_shell(op, result)

                elif op_type in {
                    "HexBolt",
                    "HexNut",
                    "Washer",
                    "SocketHeadBolt",
                    "SpurGear",
                    "BallBearing",
                    "CompressionSpring",
                    "VBelt",
                }:
                    solid = make_standard_part(op)
                    solid_id = op.get("id")
                    offset = move_offsets.get(solid_id, (0, 0, 0))
                    add(solid.moved(Location(offset)))

                elif op_type in {
                    "Move",
                    "Rotate",
                    "Mirror",
                    "Scale",
                    "LinearArray",
                    "CircularArray",
                    "BoltCircle",
                    "Union",
                    "Subtract",
                    "Intersect",
                    "Sweep",
                    "Loft",
                    "Thicken",
                    "Draft",
                    "Rib",
                    "Knurl",
                }:
                    print(f"  ! Skipping unsupported executor op for now: {op_type}")

                else:
                    raise ValueError(f"Unhandled operation: {op_type}")

            except Exception as exc:
                raise RuntimeError(f"Operation {index} failed: {op}") from exc

    return result.part


def collect_move_offsets(operations: list[dict[str, Any]]) -> dict[str, tuple[float, float, float]]:
    offsets = {}

    for op in operations:
        if op.get("type") != "Move":
            continue

        solid_id = op.get("solid")
        if not solid_id:
            continue

        offsets[solid_id] = (
            float(op.get("x", 0)),
            float(op.get("y", 0)),
            float(op.get("z", 0)),
        )

    return offsets


# ======================================================================
# SKETCH RENDERING
# ======================================================================

def execute_extrude(op: dict[str, Any], sketches: dict[str, dict[str, Any]]) -> None:
    sketch_id = op.get("sketch")
    sketch = sketches.get(sketch_id)

    if not sketch:
        raise ValueError(f"Missing sketch: {sketch_id}")

    amount = float(op.get("amount", 10))
    mode = build_mode(op.get("mode", "ADD"))

    profiles = sketch.get("profiles", [])
    if not profiles:
        raise ValueError("Sketch has no profiles")

    # Robust special cases
    if len(profiles) == 1:
        p = profiles[0]

        if p.get("op") == "Rectangle":
            Box(
                num(p, "width", 100),
                num(p, "height", 60),
                amount,
                mode=mode,
            )
            return

        if p.get("op") == "Circle":
            Cylinder(
                radius=num(p, "radius", 25),
                height=amount,
                mode=mode,
            )
            return

        if p.get("op") == "RegularPolygon":
            Cylinder(
                radius=num(p, "circumradius", 25),
                height=amount,
                vertices=integer(p, "sides", 6),
                mode=mode,
            )
            return

    # Fallback for complex sketches
    with BuildSketch(get_plane(sketch.get("plane", "XY"))):
        render_profiles(sketch)

    extrude(amount=amount, mode=mode)


def execute_revolve(op: dict[str, Any], sketches: dict[str, dict[str, Any]]) -> None:
    sketch_id = op.get("sketch")
    sketch = sketches.get(sketch_id)

    if not sketch:
        raise ValueError(f"Missing sketch: {sketch_id}")

    with BuildSketch(get_plane(sketch.get("plane", "XZ"))):
        render_profiles(sketch)

    axis = get_axis(op.get("axis", "Z"))
    angle = float(op.get("angle", 360))

    revolve(axis=axis, revolution_arc=angle)


def render_profiles(sketch: dict[str, Any]) -> None:
    profiles = sketch.get("profiles", [])

    if not profiles:
        raise ValueError("Sketch has no profiles")

    for profile in profiles:
        render_profile(profile)


def render_profile(profile: dict[str, Any]) -> None:
    op = profile.get("op")

    if op == "Rectangle":
        width = num(profile, "width", 100)
        height = num(profile, "height", 50)
        origin = profile.get("origin", [0, 0])
        with Locations((origin[0], origin[1])):
            Rectangle(width, height)

    elif op == "Circle":
        radius = num(profile, "radius", 25)
        center = profile.get("center", [0, 0])
        with Locations((center[0], center[1])):
            Circle(radius)

    elif op == "Ellipse":
        rx = num(profile, "rx", 30)
        ry = num(profile, "ry", 15)
        center = profile.get("center", [0, 0])
        with Locations((center[0], center[1])):
            Ellipse(rx, ry)

    elif op == "RegularPolygon":
        sides = integer(profile, "sides", 6)
        radius = num(profile, "circumradius", 25)
        RegularPolygon(radius, sides)

    elif op == "Slot":
        length = num(profile, "length", 50)
        width = num(profile, "width", 10)
        Rectangle(length, width)

    elif op == "CustomProfile":
        points = profile.get("points", [])
        if len(points) < 3:
            raise ValueError("CustomProfile needs at least 3 points")

        pts = [(float(x), float(y)) for x, y in points]

        with BuildLine():
            for i in range(len(pts) - 1):
                Line(pts[i], pts[i + 1])
            Line(pts[-1], pts[0])

        make_face()

    elif op == "HollowRect":
        outer_w = num(profile, "outer_w", 100)
        outer_h = num(profile, "outer_h", 60)
        wall = num(profile, "wall_thickness", 3)

        Rectangle(outer_w, outer_h)
        Rectangle(
            max(outer_w - 2 * wall, 1),
            max(outer_h - 2 * wall, 1),
            mode=Mode.SUBTRACT,
        )

    elif op == "IBeam":
        height = num(profile, "height", 120)
        flange_w = num(profile, "flange_w", 60)
        web_t = num(profile, "web_t", 8)
        flange_t = num(profile, "flange_t", 10)

        with Locations((0, height / 2 - flange_t / 2)):
            Rectangle(flange_w, flange_t)
        with Locations((0, -height / 2 + flange_t / 2)):
            Rectangle(flange_w, flange_t)
        Rectangle(web_t, height)

    elif op == "CChannel":
        height = num(profile, "height", 100)
        width = num(profile, "width", 50)
        thickness = num(profile, "thickness", 6)

        Rectangle(width, height)
        with Locations((thickness / 2, 0)):
            Rectangle(
                max(width - thickness, 1),
                max(height - 2 * thickness, 1),
                mode=Mode.SUBTRACT,
            )

    elif op == "Arc":
        radius = num(profile, "radius", 25)
        Circle(radius)

    else:
        raise ValueError(f"Unsupported profile op: {op}")


# ======================================================================
# FEATURES
# ======================================================================

def execute_hole(op: dict[str, Any]) -> None:
    diameter = num(op, "diameter", 4.2)
    depth = op.get("depth")

    if depth is None:
        depth = 1000
    else:
        depth = float(depth)

    pattern = op.get("pattern")

    if isinstance(pattern, dict) and pattern.get("type") == "grid":
        with GridLocations(
            num(pattern, "x_spacing", 40),
            num(pattern, "y_spacing", 40),
            integer(pattern, "x_count", 2),
            integer(pattern, "y_count", 2),
        ):
            Hole(radius=diameter / 2, depth=depth)
    elif isinstance(pattern, dict) and pattern.get("type") == "bolt_circle":
        pcd = num(pattern, "pcd", 40)
        count = integer(pattern, "count", 4)

        with PolarLocations(pcd / 2, count):
            Hole(radius=diameter / 2, depth=depth)
    else:
        Hole(radius=diameter / 2, depth=depth)


def execute_counterbore(op: dict[str, Any]) -> None:
    hole_d = num(op, "hole_d", 5)
    cb_d = num(op, "cb_d", 10)
    cb_depth = num(op, "cb_depth", 4)

    Hole(radius=hole_d / 2, depth=1000)
    Hole(radius=cb_d / 2, depth=cb_depth)


def execute_countersink(op: dict[str, Any]) -> None:
    hole_d = num(op, "hole_d", 5)
    cs_d = num(op, "cs_d", 10)

    Hole(radius=hole_d / 2, depth=1000)
    Hole(radius=cs_d / 2, depth=2)


def execute_thread_approx(op: dict[str, Any]) -> None:
    diameter = parse_metric_diameter(op.get("diameter", "M10"))
    depth = num(op, "depth", 20)

    Hole(radius=diameter / 2, depth=depth)


def execute_fillet(op: dict[str, Any], result: BuildPart) -> None:
    radius = num(op, "radius", 2)
    edges = str(op.get("edges", "all")).lower()

    try:
        if edges == "vertical":
            fillet(result.edges().filter_by(Axis.Z), radius=radius)
        elif edges == "top":
            fillet(result.faces().sort_by(Axis.Z)[-1].edges(), radius=radius)
        elif edges == "bottom":
            fillet(result.faces().sort_by(Axis.Z)[0].edges(), radius=radius)
        else:
            fillet(result.edges(), radius=radius)
    except Exception:
        pass


def execute_chamfer(op: dict[str, Any], result: BuildPart) -> None:
    size = num(op, "size", 1)

    try:
        chamfer(result.edges(), length=size)
    except Exception:
        pass


def execute_shell(op: dict[str, Any], result: BuildPart) -> None:
    thickness = num(op, "thickness", 3)
    open_faces = op.get("open_faces", ["top"])

    faces = []

    if "top" in open_faces:
        faces.append(result.faces().sort_by(Axis.Z)[-1])
    if "bottom" in open_faces:
        faces.append(result.faces().sort_by(Axis.Z)[0])

    offset(amount=-thickness, openings=faces)


# ======================================================================
# STANDARD PART APPROXIMATIONS
# ======================================================================

def make_standard_part(op: dict[str, Any]):
    op_type = op["type"]

    if op_type == "HexBolt":
        return make_hex_bolt(op)

    if op_type == "HexNut":
        return make_hex_nut(op)

    if op_type == "Washer":
        return make_washer(op)

    if op_type == "SocketHeadBolt":
        return make_socket_head_bolt(op)

    if op_type == "SpurGear":
        return make_spur_gear(op)

    if op_type == "BallBearing":
        return make_ball_bearing(op)

    if op_type == "CompressionSpring":
        return make_spring(op)

    if op_type == "VBelt":
        return make_vbelt(op)

    raise ValueError(f"Unsupported standard part: {op_type}")


def make_hex_bolt(op: dict[str, Any]):
    d = parse_metric_diameter(op.get("diameter", "M10"))
    length = num(op, "length", 50)
    head_af = d * 1.6
    head_h = d * 0.7

    with BuildPart() as p:
        Cylinder(radius=d / 2, height=length)
        with Locations((0, 0, length / 2 + head_h / 2)):
            Cylinder(radius=head_af / 2, height=head_h, vertices=6)

    return p.part


def make_socket_head_bolt(op: dict[str, Any]):
    d = parse_metric_diameter(op.get("diameter", "M8"))
    length = num(op, "length", 40)
    head_d = d * 1.6
    head_h = d

    with BuildPart() as p:
        Cylinder(radius=d / 2, height=length)
        with Locations((0, 0, length / 2 + head_h / 2)):
            Cylinder(radius=head_d / 2, height=head_h)
            Hole(radius=d * 0.35, depth=head_h)

    return p.part


def make_hex_nut(op: dict[str, Any]):
    d = parse_metric_diameter(op.get("diameter", "M10"))
    af = d * 1.7
    h = d * 0.8

    with BuildPart() as p:
        Cylinder(radius=af / 2, height=h, vertices=6)
        Hole(radius=d / 2, depth=h * 1.2)

    return p.part


def make_washer(op: dict[str, Any]):
    inner = num(op, "inner_d", 10.5)
    outer = num(op, "outer_d", inner * 2)
    thick = num(op, "thick", 2)

    with BuildPart() as p:
        Cylinder(radius=outer / 2, height=thick)
        Cylinder(radius=inner / 2, height=thick * 1.2, mode=Mode.SUBTRACT)

    return p.part


def make_spur_gear(op: dict[str, Any]):
    module = num(op, "module", 2)
    teeth = integer(op, "teeth", 20)
    width = num(op, "width", 10)

    pitch_d = module * teeth
    outer_d = pitch_d + 2 * module

    with BuildPart() as p:
        Cylinder(radius=outer_d / 2, height=width)
        Hole(radius=(module * 3), depth=width)

    return p.part


def make_ball_bearing(op: dict[str, Any]):
    bore = num(op, "bore", 10)
    od = num(op, "od", 30)
    width = num(op, "width", 9)

    with BuildPart() as p:
        Cylinder(radius=od / 2, height=width)
        Cylinder(radius=bore / 2, height=width * 1.2, mode=Mode.SUBTRACT)

    return p.part


def make_spring(op: dict[str, Any]):
    od = num(op, "od", 20)
    free_len = num(op, "free_len", 60)

    with BuildPart() as p:
        Cylinder(radius=od / 2, height=free_len)
        Cylinder(radius=od / 2 - 2, height=free_len * 1.1, mode=Mode.SUBTRACT)

    return p.part


def make_vbelt(op: dict[str, Any]):
    length = num(op, "length", 200)

    with BuildPart() as p:
        Box(length, 10, 6)

    return p.part


# ======================================================================
# HELPERS
# ======================================================================

def num(obj: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(obj.get(key, default))
    except Exception:
        return default


def integer(obj: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(obj.get(key, default))
    except Exception:
        return default


def parse_metric_diameter(value: Any) -> float:
    text = str(value).upper().replace("M", "").strip()

    try:
        return float(text)
    except Exception:
        return 10.0


def get_plane(value: str):
    value = str(value).upper()

    if value == "XZ":
        return Plane.XZ
    if value == "YZ":
        return Plane.YZ

    return Plane.XY


def get_axis(value: Any):
    value = str(value).upper()

    if value == "X":
        return Axis.X
    if value == "Y":
        return Axis.Y

    return Axis.Z


def build_mode(value: str):
    value = str(value).upper()

    if value == "SUBTRACT":
        return Mode.SUBTRACT
    if value == "INTERSECT":
        return Mode.INTERSECT

    return Mode.ADD


# ======================================================================
# EXPORT / DISPLAY
# ======================================================================

def run_single(description: str, idx: int, total: int):
    print(f"\n[{idx + 1}/{total}] {description}")
    print(f"  Model: {OLLAMA_MODEL}")
    print("  Generating operation JSON...")

    spec = generate_operation_spec(description)

    print("  Operation spec:")
    print(json.dumps(spec, indent=2))

    try:
        solid = execute_operation_spec(spec)

        if solid:
            bbox = solid.bounding_box()
            print(
                f"  ✓ Built: "
                f"{bbox.size.X:.1f} x {bbox.size.Y:.1f} x {bbox.size.Z:.1f} mm"
            )
            return solid

        print("  ✗ No solid produced")
        return None

    except Exception as exc:
        print(f"  ✗ Build failed: {exc}")
        return None


def export_combined_scene(solids, out_dir: str) -> str:
    combined = None
    cursor_x = 0

    for label, solid in solids:
        bbox = solid.bounding_box()
        width = bbox.size.X

        moved = solid.moved(Location((cursor_x - bbox.min.X, 0, 0)))

        if combined is None:
            combined = moved
        else:
            combined = combined + moved

        cursor_x += width + 30

    combined_path = os.path.join(out_dir, "ALL_TEST_OUTPUTS.stl")
    export_stl(combined, combined_path)

    return combined_path

def show_part(part, name="Part", color="lightblue"):
    """Display a build123d part in an interactive PyVista window."""

    # Write to temp file (PyVista needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        tmp_path = tmp.name

    export_stl(part, tmp_path)

    # Load with PyVista
    mesh = pv.read(tmp_path)

    # Clean up temp file
    os.remove(tmp_path)

    # Print info
    print(f"\n─── {name} ───")
    print(f"  Volume:  {part.volume:.2f} mm³")
    print(f"  Area:    {part.area:.2f} mm²")
    bbox = part.bounding_box()
    print(f"  Bounds:  {bbox.size}")
    print(f"  Center:  {part.center()}")
    print(f"  Triangles: {mesh.n_faces}")

    # Create plotter
    plotter = pv.Plotter(window_size=[1000, 800])
    plotter.add_mesh(
        mesh,
        color=color,
        show_edges=True,
        edge_color="darkgray",
        line_width=0.5,
        opacity=0.9,
        smooth_shading=True,
    )

    # Add axes and grid
    plotter.add_axes()
    plotter.show_grid()

    # Set title
    plotter.add_text(f"{name} — {part.volume:.0f} mm³", font_size=14, color="black")

    # Show (blocks until window closed)
    plotter.show()

    return mesh


def main():
    print("Forge — Operation Vocabulary CAD Pipeline")
    print(f"Model: {OLLAMA_MODEL}")
    print(f"Host:  {OLLAMA_HOST}")
    print(f"Running {len(TEST_PROMPTS)} hardcoded prompts")
    print("=" * 70)

    solids = []
    passed = 0

    for i, prompt in enumerate(TEST_PROMPTS):
        solid = run_single(prompt, i, len(TEST_PROMPTS))


        if not solid:
            print("build failed")
            continue
    
        show_part(solid)

        if solid:
            solids.append((prompt[:40], solid))
            passed += 1

    print("\n" + "=" * 70)
    print(f"SUCCESS: {passed}/{len(TEST_PROMPTS)} prompts built")


    out_dir = tempfile.mkdtemp(prefix="forge_ops_")
    print(f"\nExport directory:\n{out_dir}\n")

    stl_files = []

    for i, (label, solid) in enumerate(solids):
        safe_label = re.sub(r"[^a-zA-Z0-9_]+", "_", label).strip("_")

        stl_path = os.path.join(
            out_dir,
            f"part_{i:02d}_{safe_label}.stl",
        )

        try:
            export_stl(solid, stl_path)
            stl_files.append(stl_path)
            print(f"[{i:02d}] {Path(stl_path).name}")
        except Exception as exc:
            print(f"Export failed: {exc}")

    combined_path = export_combined_scene(solids, out_dir)

    print("\nCombined scene STL:")
    print(f"  {combined_path}")

    try:
        if os.name == "posix":
            for viewer in ["meshlab", "freecad", "xdg-open"]:
                result = subprocess.run(
                    ["which", viewer],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    subprocess.Popen([viewer, combined_path])
                    print(f"\nOpened combined STL in {viewer}")
                    break
    except Exception:
        pass

    print("\nGenerated STL files:")
    for path in stl_files:
        print(f"  {path}")

    print(f"\nCombined scene:\n  {combined_path}")


if __name__ == "__main__":
    main()
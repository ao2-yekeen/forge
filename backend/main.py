import asyncio
import base64
import inspect
import json
import logging
import logging.handlers
import os
import re
import traceback
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

# Load .env from the same directory as this file (won't override existing env vars)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from lxml import etree as lxml_etree

import mujoco
import numpy as np
import ollama
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import assembler
import primitives as prim_module
from builder import MechanismBuilder, repair_ground_clearance
from image_search import fetch_reference_images
from scene_semantics import (
    SCENE_SPEC_PROMPT,
    compile_scene_spec,
    extract_json_object,
    heuristic_scene_spec,
    normalize_scene_spec,
    validate_scene_spec,
)

_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_file_handler = logging.handlers.RotatingFileHandler(
    _LOG_DIR / "forge.log",
    maxBytes=5 * 1024 * 1024,  # 5 MB per file
    backupCount=5,
    encoding="utf-8",
)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(), _file_handler],
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Gemini Flash vision client (used for image relevance filtering + generation)
# --------------------------------------------------------------------------- #

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "models/gemini-2.5-flash-lite"
_gemini_client: Optional[Any] = None


def _get_gemini():
    """Return a configured google-genai Client, or None if no API key."""
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    if not GEMINI_API_KEY:
        logger.warning("[gemini] GEMINI_API_KEY not set — vision steps will be skipped")
        return None
    try:
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info(f"[gemini] {GEMINI_MODEL} client initialised")
        return _gemini_client
    except Exception as e:
        logger.error(f"[gemini] Failed to initialise Gemini client: {e}")
        return None


def _make_image_part(data: bytes):
    """Convert image bytes to a google-genai Part with inline image data."""
    from google.genai import types
    from PIL import Image
    img = Image.open(BytesIO(data))
    fmt = (img.format or "JPEG").upper()
    mime = f"image/{'jpeg' if fmt == 'JPEG' else fmt.lower()}"
    buf = BytesIO()
    img.save(buf, format=fmt)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type=mime)


async def _gemini_filter_images(
    description: str, image_data: list[bytes]
) -> list[bytes]:
    """
    Send images to Gemini Flash and return only those marked RELEVANT.
    Images that cause API errors are silently skipped.
    """
    client = _get_gemini()
    if client is None or not image_data:
        return []

    from google.genai import types

    prompt = (
        f'The user wants to build: "{description}"\n'
        "These images may or may not show the described mechanism. "
        "For each image I provide, respond with exactly RELEVANT or IRRELEVANT on a new line. "
        f"Respond with exactly {len(image_data)} lines, one per image, in order."
    )

    parts: list[Any] = [types.Part.from_text(text=prompt)]
    valid_indices: list[int] = []
    for i, data in enumerate(image_data):
        try:
            parts.append(_make_image_part(data))
            valid_indices.append(i)
        except Exception as e:
            logger.debug(f"[gemini] Skipped unreadable image {i + 1}: {e}")

    if not valid_indices:
        return []

    try:
        response = await asyncio.to_thread(
            lambda: client.models.generate_content(
                model=GEMINI_MODEL,
                contents=parts,
            )
        )
        lines = [ln.strip().upper() for ln in response.text.strip().splitlines() if ln.strip()]
        relevant: list[bytes] = []
        for idx, verdict in zip(valid_indices, lines):
            if verdict == "RELEVANT":
                relevant.append(image_data[idx])
                logger.info(f"[gemini] Image {idx + 1}/{len(image_data)} → RELEVANT")
            else:
                logger.info(f"[gemini] Image {idx + 1}/{len(image_data)} → IRRELEVANT")
        return relevant
    except Exception as e:
        logger.warning(f"[gemini] Relevance filtering failed: {e}")
        return []


async def _gemini_generate_code(
    description: str, relevant_images: list[bytes]
) -> Optional[str]:
    """
    Call Gemini Flash with description + relevant images to generate builder Python code.
    Returns the raw LLM text, or None on failure.
    """
    client = _get_gemini()
    if client is None:
        return None

    from google.genai import types

    image_context = (
        "Use the images to determine proportions, joint locations, and spatial relationships. "
        "Extract geometry from the images rather than guessing."
    )
    parts: list[Any] = [
        types.Part.from_text(text=SYSTEM_PROMPT + "\n\n" + image_context),
        types.Part.from_text(text=f"Description: {description}"),
    ]
    for i, data in enumerate(relevant_images):
        try:
            parts.append(_make_image_part(data))
        except Exception as e:
            logger.debug(f"[gemini] Skipped image {i + 1} in generation: {e}")

    try:
        response = await asyncio.to_thread(
            lambda: client.models.generate_content(
                model=GEMINI_MODEL,
                contents=parts,
            )
        )
        logger.info(f"[gemini] Vision-enhanced code generation succeeded ({len(relevant_images)} images)")
        return response.text
    except Exception as e:
        logger.warning(f"[gemini] Vision generation failed: {e}")
        return None


def _extract_python_code(text: str) -> str:
    """Strip markdown fences from LLM response and return bare Python code."""
    text = text.strip()
    m = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def _exec_builder_code(code: str) -> tuple[str, list]:
    """
    Execute LLM-generated builder Python code in a restricted namespace.
    Returns (xml_string, actuator_schedule).
    Raises ValueError on execution or validation errors.
    """
    namespace: dict = {'MechanismBuilder': MechanismBuilder}
    try:
        exec(code, namespace)  # noqa: S102 — local dev tool, LLM-generated code only
    except Exception as e:
        raise ValueError(f"Builder code execution error: {type(e).__name__}: {e}")

    xml = namespace.get('xml')
    if xml is None:
        raise ValueError("Builder code did not assign 'xml = b.build()'")
    if not isinstance(xml, str):
        raise ValueError(f"'xml' must be a string, got {type(xml).__name__}")
    try:
        xml = repair_ground_clearance(xml)
    except Exception as e:
        raise ValueError(f"Ground clearance repair failed: {type(e).__name__}: {e}")

    actuator_schedule = namespace.get('actuator_schedule', [])
    if not isinstance(actuator_schedule, list):
        actuator_schedule = []

    return xml, actuator_schedule

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are a physics simulation expert. Write Python code using the MechanismBuilder API to create a MuJoCo simulation.

MechanismBuilder is already available in scope. Do NOT import anything.

API REFERENCE:
  b = MechanismBuilder()

  b.add_body(name, shape, size, mass=1.0, color=None, free=False, axis='z', geom_offset=None, euler=None)
      shape : 'box' | 'sphere' | 'cylinder' | 'capsule'
      size  : box=(hx,hy,hz)  sphere=r  cylinder/capsule=(radius, half_height)
      free  : True for free-floating bodies (vehicles, thrown objects)
      axis  : cylinder/capsule length direction — 'x','y','z' or [x,y,z]
              'z' = vertical (default); 'y' = flat disc (wheels)
      geom_offset : (dx,dy,dz) — shift geom centre from body origin
      euler : optional body orientation in degrees, e.g. (0,30,0), for non-equilibrium starts

  b.attach_to(parent, child, joint_type='hinge', axis='z', limits=None, damping=0.5)
      parent     : body name or 'world'
      joint_type : 'hinge' | 'slide' | 'ball' | 'fixed'
      axis       : rotation axis — 'x','y','z' or [x,y,z]
      limits     : (min_deg, max_deg) for hinge; (min_m, max_m) for slide

  b.position_relative(body, reference, offset, euler=None)
      Places body at: world_position_of_reference + offset
      reference : body name or 'world' (offset is then absolute)
      offset    : (dx, dy, dz) in metres
      euler     : optional convenience body orientation in degrees

  b.add_actuator(body_name, torque=100.0)
      Motor name becomes '{body_name}_motor' — use this in actuator_schedule

  xml = b.build()   ← always last; validates with MuJoCo

COORDINATE SYSTEM: X=forward  Y=left/right  Z=up  Ground at Z=0

KEY RULES:
- Treat each described mechanism as a connected body/joint graph unless the user explicitly asks
  for independent objects. Parts that visually belong together must be connected with attach_to().
- Use joint_type='fixed' for parts that should move together, hinge/slide/ball for constrained motion.
- Use free=True only for genuinely independent moving objects: falling, dropped, thrown, loose,
  bouncing, projectile bodies, or a vehicle chassis that carries attached wheels.
- Never call attach_to() on a body created with free=True. A body is either free OR joint-attached,
  never both. For connected mechanisms, remove free=True.
- Prefer plausible defaults instead of asking: sizes, masses, damping, axes, and initial angle/offset.
- If you specify orientation, pass euler to add_body(..., euler=(...)) or
  position_relative(..., euler=(...)).
- Body origin = where the joint sits. Use geom_offset to shift geom relative to joint.
- Pendulum arm (joint at top, hangs down): geom_offset=(0,0,-half_length)
- Robot arm link (joint at bottom, reaches up): geom_offset=(0,0,+half_length)
- Gravity-driven hinged mechanisms should start away from equilibrium using euler, e.g. (0,30,0).
- Wheels: axis='y' makes a flat disc that spins with attach_to(axis='y')
- Wheeled robot: chassis free=True, position chassis z = wheel_radius + chassis_half_height
- All bodies must be above Z=0

OUTPUT FORMAT — Python code only, no markdown, no explanation:
  b = MechanismBuilder()
  ... builder calls ...
  xml = b.build()
  actuator_schedule = [...]   # or [] if no actuators

Motor name format: '{body_name}_motor'
Example schedule entry: {"name": "link1_motor", "control": 3.0, "time": 0}

EXAMPLE 1 — Simple pendulum:
b = MechanismBuilder()
b.add_body('arm', 'cylinder', (0.02, 0.75), mass=0.5, color=(0.6, 0.4, 0.2), geom_offset=(0, 0, -0.75), euler=(0, 30, 0))
b.add_body('bob', 'sphere', 0.1, mass=2.0, color=(0.8, 0.1, 0.1))
b.attach_to('world', 'arm', 'hinge', axis='y', limits=(-180, 180), damping=0.5)
b.attach_to('arm', 'bob', 'fixed')
b.position_relative('arm', 'world', (0, 0, 2.5))
b.position_relative('bob', 'arm', (0, 0, -1.5))
xml = b.build()
actuator_schedule = []

EXAMPLE 2 — Motorised 3-joint robot arm:
b = MechanismBuilder()
b.add_body('base', 'cylinder', (0.06, 0.15), mass=3.0, color=(0.3, 0.3, 0.35))
b.add_body('link1', 'capsule', (0.03, 0.2), mass=0.8, color=(0.2, 0.5, 0.8), geom_offset=(0, 0, 0.2))
b.add_body('link2', 'capsule', (0.025, 0.175), mass=0.5, color=(0.2, 0.6, 0.9), geom_offset=(0, 0, 0.175))
b.add_body('link3', 'capsule', (0.02, 0.125), mass=0.3, color=(0.1, 0.7, 0.9), geom_offset=(0, 0, 0.125))
b.attach_to('world', 'base', 'fixed')
b.attach_to('base', 'link1', 'hinge', axis='y', limits=(-90, 90), damping=1.0)
b.attach_to('link1', 'link2', 'hinge', axis='y', limits=(-90, 90), damping=1.0)
b.attach_to('link2', 'link3', 'hinge', axis='y', limits=(-90, 90), damping=0.5)
b.position_relative('base', 'world', (0, 0, 0.15))
b.position_relative('link1', 'base', (0, 0, 0.3))
b.position_relative('link2', 'link1', (0, 0, 0.4))
b.position_relative('link3', 'link2', (0, 0, 0.35))
b.add_actuator('link1', torque=3.0)
b.add_actuator('link2', torque=2.0)
b.add_actuator('link3', torque=1.5)
xml = b.build()
actuator_schedule = [
    {"name": "link1_motor", "control": 3.0, "time": 0},
    {"name": "link1_motor", "control": 3.0, "time": 10},
    {"name": "link2_motor", "control": 2.0, "time": 0},
    {"name": "link2_motor", "control": 2.0, "time": 10},
    {"name": "link3_motor", "control": 1.5, "time": 0},
    {"name": "link3_motor", "control": 1.5, "time": 10},
]

EXAMPLE 3 — 4-wheeled robot (flat disc wheels with axis='y'):
b = MechanismBuilder()
b.add_body('chassis', 'box', (0.25, 0.15, 0.05), mass=5.0, color=(0.3, 0.35, 0.4), free=True)
b.add_body('fl', 'cylinder', (0.07, 0.03), mass=0.4, color=(0.1, 0.1, 0.1), axis='y')
b.add_body('fr', 'cylinder', (0.07, 0.03), mass=0.4, color=(0.1, 0.1, 0.1), axis='y')
b.add_body('rl', 'cylinder', (0.07, 0.03), mass=0.4, color=(0.1, 0.1, 0.1), axis='y')
b.add_body('rr', 'cylinder', (0.07, 0.03), mass=0.4, color=(0.1, 0.1, 0.1), axis='y')
b.attach_to('chassis', 'fl', 'hinge', axis='y', damping=0.05)
b.attach_to('chassis', 'fr', 'hinge', axis='y', damping=0.05)
b.attach_to('chassis', 'rl', 'hinge', axis='y', damping=0.05)
b.attach_to('chassis', 'rr', 'hinge', axis='y', damping=0.05)
b.position_relative('chassis', 'world', (0, 0, 0.15))
b.position_relative('fl', 'chassis', (-0.2, 0.18, -0.08))
b.position_relative('fr', 'chassis', (0.2, 0.18, -0.08))
b.position_relative('rl', 'chassis', (-0.2, -0.18, -0.08))
b.position_relative('rr', 'chassis', (0.2, -0.18, -0.08))
b.add_actuator('fl', torque=5.0)
b.add_actuator('fr', torque=5.0)
b.add_actuator('rl', torque=5.0)
b.add_actuator('rr', torque=5.0)
xml = b.build()
actuator_schedule = [
    {"name": "fl_motor", "control": 1.0, "time": 0}, {"name": "fl_motor", "control": 1.0, "time": 10},
    {"name": "fr_motor", "control": 1.0, "time": 0}, {"name": "fr_motor", "control": 1.0, "time": 10},
    {"name": "rl_motor", "control": 1.0, "time": 0}, {"name": "rl_motor", "control": 1.0, "time": 10},
    {"name": "rr_motor", "control": 1.0, "time": 0}, {"name": "rr_motor", "control": 1.0, "time": 10},
]

Now generate Python code for the user's description. Output ONLY code, no explanation:"""


# ---- Primitives-based /generate endpoint ---------------------------------- #

# Map from LLM primitive type names to functions in prim_module
_PRIM_MAP: dict[str, Any] = {
    name: getattr(prim_module, name)
    for name in [
        "rigid_body", "ground", "revolute", "prismatic", "screw", "cylindrical",
        "spherical", "planar", "fixed", "actuator", "spring", "damper", "gravity",
        "contact_pair", "box_geom", "cylinder_geom", "sphere_geom",
        "capsule_geom", "plane_geom", "mesh_geom",
    ]
}

_PRIM_SIGS: dict[str, inspect.Signature] = {
    name: inspect.signature(fn) for name, fn in _PRIM_MAP.items()
}


def _build_primitives(raw_prims: list) -> list:
    """Convert LLM-output primitive dicts to callable-result dicts for assembler."""
    result = []
    for raw in raw_prims:
        if not isinstance(raw, dict):
            continue
        ptype = raw.get("type", "")
        fn = _PRIM_MAP.get(ptype)
        if fn is None:
            continue
        valid = set(_PRIM_SIGS[ptype].parameters)
        kwargs = {k: v for k, v in raw.items() if k in valid and v is not None}
        prim = fn(**kwargs)
        result.append(prim)
    return result


def _extract_ui_params(raw_prims: list) -> dict:
    """Build slider parameters for the UI from the primitive list."""
    params: dict[str, dict] = {}

    def _add(key: str, value: float, unit: str = "", desc: str = ""):
        if isinstance(value, (int, float)) and value != 0:
            params[key] = {
                "value": value,
                "min": round(abs(value) * 0.1, 6),
                "max": round(abs(value) * 5, 4),
                "unit": unit,
                "description": desc or key.replace("_", " "),
            }

    for raw in raw_prims:
        ptype = raw.get("type", "")
        if ptype == "rigid_body":
            if raw.get("mass"):
                _add(f'{raw["name"]}_mass', raw["mass"], "kg", f'{raw["name"]} mass')
        elif ptype == "revolute":
            if raw.get("damping"):
                _add(f'{raw["name"]}_damping', raw["damping"], "Nm·s/rad", f'{raw["name"]} damping')
        elif ptype == "actuator":
            if raw.get("gear"):
                _add(f'{raw["name"]}_gear', raw["gear"], "", f'{raw["name"]} gear ratio')
        elif ptype == "gravity":
            _add("gravity", raw.get("magnitude", 9.81), "m/s²", "gravity magnitude")

    return params


_GEOM_TYPES = {"box_geom", "cylinder_geom", "sphere_geom", "capsule_geom", "plane_geom", "mesh_geom"}
_JOINT_TYPES = {"revolute", "prismatic", "spherical", "cylindrical", "planar", "screw"}


def _validate_semantics(raw_prims: list) -> list[str]:
    """Return a list of semantic issues in the primitive list."""
    issues: list[str] = []
    body_geom_count: dict[str, int] = {}
    body_parent: dict[str, str] = {}
    body_free: dict[str, bool] = {}
    joint_names: set[str] = set()
    actuator_joints: list[str] = []
    last_body: str | None = None

    for p in raw_prims:
        ptype = p.get("type", "")
        if ptype == "rigid_body":
            name = p.get("name", "?")
            body_geom_count[name] = 0
            body_parent[name] = p.get("parent", "world")
            body_free[name] = bool(p.get("free", False))
            last_body = name
        elif ptype in _GEOM_TYPES:
            target = p.get("body") or last_body
            if target and target in body_geom_count:
                body_geom_count[target] += 1
        elif ptype in _JOINT_TYPES:
            if p.get("name"):
                joint_names.add(p["name"])
        elif ptype == "actuator":
            if p.get("joint"):
                actuator_joints.append(p["joint"])

    for bname, count in body_geom_count.items():
        if count == 0:
            issues.append(
                f"Body '{bname}' has no geom — add a sphere_geom, box_geom, cylinder_geom, or capsule_geom "
                f"with body='{bname}'. Do NOT set free=true on this body."
            )

    for bname, is_free in body_free.items():
        if is_free and body_parent.get(bname, "world") != "world":
            issues.append(
                f"Body '{bname}' has free=true but parent='{body_parent[bname]}' — "
                f"free=true is only allowed on top-level bodies (parent='world'). Remove free=true."
            )

    for jname in actuator_joints:
        if jname not in joint_names:
            issues.append(f"Actuator targets joint '{jname}' which was not defined.")

    return issues


PREVIEW_PATH = "/home/basit/Documents/forge/backend/preview.png"


def _render_preview(xml: str, actuator_schedule: list) -> bool:
    """Render a preview image after stepping the simulation briefly. Saves to PREVIEW_PATH."""
    try:
        from PIL import Image
        from collections import defaultdict

        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        # Build actuator map
        actuator_ids: dict[str, int] = {}
        for i in range(model.nu):
            aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if aname:
                actuator_ids[aname] = i
        schedule_by: dict[str, list] = defaultdict(list)
        for entry in actuator_schedule:
            schedule_by[entry["name"]].append(entry)

        # Step 150 frames (~0.3 s) so objects have started moving
        dt = float(model.opt.timestep)
        for step in range(150):
            t = step * dt
            for aname, aix in actuator_ids.items():
                if aname in schedule_by:
                    data.ctrl[aix] = _interpolate_control(schedule_by[aname], t)
            mujoco.mj_step(model, data)

        # Compute scene centre and auto-scale camera distance to fit the scene
        if model.nbody > 1:
            positions = data.xpos[1:]  # skip worldbody at index 0
            cx, cy, cz = float(np.mean(positions[:, 0])), float(np.mean(positions[:, 1])), float(np.mean(positions[:, 2]))
            # Spread = max distance from centroid → drive camera distance
            spread = float(np.max(np.linalg.norm(positions - [cx, cy, cz], axis=1)))
            spread = max(spread, 0.3)   # minimum spread so tiny scenes stay visible
        else:
            cx, cy, cz, spread = 0.0, 0.0, 0.5, 1.0

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [cx, cy, max(cz, spread * 0.3)]
        cam.distance = spread * 3.5     # scale with scene size
        cam.azimuth = 135.0
        cam.elevation = -25.0

        renderer = mujoco.Renderer(model, height=480, width=640)
        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()
        renderer.close()

        Image.fromarray(pixels).save(PREVIEW_PATH)
        return True
    except Exception as e:
        print(f"[preview] render failed: {e}")
        return False


GEOM_TYPE_MAP = {0: "plane", 2: "sphere", 3: "capsule", 4: "ellipsoid", 5: "cylinder", 6: "box"}


def _extract_geom_info(model: mujoco.MjModel) -> list[dict]:
    geoms = []
    for i in range(model.ngeom):
        gtype = model.geom_type[i]
        size = model.geom_size[i].tolist()
        pos = model.geom_pos[i].tolist()
        quat = model.geom_quat[i].tolist()  # [w, x, y, z]
        rgba = model.geom_rgba[i].tolist()
        body_id = int(model.geom_bodyid[i])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
        geoms.append({
            "name": name,
            "body_id": body_id,
            "type": GEOM_TYPE_MAP.get(gtype, "unknown"),
            "size": size,
            "pos": pos,
            "quat": quat,
            "rgba": rgba,
        })
    return geoms


def _capture_frame(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    bodies = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
        pos = data.xpos[i].tolist()
        # MuJoCo xquat is [w, x, y, z]
        quat = data.xquat[i].tolist()
        bodies.append({"name": name, "pos": pos, "quat": quat})
    return {"time": float(data.time), "bodies": bodies}


def _collect_joint_names(body: ET.Element, names: set) -> None:
    for joint in body.findall("joint"):
        n = joint.get("name")
        if n:
            names.add(n)
    for child in body.findall("body"):
        _collect_joint_names(child, names)


def _fix_bodies(body: ET.Element) -> None:
    has_freejoint = body.find("freejoint") is not None
    has_any_joint = has_freejoint or body.find("joint") is not None
    geoms = body.findall("geom")
    geom_types = [g.get("type", "box") for g in geoms]

    # Plane bodies must be static — remove any joint
    if "plane" in geom_types and has_any_joint:
        for fj in body.findall("freejoint"):
            body.remove(fj)
        for j in body.findall("joint"):
            body.remove(j)
        has_freejoint = False
        has_any_joint = False

    if has_any_joint:
        # Body with joint but no geoms — add a small sphere so it has mass
        if not geoms:
            sphere = ET.SubElement(body, "geom")
            sphere.set("type", "sphere")
            sphere.set("size", "0.05")
            sphere.set("rgba", "0.8 0.3 0.3 1")
            sphere.set("density", "1000")
            geoms = [sphere]

        # Ensure geoms have density so MuJoCo can compute mass
        has_inertial = body.find("inertial") is not None
        if not has_inertial:
            for geom in geoms:
                if geom.get("density") is None and geom.get("mass") is None:
                    geom.set("density", "1000")

    for child in body.findall("body"):
        _fix_bodies(child)


def _fix_xml(xml: str) -> str:
    if not xml or not xml.strip():
        return xml

    # Strip XML-illegal control characters (everything except tab, LF, CR, and printable ASCII)
    xml = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', xml)
    # Normalize numeric attribute values: commas → spaces, Unicode minus/dash → ASCII minus
    def _clean_attr(m):
        v = m.group(0)
        v = v.replace(',', ' ').replace('−', '-').replace('–', '-').replace('—', '-')
        # Ensure space before a minus that follows a digit: "0-9.81" → "0 -9.81"
        v = re.sub(r'(\d)(-)', r'\1 \2', v)
        return v
    xml = re.sub(r'(?<==")[^"]*(?=")', _clean_attr, xml)

    # Add pos="0 0 0" to <inertial> tags missing it
    xml = re.sub(r'(<inertial\b)(?![^>]*\bpos\b)', r'\1 pos="0 0 0"', xml)

    # Inject balanceinertia compiler option to recover from bad diaginertia values
    if "<inertial" in xml and "balanceinertia" not in xml:
        if "<compiler" in xml:
            xml = re.sub(r'(<compiler\b)(?![^>]*balanceinertia)', r'\1 balanceinertia="true"', xml)
        else:
            xml = xml.replace("<mujoco>", '<mujoco><compiler balanceinertia="true"/>', 1)
            xml = xml.replace("<mujoco/>", '<mujoco><compiler balanceinertia="true"/>', 1)

    # Escape bare ampersands that aren't part of XML entities
    xml = re.sub(r'&(?!(?:amp|lt|gt|quot|apos);)', '&amp;', xml)

    def _lenient_parse(xml_str: str):
        """Try strict ET first, fall back to lxml recovery mode."""
        try:
            return ET.fromstring(xml_str)
        except ET.ParseError:
            pass
        # lxml recovery mode repairs truncated/malformed XML
        try:
            parser = lxml_etree.XMLParser(recover=True)
            lxml_root = lxml_etree.fromstring(xml_str.encode(), parser)
            if lxml_root is not None:
                repaired = lxml_etree.tostring(lxml_root, encoding="unicode")
                return ET.fromstring(repaired)
        except Exception:
            pass
        return None

    try:
        root = _lenient_parse(xml)
        if root is None:
            raise ET.ParseError("Could not parse XML even with recovery")

        worldbody = root.find("worldbody")

        # Step 1: Move <body> elements at mujoco root level into worldbody
        for child in list(root):
            if child.tag == "body":
                root.remove(child)
                if worldbody is None:
                    worldbody = ET.SubElement(root, "worldbody")
                    root.remove(worldbody)
                    root.insert(1, worldbody)
                worldbody.append(child)

        # Step 2: Move misplaced <actuator> from inside worldbody to mujoco root
        if worldbody is not None:
            misplaced = worldbody.find("actuator")
            if misplaced is not None:
                worldbody.remove(misplaced)
                existing = root.find("actuator")
                if existing is None:
                    root.append(misplaced)
                else:
                    for child in list(misplaced):
                        existing.append(child)

        # Step 3: Remove unknown worldbody children (LLM hallucinations)
        VALID_WORLDBODY_CHILDREN = {"body", "geom", "light", "site", "camera"}
        if worldbody is not None:
            for child in list(worldbody):
                if child.tag not in VALID_WORLDBODY_CHILDREN:
                    worldbody.remove(child)

        # Step 3b: Remove bodies whose only geoms are planes — planes belong at worldbody
        # level as direct children, not wrapped in a body. Promote any plane geoms up.
        if worldbody is not None:
            for body in list(worldbody.findall("body")):
                geoms = body.findall("geom")
                all_planes = geoms and all(g.get("type", "box") == "plane" for g in geoms)
                has_children = bool(body.findall("body"))
                if all_planes and not has_children:
                    for g in geoms:
                        worldbody.append(g)
                    worldbody.remove(body)

        # Step 4: Fix invalid joint types, range attrs, axis, geom sizes
        VALID_JOINT_TYPES = {"free", "ball", "slide", "hinge"}
        if worldbody is not None:
            for body in worldbody.iter("body"):
                for joint in list(body.findall("joint")):
                    jtype = joint.get("type", "hinge")
                    if jtype not in VALID_JOINT_TYPES:
                        body.remove(joint)
                        continue
                    # Fix range attribute: strip non-numeric chars, keep two numbers
                    rng = joint.get("range", "")
                    if rng:
                        rng = re.sub(r'[°°degrada-z]', ' ', rng, flags=re.IGNORECASE)
                        rng = re.sub(r'\s+to\s+', ' ', rng, flags=re.IGNORECASE)
                        nums = re.findall(r'-?\d+\.?\d*', rng)
                        if len(nums) >= 2:
                            joint.set("range", f"{nums[0]} {nums[1]}")
                        elif len(nums) == 1:
                            joint.set("range", f"-{nums[0]} {nums[0]}")
                        else:
                            joint.attrib.pop("range", None)
                    # Normalize joint axis — zero-magnitude axis causes MuJoCo error
                    axis_str = joint.get("axis", "")
                    if axis_str:
                        try:
                            vals = [float(x) for x in axis_str.split()]
                            if len(vals) == 3:
                                mag = (vals[0]**2 + vals[1]**2 + vals[2]**2) ** 0.5
                                if mag < 1e-6:
                                    joint.set("axis", "0 0 1")
                                else:
                                    joint.set("axis", f"{vals[0]/mag:.6g} {vals[1]/mag:.6g} {vals[2]/mag:.6g}")
                        except ValueError:
                            joint.set("axis", "0 0 1")
                    else:
                        joint.set("axis", "0 0 1")
                for geom in body.findall("geom"):
                    gtype = geom.get("type", "box")
                    size_str = geom.get("size", "")
                    parts = size_str.split()
                    if gtype == "sphere" and len(parts) > 1:
                        geom.set("size", parts[0])
                    elif gtype in ("cylinder", "capsule") and len(parts) > 2:
                        geom.set("size", f"{parts[0]} {parts[1]}")
                    elif gtype == "box" and len(parts) < 3:
                        v = parts[0] if parts else "0.1"
                        geom.set("size", f"{v} {v} {v}")

        # Fix invalid <inertial> attributes — density belongs on geom, not inertial
        VALID_INERTIAL_ATTRS = {"pos", "quat", "axisangle", "xyaxes", "zaxis", "euler",
                                "mass", "diaginertia", "fullinertia"}
        if worldbody is not None:
            for inertial in worldbody.iter("inertial"):
                # Remove unknown attributes
                for attr in list(inertial.attrib):
                    if attr not in VALID_INERTIAL_ATTRS:
                        del inertial.attrib[attr]
                # If mass is missing entirely, remove the inertial element
                # (geom density will provide mass automatically)
                if inertial.get("mass") is None:
                    parent = None
                    for body in worldbody.iter("body"):
                        if inertial in list(body):
                            parent = body
                            break
                    if parent is not None:
                        parent.remove(inertial)

        # Step 5: Fix body mass/geom issues recursively
        if worldbody is not None:
            for body in worldbody.findall("body"):
                _fix_bodies(body)

        # Step 6: Final actuator cleanup — re-collect joint names AFTER all joint fixes
        joint_names: set[str] = set()
        freejoint_names: set[str] = set()
        if worldbody is not None:
            for body in worldbody.iter("body"):
                _collect_joint_names(body, joint_names)
            for fj in worldbody.iter("freejoint"):
                n = fj.get("name")
                if n:
                    freejoint_names.add(n)
                    joint_names.discard(n)

        actuator = root.find("actuator")
        if actuator is not None:
            for motor in list(actuator.findall("motor")):
                j = motor.get("joint", "")
                if j not in joint_names or j in freejoint_names or "freejoint" in j:
                    actuator.remove(motor)
            if len(actuator) == 0:
                root.remove(actuator)

        xml = ET.tostring(root, encoding="unicode")
    except ET.ParseError:
        pass

    return xml


def _validate_mjcf(root: ET.Element, description: str) -> None:
    if root.tag != "mujoco":
        raise ValueError("XML root must be <mujoco>")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Missing <worldbody> element")

    joint_names = {j.get("name") for j in root.iter("joint") if j.get("name")}
    if root.find("actuator") is not None:
        for motor in root.find("actuator").findall("motor"):
            target = motor.get("joint")
            if not target:
                raise ValueError("Motor actuator must specify a joint")
            if target not in joint_names:
                raise ValueError(f"Motor actuator targets unknown joint '{target}'")

    if not list(worldbody.iter("body")):
        raise ValueError("Generated model has no bodies — worldbody is empty. Add at least one <body> with a <geom>.")

    for body in worldbody.iter("body"):
        if body.find("freejoint") is not None and body.find("geom") is None:
            raise ValueError("A body with <freejoint/> must contain at least one <geom>")

    for joint in root.iter("joint"):
        if joint.get("type") == "hinge":
            axis = joint.get("axis", "").strip()
            if not axis:
                raise ValueError("Hinge joint must include an axis")

    pass


def _parse_llm_json(text: str) -> dict:
    text = text.strip()

    # Stage 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Stage 2: ```json block
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Stage 3: fix unescaped quotes inside XML attribute values
    # LLM sometimes writes gear="1"/> without escaping the inner quotes
    def _escape_xml_in_json_string(raw: str) -> str:
        # Find the "xml": "..." value and re-escape its contents
        def replacer(mo):
            inner = mo.group(1)
            # Escape any bare (non-backslash-preceded) double quotes inside
            inner = re.sub(r'(?<!\\)"', r'\\"', inner)
            return f'"xml": "{inner}"'
        return re.sub(r'"xml"\s*:\s*"(.*?)"(?=\s*,\s*"parameters")', replacer, raw, flags=re.DOTALL)

    try:
        return json.loads(_escape_xml_in_json_string(text))
    except (json.JSONDecodeError, Exception):
        pass

    # Stage 4: largest {} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # Stage 5: extract XML directly — recover gracefully with no parameters
    xml_m = re.search(r"(<mujoco[\s>].*?</mujoco>)", text, re.DOTALL)
    if xml_m:
        return {"xml": xml_m.group(1), "parameters": {}, "actuator_schedule": []}

    raise ValueError("Could not parse JSON from LLM response")


def _interpolate_control(entries: list[dict], t: float) -> float:
    """Linearly interpolate control value from time-keyed schedule entries."""
    if not entries:
        return 0.0
    sched = sorted(entries, key=lambda e: e["time"])
    if t <= sched[0]["time"]:
        return sched[0]["control"]
    if t >= sched[-1]["time"]:
        return sched[-1]["control"]
    for i in range(len(sched) - 1):
        t0, c0 = sched[i]["time"], sched[i]["control"]
        t1, c1 = sched[i + 1]["time"], sched[i + 1]["control"]
        if t0 <= t <= t1:
            alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return c0 + alpha * (c1 - c0)
    return sched[-1]["control"]


def _run_simulation(xml: str, duration: float, actuator_schedule: list[dict]) -> tuple:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    geoms = _extract_geom_info(model)
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
        for i in range(model.nbody)
    ]

    # Build actuator name→id map
    actuator_ids: dict[str, int] = {}
    for i in range(model.nu):
        aname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if aname:
            actuator_ids[aname] = i

    # Group schedule entries by actuator name for fast lookup
    from collections import defaultdict
    schedule_by_actuator: dict[str, list[dict]] = defaultdict(list)
    for entry in actuator_schedule:
        schedule_by_actuator[entry["name"]].append(entry)

    frames = []
    dt = float(model.opt.timestep)
    steps = int(duration / dt)
    capture_every = max(1, int(1.0 / (60.0 * dt)))  # ~60 fps capture

    for step in range(steps):
        t = step * dt
        # Apply interpolated control for each actuator at the current time
        for aname, aix in actuator_ids.items():
            if aname in schedule_by_actuator:
                data.ctrl[aix] = _interpolate_control(schedule_by_actuator[aname], t)

        mujoco.mj_step(model, data)

        if step % capture_every == 0:
            frames.append(_capture_frame(model, data))

    return geoms, body_names, frames


def _max_frame_motion(frames: list[dict]) -> float:
    """Return max body translation/rotation delta between first and last frame."""
    if len(frames) < 2:
        return 0.0

    first_by_name = {b["name"]: b for b in frames[0].get("bodies", [])}
    max_delta = 0.0
    for body in frames[-1].get("bodies", []):
        name = body.get("name")
        if name == "world" or name not in first_by_name:
            continue
        start = first_by_name[name]
        pos_delta = float(np.linalg.norm(np.array(body["pos"]) - np.array(start["pos"])))
        quat_delta = float(np.linalg.norm(np.array(body["quat"]) - np.array(start["quat"])))
        max_delta = max(max_delta, pos_delta, quat_delta)
    return max_delta


def _ensure_simulation_moves(xml: str, actuator_schedule: list[dict], duration: float = 1.0) -> None:
    """
    Catch MJCF that is valid but inert, so the LLM correction loop can fix it.
    Most failures here are top-level bodies without free joints, equilibrium poses,
    or actuators that never receive control.
    """
    _, _, frames = _run_simulation(xml, duration, actuator_schedule)
    if _max_frame_motion(frames) < 1e-4:
        raise ValueError(
            "The simulation is valid but static: no non-world body moves during playback. "
            "Add an appropriate free=True body, actuator_schedule, or non-equilibrium initial pose."
        )


def _best_effort_generation(description: str) -> dict:
    """Return a guaranteed-valid, moving approximation when LLM generation fails."""
    desc = description.lower()
    b = MechanismBuilder()
    actuator_schedule: list[dict] = []

    if any(word in desc for word in ("robot", "vehicle", "rover", "car", "wheel", "wheels")):
        b.add_body("chassis", "box", (0.25, 0.15, 0.06), mass=4.0, color=(0.25, 0.35, 0.45), free=True)
        for name, x, y in [
            ("front_left_wheel", 0.18, 0.18),
            ("front_right_wheel", 0.18, -0.18),
            ("rear_left_wheel", -0.18, 0.18),
            ("rear_right_wheel", -0.18, -0.18),
        ]:
            b.add_body(name, "cylinder", (0.07, 0.025), mass=0.4, color=(0.05, 0.05, 0.05), axis="y")
            b.attach_to("chassis", name, "hinge", axis="y", damping=0.05)
            b.position_relative(name, "chassis", (x, y, -0.08))
            b.add_actuator(name, torque=4.0)
            actuator_schedule.append({"name": f"{name}_motor", "control": 0.8, "time": 0})
            actuator_schedule.append({"name": f"{name}_motor", "control": 0.8, "time": 10})
        b.position_relative("chassis", "world", (0, 0, 0.16))
    elif any(word in desc for word in ("door", "gate", "flap")):
        b.add_body("frame", "box", (0.04, 0.06, 0.6), mass=3.0, color=(0.35, 0.25, 0.18))
        b.add_body("panel", "box", (0.45, 0.025, 0.55), mass=2.0, color=(0.7, 0.45, 0.25), geom_offset=(0.45, 0, 0), euler=(0, 0, 8))
        b.attach_to("world", "frame", "fixed")
        b.attach_to("frame", "panel", "hinge", axis="z", limits=(-120, 120), damping=0.25)
        b.position_relative("frame", "world", (0, 0, 0.65))
        b.position_relative("panel", "frame", (0.04, 0, 0))
        b.add_actuator("panel", torque=2.0)
        actuator_schedule = [{"name": "panel_motor", "control": 0.6, "time": 0}, {"name": "panel_motor", "control": 0.2, "time": 10}]
    elif any(word in desc for word in ("arm", "link", "linkage", "pendulum", "swing")):
        b.add_body("arm", "cylinder", (0.02, 0.75), mass=0.6, color=(0.55, 0.38, 0.22), geom_offset=(0, 0, -0.75), euler=(0, 30, 0))
        b.add_body("bob", "sphere", 0.12, mass=1.5, color=(0.8, 0.1, 0.1))
        b.attach_to("world", "arm", "hinge", axis="y", limits=(-180, 180), damping=0.15)
        b.attach_to("arm", "bob", "fixed")
        b.position_relative("arm", "world", (0, 0, 2.5))
        b.position_relative("bob", "arm", (0, 0, -1.5))
    elif any(word in desc for word in ("ball", "sphere")):
        b.add_body("ball", "sphere", 0.12, mass=1.0, color=(0.8, 0.1, 0.1), free=True)
        if any(word in desc for word in ("trampoline", "platform", "surface")):
            b.add_body("surface", "box", (0.65, 0.45, 0.035), mass=10.0, color=(0.15, 0.45, 0.65))
            b.attach_to("world", "surface", "fixed")
            b.position_relative("surface", "world", (0, 0, 0.2))
        b.position_relative("ball", "world", (0, 0, 2.0))
    else:
        b.add_body("body", "box", (0.15, 0.15, 0.15), mass=1.0, color=(0.3, 0.55, 0.8), free=True)
        b.position_relative("body", "world", (0, 0, 2.0))

    xml = b.build()
    _ensure_simulation_moves(xml, actuator_schedule)
    return {"xml": xml, "actuator_schedule": actuator_schedule, "fallback": True}


_INDEPENDENT_MOTION_WORDS = {
    "fall", "falling", "drop", "dropped", "dropping", "throw", "thrown", "projectile",
    "bounce", "bouncing", "loose", "grain", "grains", "sand", "material", "particle",
    "particles", "ball falling", "cube falling", "slide", "sliding", "hit", "hitting",
    "strike", "striking", "impact",
}

_VEHICLE_WORDS = {"vehicle", "robot", "car", "cart", "rover", "wheeled", "wheel", "wheels"}


def _allows_free_body(description: str) -> bool:
    desc = description.lower()
    return any(word in desc for word in _INDEPENDENT_MOTION_WORDS | _VEHICLE_WORDS)


def _body_has_descendant_body(body: ET.Element) -> bool:
    return body.find("body") is not None


def _ensure_prompt_semantics(xml: str, description: str) -> None:
    """Reject valid-but-wrong topology so the correction loop can repair it."""
    desc = description.lower()

    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    if worldbody is None:
        return

    free_bodies = [
        body
        for body in worldbody.iter("body")
        if body.find("freejoint") is not None
    ]
    if free_bodies and not _allows_free_body(desc):
        names = ", ".join(body.get("name", "unnamed") for body in free_bodies)
        raise ValueError(
            "The generated mechanism uses free=True/<freejoint/> even though the prompt describes "
            f"one connected mechanism ({names}). Connect related parts with fixed/hinge/slide/ball "
            "joints. Use free=True only for explicitly independent falling, loose, bouncing, "
            "projectile, or vehicle-chassis bodies."
        )

    if len(free_bodies) > 1 and any(word in desc for word in _VEHICLE_WORDS):
        names = ", ".join(body.get("name", "unnamed") for body in free_bodies)
        raise ValueError(
            "Vehicle-like mechanisms should usually have one free chassis carrying attached wheels "
            f"and links, not multiple independent free bodies ({names})."
        )

    for body in free_bodies:
        if any(word in desc for word in _VEHICLE_WORDS) and not _body_has_descendant_body(body):
            raise ValueError(
                f"Free vehicle body '{body.get('name', 'unnamed')}' has no attached child bodies. "
                "Use one free chassis and attach wheels/links to it with joints."
            )


def _base_assumptions(description: str) -> list[dict]:
    desc = description.lower()
    assumptions = [
        {
            "key": "topology",
            "label": "Topology",
            "value": "Treat visually related parts as one connected body/joint graph.",
        },
        {
            "key": "motion_source",
            "label": "Motion source",
            "value": "Use gravity, actuators, or joint constraints rather than detached parts.",
        },
    ]

    if any(word in desc for word in _VEHICLE_WORDS):
        assumptions.extend([
            {"key": "vehicle_base", "label": "Vehicle base", "value": "Use one free chassis carrying attached wheels."},
            {"key": "wheel_drive", "label": "Wheel drive", "value": "Attach wheels with hinge joints and add motor controls."},
        ])
    elif any(word in desc for word in _INDEPENDENT_MOTION_WORDS):
        assumptions.append({
            "key": "free_bodies",
            "label": "Independent bodies",
            "value": "Allow explicitly falling, bouncing, loose, or projectile objects to use free=True.",
        })
    else:
        assumptions.append({
            "key": "initial_pose",
            "label": "Initial pose",
            "value": "Start gravity-driven joints slightly away from equilibrium when needed.",
        })

    return assumptions


def _clarify_questions(description: str) -> list[dict]:
    desc = description.lower()
    questions: list[dict] = []
    if "robot" in desc and not any(word in desc for word in ("wheel", "leg", "arm", "gripper")):
        questions.append({
            "key": "robot_form",
            "label": "Robot form",
            "prompt": "Should this robot use wheels, legs, an arm, or another main structure?",
            "default": "wheeled chassis",
        })
    if "joint" in desc and not any(word in desc for word in ("hinge", "slide", "ball", "revolute", "prismatic")):
        questions.append({
            "key": "joint_type",
            "label": "Joint type",
            "prompt": "What joint type should ambiguous joints use?",
            "default": "hinge",
        })
    if any(word in desc for word in ("motor", "actuator", "powered")) and "speed" not in desc and "torque" not in desc:
        questions.append({
            "key": "actuation",
            "label": "Actuation",
            "prompt": "How strongly should powered joints move?",
            "default": "moderate torque",
        })
    return questions


def _compose_clarified_description(description: str, assumptions: list[dict], answers: dict | None = None) -> str:
    lines = [description.strip(), "", "Generation assumptions:"]
    for item in assumptions:
        value = item.get("value", "")
        if answers and item.get("key") in answers and str(answers[item["key"]]).strip():
            value = str(answers[item["key"]]).strip()
        lines.append(f"- {item.get('label', item.get('key', 'Assumption'))}: {value}")
    if answers:
        extra_answers = {
            key: value for key, value in answers.items()
            if key not in {item.get("key") for item in assumptions} and str(value).strip()
        }
        if extra_answers:
            lines.append("")
            lines.append("Clarification answers:")
            for key, value in extra_answers.items():
                lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def build_clarification(description: str, answers: dict | None = None) -> dict:
    assumptions = _base_assumptions(description)
    questions = _clarify_questions(description)
    return {
        "assumptions": assumptions,
        "questions": questions,
        "clarified_description": _compose_clarified_description(description, assumptions, answers),
    }


def apply_smart_defaults(description: str) -> str:
    if "Generation assumptions:" in description:
        return description
    clarification = build_clarification(description)
    return clarification["clarified_description"]


class ClarifyRequest(BaseModel):
    description: str
    answers: dict[str, Any] | None = None


@app.post("/clarify")
async def clarify(req: ClarifyRequest):
    return build_clarification(req.description, req.answers)


class GenerateRequest(BaseModel):
    description: str


class SimulateRequest(BaseModel):
    xml: str
    duration: float = 10.0
    actuator_schedule: list[dict] = []


async def _run_generation_loop(description: str, initial_raw: Optional[str] = None) -> dict:
    """
    Core generation loop. Tries up to 3 times with error feedback.
    LLM generates Python code using MechanismBuilder; code is exec'd to produce MJCF.
    If initial_raw is provided (vision path) it is used as the first attempt.
    Returns {"xml": ..., "actuator_schedule": ...} or raises RuntimeError.
    """
    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": description},
    ]
    messages = list(base_messages)
    last_error = None

    for attempt in range(3):
        if attempt == 0 and initial_raw is not None:
            raw = initial_raw
        else:
            try:
                response = await asyncio.to_thread(
                    lambda: ollama.chat(model="llama3.1:8b", messages=messages)
                )
                raw = response["message"]["content"]
            except Exception as e:
                raise RuntimeError(f"LLM error: {e}")

        logger.info(f"[generate] Attempt {attempt + 1} — executing builder code")
        code = _extract_python_code(raw)
        logger.debug(f"[generate] Code:\n{code}")

        try:
            xml, actuator_schedule = _exec_builder_code(code)
            _ensure_prompt_semantics(xml, description)
            _ensure_simulation_moves(xml, actuator_schedule)
            logger.info("[generate] Builder code produced valid MJCF")
            return {"xml": xml, "actuator_schedule": actuator_schedule}
        except Exception as e:
            last_error = str(e)
            logger.warning(f"[generate] Attempt {attempt + 1} failed: {last_error}")
            messages = list(base_messages) + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"Your code raised an error: {last_error}\n"
                        "Fix the Python code and output ONLY the corrected code."
                    ),
                },
            ]

    logger.warning(f"[generate] Falling back after 3 attempts. Last error: {last_error}")
    return _best_effort_generation(description)


async def _scene_spec_from_llm(description: str) -> dict | None:
    """Ask the local model for structured intent only; never code or MJCF."""
    messages = [
        {"role": "system", "content": SCENE_SPEC_PROMPT},
        {"role": "user", "content": description},
    ]
    try:
        response = await asyncio.to_thread(
            lambda: ollama.chat(model="llama3.1:8b", messages=messages)
        )
        return extract_json_object(response.get("message", {}).get("content", ""))
    except Exception as e:
        logger.info(f"[scene] SceneSpec LLM unavailable; using heuristic semantics: {e}")
        return None


async def _run_semantic_generation(description: str) -> dict:
    """
    Primary generation path: description -> SceneSpec -> deterministic builder.

    If the LLM returns a weak spec or is unavailable, we repair from the user's
    description and still return a working scene instead of surfacing rejection
    errors to the UI.
    """
    raw_spec = await _scene_spec_from_llm(description)
    spec = normalize_scene_spec(raw_spec, description)
    issues = validate_scene_spec(spec)
    fallback = raw_spec is None or bool(issues)

    if issues:
        logger.info(f"[scene] Repaired weak SceneSpec with heuristic defaults: {issues}")
        spec = normalize_scene_spec(heuristic_scene_spec(description), description)

    try:
        xml, actuator_schedule = compile_scene_spec(spec)
        _ensure_prompt_semantics(xml, description)
        _ensure_simulation_moves(xml, actuator_schedule)
    except Exception as e:
        logger.warning(f"[scene] Semantic compile needed fallback repair: {e}")
        spec = normalize_scene_spec(heuristic_scene_spec(description), description)
        xml, actuator_schedule = compile_scene_spec(spec)
        try:
            _ensure_simulation_moves(xml, actuator_schedule)
        except Exception as motion_error:
            logger.warning(f"[scene] Heuristic semantic scene was static; using legacy moving fallback: {motion_error}")
            result = _best_effort_generation(description)
            result["scene_spec"] = spec
            result["fallback"] = True
            return result
        fallback = True

    return {
        "xml": xml,
        "actuator_schedule": actuator_schedule,
        "scene_spec": spec,
        "fallback": fallback,
    }


@app.post("/generate")
async def generate(req: GenerateRequest):
    description = req.description.strip()
    if not description:
        description = "a falling object"

    result = await _run_semantic_generation(description)

    xml = result["xml"]
    actuator_schedule = result["actuator_schedule"]

    # Render preview (non-blocking, failure is silent)
    await asyncio.to_thread(_render_preview, xml, actuator_schedule)

    return {
        "xml": xml,
        "parameters": {},
        "actuator_schedule": actuator_schedule,
        "fallback": bool(result.get("fallback", False)),
        "scene_spec": result.get("scene_spec"),
    }


@app.websocket("/ws/simulate")
async def ws_simulate(websocket: WebSocket):
    await websocket.accept()
    try:
        raw = await websocket.receive_text()
        req = json.loads(raw)
        xml = req["xml"]
        duration = float(req.get("duration", 10.0))
        actuator_schedule = req.get("actuator_schedule", [])

        geoms, body_names, frames = await asyncio.to_thread(
            _run_simulation, xml, duration, actuator_schedule
        )

        await websocket.send_json({"type": "init", "geoms": geoms, "body_names": body_names})

        for frame in frames:
            await websocket.send_json({"type": "frame", **frame})
            await asyncio.sleep(1.0 / 60.0)

        await websocket.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass

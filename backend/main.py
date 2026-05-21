import asyncio
import json
import re
import traceback
import xml.etree.ElementTree as ET
from typing import Any

from lxml import etree as lxml_etree

import mujoco
import numpy as np
import ollama
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are a MuJoCo MJCF XML expert. Given a description of a mechanical system, you generate valid MJCF XML and return a JSON object.

MJCF structure:
<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <!-- lights -->
    <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
    <!-- ground plane -->
    <geom type="plane" size="5 5 0.1" rgba="0.5 0.5 0.5 1"/>
    <!-- bodies -->
    <body name="..." pos="x y z">
      <freejoint/>   <!-- for free-falling bodies -->
      <geom type="box|sphere|cylinder|capsule" size="..." rgba="r g b a"/>
      <inertial mass="..." pos="0 0 0" diaginertia="..."/>
    </body>
  </worldbody>
  <actuator>
    <motor name="..." joint="..." gear="1"/>
  </actuator>
</mujoco>

CRITICAL size rules:
- box: size="hx hy hz" (half-extents, 3 values)
- sphere: size="radius" (1 value)
- cylinder: size="radius half_length" (2 values)
- capsule: size="radius half_length" (2 values)
- plane: size="x y z" (3 values, z=thickness usually 0.1)

Mass rules:
- Use density attribute on geom instead of separate inertial element when possible: <geom ... density="1000"/>
- For complex bodies, add <inertial mass="X" pos="0 0 0" diaginertia="I I I"/>

Joint rules:
- Free-falling/movable bodies: add <freejoint/> inside body
- Hinged: <joint type="hinge" name="..." axis="0 0 1" range="-90 90"/>
- Static bodies: direct worldbody children with NO joint

Few-shot examples:

Example 1 - Falling box:
{"xml": "<mujoco><option gravity=\\"0 0 -9.81\\"/><worldbody><light pos=\\"0 0 3\\" dir=\\"0 0 -1\\" diffuse=\\"1 1 1\\"/><geom type=\\"plane\\" size=\\"5 5 0.1\\" rgba=\\"0.5 0.5 0.5 1\\"/><body name=\\"box\\" pos=\\"0 0 1\\"><freejoint/><geom type=\\"box\\" size=\\"0.1 0.1 0.1\\" rgba=\\"0.8 0.2 0.2 1\\" density=\\"1000\\"/></body></worldbody></mujoco>", "parameters": {"box_size": {"value": 0.1, "min": 0.05, "max": 0.5, "unit": "m", "description": "Half-extent of the box"}}, "actuator_schedule": []}

Example 2 - Hinged door:
{"xml": "<mujoco><option gravity=\\"0 0 -9.81\\"/><worldbody><light pos=\\"0 0 3\\" dir=\\"0 0 -1\\" diffuse=\\"1 1 1\\"/><geom type=\\"plane\\" size=\\"5 5 0.1\\" rgba=\\"0.5 0.5 0.5 1\\"/><body name=\\"frame\\" pos=\\"0 0 0.5\\"><geom type=\\"box\\" size=\\"0.05 0.3 0.5\\" rgba=\\"0.6 0.4 0.2 1\\" density=\\"800\\"/><body name=\\"door\\" pos=\\"0 0 0\\"><joint type=\\"hinge\\" name=\\"door_hinge\\" axis=\\"0 0 1\\" range=\\"-90 90\\"/><geom type=\\"box\\" size=\\"0.4 0.02 0.45\\" rgba=\\"0.8 0.6 0.4 1\\" density=\\"600\\"/></body></body></worldbody><actuator><motor name=\\"door_motor\\" joint=\\"door_hinge\\" gear=\\"1\\"/></actuator></mujoco>", "parameters": {"door_width": {"value": 0.8, "min": 0.4, "max": 1.5, "unit": "m", "description": "Width of the door"}, "door_height": {"value": 0.9, "min": 0.6, "max": 2.0, "unit": "m", "description": "Height of the door"}}, "actuator_schedule": [{"name": "door_motor", "control": 0.5, "time": 0.0}, {"name": "door_motor", "control": 0.5, "time": 2.0}, {"name": "door_motor", "control": 0.0, "time": 2.1}]}

Example 3 - Hopper with trapdoor:
{"xml": "<mujoco><option gravity=\\"0 0 -9.81\\"/><worldbody><light pos=\\"0 0 5\\" dir=\\"0 0 -1\\" diffuse=\\"1 1 1\\"/><geom type=\\"plane\\" size=\\"5 5 0.1\\" rgba=\\"0.5 0.5 0.5 1\\"/><body name=\\"hopper_wall_front\\" pos=\\"0 0.065 0.1\\"><geom type=\\"box\\" size=\\"0.065 0.003 0.1\\" rgba=\\"0.7 0.7 0.8 0.8\\" density=\\"800\\"/></body><body name=\\"trapdoor\\" pos=\\"0 0 0.01\\"><joint type=\\"hinge\\" name=\\"trapdoor_hinge\\" axis=\\"1 0 0\\" range=\\"0 90\\"/><geom type=\\"box\\" size=\\"0.015 0.065 0.002\\" rgba=\\"0.6 0.3 0.1 1\\" density=\\"1200\\"/></body><body name=\\"material\\" pos=\\"0 0 0.3\\"><freejoint/><geom type=\\"sphere\\" size=\\"0.02\\" rgba=\\"0.9 0.7 0.1 1\\" density=\\"500\\"/></body></worldbody><actuator><motor name=\\"trapdoor_motor\\" joint=\\"trapdoor_hinge\\" gear=\\"1\\"/></actuator></mujoco>", "parameters": {"hopper_size": {"value": 0.13, "min": 0.05, "max": 0.3, "unit": "m", "description": "Hopper width/length"}, "opening_size": {"value": 0.03, "min": 0.01, "max": 0.1, "unit": "m", "description": "Trapdoor opening size"}}, "actuator_schedule": [{"name": "trapdoor_motor", "control": 0.0, "time": 0.0}, {"name": "trapdoor_motor", "control": 2.0, "time": 1.0}, {"name": "trapdoor_motor", "control": 0.0, "time": 3.0}]}

Return ONLY a JSON object with keys: xml, parameters, actuator_schedule. No markdown, no explanation."""


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

    frames = []
    dt = float(model.opt.timestep)
    steps = int(duration / dt)
    capture_every = max(1, int(1.0 / (60.0 * dt)))  # ~60 fps capture

    for step in range(steps):
        t = step * dt
        # Apply actuator controls by interpolating schedule
        for entry in actuator_schedule:
            aname = entry["name"]
            if aname in actuator_ids:
                data.ctrl[actuator_ids[aname]] = entry["control"]

        mujoco.mj_step(model, data)

        if step % capture_every == 0:
            frames.append(_capture_frame(model, data))

    return geoms, body_names, frames


class GenerateRequest(BaseModel):
    description: str


class SimulateRequest(BaseModel):
    xml: str
    duration: float = 10.0
    actuator_schedule: list[dict] = []


@app.post("/generate")
async def generate(req: GenerateRequest):
    last_error = None
    last_text = ""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": req.description},
    ]
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            response = ollama.chat(model="llama3.2", messages=messages)
            last_text = response["message"]["content"]
            parsed = _parse_llm_json(last_text)
            xml = _fix_xml(parsed.get("xml", ""))
            if not xml or not xml.strip():
                raise ValueError("LLM returned empty XML")
            mujoco.MjModel.from_xml_string(xml)
            return {
                "xml": xml,
                "parameters": parsed.get("parameters", {}),
                "actuator_schedule": parsed.get("actuator_schedule", []),
            }
        except Exception as e:
            last_error = str(e)
            if attempt < max_attempts - 1:
                messages.append({"role": "assistant", "content": last_text})
                messages.append({
                    "role": "user",
                    "content": (
                        f"That response caused this MuJoCo validation error: {last_error}\n\n"
                        "Common causes:\n"
                        "- <actuator> must be a direct child of <mujoco>, not inside <worldbody>\n"
                        "- motors can only target named <joint> elements, not <freejoint/>\n"
                        "- cylinder/capsule size needs exactly 2 values: radius half_length\n"
                        "- box size needs exactly 3 values: hx hy hz\n"
                        "- every body with a <freejoint/> needs at least one <geom> with density\n\n"
                        "Return corrected JSON only, no explanation."
                    )
                })
                await asyncio.sleep(0.1)
    return {"error": f"Failed after {max_attempts} attempts: {last_error}"}


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

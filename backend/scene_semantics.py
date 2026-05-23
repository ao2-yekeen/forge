"""
Semantic scene planning and deterministic MuJoCo compilation.

The generator should reason about what the user described before any MJCF is
created.  This module keeps that intermediate representation small and plain:
entities have roles, parts, relationships, and motion intent; the compiler then
turns those semantics into MechanismBuilder calls using generic affordances.
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from builder import MechanismBuilder


ROLES = {"static", "free", "jointed", "driven", "passive"}
RELATION_ALIASES = {
    "falls onto": "falling_onto",
    "fall onto": "falling_onto",
    "falling on": "falling_onto",
    "falling onto": "falling_onto",
    "drops onto": "falling_onto",
    "dropped onto": "falling_onto",
    "sits on": "on",
    "rests on": "on",
    "on top of": "on",
    "over": "above",
    "attached to": "attached_to",
    "collides with": "collides_with",
    "hits": "collides_with",
    "hitting": "collides_with",
}
RELATION_TYPES = {"above", "on", "inside", "attached_to", "falling_onto", "collides_with", "supports"}

SCENE_SPEC_PROMPT = """You convert user descriptions into a SceneSpec JSON object.

Return JSON only. Do not return code, markdown, or MJCF.

Schema:
{
  "description": string,
  "entities": [
    {
      "id": "short_snake_case",
      "name": string,
      "kind": "generic object class",
      "role": "static|free|jointed|driven|passive",
      "motion_intent": string,
      "visual_intent": string,
      "parts": [
        {"id": string, "kind": string, "role": string, "shape": string}
      ]
    }
  ],
  "relationships": [
    {"type": "above|on|inside|attached_to|falling_onto|collides_with|supports", "source": "entity_id", "target": "entity_id"}
  ],
  "simulation_intent": {
    "moving_entities": ["entity_id"],
    "motion_source": "gravity|actuator|passive collision|none",
    "description": string
  }
}

Rules:
- Capture what a human expects physically, not implementation details.
- Use "free" for falling, thrown, loose, bouncing, rolling, or projectile objects.
- Use "jointed" for passive hinges/slides/balls; use "driven" when motors/actuators/opening/lifting are implied.
- Use "static" for supports, floors, houses, frames, walls, ramps, panes, and targets.
- Decompose recognizable compound objects into obvious major parts.
- Preserve relationships like X on Y, X above Y, X falling onto Y, X attached to Y.
"""


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a model response."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(1))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i, ch in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start:i + 1])
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "object"


def _entity(
    name: str,
    kind: str | None = None,
    role: str = "passive",
    motion: str = "",
    visual: str = "",
    parts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": _slug(name),
        "name": name,
        "kind": kind or name,
        "role": role,
        "motion_intent": motion,
        "visual_intent": visual,
        "parts": parts or [],
    }


def _part(pid: str, kind: str, role: str = "passive", shape: str = "box") -> dict[str, Any]:
    return {"id": pid, "kind": kind, "role": role, "shape": shape}


def _contains(desc: str, *words: str) -> bool:
    return any(word in desc for word in words)


def _add_entity_once(entities: list[dict[str, Any]], entity: dict[str, Any]) -> None:
    if not any(e.get("id") == entity["id"] for e in entities):
        entities.append(entity)


def heuristic_scene_spec(description: str) -> dict[str, Any]:
    """Build a semantic approximation without external model availability."""
    desc = description.lower()
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, str]] = []
    moving: list[str] = []
    motion_source = "gravity"

    def add(name: str, kind: str, role: str, motion: str, visual: str, parts: list[dict[str, Any]]):
        entity = _entity(name, kind, role, motion, visual, parts)
        _add_entity_once(entities, entity)
        if role in {"free", "jointed", "driven"} and entity["id"] not in moving:
            moving.append(entity["id"])
        return entity

    falling_match = re.search(
        r"(?:a |an |the )?(?P<src>[a-z0-9 \-]+?)\s+(?:falling|falls|dropping|dropped)\s+(?:on|onto|towards?|toward)\s+(?:a |an |the )?(?P<tgt>[a-z0-9 \-]+)",
        desc,
    )

    if falling_match:
        source_text = falling_match.group("src").strip()
        target_text = falling_match.group("tgt").strip()
        source = _infer_entity_from_phrase(source_text, preferred_role="free")
        target = _infer_entity_from_phrase(target_text, preferred_role="static")
        _add_entity_once(entities, source)
        _add_entity_once(entities, target)
        relationships.append({"type": "falling_onto", "source": source["id"], "target": target["id"]})
        relationships.append({"type": "above", "source": source["id"], "target": target["id"]})
        moving.append(source["id"])
    else:
        if "pendulum" in desc:
            add(
                "pendulum",
                "pendulum",
                "jointed",
                "passive gravity swing about a hinge",
                "rod with a heavier bob",
                [_part("rod", "rod", "jointed", "cylinder"), _part("bob", "mass", "passive", "sphere")],
            )
        if _contains(desc, "house", "building", "shed", "cabin"):
            add(
                "house",
                "building",
                "static",
                "static collision target",
                "box-like body with a roof",
                [_part("base", "walls", "static", "box"), _part("roof_left", "roof", "static", "box"), _part("roof_right", "roof", "static", "box")],
            )
        is_robot_arm = "robotic arm" in desc or "robot arm" in desc or "two-link" in desc or "two link" in desc
        if (not is_robot_arm) and (_contains(desc, "robot", "rover", "car", "vehicle") or "wheels" in desc):
            add(
                "robot",
                "vehicle",
                "driven",
                "motor-driven wheel rotation",
                "chassis with attached wheels",
                [_part("chassis", "chassis", "free", "box")]
                + [_part(f"wheel_{i}", "wheel", "driven", "cylinder") for i in range(4)],
            )
            motion_source = "actuator"
        if _contains(desc, "ball", "sphere"):
            role = "free" if _contains(desc, "bounce", "bouncing", "fall", "falling", "drop") else "passive"
            add("ball", "ball", role, "gravity and contact" if role == "free" else "passive object", "round sphere", [_part("body", "ball", role, "sphere")])
        if "trampoline" in desc:
            add("trampoline", "elastic_surface", "static", "static springy-looking contact surface", "thin raised surface on legs", [_part("mat", "surface", "static", "box")])
            if any(e["id"] == "ball" for e in entities):
                relationships.append({"type": "above", "source": "ball", "target": "trampoline"})
                relationships.append({"type": "collides_with", "source": "ball", "target": "trampoline"})
        if _contains(desc, "door", "gate"):
            add("door", "door", "driven", "hinge swing open", "panel attached to a frame", [_part("frame", "frame", "static", "box"), _part("panel", "panel", "driven", "box")])
            motion_source = "actuator"
        if is_robot_arm:
            add("robotic arm", "articulated_arm", "driven", "two hinged links move by motors", "base with two connected links", [_part("base", "base", "static", "cylinder"), _part("link1", "link", "driven", "capsule"), _part("link2", "link", "driven", "capsule")])
            motion_source = "actuator"
        if "ramp" in desc:
            add("ramp", "ramp", "static", "static inclined support", "tilted plane-like box", [_part("surface", "inclined surface", "static", "box")])
        if "box" in desc or "cube" in desc:
            role = "free" if _contains(desc, "slide", "sliding", "fall", "falling", "drop") else "passive"
            add("box", "box", role, "gravity and contact" if role == "free" else "passive object", "rectangular solid", [_part("body", "box", role, "box")])
            if "ramp" in desc:
                relationships.append({"type": "on", "source": "box", "target": "ramp"})
        if "hammer" in desc:
            add("hammer", "tool", "free", "gravity and contact impact", "handle with a heavy head", [_part("handle", "handle", "free", "capsule"), _part("head", "head", "free", "box")])
        if "glass" in desc or "pane" in desc:
            add("glass pane", "pane", "static", "static collision target", "thin upright transparent panel", [_part("panel", "pane", "static", "box")])
            if any(e["id"] == "hammer" for e in entities):
                relationships.append({"type": "collides_with", "source": "hammer", "target": "glass_pane"})
                relationships.append({"type": "above", "source": "hammer", "target": "glass_pane"})
        if "crane" in desc:
            add("crane", "crane", "driven", "load lifted by actuator", "base, mast, boom, and suspended load", [_part("base", "base", "static", "box"), _part("mast", "mast", "static", "box"), _part("boom", "boom", "static", "capsule"), _part("load", "load", "driven", "box")])
            motion_source = "actuator"

    if not entities:
        add("object", "object", "free", "gravity fall", "simple visible object", [_part("body", "object", "free", "box")])

    if not moving:
        for entity in entities:
            if entity.get("role") in {"free", "jointed", "driven"}:
                moving.append(entity["id"])
    if not moving and entities:
        entities[0]["role"] = "free"
        entities[0]["motion_intent"] = "gravity fall"
        moving.append(entities[0]["id"])

    return {
        "description": description,
        "entities": entities,
        "relationships": relationships,
        "simulation_intent": {
            "moving_entities": moving,
            "motion_source": motion_source if moving else "none",
            "description": "simulate the described motion with contact and constraints",
        },
        "source": "heuristic",
    }


def _infer_entity_from_phrase(phrase: str, preferred_role: str) -> dict[str, Any]:
    phrase = phrase.strip().lower()
    if "pendulum" in phrase:
        return _entity(
            "pendulum",
            "pendulum",
            preferred_role,
            "falling compound pendulum assembly" if preferred_role == "free" else "hinged pendulum swing",
            "rod with a heavier bob",
            [_part("rod", "rod", preferred_role, "cylinder"), _part("bob", "mass", "passive", "sphere")],
        )
    if any(word in phrase for word in ("house", "building", "shed", "cabin")):
        return _entity(
            "house",
            "building",
            "static",
            "static collision target",
            "box-like body with a roof",
            [_part("base", "walls", "static", "box"), _part("roof_left", "roof", "static", "box"), _part("roof_right", "roof", "static", "box")],
        )
    if "ball" in phrase:
        return _entity("ball", "ball", preferred_role, "gravity and contact", "round sphere", [_part("body", "ball", preferred_role, "sphere")])
    if "hammer" in phrase:
        return _entity("hammer", "tool", preferred_role, "gravity and contact impact", "handle with heavy head", [_part("handle", "handle", preferred_role, "capsule"), _part("head", "head", "passive", "box")])
    if "box" in phrase or "cube" in phrase:
        return _entity("box", "box", preferred_role, "gravity and contact", "rectangular solid", [_part("body", "box", preferred_role, "box")])
    return _entity(phrase or "object", "object", preferred_role, "gravity and contact" if preferred_role == "free" else "static support", "simple visible object", [_part("body", "object", preferred_role, "box")])


def normalize_scene_spec(raw: dict[str, Any] | None, description: str) -> dict[str, Any]:
    """Repair weak/missing specs into a complete role-bearing SceneSpec."""
    heuristic = heuristic_scene_spec(description)
    if not isinstance(raw, dict):
        return heuristic

    spec = deepcopy(raw)
    spec.setdefault("description", description)
    spec.setdefault("entities", [])
    spec.setdefault("relationships", [])
    spec.setdefault("simulation_intent", {})
    if not isinstance(spec["entities"], list) or not spec["entities"]:
        return heuristic

    repaired_entities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in spec["entities"]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("id") or item.get("kind") or "object")
        eid = _slug(str(item.get("id") or name))
        if eid in seen:
            continue
        seen.add(eid)
        role = str(item.get("role") or "").lower()
        if role not in ROLES:
            role = "passive"
        kind = str(item.get("kind") or name).lower()
        inferred = _infer_entity_from_phrase(f"{name} {kind}", role)
        parts = item.get("parts")
        weak_parts = (
            not isinstance(parts, list)
            or not parts
            or (
                len(parts) == 1
                and isinstance(parts[0], dict)
                and str(parts[0].get("kind", "")).lower() in {"object", "body", ""}
                and str(parts[0].get("shape", "")).lower() in {"box", ""}
            )
        )
        generic_kind = kind in {"generic object class", "generic object", "generic", "object"}
        if weak_parts:
            parts = inferred["parts"]
        if generic_kind:
            kind = inferred["kind"]
        kind, parts = _repair_recognizable_parts(f"{eid} {name}", kind, role, parts)
        repaired_entities.append({
            "id": eid,
            "name": name,
            "kind": kind,
            "role": role,
            "motion_intent": str(item.get("motion_intent") or inferred.get("motion_intent") or ""),
            "visual_intent": str(item.get("visual_intent") or inferred.get("visual_intent") or ""),
            "parts": parts,
        })

    if not repaired_entities:
        return heuristic

    relations = []
    ids = {e["id"] for e in repaired_entities}
    for rel in spec.get("relationships", []):
        if not isinstance(rel, dict):
            continue
        rtype = str(rel.get("type") or "").lower().strip().replace("-", "_")
        rtype = RELATION_ALIASES.get(rtype, rtype)
        src = _slug(str(rel.get("source") or ""))
        tgt = _slug(str(rel.get("target") or ""))
        if rtype in RELATION_TYPES and src in ids and tgt in ids:
            relations.append({"type": rtype, "source": src, "target": tgt})

    sim = spec.get("simulation_intent") if isinstance(spec.get("simulation_intent"), dict) else {}
    moving = [
        _slug(str(x)) for x in sim.get("moving_entities", [])
        if _slug(str(x)) in ids
    ] if isinstance(sim.get("moving_entities"), list) else []
    for entity in repaired_entities:
        if entity["role"] in {"free", "jointed", "driven"} and entity["id"] not in moving:
            moving.append(entity["id"])
    if not moving:
        return heuristic

    return {
        "description": spec["description"],
        "entities": repaired_entities,
        "relationships": relations,
        "simulation_intent": {
            "moving_entities": moving,
            "motion_source": str(sim.get("motion_source") or "gravity"),
            "description": str(sim.get("description") or "simulate the described motion"),
        },
        "source": spec.get("source", "llm"),
    }


def _repair_recognizable_parts(
    name: str,
    kind: str,
    role: str,
    parts: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    phrase = f"{name} {kind}".lower()
    part_text = " ".join(
        f"{part.get('id', '')} {part.get('kind', '')}" for part in parts if isinstance(part, dict)
    ).lower()

    def missing(*needles: str) -> bool:
        return not any(needle in part_text for needle in needles)

    if any(word in phrase for word in ("house", "building", "shed", "cabin", "structure")):
        if missing("roof") or missing("base", "wall"):
            repaired = _infer_entity_from_phrase("house", "static")
            parts = repaired["parts"]
        return "building", parts
    if "pendulum" in phrase:
        if missing("rod", "arm") or missing("bob", "mass"):
            repaired = _infer_entity_from_phrase("pendulum", role)
            parts = repaired["parts"]
        return "pendulum", parts
    if any(word in phrase for word in ("vehicle", "robot", "rover", "car")) and "arm" not in phrase:
        if missing("chassis") or missing("wheel"):
            parts = [_part("chassis", "chassis", "free", "box")] + [
                _part(f"wheel_{i}", "wheel", "driven", "cylinder") for i in range(4)
            ]
        return "vehicle", parts
    if "door" in phrase:
        if missing("frame") or missing("panel"):
            parts = [_part("frame", "frame", "static", "box"), _part("panel", "panel", "driven", "box")]
        return "door", parts
    if "arm" in phrase and ("robot" in phrase or "two" in phrase):
        if missing("link"):
            parts = [_part("base", "base", "static", "cylinder"), _part("link1", "link", "driven", "capsule"), _part("link2", "link", "driven", "capsule")]
        return "articulated_arm", parts
    return kind, parts


def validate_scene_spec(spec: dict[str, Any]) -> list[str]:
    """Return semantic issues that would make deterministic compilation unreliable."""
    issues: list[str] = []
    entities = spec.get("entities", [])
    if not isinstance(entities, list) or not entities:
        return ["SceneSpec has no entities."]

    ids: set[str] = set()
    for entity in entities:
        eid = entity.get("id")
        if not eid:
            issues.append("Entity missing id.")
            continue
        if eid in ids:
            issues.append(f"Duplicate entity id '{eid}'.")
        ids.add(eid)
        if entity.get("role") not in ROLES:
            issues.append(f"Entity '{eid}' has no valid role.")
        if not entity.get("parts"):
            issues.append(f"Entity '{eid}' has no parts.")
        part_text = " ".join(
            f"{part.get('id', '')} {part.get('kind', '')}"
            for part in entity.get("parts", [])
            if isinstance(part, dict)
        ).lower()
        phrase = f"{entity.get('name', '')} {entity.get('kind', '')}".lower()
        if any(word in phrase for word in ("house", "building")) and "roof" not in part_text:
            issues.append(f"Entity '{eid}' should include a roof part.")
        if "pendulum" in phrase and not ("rod" in part_text and ("bob" in part_text or "mass" in part_text)):
            issues.append(f"Entity '{eid}' should include rod and bob parts.")

    for rel in spec.get("relationships", []):
        src = rel.get("source")
        tgt = rel.get("target")
        if src not in ids or tgt not in ids:
            issues.append(f"Relationship '{rel.get('type')}' references unknown entities.")
        if rel.get("type") in {"falling_onto", "above"}:
            source = next((e for e in entities if e.get("id") == src), {})
            if source.get("role") not in {"free", "passive"}:
                issues.append(f"Entity '{src}' should be free/passive for relation '{rel.get('type')}'.")

    moving = spec.get("simulation_intent", {}).get("moving_entities", [])
    if not moving:
        issues.append("Simulation intent has no moving entities.")
    for eid in moving:
        if eid not in ids:
            issues.append(f"Moving entity '{eid}' is unknown.")
    return issues


def compile_scene_spec(spec: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Compile a valid-ish SceneSpec into MJCF and an actuator schedule."""
    b = MechanismBuilder()
    schedule: list[dict[str, Any]] = []
    entities = {e["id"]: e for e in spec.get("entities", [])}
    compiled: set[str] = set()

    for rel in spec.get("relationships", []):
        if rel.get("type") == "falling_onto":
            target = entities.get(rel.get("target"))
            source = entities.get(rel.get("source"))
            if target and source:
                _compile_static_entity(b, target, (0, 0, 0))
                compiled.add(target["id"])
                top_z = _entity_top_z(target)
                _compile_free_entity(b, source, (0, 0, top_z + 1.8))
                compiled.add(source["id"])

    for rel in spec.get("relationships", []):
        if rel.get("type") in {"above", "on"} and rel.get("source") not in compiled:
            target = entities.get(rel.get("target"))
            source = entities.get(rel.get("source"))
            if target and target["id"] not in compiled:
                _compile_static_entity(b, target, (0, 0, 0))
                compiled.add(target["id"])
            if source:
                z = _entity_top_z(target) + (0.9 if rel.get("type") == "above" else 0.28)
                if source.get("role") == "static":
                    _compile_static_entity(b, source, (0, 0, z))
                else:
                    _compile_free_entity(b, source, (0, 0, z))
                compiled.add(source["id"])

    x_cursor = -1.0 if compiled else 0.0
    for entity in spec.get("entities", []):
        if entity["id"] in compiled:
            continue
        role = entity.get("role")
        kind = entity.get("kind", "")
        origin = (x_cursor, 0, 0)
        if kind in {"pendulum"} and role == "jointed":
            _compile_hinged_pendulum(b, entity, (x_cursor, 0, 2.4))
        elif kind in {"vehicle"} or "wheel" in entity.get("visual_intent", ""):
            schedule.extend(_compile_vehicle(b, entity, (x_cursor, 0, 0.16)))
        elif kind in {"door"}:
            schedule.extend(_compile_door(b, entity, (x_cursor, 0, 0)))
        elif kind in {"articulated_arm"}:
            schedule.extend(_compile_two_link_arm(b, entity, (x_cursor, 0, 0)))
        elif kind in {"ramp"}:
            _compile_ramp(b, entity, (x_cursor, 0, 0))
        elif kind in {"crane"}:
            schedule.extend(_compile_crane(b, entity, (x_cursor, 0, 0)))
        elif role == "static":
            _compile_static_entity(b, entity, origin)
        elif role in {"free", "passive"}:
            _compile_free_entity(b, entity, (x_cursor, 0, 1.2))
        elif role == "driven":
            schedule.extend(_compile_two_link_arm(b, entity, (x_cursor, 0, 0)))
        else:
            _compile_free_entity(b, entity, (x_cursor, 0, 1.2))
        compiled.add(entity["id"])
        x_cursor += 1.3

    xml = b.build()
    return xml, _valid_schedule_for_xml(xml, schedule)


def _entity_top_z(entity: dict[str, Any]) -> float:
    kind = entity.get("kind", "")
    if kind in {"building", "house"} or "house" in entity.get("name", ""):
        return 1.05
    if kind == "elastic_surface":
        return 0.42
    if kind == "ramp":
        return 0.45
    if kind == "pane":
        return 1.2
    return 0.3


def _compile_static_entity(b: MechanismBuilder, entity: dict[str, Any], origin: tuple[float, float, float]) -> None:
    kind = entity.get("kind", "")
    name = entity["id"]
    x, y, z = origin
    if kind in {"building", "house"} or "house" in entity.get("name", ""):
        b.add_body(f"{name}_body", "box", (0.8, 0.55, 0.45), mass=100.0, color=(0.42, 0.34, 0.26))
        b.add_body(f"{name}_roof_left", "box", (0.55, 0.6, 0.08), mass=30.0, color=(0.35, 0.16, 0.12), euler=(0, 25, 0))
        b.add_body(f"{name}_roof_right", "box", (0.55, 0.6, 0.08), mass=30.0, color=(0.35, 0.16, 0.12), euler=(0, -25, 0))
        b.attach_to("world", f"{name}_body", "fixed")
        b.attach_to(f"{name}_body", f"{name}_roof_left", "fixed")
        b.attach_to(f"{name}_body", f"{name}_roof_right", "fixed")
        b.position_relative(f"{name}_body", "world", (x, y, z + 0.45))
        b.position_relative(f"{name}_roof_left", f"{name}_body", (-0.22, 0, 0.55))
        b.position_relative(f"{name}_roof_right", f"{name}_body", (0.22, 0, 0.55))
    elif kind == "elastic_surface":
        b.add_body(f"{name}_mat", "box", (0.7, 0.45, 0.035), mass=20.0, color=(0.1, 0.45, 0.65))
        b.attach_to("world", f"{name}_mat", "fixed")
        b.position_relative(f"{name}_mat", "world", (x, y, z + 0.38))
    elif kind == "ramp":
        _compile_ramp(b, entity, origin)
    elif kind == "pane":
        b.add_body(f"{name}_panel", "box", (0.04, 0.65, 0.75), mass=8.0, color=(0.65, 0.85, 1.0, 0.45))
        b.attach_to("world", f"{name}_panel", "fixed")
        b.position_relative(f"{name}_panel", "world", (x, y, z + 0.75))
    else:
        b.add_body(name, "box", (0.35, 0.25, 0.18), mass=10.0, color=(0.45, 0.45, 0.42))
        b.attach_to("world", name, "fixed")
        b.position_relative(name, "world", (x, y, z + 0.18))


def _compile_free_entity(b: MechanismBuilder, entity: dict[str, Any], origin: tuple[float, float, float]) -> None:
    kind = entity.get("kind", "")
    name = entity["id"]
    x, y, z = origin
    if kind == "pendulum":
        b.add_body(f"{name}_rod", "cylinder", (0.025, 0.6), mass=0.6, color=(0.55, 0.38, 0.22), free=True, geom_offset=(0, 0, -0.6), euler=(0, 20, 0))
        b.add_body(f"{name}_bob", "sphere", 0.14, mass=2.0, color=(0.75, 0.08, 0.08))
        b.attach_to(f"{name}_rod", f"{name}_bob", "fixed")
        b.position_relative(f"{name}_rod", "world", (x, y, z))
        b.position_relative(f"{name}_bob", f"{name}_rod", (0, 0, -1.2))
    elif kind == "ball":
        b.add_body(name, "sphere", 0.15, mass=1.0, color=(0.82, 0.12, 0.1), free=True)
        b.position_relative(name, "world", (x, y, z))
    elif kind in {"box", "object"}:
        b.add_body(name, "box", (0.18, 0.18, 0.18), mass=1.2, color=(0.25, 0.55, 0.8), free=True)
        b.position_relative(name, "world", (x, y, z))
    elif kind == "tool":
        b.add_body(f"{name}_handle", "capsule", (0.035, 0.45), mass=0.8, color=(0.45, 0.28, 0.12), free=True, axis="x", euler=(0, 20, 0))
        b.add_body(f"{name}_head", "box", (0.18, 0.08, 0.09), mass=2.0, color=(0.45, 0.45, 0.48))
        b.attach_to(f"{name}_handle", f"{name}_head", "fixed")
        b.position_relative(f"{name}_handle", "world", (x, y, z))
        b.position_relative(f"{name}_head", f"{name}_handle", (0.45, 0, 0))
    else:
        b.add_body(name, "box", (0.18, 0.18, 0.18), mass=1.0, color=(0.3, 0.55, 0.8), free=True)
        b.position_relative(name, "world", (x, y, z))


def _compile_hinged_pendulum(b: MechanismBuilder, entity: dict[str, Any], origin: tuple[float, float, float]) -> None:
    name = entity["id"]
    b.add_body(f"{name}_rod", "cylinder", (0.025, 0.7), mass=0.6, color=(0.55, 0.38, 0.22), geom_offset=(0, 0, -0.7), euler=(0, 30, 0))
    b.add_body(f"{name}_bob", "sphere", 0.14, mass=2.0, color=(0.75, 0.08, 0.08))
    b.attach_to("world", f"{name}_rod", "hinge", axis="y", limits=(-180, 180), damping=0.08)
    b.attach_to(f"{name}_rod", f"{name}_bob", "fixed")
    b.position_relative(f"{name}_rod", "world", origin)
    b.position_relative(f"{name}_bob", f"{name}_rod", (0, 0, -1.4))


def _compile_vehicle(b: MechanismBuilder, entity: dict[str, Any], origin: tuple[float, float, float]) -> list[dict[str, Any]]:
    name = entity["id"]
    x, y, z = origin
    schedule: list[dict[str, Any]] = []
    b.add_body(f"{name}_chassis", "box", (0.32, 0.18, 0.07), mass=5.0, color=(0.25, 0.35, 0.42), free=True)
    b.position_relative(f"{name}_chassis", "world", (x, y, z))
    for wheel, dx, dy in [
        ("front_left_wheel", 0.23, 0.2),
        ("front_right_wheel", 0.23, -0.2),
        ("rear_left_wheel", -0.23, 0.2),
        ("rear_right_wheel", -0.23, -0.2),
    ]:
        wname = f"{name}_{wheel}"
        b.add_body(wname, "cylinder", (0.075, 0.028), mass=0.45, color=(0.04, 0.04, 0.04), axis="y")
        b.attach_to(f"{name}_chassis", wname, "hinge", axis="y", damping=0.05)
        b.position_relative(wname, f"{name}_chassis", (dx, dy, -0.08))
        b.add_actuator(wname, torque=5.0)
        schedule.extend([{"name": f"{wname}_motor", "control": 0.9, "time": 0}, {"name": f"{wname}_motor", "control": 0.9, "time": 10}])
    return schedule


def _compile_door(b: MechanismBuilder, entity: dict[str, Any], origin: tuple[float, float, float]) -> list[dict[str, Any]]:
    name = entity["id"]
    x, y, z = origin
    b.add_body(f"{name}_frame", "box", (0.04, 0.08, 0.7), mass=8.0, color=(0.32, 0.24, 0.18))
    b.add_body(f"{name}_panel", "box", (0.5, 0.025, 0.62), mass=3.0, color=(0.65, 0.42, 0.22), geom_offset=(0.5, 0, 0), euler=(0, 0, 10))
    b.attach_to("world", f"{name}_frame", "fixed")
    b.attach_to(f"{name}_frame", f"{name}_panel", "hinge", axis="z", limits=(-120, 120), damping=0.2)
    b.position_relative(f"{name}_frame", "world", (x, y, z + 0.7))
    b.position_relative(f"{name}_panel", f"{name}_frame", (0.04, 0, 0))
    b.add_actuator(f"{name}_panel", torque=2.5)
    return [{"name": f"{name}_panel_motor", "control": 0.7, "time": 0}, {"name": f"{name}_panel_motor", "control": 0.25, "time": 10}]


def _compile_two_link_arm(b: MechanismBuilder, entity: dict[str, Any], origin: tuple[float, float, float]) -> list[dict[str, Any]]:
    name = entity["id"]
    x, y, z = origin
    b.add_body(f"{name}_base", "cylinder", (0.08, 0.12), mass=4.0, color=(0.25, 0.25, 0.28))
    b.add_body(f"{name}_link1", "capsule", (0.035, 0.28), mass=1.0, color=(0.15, 0.45, 0.75), geom_offset=(0, 0, 0.28), euler=(0, 20, 0))
    b.add_body(f"{name}_link2", "capsule", (0.03, 0.24), mass=0.7, color=(0.15, 0.58, 0.82), geom_offset=(0, 0, 0.24), euler=(0, -18, 0))
    b.attach_to("world", f"{name}_base", "fixed")
    b.attach_to(f"{name}_base", f"{name}_link1", "hinge", axis="y", limits=(-110, 110), damping=0.5)
    b.attach_to(f"{name}_link1", f"{name}_link2", "hinge", axis="y", limits=(-120, 120), damping=0.45)
    b.position_relative(f"{name}_base", "world", (x, y, z + 0.12))
    b.position_relative(f"{name}_link1", f"{name}_base", (0, 0, 0.24))
    b.position_relative(f"{name}_link2", f"{name}_link1", (0, 0, 0.56))
    b.add_actuator(f"{name}_link1", torque=3.0)
    b.add_actuator(f"{name}_link2", torque=2.0)
    return [
        {"name": f"{name}_link1_motor", "control": 0.8, "time": 0},
        {"name": f"{name}_link1_motor", "control": -0.2, "time": 10},
        {"name": f"{name}_link2_motor", "control": -0.6, "time": 0},
        {"name": f"{name}_link2_motor", "control": 0.4, "time": 10},
    ]


def _compile_ramp(b: MechanismBuilder, entity: dict[str, Any], origin: tuple[float, float, float]) -> None:
    name = entity["id"]
    x, y, z = origin
    b.add_body(name, "box", (0.8, 0.35, 0.05), mass=20.0, color=(0.38, 0.38, 0.34), euler=(0, -18, 0))
    b.attach_to("world", name, "fixed")
    b.position_relative(name, "world", (x, y, z + 0.32))


def _compile_crane(b: MechanismBuilder, entity: dict[str, Any], origin: tuple[float, float, float]) -> list[dict[str, Any]]:
    name = entity["id"]
    x, y, z = origin
    b.add_body(f"{name}_base", "box", (0.28, 0.22, 0.08), mass=30.0, color=(0.42, 0.38, 0.24))
    b.add_body(f"{name}_mast", "box", (0.05, 0.05, 0.75), mass=20.0, color=(0.75, 0.62, 0.22))
    b.add_body(f"{name}_boom", "capsule", (0.035, 0.55), mass=12.0, color=(0.75, 0.62, 0.22), axis="x")
    b.add_body(f"{name}_load", "box", (0.16, 0.16, 0.16), mass=4.0, color=(0.32, 0.32, 0.34))
    b.attach_to("world", f"{name}_base", "fixed")
    b.attach_to(f"{name}_base", f"{name}_mast", "fixed")
    b.attach_to(f"{name}_mast", f"{name}_boom", "fixed")
    b.attach_to(f"{name}_boom", f"{name}_load", "slide", axis="z", limits=(-0.5, 0.15), damping=0.2)
    b.position_relative(f"{name}_base", "world", (x, y, z + 0.08))
    b.position_relative(f"{name}_mast", f"{name}_base", (0, 0, 0.83))
    b.position_relative(f"{name}_boom", f"{name}_mast", (0.48, 0, 0.65))
    b.position_relative(f"{name}_load", f"{name}_boom", (0.48, 0, -0.55))
    b.add_actuator(f"{name}_load", torque=5.0)
    return [{"name": f"{name}_load_motor", "control": 0.9, "time": 0}, {"name": f"{name}_load_motor", "control": -0.2, "time": 10}]


def _valid_schedule_for_xml(xml: str, schedule: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = set(re.findall(r'<motor\s+name="([^"]+)"', xml))
    return [entry for entry in schedule if entry.get("name") in names]

"""
Assembles a flat list of primitive dicts (from primitives.py) into a valid MJCF XML string.
Validates with MuJoCo before returning. Raises ValueError on structural errors.
"""
from typing import Any
import mujoco


def _flatten(primitives: list) -> list:
    """Flatten nested lists (cylindrical/planar return lists of joints)."""
    out = []
    for p in primitives:
        if isinstance(p, list):
            out.extend(p)
        else:
            out.append(p)
    return out


def assemble(primitives: list) -> str:
    """
    Convert a list of primitive dicts to a complete, validated MJCF XML string.

    Ordering rules:
    - rigid_body() defines the body hierarchy via parent= field.
    - joint/geom prims attach to the body named in their body= field,
      or to the most recently defined body if body= is omitted.
    - actuator, contact_pair, spring, damper go into their respective XML sections.
    - gravity() sets the <option> gravity attribute.
    - ground() adds a geom directly to worldbody.
    """
    flat = _flatten(primitives)

    # ---- Collect by kind ----
    gravity_vec = "0 0 -9.81"
    world_geoms: list[dict] = []
    bodies: dict[str, dict] = {}          # name → prim dict
    body_order: list[str] = []
    body_joints: dict[str, list] = {}     # name → [joint dicts]
    body_geoms: dict[str, list] = {}      # name → [geom dicts]
    body_children: dict[str, list] = {"world": []}
    actuators: list[dict] = []
    contact_pairs: list[dict] = []
    last_body: str | None = None

    for p in flat:
        kind = p.get("_kind", "")

        if kind == "option_gravity":
            gravity_vec = p["gravity"]

        elif kind == "world_geom":
            world_geoms.append(p)

        elif kind == "body":
            name = p["name"]
            if name in bodies:
                raise ValueError(f"Duplicate body name: {name!r}")
            bodies[name] = p
            body_order.append(name)
            body_joints[name] = []
            body_geoms[name] = []
            parent = p.get("parent", "world")
            body_children.setdefault(parent, []).append(name)
            body_children.setdefault(name, [])
            last_body = name

        elif kind == "joint":
            target = p.get("body") or last_body
            if not target:
                raise ValueError(f"Joint {p.get('name')!r} has no body to attach to")
            body_joints.setdefault(target, []).append(p)

        elif kind == "geom":
            target = p.get("body") or last_body
            if target and target in body_geoms:
                body_geoms[target].append(p)
            else:
                world_geoms.append(p)

        elif kind == "fixed_marker":
            pass  # fixed = body with no joint — nothing to add

        elif kind == "actuator":
            actuators.append(p)

        elif kind == "contact_pair":
            contact_pairs.append(p)

        elif kind == "joint_spring":
            # Inject stiffness into the target joint dict
            jname = p["joint"]
            for joints in body_joints.values():
                for j in joints:
                    if j.get("name") == jname:
                        j["stiffness"] = p["stiffness"]
                        if p.get("rest_length"):
                            j["springref"] = p["rest_length"]

        elif kind == "joint_damper":
            # Augment damping on the target joint dict
            jname = p["joint"]
            for joints in body_joints.values():
                for j in joints:
                    if j.get("name") == jname:
                        j["damping"] = float(j.get("damping", 0)) + p["damping"]

    # ---- XML render helpers ----

    # Expected number of size values per geom type (MuJoCo requirement)
    _SIZE_COUNTS = {"box": 3, "sphere": 1, "cylinder": 2, "capsule": 2, "plane": 3, "ellipsoid": 3}

    def _fix_size(gt: str, raw_size) -> str:
        """Ensure size string has the right number of numeric tokens for the geom type."""
        if raw_size is None:
            raw_size = "0.1"
        if isinstance(raw_size, (list, tuple)):
            tokens = [str(x) for x in raw_size]
        else:
            tokens = str(raw_size).split()
        # Strip non-numeric tokens
        clean = []
        for t in tokens:
            try:
                float(t)
                clean.append(t)
            except ValueError:
                pass
        if not clean:
            clean = ["0.1"]
        n = _SIZE_COUNTS.get(gt, 3)
        # Trim extras or pad with last value
        if len(clean) > n:
            clean = clean[:n]
        while len(clean) < n:
            clean.append(clean[-1])
        return " ".join(clean)

    def _fix_range(raw) -> str | None:
        """Sanitize joint range to exactly two numbers, stripping units/symbols."""
        if raw is None:
            return None
        import re as _re
        s = str(raw)
        # Strip degree signs, letters, replace commas/brackets
        s = _re.sub(r'[°°\[\]()degrada-zA-Z]', ' ', s)
        s = s.replace(',', ' ').replace('−', '-').replace('–', '-')
        nums = _re.findall(r'-?\d+\.?\d*', s)
        if len(nums) >= 2:
            return f"{nums[0]} {nums[1]}"
        if len(nums) == 1:
            return f"-{nums[0]} {nums[0]}"
        return None

    def _fix_axis(raw) -> str:
        """Ensure joint axis is a valid unit vector string."""
        if not raw:
            return "0 0 1"
        try:
            vals = [float(x) for x in str(raw).split()]
            if len(vals) == 3:
                mag = (vals[0]**2 + vals[1]**2 + vals[2]**2) ** 0.5
                if mag > 1e-6:
                    return f"{vals[0]/mag:.6g} {vals[1]/mag:.6g} {vals[2]/mag:.6g}"
        except ValueError:
            pass
        return "0 0 1"

    def _joint_xml(j: dict, pad: str) -> str:
        jt = j.get("joint_type", "hinge")
        parts = [f'type="{jt}"']
        if j.get("name") is not None:
            parts.append(f'name="{j["name"]}"')
        parts.append(f'axis="{_fix_axis(j.get("axis"))}"')
        rng = _fix_range(j.get("range"))
        if rng is not None:
            parts.append(f'range="{rng}"')
        for k in ("damping", "stiffness", "springref"):
            v = j.get(k)
            if v is not None:
                parts.append(f'{k}="{v}"')
        return f'{pad}<joint {" ".join(parts)}/>'

    def _fix_rgba(raw) -> str:
        """Clamp all four rgba channels to [0, 1]."""
        if raw is None:
            return "0.7 0.7 0.7 1"
        tokens = str(raw).split()
        clamped = []
        for t in tokens:
            try:
                clamped.append(f"{max(0.0, min(1.0, float(t))):.3f}")
            except ValueError:
                clamped.append("0.700")
        while len(clamped) < 4:
            clamped.append("1.000")
        return " ".join(clamped[:4])

    def _geom_xml(g: dict, pad: str) -> str:
        gt = g.get("geom_type", "box")
        parts = [f'type="{gt}"']
        # Sanitize size and rgba before rendering
        g_copy = dict(g)
        g_copy["size"] = _fix_size(gt, g.get("size"))
        g_copy["rgba"] = _fix_rgba(g.get("rgba"))
        for k in ("name", "size", "pos", "rgba", "density", "mesh", "euler"):
            v = g_copy.get(k)
            if v is not None:
                parts.append(f'{k}="{v}"')
        return f'{pad}<geom {" ".join(parts)}/>'

    def _body_xml(name: str, depth: int) -> str:
        b = bodies[name]
        pad = "  " * depth
        inner = "  " * (depth + 1)

        bparts = [f'name="{name}"']
        if b.get("pos"):
            bparts.append(f'pos="{b["pos"]}"')
        if b.get("euler"):
            bparts.append(f'euler="{b["euler"]}"')

        lines = [f'{pad}<body {" ".join(bparts)}>']

        if b.get("mass"):
            m = float(b["mass"])
            ii = max(m * 0.001, 1e-5)
            lines.append(
                f'{inner}<inertial mass="{m}" pos="0 0 0" diaginertia="{ii:.6g} {ii:.6g} {ii:.6g}"/>'
            )

        if b.get("free"):
            lines.append(f"{inner}<freejoint/>")

        for j in body_joints.get(name, []):
            lines.append(_joint_xml(j, inner))

        for g in body_geoms.get(name, []):
            lines.append(_geom_xml(g, inner))

        for child in body_children.get(name, []):
            lines.append(_body_xml(child, depth + 1))

        lines.append(f"{pad}</body>")
        return "\n".join(lines)

    # ---- Assemble worldbody ----
    wb: list[str] = []
    wb.append('    <light pos="0 0 5" dir="0 0 -1" diffuse="1 1 1"/>')
    for wg in world_geoms:
        wb.append(_geom_xml(wg, "    "))
    for top in body_children.get("world", []):
        wb.append(_body_xml(top, 2))

    # ---- Actuator section ----
    act: list[str] = []
    for a in actuators:
        aparts = [f'name="{a["name"]}"', f'joint="{a["joint"]}"']
        if a.get("gear") is not None:
            aparts.append(f'gear="{a["gear"]}"')
        if a.get("ctrllimited"):
            aparts.append('ctrllimited="true"')
        if a.get("ctrlrange"):
            aparts.append(f'ctrlrange="{a["ctrlrange"]}"')
        act.append(f'    <motor {" ".join(aparts)}/>')

    # ---- Contact section ----
    cp: list[str] = []
    for c in contact_pairs:
        cp.append(f'    <pair body1="{c["body1"]}" body2="{c["body2"]}"/>')

    # ---- Build final XML ----
    lines: list[str] = [
        "<mujoco>",
        f'  <option gravity="{gravity_vec}" timestep="0.002"/>',
        '  <compiler balanceinertia="true" boundmass="0.001" boundinertia="0.0001"/>',
        "  <worldbody>",
        *wb,
        "  </worldbody>",
    ]
    if act:
        lines += ["  <actuator>", *act, "  </actuator>"]
    if cp:
        lines += ["  <contact>", *cp, "  </contact>"]
    lines.append("</mujoco>")

    xml = "\n".join(lines)

    # Validate — raises mujoco.FatalError on bad XML
    mujoco.MjModel.from_xml_string(xml)

    return xml

"""
MechanismBuilder — fluent Python API for constructing MuJoCo MJCF scenes.

The LLM generates Python code using this API. The builder handles coordinate
transforms, geom orientation, and XML validation. The LLM never writes raw
MJCF, coordinates, or quaternions directly.
"""
import math
from typing import Optional
import xml.etree.ElementTree as ET
import mujoco


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _axis(a) -> list[float]:
    """Accept 'x'/'y'/'z' shortcuts or a vector; return a normalised unit vector."""
    shortcuts = {
        'x': [1., 0., 0.], '+x': [1., 0., 0.], '-x': [-1., 0., 0.],
        'y': [0., 1., 0.], '+y': [0., 1., 0.], '-y': [0., -1., 0.],
        'z': [0., 0., 1.], '+z': [0., 0., 1.], '-z': [0., 0., -1.],
    }
    if isinstance(a, str) and a.lower() in shortcuts:
        return shortcuts[a.lower()]
    v = [float(x) for x in (a.split() if isinstance(a, str) else a)]
    m = math.sqrt(sum(x * x for x in v))
    return [x / m for x in v] if m > 1e-8 else [0., 0., 1.]


def _size(shape: str, s) -> list[float]:
    """Normalise size to the correct number of floats for the given geom shape."""
    if isinstance(s, (int, float)):
        s = [float(s)]
    elif isinstance(s, str):
        s = [float(x) for x in s.split()]
    else:
        s = [float(x) for x in s]
    n = {'box': 3, 'sphere': 1, 'cylinder': 2, 'capsule': 2,
         'plane': 3, 'ellipsoid': 3}.get(shape, 1)
    while len(s) < n:
        s.append(s[-1])
    return s[:n]


def _geom_quat(body_axis: list[float]) -> Optional[list[float]]:
    """
    MuJoCo quaternion [w, x, y, z] that rotates the cylinder/capsule default
    axis (+Z) to align with the given body_axis direction.
    Returns None if already aligned with +Z (no rotation needed).
    """
    dot = body_axis[2]  # dot product with (0, 0, 1)
    if dot > 0.9999:
        return None
    if dot < -0.9999:
        return [0., 1., 0., 0.]  # 180° around X

    # cross((0,0,1), body_axis) = (-body_axis[1], body_axis[0], 0)
    cx, cy = -body_axis[1], body_axis[0]
    sin_a = math.sqrt(cx * cx + cy * cy)
    angle = math.atan2(sin_a, dot)
    cx, cy = cx / sin_a, cy / sin_a
    half = angle / 2.0
    w, s = math.cos(half), math.sin(half)
    return [w, s * cx, s * cy, 0.]


def _rgba(color) -> str:
    if color is None:
        return "0.7 0.7 0.7 1"
    c = list(color)
    while len(c) < 4:
        c.append(1.0)
    return " ".join(f"{min(1., max(0., v)):.3f}" for v in c[:4])


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _parse_vec(raw: str | None, n: int = 3) -> list[float]:
    vals = [float(x) for x in raw.split()] if raw else []
    while len(vals) < n:
        vals.append(0.0)
    return vals[:n]


def _fmt_vec(vals: list[float]) -> str:
    return " ".join(f"{v:.6g}" for v in vals)


def _geom_low_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float | None:
    """Conservative lower world-Z bound for a non-plane geom."""
    gtype = int(model.geom_type[geom_id])
    if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return None

    center_z = float(data.geom_xpos[geom_id][2])
    size = model.geom_size[geom_id]
    mat = data.geom_xmat[geom_id].reshape(3, 3)
    zrow = [float(mat[2, 0]), float(mat[2, 1]), float(mat[2, 2])]

    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        extent = float(size[0])
    elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        extent = sum(abs(zrow[i]) * float(size[i]) for i in range(3))
    elif gtype in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
        radius = float(size[0])
        half_length = float(size[1])
        axial = abs(zrow[2]) * half_length
        radial = radius * math.sqrt(max(0.0, 1.0 - zrow[2] * zrow[2]))
        extent = axial + radial
    elif gtype == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        extent = sum(abs(zrow[i]) * float(size[i]) for i in range(3))
    else:
        extent = float(model.geom_rbound[geom_id])

    return center_z - extent


def repair_ground_clearance(xml: str, clearance: float = 0.01) -> str:
    """
    Lift top-level world bodies so visible non-plane geometry starts above Z=0.
    Returns the original XML unchanged when no repair is needed.
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    lows = [
        low for i in range(model.ngeom)
        if (low := _geom_low_z(model, data, i)) is not None
    ]
    if not lows:
        return xml

    min_z = min(lows)
    if min_z >= clearance:
        return xml

    lift = clearance - min_z
    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    if worldbody is None:
        return xml

    top_bodies = [child for child in list(worldbody) if child.tag == "body"]
    if not top_bodies:
        return xml

    for body in top_bodies:
        pos = _parse_vec(body.get("pos"), 3)
        pos[2] += lift
        body.set("pos", _fmt_vec(pos))

    repaired = ET.tostring(root, encoding="unicode")
    mujoco.MjModel.from_xml_string(repaired)
    return repaired


# --------------------------------------------------------------------------- #
# Public class
# --------------------------------------------------------------------------- #

class MechanismBuilder:
    """
    Fluent builder for MuJoCo MJCF physics simulations.

    Coordinate system: X = forward, Y = left/right, Z = up.
    Ground plane is at Z = 0. Gravity acts in the -Z direction.

    Typical usage::

        b = MechanismBuilder()
        b.add_body('arm', 'cylinder', (0.02, 0.75), mass=0.5,
                   geom_offset=(0, 0, -0.75), euler=(0, 30, 0))
        b.attach_to('world', 'arm', 'hinge', axis='y', limits=(-180, 180))
        b.position_relative('arm', 'world', (0, 0, 2.5))
        xml = b.build()
    """

    def __init__(self, gravity=(0., 0., -9.81), timestep: float = 0.002):
        self._bodies: dict = {}      # name → config dict
        self._parents: dict = {}     # child_name → parent_name
        self._joints: dict = {}      # child_name → joint config
        self._positions: dict = {}   # name → (reference_name, [dx, dy, dz])
        self._actuators: list = []
        self._gravity = list(gravity)
        self._timestep = float(timestep)

    # ------------------------------------------------------------------ #
    # Fluent public API
    # ------------------------------------------------------------------ #

    def add_body(
        self,
        name: str,
        shape: str,
        size,
        mass: float = 1.0,
        color=None,
        free: bool = False,
        axis=None,
        geom_offset=None,
        euler=None,
    ) -> 'MechanismBuilder':
        """
        Define a body.

        Parameters
        ----------
        name      : unique identifier for this body
        shape     : 'box' | 'sphere' | 'cylinder' | 'capsule'
        size      : box → (hx, hy, hz); sphere → r;
                    cylinder/capsule → (radius, half_height)
        mass      : kg
        color     : (r, g, b) or (r, g, b, a) in [0, 1]; None = grey
        free      : True for free-floating bodies (vehicles, thrown objects)
        axis      : cylinder/capsule length direction — 'x', 'y', 'z', or [x,y,z]
                    'z' (default) = vertical; 'y' = flat disc (wheel)
        geom_offset : (dx, dy, dz) shift of the geom centre from the body origin.
                    Use for arms/pendulums where the joint is at one end.
                    Pendulum arm: geom_offset=(0, 0, -half_length)
                    Robot link:   geom_offset=(0, 0, +half_length)
        euler     : optional body orientation in degrees, e.g. (0, 30, 0).
                    Use this for non-equilibrium starting poses in hinged mechanisms.
        """
        self._bodies[name] = {
            'shape': shape.lower(),
            'size': _size(shape.lower(), size),
            'mass': float(mass),
            'color': color,
            'free': bool(free),
            'axis': _axis(axis) if axis is not None else None,
            'geom_offset': [float(v) for v in geom_offset] if geom_offset else None,
            'euler': [float(v) for v in euler] if euler is not None else None,
        }
        return self

    def attach_to(
        self,
        parent: str,
        child: str,
        joint_type: str = 'hinge',
        axis='z',
        limits=None,
        damping: float = 0.5,
        name: str = None,
    ) -> 'MechanismBuilder':
        """
        Attach child to parent via a joint.

        Parameters
        ----------
        parent     : body name or 'world'
        child      : body name (must have been defined with add_body first)
        joint_type : 'hinge' | 'slide' | 'ball' | 'fixed'
        axis       : rotation/slide axis — 'x', 'y', 'z', or [x,y,z]
        limits     : (min, max) — degrees for hinge, metres for slide; None = unlimited
        damping    : joint damping coefficient
        name       : override joint name (default: '{child}_joint')
        """
        if parent != 'world' and parent not in self._bodies:
            raise ValueError(f"Parent body '{parent}' must be defined before attaching.")
        if child not in self._bodies:
            raise ValueError(f"Child body '{child}' must be defined before attaching.")
        if self._bodies[child].get('free'):
            self._bodies[child]['free'] = False
        self._parents[child] = parent
        self._joints[child] = {
            'type': joint_type.lower(),
            'axis': _axis(axis),
            'limits': limits,
            'damping': float(damping),
            'name': name or f"{child}_joint",
        }
        return self

    def add_actuator(
        self,
        joint_or_body: str,
        torque: float = 100.0,
        name: str = None,
    ) -> 'MechanismBuilder':
        """
        Add a motor actuator.

        Parameters
        ----------
        joint_or_body : body name (auto-resolves to '{body}_joint') or explicit joint name.
                        The motor will be named '{body}_motor' (use this in actuator_schedule).
        torque        : peak torque / gear ratio
        name          : override motor name
        """
        self._actuators.append({
            'ref': str(joint_or_body),
            'torque': float(torque),
            'name': name,
        })
        return self

    def position_relative(
        self,
        body: str,
        reference: str,
        offset,
        euler=None,
    ) -> 'MechanismBuilder':
        """
        Set the world position of body to: reference_world_position + offset.

        Parameters
        ----------
        body      : body to position
        reference : reference body name, or 'world' for an absolute position
        offset    : (dx, dy, dz) in metres; Z is up, ground is Z=0
        euler     : optional convenience orientation in degrees for this body
        """
        off = [float(v) for v in offset]
        while len(off) < 3:
            off.append(0.0)
        self._positions[body] = (reference, off[:3])
        if euler is not None:
            if body not in self._bodies:
                raise ValueError(f"Body '{body}' must be defined before setting euler.")
            self._bodies[body]['euler'] = [float(v) for v in euler]
        return self

    def build(self) -> str:
        """Assemble, validate with MuJoCo, and return the MJCF XML string."""
        abs_pos = self._resolve_positions()
        children = self._build_tree()
        xml = "\n".join(self._render(abs_pos, children))
        try:
            xml = repair_ground_clearance(xml)
            mujoco.MjModel.from_xml_string(xml)
        except Exception as exc:
            raise ValueError(f"MJCF validation failed: {exc}\n\n--- XML ---\n{xml}")
        return xml

    # ------------------------------------------------------------------ #
    # Internal construction helpers
    # ------------------------------------------------------------------ #

    def _resolve_positions(self) -> dict:
        cache: dict = {'world': [0., 0., 0.]}

        def resolve(n: str) -> list:
            if n in cache:
                return cache[n]
            ref, off = self._positions.get(n, ('world', [0., 0., 0.]))
            p = resolve(ref)
            cache[n] = [p[i] + off[i] for i in range(3)]
            return cache[n]

        for name in self._bodies:
            resolve(name)
        return cache

    def _build_tree(self) -> dict:
        ch: dict = {'world': []}
        for child, parent in self._parents.items():
            ch.setdefault(parent, []).append(child)
            ch.setdefault(child, [])
        for name in self._bodies:
            if name not in self._parents:
                ch['world'].append(name)
                ch.setdefault(name, [])
        return ch

    def _joint_name(self, ref: str) -> str:
        if ref in self._joints:
            return self._joints[ref]['name']
        return ref

    def _motor_name(self, ref: str, custom: Optional[str]) -> str:
        if custom:
            return custom
        return f"{ref}_motor" if ref in self._bodies else f"motor_{ref}"

    def _render(self, abs_pos: dict, ch: dict) -> list[str]:
        grav = " ".join(f"{v:.4g}" for v in self._gravity)
        out = [
            "<mujoco>",
            f'  <option gravity="{grav}" timestep="{self._timestep}"/>',
            '  <compiler balanceinertia="true" boundmass="0.001" boundinertia="0.0001"/>',
            "  <worldbody>",
            '    <light pos="0 0 5" dir="0 0 -1" diffuse="1 1 1"/>',
            '    <geom type="plane" size="10 10 0.1" rgba="0.3 0.3 0.35 1"/>',
        ]
        for top in ch.get('world', []):
            out += self._body_lines(top, 2, abs_pos, ch)
        out.append("  </worldbody>")
        if self._actuators:
            actuator_lines = []
            valid_joint_names = {j['name'] for j in self._joints.values()}
            for a in self._actuators:
                jn = self._joint_name(a['ref'])
                if jn not in valid_joint_names:
                    continue
                mn = self._motor_name(a['ref'], a['name'])
                actuator_lines.append(
                    f'    <motor name="{mn}" joint="{jn}" '
                    f'gear="{a["torque"]:.4g}" ctrllimited="true" ctrlrange="-1 1"/>'
                )
            if actuator_lines:
                out.append("  <actuator>")
                out.extend(actuator_lines)
                out.append("  </actuator>")
        out.append("</mujoco>")
        return out

    def _body_lines(self, name: str, depth: int, abs_pos: dict, ch: dict) -> list[str]:
        b = self._bodies[name]
        pad = "  " * depth
        inn = "  " * (depth + 1)

        parent = self._parents.get(name, 'world')
        my = abs_pos.get(name, [0., 0., 0.])
        pa = abs_pos.get(parent, [0., 0., 0.])
        rel = [my[i] - pa[i] for i in range(3)]
        pos_str = " ".join(f"{v:.6g}" for v in rel)

        j = self._joints.get(name)
        direct_children = ch.get(name, [])
        body_euler = b.get('euler')
        if body_euler is None and j and j['type'] in ('hinge', 'slide') and direct_children:
            # Linked gravity-driven mechanisms should not default to perfect equilibrium.
            # Pick a small initial angle around the declared joint axis.
            body_euler = [25.0 * abs(v) for v in j['axis']]

        body_parts = [f'name="{name}"', f'pos="{pos_str}"']
        if body_euler is not None:
            body_parts.append('euler="' + " ".join(f"{v:.6g}" for v in body_euler) + '"')

        out = [f'{pad}<body {" ".join(body_parts)}>']

        m = b['mass']
        ii = max(m * 0.001, 1e-5)
        out.append(
            f'{inn}<inertial mass="{m:.4g}" pos="0 0 0" '
            f'diaginertia="{ii:.6g} {ii:.6g} {ii:.6g}"/>'
        )

        if b.get('free'):
            out.append(f'{inn}<freejoint/>')

        if j and j['type'] != 'fixed':
            jtype = j['type']
            # ball joints on cylinder/capsule bodies make no physical sense as wheels;
            # treat as hinge around the declared axis so the disc spins correctly
            shape_here = b['shape']
            if jtype == 'ball' and shape_here in ('cylinder', 'capsule') and not b.get('geom_offset'):
                jtype = 'hinge'
            ax = " ".join(f"{v:.6g}" for v in j['axis'])
            parts = [f'name="{j["name"]}"', f'type="{jtype}"', f'axis="{ax}"']
            if j['limits'] is not None:
                lo, hi = j['limits']
                parts.append(f'range="{lo} {hi}"')
            if j['damping']:
                parts.append(f'damping="{j["damping"]:.4g}"')
            out.append(f'{inn}<joint {" ".join(parts)}/>')

        shape = b['shape']
        size_values = list(b['size'])

        inferred_axis = None
        inferred_offset = None
        direct_children = ch.get(name, [])
        j_here = self._joints.get(name)
        if (
            shape in ('cylinder', 'capsule')
            and not b.get('geom_offset')
            and b.get('axis') is None
            and j_here
            and j_here['type'] != 'fixed'
            and direct_children
        ):
            child = direct_children[0]
            child_pos = abs_pos.get(child, my)
            link_vec = [child_pos[i] - my[i] for i in range(3)]
            link_len = _norm(link_vec)
            if link_len > 1e-6:
                inferred_axis = [v / link_len for v in link_vec]
                inferred_offset = [v * 0.5 for v in link_vec]
                size_values[1] = link_len * 0.5

        sz = " ".join(f"{v:.6g}" for v in size_values)
        gp = [f'type="{shape}"', f'size="{sz}"', f'rgba="{_rgba(b.get("color"))}"']

        if b.get('geom_offset') or inferred_offset:
            off = b.get('geom_offset') or inferred_offset
            gp.append(f'pos="{off[0]:.6g} {off[1]:.6g} {off[2]:.6g}"')

        if shape in ('cylinder', 'capsule'):
            body_axis = b.get('axis')  # None if LLM didn't specify
            if body_axis is None and inferred_axis is not None:
                body_axis = inferred_axis
            if body_axis is None:
                if (
                    j_here
                    and j_here['type'] in ('hinge', 'slide', 'ball')
                    and not b.get('geom_offset')
                    and not direct_children
                ):
                    # No geom_offset → body IS the spinning element (wheel); match joint axis
                    body_axis = j_here['axis']
                else:
                    body_axis = [0., 0., 1.]  # vertical rod / arm default
            q = _geom_quat(body_axis)
            if q is not None:
                gp.append(f'quat="{" ".join(f"{v:.6g}" for v in q)}"')

        out.append(f'{inn}<geom {" ".join(gp)}/>')

        for child in ch.get(name, []):
            out += self._body_lines(child, depth + 1, abs_pos, ch)

        out.append(f'{pad}</body>')
        return out

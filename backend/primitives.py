"""
20 MJCF fragment functions — one per primitive.
Each returns a structured dict. Pass a list to assembler.assemble() to get valid MJCF.
"""
import math
from typing import Optional, Union


def _v(val) -> Optional[str]:
    """Convert list/tuple/scalar to space-separated string, or pass through strings."""
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        return " ".join(str(x) for x in val)
    return str(val)


def _p(kind: str, **kwargs) -> dict:
    return {"_kind": kind, **{k: v for k, v in kwargs.items() if v is not None}}


# ============================================================
# Bodies
# ============================================================

def rigid_body(
    name: str,
    pos: Union[str, list] = "0 0 0",
    mass: Optional[float] = None,
    euler: Optional[Union[str, list]] = None,
    parent: str = "world",
    free: bool = False,
) -> dict:
    """Rigid body. parent='world' → direct worldbody child. free=True adds <freejoint/>."""
    return _p("body", name=name, pos=_v(pos), mass=mass, euler=_v(euler),
              parent=parent, free=free or None)


def ground(size: float = 5.0, rgba: str = "0.5 0.5 0.5 1") -> dict:
    """Flat ground plane, placed directly in worldbody."""
    return _p("world_geom", name="ground", geom_type="plane",
              size=f"{size} {size} 0.1", rgba=rgba)


# ============================================================
# Joints  (attach to a body via body= or last-defined-body convention)
# ============================================================

def revolute(
    name: str,
    axis: Union[str, list] = "0 0 1",
    range: str = "-180 180",
    damping: float = 0.5,
    body: Optional[str] = None,
) -> dict:
    """Hinge (revolute) joint — one rotational DOF."""
    return _p("joint", joint_type="hinge", name=name, axis=_v(axis),
              range=range, damping=damping, body=body)


def prismatic(
    name: str,
    axis: Union[str, list] = "1 0 0",
    range: str = "-1 1",
    damping: float = 0.1,
    body: Optional[str] = None,
) -> dict:
    """Slide (prismatic) joint — one translational DOF."""
    return _p("joint", joint_type="slide", name=name, axis=_v(axis),
              range=range, damping=damping, body=body)


def screw(
    name: str,
    axis: Union[str, list] = "0 0 1",
    pitch: float = 0.01,
    body: Optional[str] = None,
) -> dict:
    """
    Screw joint (slide along axis with coupled rotation).
    MuJoCo has no native screw joint; approximated as a slide joint.
    """
    return _p("joint", joint_type="slide", name=name, axis=_v(axis), body=body)


def cylindrical(
    name: str,
    axis: Union[str, list] = "0 0 1",
    body: Optional[str] = None,
) -> list:
    """Cylindrical joint (rotate + slide on same axis). Returns two joint dicts."""
    ax = _v(axis)
    return [
        _p("joint", joint_type="hinge", name=f"{name}_rot", axis=ax, body=body),
        _p("joint", joint_type="slide", name=f"{name}_slide", axis=ax, body=body),
    ]


def spherical(name: str, body: Optional[str] = None) -> dict:
    """Ball (spherical) joint — three rotational DOFs."""
    return _p("joint", joint_type="ball", name=name, body=body)


def planar(name: str, body: Optional[str] = None) -> list:
    """Planar joint (2 translational + 1 rotational). Returns three joint dicts."""
    return [
        _p("joint", joint_type="slide", name=f"{name}_x", axis="1 0 0", body=body),
        _p("joint", joint_type="slide", name=f"{name}_y", axis="0 1 0", body=body),
        _p("joint", joint_type="hinge", name=f"{name}_z", axis="0 0 1", body=body),
    ]


def fixed(name: str, body: Optional[str] = None) -> dict:
    """Fixed (weld) constraint — in MuJoCo this means adding no joint to the body."""
    return _p("fixed_marker", name=name, body=body)


# ============================================================
# Force elements
# ============================================================

def actuator(
    name: str,
    joint: str,
    gear: float = 1.0,
    ctrllimited: bool = False,
    ctrlrange: Optional[str] = None,
) -> dict:
    """Motor actuator driving a named joint."""
    return _p("actuator", name=name, joint=joint, gear=gear,
              ctrllimited=ctrllimited or None, ctrlrange=ctrlrange)


def spring(
    name: str,
    joint: str,
    stiffness: float = 100.0,
    rest_length: float = 0.0,
) -> dict:
    """Spring on a joint — injects stiffness into the joint element."""
    return _p("joint_spring", name=name, joint=joint,
              stiffness=stiffness, rest_length=rest_length or None)


def damper(name: str, joint: str, damping: float = 10.0) -> dict:
    """Damper on a joint — augments the joint's damping attribute."""
    return _p("joint_damper", name=name, joint=joint, damping=damping)


def gravity(direction: str = "0 0 -1", magnitude: float = 9.81) -> dict:
    """Set the gravity vector."""
    parts = [float(x) for x in str(direction).split()]
    norm = math.sqrt(sum(p ** 2 for p in parts)) or 1.0
    scaled = [p / norm * magnitude for p in parts]
    return _p("option_gravity", gravity=" ".join(f"{x:.6f}" for x in scaled))


def contact_pair(body1: str, body2: str) -> dict:
    """Explicit contact pair between two named bodies."""
    return _p("contact_pair", body1=body1, body2=body2)


# ============================================================
# Geometry  (attach to a body via body= or last-defined-body convention)
# ============================================================

def box_geom(
    size: Union[str, list, float] = 0.1,
    pos: Union[str, list] = "0 0 0",
    rgba: str = "0.7 0.7 0.7 1",
    density: float = 1000,
    name: Optional[str] = None,
    body: Optional[str] = None,
) -> dict:
    """Box geometry. size = [hx, hy, hz] half-extents."""
    if isinstance(size, (list, tuple)):
        sz = " ".join(str(x) for x in size)
    elif isinstance(size, (int, float)):
        sz = f"{size} {size} {size}"
    else:
        sz = str(size)
    return _p("geom", geom_type="box", size=sz, pos=_v(pos),
              rgba=rgba, density=density, name=name, body=body)


def cylinder_geom(
    radius: float,
    length: float,
    pos: Union[str, list] = "0 0 0",
    rgba: str = "0.7 0.7 0.7 1",
    density: float = 1000,
    name: Optional[str] = None,
    body: Optional[str] = None,
) -> dict:
    """Cylinder geometry. MuJoCo size = radius + half_length."""
    return _p("geom", geom_type="cylinder", size=f"{radius} {length / 2:.4f}",
              pos=_v(pos), rgba=rgba, density=density, name=name, body=body)


def sphere_geom(
    radius: float,
    pos: Union[str, list] = "0 0 0",
    rgba: str = "0.7 0.7 0.7 1",
    density: float = 1000,
    name: Optional[str] = None,
    body: Optional[str] = None,
) -> dict:
    """Sphere geometry."""
    return _p("geom", geom_type="sphere", size=str(radius), pos=_v(pos),
              rgba=rgba, density=density, name=name, body=body)


def capsule_geom(
    radius: float,
    length: float,
    pos: Union[str, list] = "0 0 0",
    rgba: str = "0.7 0.7 0.7 1",
    density: float = 1000,
    name: Optional[str] = None,
    body: Optional[str] = None,
) -> dict:
    """Capsule geometry. MuJoCo size = radius + half_length."""
    return _p("geom", geom_type="capsule", size=f"{radius} {length / 2:.4f}",
              pos=_v(pos), rgba=rgba, density=density, name=name, body=body)


def plane_geom(
    size: float = 5.0,
    rgba: str = "0.5 0.5 0.5 1",
    name: Optional[str] = None,
    body: Optional[str] = None,
) -> dict:
    """Infinite ground plane. Works best as a world_geom (use ground() instead)."""
    return _p("geom", geom_type="plane", size=f"{size} {size} 0.1",
              rgba=rgba, name=name, body=body)


def mesh_geom(
    file: str,
    pos: Union[str, list] = "0 0 0",
    rgba: str = "0.7 0.7 0.7 1",
    name: Optional[str] = None,
    body: Optional[str] = None,
) -> dict:
    """Mesh geometry loaded from a file path."""
    return _p("geom", geom_type="mesh", mesh=file, pos=_v(pos),
              rgba=rgba, name=name, body=body)

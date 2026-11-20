"""LEGO bricks for Drake: SDF generation, brick metadata, and grid placement.

Rigid-body simulation cannot model the press fit that holds real LEGO
together. With collision="interlock" (default) a brick is a hollow shell with
studs and underside tubes, so a seated brick is locked sideways by geometry
but held down only by gravity; "box" is a single box held by friction.

Brick frame B: origin at the center of the bottom face, +z up, +x along the
long side (nx studs), +y along the short side (ny studs).
"""

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

from physical.lego import specifications as lego
from physical.lego.specifications import (
    ABS_DENSITY,
    BRICK_H_MM,
    COLORS,
    FILL_FRACTION,
    PITCH_MM,
    PLATE_H_MM,
    SIDE_GAP_MM,
    STUD_D_MM,
    STUD_H_MM,
    TOP_THICK_MM,
    SimParams,
)

PRESETS = {"2x2": (2, 2), "2x4": (4, 2)}


# ----------------------------------------------------------------------------
# Specs
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class BrickSpec:
    """What kind of brick. nx = studs along x (length), ny = studs along y."""
    nx: int
    ny: int
    color: str = "red"

    @staticmethod
    def from_string(kind: str, color: str = "red") -> "BrickSpec":
        """BrickSpec.from_string('2x4', 'blue'). Accepts 'WxL' in either order;
        the longer side is always put along x."""
        a, b = (int(v) for v in kind.lower().split("x"))
        return BrickSpec(nx=max(a, b), ny=min(a, b), color=color)

    @property
    def kind(self) -> str:
        return f"{self.ny}x{self.nx}"

    def default_name(self) -> str:
        return f"lego_{self.kind}_{self.color}"


@dataclass
class BrickInfo:
    """Metadata the planners needs (all SI, body frame B)."""
    name: str
    kind: str
    nx: int
    ny: int
    color: str
    rgba: Tuple[float, float, float, float]
    scale: float
    collision: str
    pitch: float                 # stud spacing (m)
    length: float                # outer size along x (m)
    width: float                 # outer size along y (m)
    height: float                # body height without studs (m) = stacking increment
    stud_height: float
    stud_radius: float
    mass: float
    inertia_diag: Tuple[float, float, float]
    com: Tuple[float, float, float]
    stud_positions: List[Tuple[float, float, float]] = field(default_factory=list)  # tops of studs
    # Suggested top-down grasp: fingers close along y (short side) at the COM height.
    grasp_width: float = 0.0
    grasp_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------
def _mm(v: float, scale: float) -> float:
    return v * scale * 1e-3


def _stud_xy(nx: int, ny: int, pitch: float) -> List[Tuple[float, float]]:
    return [((i - (nx - 1) / 2.0) * pitch, (j - (ny - 1) / 2.0) * pitch)
            for i in range(nx) for j in range(ny)]


def _tube_xy(nx: int, ny: int, pitch: float) -> List[Tuple[float, float]]:
    """Underside tubes sit at the interior lattice points between 4 studs."""
    if nx < 2 or ny < 2:
        return []
    return [((i - (nx - 2) / 2.0) * pitch, (j - (ny - 2) / 2.0) * pitch)
            for i in range(nx - 1) for j in range(ny - 1)]


def brick_info(spec: BrickSpec, params: SimParams = SimParams(),
               name: Optional[str] = None) -> BrickInfo:
    s = params.scale
    pitch = _mm(PITCH_MM, s)
    L = _mm(spec.nx * PITCH_MM - 2 * SIDE_GAP_MM, s)
    W = _mm(spec.ny * PITCH_MM - 2 * SIDE_GAP_MM, s)
    H = _mm(BRICK_H_MM, s)
    stud_r = _mm(STUD_D_MM / 2, s)
    stud_h = _mm(STUD_H_MM, s)

    if params.mass_override is not None:
        mass = params.mass_override
    else:
        mass = FILL_FRACTION * ABS_DENSITY * L * W * H
    # Solid-box inertia about the COM. The real shell has slightly larger
    # inertia, but this is well within what matters for manipulation.
    Ixx = mass / 12.0 * (W**2 + H**2)
    Iyy = mass / 12.0 * (L**2 + H**2)
    Izz = mass / 12.0 * (L**2 + W**2)
    # A hollow-bottom brick's COM sits above mid-height; approximate it.
    com_z = 0.5 * H if params.collision == "box" else 0.58 * H

    studs = [(x, y, H + stud_h) for x, y in _stud_xy(spec.nx, spec.ny, pitch)]
    return BrickInfo(
        name=name or spec.default_name(), kind=spec.kind, nx=spec.nx, ny=spec.ny,
        color=spec.color, rgba=_rgba(spec.color), scale=s, collision=params.collision,
        pitch=pitch, length=L, width=W, height=H, stud_height=stud_h,
        stud_radius=stud_r, mass=mass, inertia_diag=(Ixx, Iyy, Izz),
        com=(0.0, 0.0, com_z), stud_positions=studs,
        grasp_width=W, grasp_point=(0.0, 0.0, com_z),
    )


def _rgba(color) -> Tuple[float, float, float, float]:
    if isinstance(color, str):
        if color not in COLORS:
            raise ValueError(f"Unknown color '{color}'. Options: {list(COLORS)}")
        return COLORS[color]
    return tuple(color)  # already RGBA


def _primitives(spec: BrickSpec, params: SimParams):
    """Yields (name, kind, pose_xyz, dims) for the collision geometry.
    kind 'box' -> dims=(sx, sy, sz); kind 'cyl' -> dims=(radius, length)."""
    s = params.scale
    info = brick_info(spec, params)
    L, W, H = info.length, info.width, info.height
    pitch, stud_r, stud_h = info.pitch, info.stud_radius, info.stud_height
    c = _mm(params.clearance_mm, s)

    prims = []
    if params.collision == "box":
        prims.append(("body", "box", (0, 0, H / 2), (L, W, H)))
    elif params.collision == "interlock":
        top_t = _mm(TOP_THICK_MM, s)
        # Wall thickness so the inner face sits just outside the stud reach.
        stud_reach_y = (spec.ny - 1) / 2.0 * pitch + stud_r
        wall = W / 2 - stud_reach_y - c
        if wall <= 0:
            raise ValueError("Clearance too large for wall thickness.")
        cav_h = H - top_t
        prims.append(("top", "box", (0, 0, H - top_t / 2), (L, W, top_t)))
        prims.append(("wall_px", "box", (L / 2 - wall / 2, 0, cav_h / 2), (wall, W, cav_h)))
        prims.append(("wall_nx", "box", (-L / 2 + wall / 2, 0, cav_h / 2), (wall, W, cav_h)))
        inner_L = L - 2 * wall
        prims.append(("wall_py", "box", (0, W / 2 - wall / 2, cav_h / 2), (inner_L, wall, cav_h)))
        prims.append(("wall_ny", "box", (0, -W / 2 + wall / 2, cav_h / 2), (inner_L, wall, cav_h)))
        # Solid tubes touching the 4 surrounding studs (minus clearance).
        tube_r = math.sqrt(2) / 2 * pitch - stud_r - c
        for k, (x, y) in enumerate(_tube_xy(spec.nx, spec.ny, pitch)):
            prims.append((f"tube_{k}", "cyl", (x, y, cav_h / 2), (tube_r, cav_h)))
    else:
        raise ValueError(f"collision must be 'interlock' or 'box', got {params.collision}")

    studs = [(f"stud_{k}", "cyl", (x, y, H + stud_h / 2), (stud_r, stud_h))
             for k, (x, y) in enumerate(_stud_xy(spec.nx, spec.ny, pitch))]
    return prims, studs


# ----------------------------------------------------------------------------
# SDF generation
# ----------------------------------------------------------------------------
def _geom_xml(kind: str, dims) -> str:
    if kind == "box":
        return f"<box><size>{dims[0]:.6g} {dims[1]:.6g} {dims[2]:.6g}</size></box>"
    return (f"<cylinder><radius>{dims[0]:.6g}</radius>"
            f"<length>{dims[1]:.6g}</length></cylinder>")


def _proximity_xml(params: SimParams, info: BrickInfo) -> str:
    lines = [
        f"<drake:mu_static>{params.mu_static}</drake:mu_static>",
        f"<drake:mu_dynamic>{params.mu_dynamic}</drake:mu_dynamic>",
        f"<drake:hunt_crossley_dissipation>{params.hunt_crossley_dissipation}"
        f"</drake:hunt_crossley_dissipation>",
    ]
    if params.contact_model == "hydroelastic":
        lines += [
            "<drake:compliant_hydroelastic/>",
            f"<drake:hydroelastic_modulus>{params.hydroelastic_modulus:.6g}"
            f"</drake:hydroelastic_modulus>",
            # Needed for cylinders; ~1/4 stud radius keeps meshes small.
            f"<drake:mesh_resolution_hint>{info.stud_radius / 2:.6g}"
            f"</drake:mesh_resolution_hint>",
        ]
    elif params.contact_model != "point":
        raise ValueError("contact_model must be 'point' or 'hydroelastic'")
    return "<drake:proximity_properties>" + "".join(lines) + "</drake:proximity_properties>"


def generate_brick_sdf(spec: BrickSpec, params: SimParams = SimParams(),
                       name: Optional[str] = None) -> str:
    """Return an SDFormat string for one brick model (one link named 'brick')."""
    info = brick_info(spec, params, name)
    prims, studs = _primitives(spec, params)
    r, g, b, a = info.rgba
    material = (f"<material><diffuse>{r} {g} {b} {a}</diffuse>"
                f"<ambient>{r} {g} {b} {a}</ambient></material>")
    prox = _proximity_xml(params, info)

    out = [
        '<?xml version="1.0"?>',
        '<sdf version="1.9" xmlns:drake="http://drake.mit.edu">',
        f'  <model name="{info.name}">',
        '    <link name="brick">',
        '      <inertial>',
        f'        <pose>{info.com[0]} {info.com[1]} {info.com[2]:.6g} 0 0 0</pose>',
        f'        <mass>{info.mass:.6g}</mass>',
        '        <inertia>',
        f'          <ixx>{info.inertia_diag[0]:.6g}</ixx><iyy>{info.inertia_diag[1]:.6g}</iyy>'
        f'<izz>{info.inertia_diag[2]:.6g}</izz>',
        '          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>',
        '        </inertia>',
        '      </inertial>',
    ]

    # Visuals: in interlock mode the visual == collision geometry (looks like
    # a real brick from below); in box mode we draw the box + studs.
    for pname, kind, (x, y, z), dims in prims + studs:
        out.append(f'      <visual name="v_{pname}"><pose>{x:.6g} {y:.6g} {z:.6g} 0 0 0</pose>'
                   f'<geometry>{_geom_xml(kind, dims)}</geometry>{material}</visual>')

    # Collisions: studs only collide in interlock mode.
    coll = prims + (studs if params.collision == "interlock" else [])
    for pname, kind, (x, y, z), dims in coll:
        out.append(f'      <collision name="c_{pname}"><pose>{x:.6g} {y:.6g} {z:.6g} 0 0 0</pose>'
                   f'<geometry>{_geom_xml(kind, dims)}</geometry>{prox}</collision>')

    out += ['    </link>', '  </model>', '</sdf>', '']
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Drake helpers
# ----------------------------------------------------------------------------
def add_brick(plant, spec: BrickSpec, name: Optional[str] = None,
              params: SimParams = SimParams(), parser=None):
    """Parse a brick into `plant` and return its ModelInstanceIndex.
    `name` must be unique in the plant (e.g. 'brick_7')."""
    from pydrake.multibody.parsing import Parser
    parser = parser or Parser(plant)
    sdf = generate_brick_sdf(spec, params, name=name or spec.default_name())
    (instance,) = parser.AddModelsFromString(sdf, "sdf")
    return instance


def grid_pose(info: BrickInfo, gx: int, gy: int, level: int, yaw_deg: int = 0,
              origin=(0.0, 0.0, 0.0)):
    """Pose X_WB of a brick placed on the stud grid.

    The brick covers stud cells [gx, gx+nx') x [gy, gy+ny') at layer `level`
    (0 = resting on the baseplate surface at origin z), where (nx', ny') are
    (nx, ny) rotated by `yaw_deg` (must be a multiple of 90).
    """
    from pydrake.math import RigidTransform, RollPitchYaw
    if yaw_deg % 90 != 0:
        raise ValueError("yaw_deg must be a multiple of 90")
    rotated = (yaw_deg // 90) % 2 == 1
    nx, ny = (info.ny, info.nx) if rotated else (info.nx, info.ny)
    x = origin[0] + (gx + nx / 2.0) * info.pitch
    y = origin[1] + (gy + ny / 2.0) * info.pitch
    z = origin[2] + level * info.height
    return RigidTransform(RollPitchYaw(0, 0, math.radians(yaw_deg)), [x, y, z])


def generate_baseplate_sdf(nx: int = 16, ny: int = 16, params: SimParams = SimParams(),
                           color: str = "green", name: str = "baseplate",
                           thickness_mm: float = PLATE_H_MM) -> str:
    """Static baseplate with collision studs, top surface at z=0 of its frame
    (so bricks at grid_pose(level=0) sit on it). Weld it to the world."""
    s = params.scale
    pitch = _mm(PITCH_MM, s)
    t = _mm(thickness_mm, s)
    stud_r = _mm(STUD_D_MM / 2, s)
    stud_h = _mm(STUD_H_MM, s)
    L, W = nx * pitch, ny * pitch
    r, g, b, a = _rgba(color)
    material = f"<material><diffuse>{r} {g} {b} {a}</diffuse></material>"
    info_stub = brick_info(BrickSpec(1, 1), params)
    prox = _proximity_xml(params, info_stub)
    collide_studs = params.collision == "interlock"
    out = [
        '<?xml version="1.0"?>',
        '<sdf version="1.9" xmlns:drake="http://drake.mit.edu">',
        f'  <model name="{name}"><link name="plate">',
        f'    <visual name="v_plate"><pose>{L/2:.6g} {W/2:.6g} {-t/2:.6g} 0 0 0</pose>'
        f'<geometry>{_geom_xml("box", (L, W, t))}</geometry>{material}</visual>',
        f'    <collision name="c_plate"><pose>{L/2:.6g} {W/2:.6g} {-t/2:.6g} 0 0 0</pose>'
        f'<geometry>{_geom_xml("box", (L, W, t))}</geometry>{prox}</collision>',
    ]
    for i in range(nx):
        for j in range(ny):
            x, y = (i + 0.5) * pitch, (j + 0.5) * pitch
            pose = f'<pose>{x:.6g} {y:.6g} {stud_h/2:.6g} 0 0 0</pose>'
            geom = f'<geometry>{_geom_xml("cyl", (stud_r, stud_h))}</geometry>'
            out.append(f'    <visual name="v_s{i}_{j}">{pose}{geom}{material}</visual>')
            if collide_studs:
                out.append(f'    <collision name="c_s{i}_{j}">{pose}{geom}{prox}</collision>')
    out += ['  </link></model>', '</sdf>', '']
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Placement on the board and the supply plate
# ----------------------------------------------------------------------------
def all_preset_specs(colors: Iterable[str] = COLORS) -> List[BrickSpec]:
    return [BrickSpec(nx, ny, c) for (nx, ny) in PRESETS.values() for c in colors]


def world_pose(W, L, x, y, rotation, level=0, origin=lego.BOARD_ORIGIN):
    """World pose of a seated W x L brick whose corner is at stud cell (x, y)."""
    info = brick_info(BrickSpec(nx=W, ny=L), lego.PARAMS)
    return grid_pose(info, x, y, level, yaw_deg=90 * rotation, origin=origin)


def supply_cells(final_configuration):
    """Stud cell (x, y) on the supply plate for each brick, packed in rows in order."""
    cells, x, y, row_depth = [], 0, 0, 0
    for W, L, *_ in final_configuration:
        if x + W > lego.SUPPLY_STUDS[0]:
            x, y, row_depth = 0, y + row_depth + lego.SUPPLY_GAP, 0
        if y + L > lego.SUPPLY_STUDS[1]:
            raise ValueError("too many bricks for the supply plate")
        cells.append((x, y))
        x += W + lego.SUPPLY_GAP
        row_depth = max(row_depth, L)
    return cells


def _cached(name, sdf):
    """Write an SDF into the cache and return the file:// URL a directive can load."""
    lego.GENERATED.mkdir(parents=True, exist_ok=True)
    path = lego.GENERATED / f"{name}.sdf"
    path.write_text(sdf)
    return path.as_uri()


def brick_url(W, L, color):
    name = f"lego_{W}x{L}_{color}"
    return _cached(name, generate_brick_sdf(BrickSpec(W, L, color), lego.PARAMS, name=name))


def plate_url(name, studs, color):
    return _cached(name, generate_baseplate_sdf(*studs, params=lego.PARAMS, color=color, name=name))

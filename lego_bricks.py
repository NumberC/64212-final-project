"""
lego_bricks.py — procedural LEGO brick model generator for Drake.

Generates SDFormat (SDF 1.9) models for N x M LEGO bricks at a configurable
scale, plus metadata (dimensions, stud positions, mass, grasp width) that the
planner / config interpreter can use.

------------------------------------------------------------------------------
Stacking mechanics (why not press-fit?)
------------------------------------------------------------------------------
Real LEGO holds together through elastic deformation of the stud/tube
interface (an interference fit). Rigid-body simulators — Drake, Gazebo,
PyBullet, MuJoCo — cannot model that; even the CMU LEGO group (Liu et al.,
StableLego, arXiv:2402.10711) states existing simulators "are not able to
simulate the interlocking mechanism of Lego". The related work we cite
(arXiv:2309.11010) only uses Gazebo for visual verification, not physics.

We therefore offer two rigid approximations:

  collision="interlock"  (default)
      Hollow shell (top plate + 4 side walls), collision studs on top, and
      solid underside tubes, all built from Drake primitives. The geometry
      matches real LEGO proportions except for a small `clearance`, so a
      seated brick is *laterally* locked by stud/wall/tube contact but held
      *vertically* only by gravity (it can be lifted off with zero force).
      Misaligned placements land on top of studs, so placement accuracy
      matters — that's a feature for comparing controllers.

  collision="box"
      One collision box (+ visual-only studs). Fast and very stable;
      bricks are held only by friction. Good for debugging the pipeline.

If you later need bricks that resist pull-off, activate a Drake weld
constraint between bricks at placement time (MultibodyPlant.AddWeldConstraint
+ SetConstraintActiveStatus, SAP solver) — that is a simulator-side choice
and doesn't change these models.

------------------------------------------------------------------------------
Conventions
------------------------------------------------------------------------------
* Units: SI (meters, kg). Real LEGO dimensions are in mm and multiplied by
  `scale` (default 2.0 -> 16 mm stud pitch, DUPLO-sized).
* Brick body frame B: origin at the center of the BOTTOM face, +z up,
  +x along the long side (length = nx studs), +y along the short side
  (width = ny studs). So a "2x4" brick is BrickSpec(nx=4, ny=2).
  Stacking brick k on brick j that sits at z=z_j gives z_k = z_j + height.
* Grid coordinates: a brick at grid cell (gx, gy, level) with yaw in
  {0, 90, 180, 270} deg occupies stud cells [gx, gx+nx) x [gy, gy+ny)
  (nx/ny swapped for 90/270). See `grid_pose()`.

Usage
-----
    from lego_bricks import BrickSpec, generate_brick_sdf, add_brick

    spec = BrickSpec(nx=4, ny=2, color="red")
    sdf = generate_brick_sdf(spec)                 # SDF string
    model = add_brick(plant, spec, name="brick_0") # parse into a plant

    # or write files + metadata for the config loader:
    write_brick_library("models/", [BrickSpec(4, 2, "red"), BrickSpec(2, 2, "blue")])
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

# ----------------------------------------------------------------------------
# Real LEGO dimensions (mm). Source: standard LEGO brick specification.
# ----------------------------------------------------------------------------
PITCH_MM = 8.0            # stud-to-stud spacing
SIDE_GAP_MM = 0.1         # each outer face is inset 0.1 mm from the grid
BRICK_H_MM = 9.6          # brick height (without stud)
PLATE_H_MM = 3.2          # plate height (1/3 brick)
STUD_D_MM = 4.8           # stud diameter
STUD_H_MM = 1.7           # stud height
TOP_THICK_MM = 1.0        # thickness of the top deck
# Outer wall thickness is derived so that a stud just touches the inner wall
# (half-width 3.9 - stud reach 2.4 = 1.5 mm); clearance is then removed.
ABS_DENSITY = 1050.0      # kg/m^3
# Real 2x4 brick weighs ~2.3 g, i.e. ~46% of its solid envelope.
FILL_FRACTION = 0.46

# Colors (RGBA, 0..1). Distinct hues to make the classifier's job tractable.
COLORS: Dict[str, Tuple[float, float, float, float]] = {
    "red":    (0.80, 0.07, 0.07, 1.0),
    "blue":   (0.05, 0.30, 0.75, 1.0),
    "yellow": (0.98, 0.80, 0.05, 1.0),
    "green":  (0.10, 0.55, 0.20, 1.0),
    "white":  (0.95, 0.95, 0.95, 1.0),
    "black":  (0.08, 0.08, 0.08, 1.0),
    "orange": (0.95, 0.45, 0.05, 1.0),
    "purple": (0.45, 0.20, 0.60, 1.0),
}

# Presets requested for the project.
PRESETS = {
    "2x2": (2, 2),
    "2x4": (4, 2),
}


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


@dataclass(frozen=True)
class SimParams:
    """How the brick is simulated (shared by every brick in a scene)."""
    scale: float = 2.0
    collision: str = "interlock"        # "interlock" | "box"
    clearance_mm: float = 0.15          # real-scale mm, removed from walls/tubes
    mu_static: float = 0.6              # ABS-on-ABS is ~0.3-0.5; a bit higher helps grasping
    mu_dynamic: float = 0.5
    contact_model: str = "point"        # "point" | "hydroelastic"
    hydroelastic_modulus: float = 5.0e7 # Pa, only used for hydroelastic
    hunt_crossley_dissipation: float = 1.0
    mass_override: Optional[float] = None  # kg; None = derived from size


@dataclass
class BrickInfo:
    """Metadata the planner / config loader needs (all SI, body frame B)."""
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
# File output
# ----------------------------------------------------------------------------
def write_brick_library(out_dir: str, specs: Iterable[BrickSpec],
                        params: SimParams = SimParams(),
                        baseplate: Optional[Tuple[int, int]] = (16, 16)) -> Dict[str, BrickInfo]:
    """Write one .sdf per spec plus `bricks.json` metadata into out_dir.
    Returns {model_name: BrickInfo}."""
    os.makedirs(out_dir, exist_ok=True)
    infos = {}
    for spec in specs:
        info = brick_info(spec, params)
        with open(os.path.join(out_dir, f"{info.name}.sdf"), "w") as f:
            f.write(generate_brick_sdf(spec, params))
        infos[info.name] = info
    if baseplate:
        with open(os.path.join(out_dir, "baseplate.sdf"), "w") as f:
            f.write(generate_baseplate_sdf(*baseplate, params=params))
    meta = {"sim_params": asdict(params),
            "bricks": {k: asdict(v) for k, v in infos.items()}}
    with open(os.path.join(out_dir, "bricks.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return infos


def all_preset_specs(colors: Iterable[str] = COLORS) -> List[BrickSpec]:
    return [BrickSpec(nx, ny, c) for (nx, ny) in PRESETS.values() for c in colors]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate LEGO brick SDF models.")
    ap.add_argument("--out", default="models/lego")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--collision", choices=["interlock", "box"], default="interlock")
    ap.add_argument("--contact", choices=["point", "hydroelastic"], default="point")
    ap.add_argument("--colors", nargs="+", default=list(COLORS))
    a = ap.parse_args()
    p = SimParams(scale=a.scale, collision=a.collision, contact_model=a.contact)
    infos = write_brick_library(a.out, all_preset_specs(a.colors), p)
    print(f"Wrote {len(infos)} bricks + baseplate to {a.out}/")

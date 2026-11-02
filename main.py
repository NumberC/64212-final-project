# config file and interpretor
"""
The config file specifies what block (what size) should be placed where and in what orientation.
If you are looking at the lego board, a 2x4 (WxL) block is the equivalent of a 4x2 block rotated 90 degrees.
To reduce ambiguity, W will always be >= L. so if you are placing a 2x4, you would say you are rotating a 4x2.
There are only 2 rotations that are possible. 0 or 90 degrees. The lego platform is a grid, so only multiples of 90 are allowed.
Due to legos being rectangles there is no difference from 90 and 270 or 0 and 180.

This means the format is:
W, L, x, y, rotation(only 2 options, 0, 90; 0 for 0 and 1 for 90)[, level[, color]]

Optional extra columns (old 5-column files still work):
  level  stacking layer, 0 = sitting on the baseplate (default 0)
  color  one of lego_bricks.COLORS (default: cycles through the colors)
Blank lines and lines starting with '#' are ignored.

Units: x, y are stud cells on the baseplate. A brick of size W x L at rotation 0
covers cells [x, x+W) along the board's x axis and [y, y+L) along its y axis;
at rotation 1 it covers [x, x+L) x [y, y+W).

The pivot point for location is the bottom-left corner regardless of vertical or horizontal placemntn 

Legos are placed out in the order they appear in the file. So line 1 will be placed before line 2. A series of lines will build our structure.
"""

import csv
import sys
from pathlib import Path

import lego_bricks as lb

from manipulation.station import (
    AppendDirectives,
    LoadScenario,
    MakeHardwareStation,
    MakeMultibodyPlant,
    Scenario,
)
from pydrake.all import (
    AddFrameTriadIllustration,
    ConstantVectorSource,
    DiagramBuilder,
    Integrator,
    InverseKinematics,
    JacobianWrtVariable,
    PiecewisePolynomial,
    PiecewisePose,
    RigidTransform,
    RotationMatrix,
    Simulator,
    Solve,
    TrajectorySource,
)
from utils.viz import HEADLESS, get_meshcat, keep_meshcat_open


# classic IIWA with gripper
ROBOT = """directives:
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf
    default_joint_positions:
        iiwa_joint_1: [0]
        iiwa_joint_2: [0.4]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.6]
        iiwa_joint_5: [0]
        iiwa_joint_6: [1.2]
        iiwa_joint_7: [0]
- add_weld:
    parent: world
    child: iiwa::iiwa_link_0
- add_model:
    name: wsg
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy { deg: [90, 0, 90] }

model_drivers:
    iiwa: !IiwaDriver
      control_mode: position_only
      hand_model_name: wsg
    wsg: !SchunkWsgDriver {}
"""

# ---------------------------------------------------------------------------
# LEGO models (geometry comes from lego_bricks.py)
# ---------------------------------------------------------------------------
# Generated .sdf files go here; directives point at them with file:// URLs.
ASSET_DIR = Path(__file__).resolve().parent / "models" / "generated"

# Shared brick settings: scale 2.0 (16 mm stud pitch), "interlock" collision
# geometry (studs + hollow underside so seated bricks can't slide sideways).
LEGO_PARAMS = lb.SimParams()

# Baseplate size in studs, and where its (0, 0) corner sits in the world.
# Its top surface is at BOARD_ORIGIN z. The default puts the board's center
# about 0.6 m in front of the iiwa base; raise z if you add a table.
BOARD_STUDS = (16, 16)
_board_len = BOARD_STUDS[0] * lb.PITCH_MM * LEGO_PARAMS.scale * 1e-3
_board_wid = BOARD_STUDS[1] * lb.PITCH_MM * LEGO_PARAMS.scale * 1e-3
BOARD_ORIGIN = [0.6 - _board_len / 2, -_board_wid / 2, 0.0]

# Bricks spawn this far above their seated height so nothing starts in
# penetration; gravity settles them onto the studs.
SPAWN_CLEARANCE = 0.001

LEGO_COLORS = list(lb.COLORS)


def writeLegoSDF(W, L, color="red"):
    """SDF string for one W x L brick (W studs along x, L along y at rotation 0)."""
    spec = lb.BrickSpec(nx=W, ny=L, color=color)
    return lb.generate_brick_sdf(spec, LEGO_PARAMS, name=f"lego_{W}x{L}_{color}")


def _lego_sdf_url(W, L, color):
    """Writes the brick's SDF to ASSET_DIR (once per size/color) and returns
    the file:// URL a directive can load."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSET_DIR / f"lego_{W}x{L}_{color}.sdf"
    path.write_text(writeLegoSDF(W, L, color))
    return path.as_uri()


def lego_world_pose(W, L, x, y, rotation, level=0):
    """World pose (RigidTransform) of a seated brick's body frame (center of its bottom face)."""
    info = lb.brick_info(lb.BrickSpec(nx=W, ny=L), LEGO_PARAMS)
    return lb.grid_pose(info, x, y, level, yaw_deg=90 * rotation, origin=BOARD_ORIGIN)


def generateLegoDirective(W, L, x, y, rotation, level=0, color="red", name=None):
    """Directive YAML that adds one free-floating brick at a board position.

    The brick is NOT welded: it is a free body that starts just above the
    given cell and settles under gravity, so the robot can pick it up later.
    `name` must be unique within the scenario (e.g. "lego_3").
    """
    name = name or f"lego_{W}x{L}_{x}_{y}_{level}"
    X = lego_world_pose(W, L, x, y, rotation, level)
    tx, ty, tz = X.translation()
    tz += SPAWN_CLEARANCE * (level + 1)
    return f"""directives:
- add_model:
    name: {name}
    file: {_lego_sdf_url(W, L, color)}
    default_free_body_pose:
        brick:
            translation: [{tx:.6f}, {ty:.6f}, {tz:.6f}]
            rotation: !Rpy {{ deg: [0, 0, {90 * rotation}] }}
"""


def generateBaseplateDirective(color="green"):
    """Directive YAML that adds the studded baseplate, welded to the world."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSET_DIR / "baseplate.sdf"
    path.write_text(lb.generate_baseplate_sdf(*BOARD_STUDS, params=LEGO_PARAMS, color=color))
    bx, by, bz = BOARD_ORIGIN
    return f"""directives:
- add_model:
    name: baseplate
    file: {path.as_uri()}
- add_weld:
    parent: world
    child: baseplate::plate
    X_PC:
        translation: [{bx:.6f}, {by:.6f}, {bz:.6f}]
"""

def config_interpretor(file_path: str):
    """Returns the legos to place, in order, as tuples
    (W, L, x, y, rotation, level, color)."""
    legos = []
    with open(file_path, "r") as f:
        for row in csv.reader(f):
            row = [c.strip() for c in row]
            if not row or not row[0] or row[0].startswith("#"):
                continue
            W, L, x, y, rot = map(int, row[:5])
            if W < L:
                raise ValueError(f"{file_path}: W must be >= L, got {W}x{L}")
            if rot not in (0, 1):
                raise ValueError(f"{file_path}: rotation must be 0 or 1, got {rot}")
            level = int(row[5]) if len(row) > 5 and row[5] else 0
            color = row[6] if len(row) > 6 and row[6] else LEGO_COLORS[len(legos) % len(LEGO_COLORS)]
            legos.append((W, L, x, y, rot, level, color))
    return legos

def make_scenario() -> Scenario:
    """The robot"""
    # ASSET_DIR.mkdir(parents=True, exist_ok=True)
    scenario = LoadScenario(data=ROBOT)
    return scenario


# helpers cut down a bit from PS2
def build_static_system(scenario):
    """The scene, with the arm parked at the scenario's own pose."""
    builder = DiagramBuilder()
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat=get_meshcat()))
    plant = station.GetSubsystemByName("plant")
    # The station's command inputs must be connected or the simulation errors
    # out: hold the iiwa at its default joint positions and the WSG open.
    q0 = plant.GetPositions(plant.CreateDefaultContext(), plant.GetModelInstanceByName("iiwa"))
    hold_arm = builder.AddSystem(ConstantVectorSource(q0))
    builder.Connect(hold_arm.get_output_port(), station.GetInputPort("iiwa.position"))
    open_wsg = builder.AddSystem(ConstantVectorSource([0.1]))
    builder.Connect(open_wsg.get_output_port(), station.GetInputPort("wsg.position"))
    return builder.Build()

def show_scene(system, duration: float = 2.0):
    """Simulate a system in real time, so it can be watched in Meshcat."""
    simulator = Simulator(system)
    if not HEADLESS:
        simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(duration)

def main():
    mainScenario = make_scenario()
    # build scenario with legos
    config = sys.argv[1] if len(sys.argv) > 1 else "./sample_configs/4by4Square.csv"
    legosToBePlaced = config_interpretor(config)
    mainScenario = AppendDirectives(mainScenario, data=generateBaseplateDirective())
    for i, (W, L, x, y, is90Deg, level, color) in enumerate(legosToBePlaced):
        mainScenario = AppendDirectives(
            mainScenario,
            data=generateLegoDirective(W, L, x, y, is90Deg, level, color, name=f"lego_{i}"),
        )

    show_scene(build_static_system(mainScenario))


if __name__ == "__main__":
    main()
    keep_meshcat_open()
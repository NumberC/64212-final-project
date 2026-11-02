# config file and interpretor
"""
The config file specifies what block (what size) should be placed where and in what orientation.
If you are looking at the lego board, a 2x4 (WxL) block is the equivalent of a 4x2 block rotated 90 degrees.
To reduce ambiguity, W will always be >= L. so if you are placing a 2x4, you would say you are rotating a 4x2.
There are only 2 rotations that are possible. 0 or 90 degrees. The lego platform is a grid, so only multiples of 90 are allowed.
Due to legos being rectangles there is no difference from 90 and 270 or 0 and 180.

This means the format is:
W, L, x, y, rotation(only 2 options, 0, 90; 0 for 0 and 1 for 90)

The pivot point for location is the bottom-left corner regardless of vertical or horizontal placemntn 

Legos are placed out in the order they appear in the file. So line 1 will be placed before line 2. A series of lines will build our structure.
"""

import csv

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

# TODO
def writeLegoSDF(W, L):
    # Code that returns a string corresponding to a lego block's SDF (3D model description)
    # pick_and_place.py has this in PS2
    raise NotImplementedError

# TODO
def generateLegoDirective(W, L, x, y, rotation):
    # returns a string for the directive (position) of the lego. pi
    # look at table_directive() function in pick_and_play.py in PS2
    # use writeLegoSDF()
    raise NotImplementedError

def config_interpretor(file_path: str):
    # returns a list of which legos need to be placed in order
    pythonFile = open(file_path, "r")
    configFile = csv.reader(pythonFile)

    # W, L, x, y, is0degrees (0/1)
    return [tuple(map(int, row)) for row in configFile]

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
    # plant = station.GetSubsystemByName("plant")?
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
    legosToBePlaced = config_interpretor("./sample_configs/4by4Square.csv")
    for lego in legosToBePlaced:
        W, L, x, y, is90Deg = lego
        # mainScenario = AppendDirectives(mainScenario, data=generateLegoDirective(W, L, x, y, is90Deg))
    
    show_scene(build_static_system(mainScenario))

main()
# keep_meshcat_open()
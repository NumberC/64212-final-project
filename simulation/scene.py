"""Drake scene construction: the robot, the two plates and the bricks, plus the
Meshcat viewer. Nothing here plans or moves the arm."""

import functools
import time

from manipulation.station import AppendDirectives, LoadScenario, MakeHardwareStation
from pydrake.all import (
    ConstantVectorSource,
    DiagramBuilder,
    Simulator,
    StartMeshcat,
    TrajectorySource,
)

from physical.arm.specifications import ROBOT
from physical.lego import brick_models
from physical.lego import specifications as lego

BROWSER_WAIT = 15.0  # seconds a new viewer waits for a browser to connect


@functools.cache
def viewer():
    """The session's one Meshcat server."""
    server = StartMeshcat()
    print(f"Meshcat: {server.web_url()}")
    deadline = time.time() + BROWSER_WAIT
    while time.time() < deadline and server.GetNumActiveConnections() == 0:
        time.sleep(0.1)
    return server


def _plate(name, studs, origin, color):
    """Directive for a studded plate welded to the world."""
    x, y, z = origin
    return f"""directives:
- add_model:
    name: {name}
    file: {brick_models.plate_url(name, studs, color)}
- add_weld:
    parent: world
    child: {name}::plate
    X_PC:
        translation: [{x:.6f}, {y:.6f}, {z:.6f}]
"""


def _brick(name, W, L, x, y, rotation, level, color, origin):
    """Directive for a free brick dropped just above a seat."""
    tx, ty, tz = brick_models.world_pose(W, L, x, y, rotation, level, origin).translation()
    tz += lego.SPAWN_CLEARANCE * (level + 1)
    return f"""directives:
- add_model:
    name: {name}
    file: {brick_models.brick_url(W, L, color)}
    default_free_body_pose:
        brick:
            translation: [{tx:.6f}, {ty:.6f}, {tz:.6f}]
            rotation: !Rpy {{ deg: [0, 0, {90 * rotation}] }}
"""


def make_scenario(final_configuration, state="start"):
    """The robot, both plates and brick i as model lego_i: on the supply plate
    ("start") or at its target ("goal")."""
    parts = [_plate("baseplate", lego.BOARD_STUDS, lego.BOARD_ORIGIN, "green"),
             _plate("supply", lego.SUPPLY_STUDS, lego.SUPPLY_ORIGIN, (0.6, 0.6, 0.6, 1))]
    for i, ((W, L, x, y, rot, level, color), (sx, sy)) in enumerate(
            zip(final_configuration, brick_models.supply_cells(final_configuration))):
        if state == "goal":
            parts.append(_brick(f"lego_{i}", W, L, x, y, rot, level, color, lego.BOARD_ORIGIN))
        else:
            parts.append(_brick(f"lego_{i}", W, L, sx, sy, 0, 0, color, lego.SUPPLY_ORIGIN))
    scenario = LoadScenario(data=ROBOT)
    for part in parts:
        scenario = AppendDirectives(scenario, data=part)
    return scenario


def build_diagram(scenario, viewer=None, q=None, opening=None):
    """Diagram of the scene. The arm follows q(t) and the fingers opening(t);
    without them the arm holds its home pose with the fingers open."""
    builder = DiagramBuilder()
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat=viewer))
    if q is None:
        plant = station.GetSubsystemByName("plant")
        iiwa = plant.GetModelInstanceByName("iiwa")
        home = plant.GetPositions(plant.CreateDefaultContext(), iiwa)
        sources = ConstantVectorSource(home), ConstantVectorSource([0.1])
    else:
        sources = TrajectorySource(q), TrajectorySource(opening)
    for source, port in zip(sources, ("iiwa.position", "wsg.position")):
        builder.Connect(builder.AddSystem(source).get_output_port(), station.GetInputPort(port))
    return builder.Build()


def simulate(system, duration, realtime=False):
    simulator = Simulator(system)
    if realtime:
        simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(duration)
    return simulator

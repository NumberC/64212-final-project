"""Pick-and-place macro action for one LEGO brick.

pick_and_place() turns "the brick seated at X_WB_pick goes to X_WB_place" into
gripper Waypoints; build_system() turns Waypoints into a Drake diagram that
executes them.

Seated studs lock a brick sideways, so every motion near a stud surface is
straight up or straight down. Sideways travel happens only at CLEARANCE above
the pick and the place. To keep those lines straight, the gripper path is
sampled every PATH_STEP and each sample is solved with IK; the arm then
follows those joint angles under position control. (PS2's open-loop velocity
integration drifts by more than the ~0.3 mm stud clearance.)
"""

from typing import NamedTuple

import numpy as np
from manipulation.station import LoadScenario, MakeHardwareStation, MakeMultibodyPlant
from pydrake.all import (
    DiagramBuilder,
    InverseKinematics,
    PiecewisePolynomial,
    PiecewisePose,
    RigidTransform,
    RotationMatrix,
    Solve,
    TrajectorySource,
)

import lego_bricks as lb
from utils.viz import get_meshcat

# Gripper geometry, from the WSG model (see PS2): the fingers slide along G's
# x axis and reach out along G's +y axis; the 20 mm pads are centered 108 mm out.
GRIPPER_REACH = 0.108
PAD_HALF_LENGTH = 0.010
FINGER_THICKNESS = 0.015    # finger + tip, along the closing direction
FINGER_WIDTH = 0.020        # finger extent across the closing direction
FINGER_OPEN_MARGIN = 0.02   # opening beyond the brick width on the way in
FINGER_SQUEEZE = 0.02       # commanded closing past the brick width

CLEARANCE = 0.10    # hover height above a pick or a place
PLACE_GAP = 0.001   # let go this far above the seated pose
PATH_STEP = 0.005   # IK sample spacing along the gripper path (m)
SPEED = 0.05        # gripper speed (m/s)
TURN_SPEED = 0.5    # gripper turning speed (rad/s)
HOLD_SECONDS = 1.0  # time given to the fingers to open or close


class Waypoint(NamedTuple):
    """A gripper pose, and the finger opening to hold there."""

    X_WG: RigidTransform
    opening: float


def overhead_grasp(info: lb.BrickInfo, axis: str = "short") -> tuple[RigidTransform, float]:
    """X_BG for a top-down grasp, and the brick's width across the fingers.

    `axis` is the side the fingers close across: "short" (brick y) or "long"
    (brick x). The pads sit high on the brick, so their lower edge clears the
    studs of the level below, which poke up beside the brick.
    """
    if axis not in ("short", "long"):
        raise ValueError(f"axis must be 'short' or 'long', got {axis!r}")
    x_G = np.array([0.0, 1.0, 0.0] if axis == "short" else [1.0, 0.0, 0.0])
    y_G = np.array([0.0, 0.0, -1.0])
    R_BG = RotationMatrix(np.column_stack([x_G, y_G, np.cross(x_G, y_G)]))
    pad_center = np.array([0.0, 0.0, info.stud_height + 1.5 * PAD_HALF_LENGTH])
    p_BG = pad_center - R_BG @ np.array([0.0, GRIPPER_REACH, 0.0])
    width = info.width if axis == "short" else info.length
    return RigidTransform(R_BG, p_BG), width


def pick_and_place(info, X_WB_pick, X_WB_place, axis="short") -> list[Waypoint]:
    """Waypoints that carry the brick seated at X_WB_pick to X_WB_place.

    Both poses are of the brick body frame B (center of its bottom face) when
    seated. Hover, descend, close, lift, carry, descend, open, lift.
    """
    X_BG, width = overhead_grasp(info, axis)
    lift = RigidTransform([0, 0, CLEARANCE])
    X_WG_pick = X_WB_pick @ X_BG
    X_WG_place = RigidTransform([0, 0, PLACE_GAP]) @ X_WB_place @ X_BG
    opened, closed = width + FINGER_OPEN_MARGIN, max(width - FINGER_SQUEEZE, 0.0)
    return [
        Waypoint(lift @ X_WG_pick, opened),
        Waypoint(X_WG_pick, opened),
        Waypoint(X_WG_pick, closed),
        Waypoint(lift @ X_WG_pick, closed),
        Waypoint(lift @ X_WG_place, closed),
        Waypoint(X_WG_place, closed),
        Waypoint(X_WG_place, opened),
        Waypoint(lift @ X_WG_place, opened),
    ]


def leg_seconds(X_a: RigidTransform, X_b: RigidTransform) -> float:
    """How long the gripper takes from X_a to X_b, at least HOLD_SECONDS."""
    distance = np.linalg.norm(X_b.translation() - X_a.translation())
    angle = (X_a.rotation().inverse() @ X_b.rotation()).ToAngleAxis().angle()
    return max(distance / SPEED, angle / TURN_SPEED, HOLD_SECONDS)


def joint_trajectories(robot_yaml: str, waypoints, q_start):
    """(q(t) for the iiwa, opening(t) for the wsg) that walk through `waypoints`.

    Each leg after the first is a straight line (slerp for the rotation)
    sampled every PATH_STEP; each sample is solved by IK, seeded with and
    pulled toward the previous solution so the arm does not jump between IK
    branches.
    """
    plant = MakeMultibodyPlant(LoadScenario(data=robot_yaml))
    G = plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg")).body_frame()
    iiwa = plant.GetModelInstanceByName("iiwa")
    context = plant.CreateDefaultContext()
    plant.SetPositions(context, iiwa, q_start)
    q = plant.GetPositions(context)

    def solve(X_WG):
        ik = InverseKinematics(plant)
        p = X_WG.translation()
        ik.AddPositionConstraint(G, [0, 0, 0], plant.world_frame(), p, p)
        ik.AddOrientationConstraint(G, RotationMatrix(), plant.world_frame(), X_WG.rotation(), 0.0)
        prog = ik.get_mutable_prog()
        prog.AddQuadraticErrorCost(np.eye(len(q)), q, ik.q())
        prog.SetInitialGuess(ik.q(), q)
        result = Solve(prog)
        if not result.is_success():
            raise RuntimeError(f"no IK solution for gripper pose\n{X_WG}")
        return result.GetSolution(ik.q())

    X_WG = plant.CalcRelativeTransform(context, plant.world_frame(), G)
    times, qs = [0.0], [q_start]
    opening_times, openings = [], []
    for k, waypoint in enumerate(waypoints):
        # A waypoint's opening applies from the start of the leg toward it, so
        # on a stationary leg (same pose, new opening) the fingers move while
        # the arm holds still, before the next leg lifts or retreats.
        opening_times.append(times[-1])
        openings.append(waypoint.opening)
        duration = leg_seconds(X_WG, waypoint.X_WG)
        # The first leg leaves the arm's start pose through free space, where a
        # straight line is not needed (and may leave the workspace): solve
        # only its end and move there in joint space.
        n = 1 if k == 0 else max(1, int(np.ceil(SPEED * duration / PATH_STEP)))
        leg = PiecewisePose.MakeLinear([0, 1], [X_WG, waypoint.X_WG])
        for s in np.linspace(0, 1, n + 1)[1:]:
            q = solve(leg.GetPose(s))
            plant.SetPositions(context, q)
            times.append(times[-1] + duration / n)
            qs.append(plant.GetPositions(context, iiwa))
        X_WG = waypoint.X_WG
    times.append(times[-1] + HOLD_SECONDS)
    qs.append(qs[-1])
    opening_times.append(times[-1])
    openings.append(openings[-1])
    q_traj = PiecewisePolynomial.FirstOrderHold(times, np.array(qs).T)
    wsg_traj = PiecewisePolynomial.ZeroOrderHold(opening_times, np.array([openings]))
    return q_traj, wsg_traj


def build_system(scenario, robot_yaml, waypoints):
    """The scene, with the arm and the hand walking through `waypoints`.

    Returns the diagram and how long its plan runs for.
    """
    builder = DiagramBuilder()
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat=get_meshcat()))
    plant = station.GetSubsystemByName("plant")
    q_start = plant.GetPositions(
        plant.CreateDefaultContext(), plant.GetModelInstanceByName("iiwa")
    )
    q_traj, wsg_traj = joint_trajectories(robot_yaml, waypoints, q_start)
    q_source = builder.AddSystem(TrajectorySource(q_traj))
    wsg_source = builder.AddSystem(TrajectorySource(wsg_traj))
    builder.Connect(q_source.get_output_port(), station.GetInputPort("iiwa.position"))
    builder.Connect(wsg_source.get_output_port(), station.GetInputPort("wsg.position"))
    return builder.Build(), q_traj.end_time()

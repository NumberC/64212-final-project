"""Arm primitives that execute any plan: waypoints, their timing, and IK.

Seated studs leave ~0.3 mm of play, so the gripper path between waypoints is
a straight line sampled every PATH_STEP and solved by IK; the arm follows
those joint angles under position arm_motion.
"""

import functools
from typing import NamedTuple

import numpy as np
from manipulation.station import LoadScenario, MakeMultibodyPlant
from pydrake.all import (
    InverseKinematics,
    PiecewisePolynomial,
    PiecewisePose,
    RigidTransform,
    RotationMatrix,
    Solve,
)

from physical.arm.specifications import HOLD_SECONDS, PATH_STEP, ROBOT, SPEED, TURN_SPEED


class Waypoint(NamedTuple):
    """A gripper pose, and the finger opening to hold there. With joint angles
    q (whose gripper pose is X_WG), the arm moves straight to q in joint space
    instead of drawing a straight line with the gripper."""

    X_WG: RigidTransform
    opening: float
    q: tuple | None = None


@functools.cache
def _robot():
    """(plant, iiwa model instance, gripper frame G) of the robot alone."""
    plant = MakeMultibodyPlant(LoadScenario(data=ROBOT))
    iiwa, wsg = map(plant.GetModelInstanceByName, ("iiwa", "wsg"))
    return plant, iiwa, plant.GetBodyByName("body", wsg).body_frame()


def home_pose() -> RigidTransform:
    """X_WG with the arm at its home joint angles."""
    plant, _, G = _robot()
    return plant.CalcRelativeTransform(plant.CreateDefaultContext(), plant.world_frame(), G)


def travel_seconds(X_a: RigidTransform, X_b: RigidTransform) -> float:
    """How long the gripper takes from X_a to X_b, at least HOLD_SECONDS."""
    distance = np.linalg.norm(X_b.translation() - X_a.translation())
    angle = (X_a.rotation().inverse() @ X_b.rotation()).ToAngleAxis().angle()
    return max(distance / SPEED, angle / TURN_SPEED, HOLD_SECONDS)


def home_joints():
    plant, iiwa, _ = _robot()
    return plant.GetPositions(plant.CreateDefaultContext(), iiwa)


def inverse_kinematics(X_WG, near):
    """Joint angles that put the gripper at X_WG, as close to `near` as they
    can be so the arm does not jump between IK branches; None if there are none."""
    plant, iiwa, G = _robot()
    context = plant.CreateDefaultContext()
    plant.SetPositions(context, iiwa, near)
    q = plant.GetPositions(context)
    ik = InverseKinematics(plant)
    p = X_WG.translation()
    ik.AddPositionConstraint(G, [0, 0, 0], plant.world_frame(), p, p)
    ik.AddOrientationConstraint(G, RotationMatrix(), plant.world_frame(), X_WG.rotation(), 0.0)
    ik.get_mutable_prog().AddQuadraticErrorCost(np.eye(len(q)), q, ik.q())
    ik.get_mutable_prog().SetInitialGuess(ik.q(), q)
    result = Solve(ik.prog())
    if result.is_success():
        return plant.GetPositionsFromArray(iiwa, result.GetSolution(ik.q()))


def joint_trajectories(waypoints):
    """(q(t) for the iiwa, opening(t) for the wsg) that walk through `waypoints` from home."""
    q, X_WG = home_joints(), home_pose()
    times, qs = [0.0], [q]
    opening_times, openings = [], []
    for k, waypoint in enumerate(waypoints):
        # A waypoint's opening applies from the start of the leg toward it, so
        # on a stationary leg the fingers move while the arm holds still.
        opening_times.append(times[-1])
        openings.append(waypoint.opening)
        duration = travel_seconds(X_WG, waypoint.X_WG)
        # The first leg leaves home through free space, so like a leg given
        # joint angles it is one joint-space move, not a gripper line.
        joint_move = k == 0 or waypoint.q is not None
        n = 1 if joint_move else max(1, int(np.ceil(SPEED * duration / PATH_STEP)))
        leg = PiecewisePose.MakeLinear([0, 1], [X_WG, waypoint.X_WG])
        for s in np.linspace(0, 1, n + 1)[1:]:
            q = waypoint.q if waypoint.q is not None else inverse_kinematics(leg.GetPose(s), q)
            if q is None:
                raise RuntimeError(f"no IK solution for gripper pose\n{leg.GetPose(s)}")
            times.append(times[-1] + duration / n)
            qs.append(q)
        X_WG = waypoint.X_WG
    times.append(times[-1] + HOLD_SECONDS)
    qs.append(qs[-1])
    opening_times.append(times[-1])
    openings.append(openings[-1])
    q_traj = PiecewisePolynomial.FirstOrderHold(times, np.array(qs).T)
    wsg_traj = PiecewisePolynomial.ZeroOrderHold(opening_times, np.array([openings]))
    return q_traj, wsg_traj

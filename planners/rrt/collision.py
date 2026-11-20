"""What the robot, as Drake models it, would hit at given joint angles.

Every brick is an obstacle in one of two places: on the supply plate until it
is picked, at its target once it is placed. So what a motion collides with
does not depend on the build state, and is found once as (need, forbid) brick
bitmasks: the motion is free in exactly the build states where every `need`
brick has left the supply and no `forbid` brick has been placed.
"""

import functools
from dataclasses import dataclass

import numpy as np
from manipulation.station import LoadScenario
from manipulation.utils import ConfigureParser
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Box,
    CoulombFriction,
    DiagramBuilder,
    ModelDirectives,
    Parser,
    ProcessModelDirectives,
    RigidTransform,
)

from physical.arm import specifications as arm
from planners.build_task import overhead_grasp

MARGIN = 0.005     # bricks grow by this much on every side (m)
JOINT_STEP = 0.01  # most a joint turns between collision samples (rad)


@dataclass(frozen=True)
class Hand:
    """What sweeps through space: the finger opening and, if a brick is held,
    its (half length along the closing axis, half length across, bottom, top),
    heights relative to the gripper frame."""

    opening: float
    load: tuple | None = None


def holding(info, axis):
    """The hand gripping a brick with overhead_grasp(info, axis)."""
    X_BG, width = overhead_grasp(info, axis)
    across = info.length if axis == "short" else info.width
    bottom = -X_BG.translation()[2]
    return Hand(width, (width / 2, across / 2, bottom, bottom + info.height + info.stud_height))


def _box(info, X_WB):
    """A seated brick as (x, y, half x, half y, bottom, top), grown by MARGIN."""
    hx, hy = np.abs(X_WB.rotation().matrix()[:2, :2]) @ [info.length / 2, info.width / 2]
    x, y, z = X_WB.translation()
    return (x, y, hx + MARGIN, hy + MARGIN,
            z - MARGIN, z + info.height + info.stud_height + MARGIN)


def is_free(blocking_bricks, gone, placed):
    """Is a motion free once the bricks `gone` have left the supply and the
    bricks `placed` sit at their targets?"""
    need, forbid = blocking_bricks
    return not (need & ~gone or forbid & placed)


def _bits(hits, n):
    """(need, forbid) from which of the 2n brick boxes were hit."""
    return tuple(sum(1 << int(j) for j in np.flatnonzero(half)) for half in (hits[:n], hits[n:]))


class RobotCollision:
    def __init__(self, seats, loads):
        """seats: per brick (info, pose on the supply, pose at its target);
        loads: the held-brick boxes of every Hand that will be asked about."""
        # Rows 0..n-1 are the bricks on the supply, n..2n-1 at their targets.
        boxes = [_box(seat[0], seat[k]) for k in (1, 2) for seat in seats]
        builder = DiagramBuilder()
        plant, scene_graph = self.plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        parser = Parser(plant)
        ConfigureParser(parser)
        directives = LoadScenario(data=arm.ROBOT).directives
        ProcessModelDirectives(ModelDirectives(directives=directives), plant, parser)
        self.iiwa, self.wsg = map(plant.GetModelInstanceByName, ("iiwa", "wsg"))
        self.G = plant.GetBodyByName("body", self.wsg)
        add = lambda body, X, shape, name: plant.RegisterCollisionGeometry(
            body, X, shape, name, CoulombFriction())
        world = plant.world_body()
        self.row = {add(world, RigidTransform([x, y, (lo + hi) / 2]),
                        Box(2 * hx, 2 * hy, hi - lo), f"box{j}"): j
                    for j, (x, y, hx, hy, lo, hi) in enumerate(boxes)}
        # A held brick hangs below the gripper frame, whose y axis points down.
        self.load = {load: add(self.G, RigidTransform([0, -(load[2] + load[3]) / 2, 0]),
                               Box(2 * load[0], load[3] - load[2], 2 * load[1]), f"load{k}")
                     for k, load in enumerate(loads)}
        self.floor = add(world, RigidTransform([0, 0, -0.5]), Box(4, 4, 1), "floor")
        plant.Finalize()
        geometries = lambda model: {g for k in plant.GetBodyIndices(model) for g in
                                    plant.GetCollisionGeometriesForBody(plant.get_body(k))}
        self.links = geometries(self.iiwa)
        self.fingers = geometries(self.wsg) - set(self.load.values())  # and the gripper body
        self.root = builder.Build().CreateDefaultContext()
        self.context = plant.GetMyMutableContextFromRoot(self.root)
        self.query = scene_graph.get_query_output_port()
        self.scene = scene_graph.GetMyContextFromRoot(self.root)
        self.limits = [plant.GetPositionsFromArray(self.iiwa, limit) for limit in
                       (plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits())]
        self.rest = set(self._pairs())  # contacts of the parked robot are not collisions
        self.blocking_bricks = functools.cache(self._blocking_bricks)

    def _pairs(self):
        return [frozenset((pair.id_A, pair.id_B))
                for pair in self.query.Eval(self.scene).ComputePointPairPenetration()]

    def gripper(self, q):
        """X_WG at joint angles q."""
        self.plant.SetPositions(self.context, self.iiwa, q)
        return self.plant.EvalBodyPoseInWorld(self.context, self.G)

    def _blocking_bricks(self, hand, a, b=None):
        """(need, forbid) of the bricks touched at joint angles a, or turning
        every joint steadily from a to b: by the whole robot with `hand`, or
        with hand None by the arm's links alone. Any other collision (the
        floor, the robot itself) is a need no build state meets.
        """
        robot = set(self.links)
        if hand:
            robot |= self.fingers | ({self.load[hand.load]} if hand.load else set())
            self.plant.SetPositions(self.context, self.wsg, [-hand.opening / 2, hand.opening / 2])
        solid = robot | {self.floor}
        hits, n = np.zeros(len(self.row) + 1, dtype=bool), len(self.row) // 2
        b = a if b is None else b
        for q in np.linspace(a, b, 2 + int(np.abs(np.subtract(b, a)).max() / JOINT_STEP)):
            self.plant.SetPositions(self.context, self.iiwa, q)
            for pair in self._pairs():
                for brick in pair & self.row.keys():
                    hits[self.row[brick]] |= bool(pair & robot)
                hits[-1] |= pair <= solid and pair not in self.rest
        need, forbid = _bits(hits[:-1], n)
        return need | int(hits[-1]) << n, forbid

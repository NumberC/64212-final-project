"""Sheet-RRT: one random tree, in the arm's joint angles, from the empty board
to the final configuration.

A tree node is (joint angles, mode). A mode is a build state plus what the
hand is doing: travelling empty, or carrying one brick. Inside a mode the tree
grows by RRT-Connect, stepping toward fresh samples to meet a small tree grown
back from the chimney it is heading for, with the whole robot checked against
the bricks standing in that mode. Modes are joined by
chimneys: the straight down and up over a brick's seat on the supply (grasp)
or at its target (release), which BuildEnv's rules allow or forbid; chimneys
that lead the build into a dead end of those rules are never taken. Reaching
the top of a chimney opens the next mode, and the tree grows on from there.

Each motion's collisions are found once and labelled with the bricks hit
(collision.py), so a motion met again in another mode costs two bit operations.
"""

import dataclasses
import functools
import itertools
from dataclasses import dataclass, field

import numpy as np
from pydrake.all import RigidTransform

from planners import build_task
from planners.rrt.collision import Hand, RobotCollision, holding, is_free
from simulation import arm_motion
from simulation.arm_motion import Waypoint

LIFTS = (0.02, 0.04, 0.06, 0.08, build_task.CLEARANCE)  # chimney heights tried (m)
STEP = 0.2         # most a joint turns along one tree edge (rad)
WANDER = 0.6       # how far random samples stray from a mode's tree (rad)
GOAL_BIAS = 0.5    # chance a round aims at the chimney's tops, not a random sample
DEPTH_BIAS = 0.7   # chance it grows the mode with the most bricks placed
ITERATIONS = 3000  # extensions before settling for the best build so far


@dataclass(eq=False)
class Mode:
    mask: int      # the build state: bit j is set once brick j is placed
    held: tuple    # the (brick, axis) being carried, or None
    hand: Hand     # what is collision checked while travelling
    opening: float # finger command while travelling
    root: tuple    # joint angles the mode is entered at
    via: tuple     # (the mode before, waypoints of the chimney between), or None
    exits: list    # (brick, axis) chimneys still to try
    parent: dict = field(default_factory=dict)  # the tree: joint angles -> their parent's
    back: dict = field(default_factory=dict)    # per exit, the tree grown back from its tops

    def __post_init__(self):
        self.parent[self.root] = None
        self.gone = self.mask | (1 << self.held[0] if self.held else 0)  # bricks off the supply


def shortcut(path, ok):
    """Skip ahead to the furthest vertex each vertex can reach directly."""
    out, i = [path[0]], 0
    while i < len(path) - 1:
        i = next(j for j in range(len(path) - 1, i, -1) if j == i + 1 or ok(path[i], path[j]))
        out.append(path[i])
    return out


class SheetRRT:
    def __init__(self, final_configuration, seed=0, iterations=ITERATIONS):
        env = self.env = build_task.BuildEnv(final_configuration)
        self.rng, self.iterations = np.random.default_rng(seed), iterations
        # Per (brick, axis): seat poses, the open hand that comes for the brick,
        # the finger command that grips it and the loaded hand that leaves.
        self.seat, self.open, self.closed, self.load = {}, {}, {}, {}
        for key, w in env.waypoints.items():
            self.seat["pick", key], self.seat["place", key] = w[1].X_WG, w[5].X_WG
            self.open[key], self.closed[key] = Hand(w[0].opening), w[2].opening
            self.load[key] = holding(env.seats[key[0]][0], build_task.AXES[key[1]])
        self.robot = RobotCollision(env.seats, {hand.load for hand in self.load.values()})
        self.top = functools.cache(self._top)
        self.tops = functools.cache(self._tops)
        self.allowed = functools.cache(self._allowed)
        self.reach = functools.cache(self._reach)

    def _allowed(self, mask):
        """The (brick, axis) the rules allow next in a build state."""
        placed = np.array([mask >> j & 1 for j in range(self.env.n)], dtype=bool)
        return [tuple(key) for key in np.argwhere(self.env.feasible(placed)).tolist()]

    def _reach(self, mask):
        """The most bricks the rules let any order place, starting from a build state."""
        return max([mask.bit_count()] + [self.reach(mask | 1 << i) for i, _ in self.allowed(mask)])

    def exits(self, mask):
        """The allowed (brick, axis) that do not lead the build into a dead end."""
        return [key for key in self.allowed(mask)
                if self.reach(mask | 1 << key[0]) == self.reach(0)]

    def clear(self, mode, a, b=None, hand=None):
        return is_free(self.robot.blocking_bricks(hand or mode.hand, a, b), mode.gone, mode.mask)

    def _top(self, side, key, lift):
        """Joint angles holding the gripper `lift` above a seat, or None."""
        X = RigidTransform([0, 0, lift]) @ self.seat[side, key]
        q = arm_motion.inverse_kinematics(X, arm_motion.home_joints())
        return q if q is None else tuple(q.tolist())

    def _tops(self, mode, key):
        """(the mode entered by taking `key`'s chimney out of `mode`, the clear
        chimney tops as (lift, joint angles) from the lowest up); no tops if
        the chimney cannot be used."""
        bit, side = 1 << key[0], "place" if mode.held else "pick"
        if mode.held:  # release: the brick is at its target from here on
            after = Mode(mode.mask | bit, None, self.open[key], self.open[key].opening,
                         None, None, None)
        else:          # grasp: the brick is off the supply from here on
            after = Mode(mode.mask, key, self.load[key], self.closed[key], None, None, [key])
        tops = [(lift, q) for lift in LIFTS if (q := self.top(side, key, lift))
                and self.clear(mode, q) and self.clear(mode, q, hand=self.open[key])
                and self.clear(after, q)]
        # The hand's fit inside a chimney is the rules' business; the arm's is checked.
        if tops:
            high = tops[-1][1]
            seated = arm_motion.inverse_kinematics(self.seat[side, key], high)
            if seated is None or not is_free(self.robot.blocking_bricks(
                    None, high, tuple(seated.tolist())), after.gone, after.mask):
                tops = []
        return after, tops

    def enter(self, mode, key, lift, q):
        """The mode after `key`'s chimney, taken from joint angles q at its top."""
        after, tops = self.tops(mode, key)
        exits = after.exits or self.exits(after.mask)
        after = dataclasses.replace(after, root=q, exits=list(exits), parent={}, back={})
        X = self.seat["place" if mode.held else "pick", key]
        X_top = RigidTransform([0, 0, lift]) @ X
        opened, closed = self.open[key].opening, self.closed[key]
        if mode.held:
            moves = [Waypoint(X, closed), Waypoint(X, opened), Waypoint(X_top, opened)]
        else:
            reshape = [Waypoint(X_top, opened, q)] if opened != mode.opening else []
            moves = reshape + [Waypoint(X, opened), Waypoint(X, closed), Waypoint(X_top, closed)]
        after.via = mode, moves
        # The tree may leave from any higher top it can climb to.
        for (_, low), (_, high) in itertools.pairwise(top for top in tops if top[0] >= lift):
            if not self.clear(after, low, high):
                break
            after.parent[high] = low
        return after

    def extend(self, mode, tree, target):
        """Grow a tree of `mode` from its nearest node toward `target`, STEP at
        a time, until something is in the way; returns the node it got to."""
        nodes = list(tree)
        near = nodes[int(np.argmin(np.abs(np.array(nodes) - target).max(1)))]
        while near != target:
            gap = np.subtract(target, near)
            far = np.abs(gap).max()
            new = target if far <= STEP else tuple((near + gap * STEP / far).tolist())
            if not self.clear(mode, near, new):
                break
            tree[new] = near
            near = new
        return near

    def connect(self, mode, key):
        """One RRT-Connect round between `mode`'s tree and the tree grown back
        from `key`'s chimney tops; returns the (lift, joint angles) of the top
        the two now join at, or None."""
        tops = self.tops(mode, key)[1]
        back = mode.back.setdefault(key, {q: None for _, q in tops})
        ends = np.array([mode.root, *back])
        lo, hi = np.clip([ends.min(0) - WANDER, ends.max(0) + WANDER], *self.robot.limits)
        if self.rng.random() < GOAL_BIAS:  # aim straight at the tops, lowest first
            return next((top for top in tops
                         if self.extend(mode, mode.parent, top[1]) == top[1]), None)
        target = tuple(self.rng.uniform(lo, hi).tolist())
        for first, second in ((mode.parent, back), (back, mode.parent)):
            tip = self.extend(mode, first, target)
            if self.extend(mode, second, tip) == tip:
                while (after := back[tip]) is not None:  # hang the joining branch on the tree
                    mode.parent[after] = tip
                    tip = after
                return next(top for top in tops if top[1] == tip)
        return None

    def grow(self):
        """Grow the tree; returns the empty-handed mode with the most bricks placed."""
        wide = max(self.open.values(), key=lambda hand: hand.opening)
        start = Mode(0, None, wide, wide.opening, tuple(arm_motion.home_joints().tolist()),
                     None, self.exits(0))
        modes, best = [start], start
        depth = lambda mode: (mode.mask.bit_count(), mode.held is not None)
        for _ in range(self.iterations):
            live = [mode for mode in modes if mode.exits]
            if not live or best.mask.bit_count() == self.reach(0):
                break
            deep = self.rng.random() < DEPTH_BIAS
            mode = max(live, key=depth) if deep else live[self.rng.integers(len(live))]
            key = mode.exits[self.rng.integers(len(mode.exits))]
            top = self.tops(mode, key)[1] and self.connect(mode, key)
            if top == []:  # the chimney cannot be used
                mode.exits.remove(key)
            elif top:
                mode.exits.remove(key)
                modes.append(after := self.enter(mode, key, *top))
                if not after.held and after.mask.bit_count() > best.mask.bit_count():
                    best = after
        return best

    def plan(self):
        """(order, waypoints) of the build that placed the most bricks, each
        trip between chimneys shortcut smoothed."""
        mode, order, waypoints = self.grow(), [], []
        while mode.via:
            before, chimney = mode.via
            route = [mode.root]
            while route[-1] != before.root:
                route.append(before.parent[route[-1]])
            route = shortcut(route[::-1], lambda a, b: self.clear(before, a, b))
            trip = [Waypoint(self.robot.gripper(q), before.opening, q) for q in route[1:]]
            waypoints = trip + chimney + waypoints
            order = ([before.held] if before.held else []) + order
            mode = before
        return order, waypoints

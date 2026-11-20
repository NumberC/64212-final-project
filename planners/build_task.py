"""The task both approaches solve: build a final configuration one brick at a time.

BuildEnv is a fast, symbolic stand-in for the simulation. Each step places one
brick at its target with one grasp axis; three rules, taken from the scene's
geometry, decide whether that can succeed:

  * covered:   no placed brick may sit above it (bricks only go on top),
  * supported: off the baseplate, its center must lie over placed bricks
               directly below (gravity alone holds bricks),
  * reachable: the fingers come down beside the brick and close on it; no
               placed brick at its level or higher may be in their way.

Time is arm_motion.travel_seconds over the waypoints, so an episode's seconds are
what the simulation takes to execute it. A final configuration is a list of
bricks (W, L, x, y, rotation, level, color); a Plan is what every planner returns.
"""

from dataclasses import dataclass

import numpy as np
from pydrake.all import RigidTransform, RotationMatrix

from physical.arm import specifications as arm
from physical.lego import brick_models
from physical.lego import specifications as lego
from simulation import arm_motion

AXES = ("short", "long")
SHAPES = [(1, 1), (2, 2), (4, 2)]
MAX_LEVEL = 2
TIME_COST = 0.01  # reward per second of robot time; a placed brick is worth 1
GRAZE = 0.0005    # a pad passing this close to a neighbor's face hits it (m)
CLEARANCE = 0.10  # hover height above a pick or a place (m)
PLACE_GAP = 0.001 # let go this far above the seated pose (m)

_GAP = lego.SIDE_GAP_MM / lego.PITCH_MM  # brick face inset from its cells, in studs


@dataclass
class Plan:
    order: list      # (brick, axis) steps
    waypoints: list  # arm_motion.Waypoint, for the whole build

    @property
    def seconds(self):
        poses = [arm_motion.home_pose()] + [w.X_WG for w in self.waypoints]
        return sum(arm_motion.travel_seconds(u, v) for u, v in zip(poses, poses[1:]))


def overhead_grasp(info, axis: str = "short") -> tuple[RigidTransform, float]:
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
    pad_center = np.array([0.0, 0.0, info.stud_height + 1.5 * arm.PAD_HALF_LENGTH])
    p_BG = pad_center - R_BG @ np.array([0.0, arm.GRIPPER_REACH, 0.0])
    width = info.width if axis == "short" else info.length
    return RigidTransform(R_BG, p_BG), width


def pick_and_place(info, X_WB_pick, X_WB_place, axis="short") -> list:
    """Waypoints that carry the brick seated at X_WB_pick to X_WB_place.

    Both poses are of the brick body frame B (center of its bottom face) when
    seated. Hover, descend, close, lift, carry, descend, open, lift.
    """
    X_BG, width = overhead_grasp(info, axis)
    lift = RigidTransform([0, 0, CLEARANCE])
    X_WG_pick = X_WB_pick @ X_BG
    X_WG_place = RigidTransform([0, 0, PLACE_GAP]) @ X_WB_place @ X_BG
    opened, closed = width + arm.FINGER_OPEN_MARGIN, max(width - arm.FINGER_SQUEEZE, 0.0)
    return [
        arm_motion.Waypoint(lift @ X_WG_pick, opened),
        arm_motion.Waypoint(X_WG_pick, opened),
        arm_motion.Waypoint(X_WG_pick, closed),
        arm_motion.Waypoint(lift @ X_WG_pick, closed),
        arm_motion.Waypoint(lift @ X_WG_place, closed),
        arm_motion.Waypoint(X_WG_place, closed),
        arm_motion.Waypoint(X_WG_place, opened),
        arm_motion.Waypoint(lift @ X_WG_place, opened),
    ]


def footprint(brick):
    """The (x, y) stud cells a brick covers."""
    W, L, x, y, rot = brick[:5]
    sx, sy = (L, W) if rot else (W, L)
    return {(x + i, y + j) for i in range(sx) for j in range(sy)}


def center(brick):
    W, L, x, y, rot = brick[:5]
    sx, sy = (L, W) if rot else (W, L)
    return x + sx / 2, y + sy / 2


def finger_cells(brick, axis):
    """The (x, y) cells, other than its own, the fingers pass through when
    placing `brick`.

    Each finger is a FINGER_THICKNESS x FINGER_WIDTH box that travels from
    FINGER_OPEN_MARGIN/2 out (open) to the brick face it closes on. A cell
    counts if a brick seated in it would come within GRAZE of that box: the
    pads are wider than a 1x1, so they graze any neighbor flush with its faces.
    """
    W, L, _, _, rot = brick[:5]
    cx, cy = center(brick)
    along_x = (axis == "long") == (rot == 0)  # do the fingers close along x?
    half = ((L, W) if rot else (W, L))[0 if along_x else 1] / 2
    near = half - _GAP - GRAZE / lego.PITCH
    far = half - _GAP + (arm.FINGER_OPEN_MARGIN / 2 + arm.FINGER_THICKNESS) / lego.PITCH
    across = arm.FINGER_WIDTH / 2 / lego.PITCH
    cells = set()
    for side in (-1, 1):
        lo, hi = sorted((side * near, side * far))
        u_range, v_range = (lo, hi), (-across, across)
        (x0, x1), (y0, y1) = (u_range, v_range) if along_x else (v_range, u_range)
        for i in range(int(np.floor(cx + x0)), int(np.ceil(cx + x1))):
            for j in range(int(np.floor(cy + y0)), int(np.ceil(cy + y1))):
                if (min(i + 1 - _GAP, cx + x1) > max(i + _GAP, cx + x0)
                        and min(j + 1 - _GAP, cy + y1) > max(j + _GAP, cy + y0)):
                    cells.add((i, j))
    return cells - footprint(brick)


def center_supported(point, cells):
    """Is `point` inside the bounding box of the supporting `cells`?"""
    if not cells:
        return False
    xs, ys = zip(*cells)
    return min(xs) <= point[0] <= max(xs) + 1 and min(ys) <= point[1] <= max(ys) + 1


class BuildEnv:
    """Place the bricks of one final configuration, one (brick, axis) action at a time."""

    def __init__(self, final_configuration):
        self.final_configuration = final_configuration
        n = self.n = len(final_configuration)
        feet = [footprint(b) for b in final_configuration]
        level = [b[5] for b in final_configuration]
        fingers = [[finger_cells(b, a) for a in AXES] for b in final_configuration]
        # blocks[i, a, j]: once j is placed, i can't be placed with axis a.
        self.blocks = np.array([[[j != i and level[j] >= level[i] and bool(feet[j] & fingers[i][a])
                                  for j in range(n)] for a in range(2)] for i in range(n)],
                               dtype=bool).reshape(n, 2, n)
        # covers[i, j]: j sits above i, so i must go first.
        self.covers = np.array([[level[j] > level[i] and bool(feet[j] & feet[i])
                                 for j in range(n)] for i in range(n)], dtype=bool).reshape(n, n)
        # below[i]: (j, cells of i resting on j) for bricks directly under i.
        self.below = [[(j, feet[i] & feet[j]) for j in range(n)
                       if level[j] == level[i] - 1 and feet[i] & feet[j]] for i in range(n)]
        self.centers = [center(b) for b in final_configuration]

        # Every macro's waypoints are fixed by its brick and axis; only the
        # leg from the previous hand pose depends on the order.
        cells = brick_models.supply_cells(final_configuration)
        self.waypoints, self.first, self.last, self.inner = {}, {}, {}, {}
        self.seats = []  # per brick: (info, seated pose on the supply, at its target)
        for i, ((W, L, x, y, rot, lvl, _), (sx, sy)) in enumerate(zip(final_configuration, cells)):
            info = brick_models.brick_info(brick_models.BrickSpec(W, L), lego.PARAMS)
            X_pick = brick_models.world_pose(W, L, sx, sy, 0, 0, lego.SUPPLY_ORIGIN)
            X_place = brick_models.world_pose(W, L, x, y, rot, lvl)
            self.seats.append((info, X_pick, X_place))
            for a, axis in enumerate(AXES):
                wps = self.waypoints[i, a] = pick_and_place(info, X_pick, X_place, axis)
                self.first[i, a], self.last[i, a] = wps[0].X_WG, wps[-1].X_WG
                self.inner[i, a] = sum(arm_motion.travel_seconds(u.X_WG, v.X_WG)
                                       for u, v in zip(wps, wps[1:]))
        self.reset()

    def reset(self):
        self.placed = np.zeros(self.n, dtype=bool)
        self.X_WG = arm_motion.home_pose()
        self.seconds = 0.0
        self.order = []  # the (brick, axis) steps taken
        return self

    def feasible(self, placed=None):
        """(n, 2) mask of the (brick, axis) actions that can succeed now."""
        p = self.placed if placed is None else placed
        ok = ~p[:, None] & ~(self.blocks & p).any(-1) & ~(self.covers & p).any(-1)[:, None]
        for i in range(self.n):
            if self.below[i] and ok[i].any():
                support = set().union(*(c for j, c in self.below[i] if p[j]))
                ok[i] &= center_supported(self.centers[i], support)
        return ok

    @property
    def done(self):
        return self.placed.all() or not self.feasible().any()

    def step(self, i, a):
        """Place brick i with axis a; returns (reward, done, seconds)."""
        if not self.feasible()[i, a]:
            raise ValueError(f"brick {i} with axis {AXES[a]} is not feasible now")
        seconds = arm_motion.travel_seconds(self.X_WG, self.first[i, a]) + self.inner[i, a]
        self.placed[i] = True
        self.order.append((i, a))
        self.X_WG = self.last[i, a]
        self.seconds += seconds
        return 1.0 - TIME_COST * seconds, self.done, seconds


def rollout(final_configuration, choose):
    """Build `final_configuration` choosing each (brick, axis) with `choose`; returns the env."""
    env = BuildEnv(final_configuration)
    while not env.done:
        env.step(*choose(env))
    return env


def plan_with_fixed_motion(final_configuration, choose):
    """A Plan: the order from a chooser, env -> (brick, axis); the motion from
    pick_and_place."""
    env = rollout(final_configuration, choose)
    return Plan(env.order, [w for key in env.order for w in env.waypoints[key]])


def scripted_order(env):
    """Lowest level first, then file order; short axis unless it's blocked."""
    ok = env.feasible()
    for i in sorted(range(env.n), key=lambda i: (env.final_configuration[i][5], i)):
        if ok[i].any():
            return i, int(not ok[i, 0])
    return None


def max_placeable(final_configuration):
    """Most bricks any order can place (search over placed sets)."""
    env = BuildEnv(final_configuration)
    frontier, seen = [np.zeros(env.n, dtype=bool)], set()
    best = 0
    while frontier:
        placed = frontier.pop()
        best = max(best, int(placed.sum()))
        for i in np.flatnonzero(env.feasible(placed).any(1)):
            nxt = placed.copy()
            nxt[i] = True
            key = nxt.tobytes()
            if key not in seen:
                seen.add(key)
                frontier.append(nxt)
    return best


def random_final_configuration(rng, n_bricks=(4, 10), patch=6, corner=(5, 5)):
    """A random buildable final configuration: bricks dropped into a patch x patch area.

    Bricks are stacked up to MAX_LEVEL with supported centers, shuffled so the
    file order says nothing, and kept only if some order can place them all.
    """
    colors = list(lego.COLORS)
    while True:
        n = rng.integers(n_bricks[0], n_bricks[1] + 1)
        final_configuration, height = [], {}
        for _ in range(50 * n):
            if len(final_configuration) == n:
                break
            W, L = SHAPES[rng.integers(len(SHAPES))]
            rot = int(rng.integers(2))
            sx, sy = (L, W) if rot else (W, L)
            x = corner[0] + int(rng.integers(patch - sx + 1))
            y = corner[1] + int(rng.integers(patch - sy + 1))
            brick = (W, L, x, y, rot, 0, colors[len(final_configuration) % len(colors)])
            feet = footprint(brick)
            level = max(height.get(c, -1) for c in feet) + 1
            below = {c for c in feet if height.get(c) == level - 1}
            if level > MAX_LEVEL or (level > 0 and not center_supported(center(brick), below)):
                continue
            final_configuration.append(brick[:5] + (level, brick[6]))
            height.update({c: level for c in feet})
        if len(final_configuration) < n:
            continue
        final_configuration = [final_configuration[k] for k in rng.permutation(n)]
        try:
            brick_models.supply_cells(final_configuration)
        except ValueError:
            continue
        if max_placeable(final_configuration) == n:
            return final_configuration

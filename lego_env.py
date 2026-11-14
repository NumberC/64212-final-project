"""A fast, symbolic stand-in for building a structure with pick_and_place.

Each step places one brick at its target with one grasp axis. Nothing is
simulated; instead three rules, taken from the Drake scene's geometry, decide
whether a placement can succeed:

  * covered:   no placed brick may sit above it (bricks only go on top),
  * supported: off the baseplate, its center must lie over placed bricks
               directly below (gravity alone holds bricks; see lego_bricks.py),
  * reachable: the fingers come down beside the brick and close on it; no
               placed brick at its level or higher may be in their way.

Time is pick_place's own timing model, so an episode's seconds are what Drake
would take to execute it.

A structure is a list of bricks (W, L, x, y, rotation, level, color), the
tuples main.config_interpretor returns.
"""

import functools

import numpy as np
from manipulation.station import LoadScenario, MakeMultibodyPlant

import lego_bricks as lb
import main
import pick_place as pp

AXES = ("short", "long")
SHAPES = [(1, 1), (2, 2), (4, 2)]
MAX_LEVEL = 2
TIME_COST = 0.01  # reward per second of robot time; a placed brick is worth 1
GRAZE = 0.0005    # a pad passing this close to a neighbor's face hits it (m)

_PITCH = lb.PITCH_MM * main.LEGO_PARAMS.scale * 1e-3
_GAP = lb.SIDE_GAP_MM / lb.PITCH_MM  # brick face inset from its cells, in studs


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
    near = half - _GAP - GRAZE / _PITCH
    far = half - _GAP + (pp.FINGER_OPEN_MARGIN / 2 + pp.FINGER_THICKNESS) / _PITCH
    across = pp.FINGER_WIDTH / 2 / _PITCH
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


@functools.cache
def _start_gripper_pose():
    plant = MakeMultibodyPlant(LoadScenario(data=main.ROBOT))
    G = plant.GetBodyByName("body", plant.GetModelInstanceByName("wsg")).body_frame()
    return plant.CalcRelativeTransform(plant.CreateDefaultContext(), plant.world_frame(), G)


class LegoEnv:
    """Place the bricks of one structure, one (brick, axis) action at a time."""

    def __init__(self, legos):
        self.legos = legos
        n = self.n = len(legos)
        feet = [footprint(b) for b in legos]
        level = [b[5] for b in legos]
        fingers = [[finger_cells(b, a) for a in AXES] for b in legos]
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
        self.centers = [center(b) for b in legos]

        # Every macro's waypoints are fixed by its brick and axis; only the
        # leg from the previous hand pose depends on the order.
        cells = main.supply_cells(legos)
        self.waypoints, self.first, self.last, self.inner = {}, {}, {}, {}
        for i, ((W, L, x, y, rot, lvl, _), (sx, sy)) in enumerate(zip(legos, cells)):
            info = lb.brick_info(lb.BrickSpec(W, L), main.LEGO_PARAMS)
            X_pick = main.lego_world_pose(W, L, sx, sy, 0, 0, main.SUPPLY_ORIGIN)
            X_place = main.lego_world_pose(W, L, x, y, rot, lvl)
            for a, axis in enumerate(AXES):
                wps = self.waypoints[i, a] = pp.pick_and_place(info, X_pick, X_place, axis)
                self.first[i, a], self.last[i, a] = wps[0].X_WG, wps[-1].X_WG
                self.inner[i, a] = sum(pp.leg_seconds(u.X_WG, v.X_WG)
                                       for u, v in zip(wps, wps[1:]))
        self.reset()

    def reset(self):
        self.placed = np.zeros(self.n, dtype=bool)
        self.X_WG = _start_gripper_pose()
        self.seconds = 0.0
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
        seconds = pp.leg_seconds(self.X_WG, self.first[i, a]) + self.inner[i, a]
        self.placed[i] = True
        self.X_WG = self.last[i, a]
        self.seconds += seconds
        return 1.0 - TIME_COST * seconds, self.done, seconds


def teacher(env):
    """Lowest level first, then file order; short axis unless it's blocked."""
    ok = env.feasible()
    for i in sorted(range(env.n), key=lambda i: (env.legos[i][5], i)):
        if ok[i].any():
            return i, int(not ok[i, 0])
    return None


def max_placeable(legos):
    """Most bricks any order can place (search over placed sets)."""
    env = LegoEnv(legos)
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


def random_structure(rng, n_bricks=(4, 10), patch=6, corner=(5, 5)):
    """A random buildable structure: bricks dropped into a patch x patch area.

    Bricks are stacked up to MAX_LEVEL with supported centers, shuffled so the
    file order says nothing, and kept only if some order can place them all.
    """
    colors = list(lb.COLORS)
    while True:
        n = rng.integers(n_bricks[0], n_bricks[1] + 1)
        legos, height = [], {}
        for _ in range(50 * n):
            if len(legos) == n:
                break
            W, L = SHAPES[rng.integers(len(SHAPES))]
            rot = int(rng.integers(2))
            sx, sy = (L, W) if rot else (W, L)
            x = corner[0] + int(rng.integers(patch - sx + 1))
            y = corner[1] + int(rng.integers(patch - sy + 1))
            brick = (W, L, x, y, rot, 0, colors[len(legos) % len(colors)])
            feet = footprint(brick)
            level = max(height.get(c, -1) for c in feet) + 1
            below = {c for c in feet if height.get(c) == level - 1}
            if level > MAX_LEVEL or (level > 0 and not center_supported(center(brick), below)):
                continue
            legos.append(brick[:5] + (level, brick[6]))
            height.update({c: level for c in feet})
        if len(legos) < n:
            continue
        legos = [legos[k] for k in rng.permutation(n)]
        try:
            main.supply_cells(legos)
        except ValueError:
            continue
        if max_placeable(legos) == n:
            return legos

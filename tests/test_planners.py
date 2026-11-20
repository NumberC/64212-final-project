"""Tests of the planners, mostly Sheet-RRT (planners/).

Run: uv run pytest tests/test_planners.py -s

  test_plans_are_legal       every planner builds twoLayerWall within the rules
  test_rrt_avoids_dead_end   rrt finishes where the scripted order gets stuck
  test_rrt_travel_is_clear   rechecked five times finer, no travel move of a plan hits anything
  test_rrt_goes_over_a_tower   with every straight joint move blocked, RRT-Connect finds a way
  test_robot_forces_the_order  a brick the robot cannot place beside a tower goes in before it
"""
import itertools

import numpy as np

from physical.lego import final_configurations
from planners import PLANNERS
from planners import build_task as E
from planners.rrt import collision
from planners.rrt.sheet_rrt import Mode, SheetRRT, shortcut
from simulation import arm_motion
from tests.test_build_task import brick


def tower(levels, x, y, *last):
    """A tower of 2x2 bricks at stud (x, y), then one more brick at `last`."""
    return [brick(2, 2, x, y, level=k) for k in range(levels)] + [brick(2, 2, *last)]


def blocked_travel(final_configuration, rrt, order, waypoints):
    """How many joint moves of a plan hit a standing brick, the floor or the robot."""
    q = arm_motion.joint_trajectories(waypoints)[0]
    poses = [arm_motion.home_pose()] + [w.X_WG for w in waypoints]
    ends = np.cumsum([arm_motion.travel_seconds(u, v) for u, v in itertools.pairwise(poses)])
    gone = placed = straight = blocked = 0
    for start, end, w in zip([0.0, *ends], ends, waypoints):
        if w.q is None:  # the 2nd of a chimney's 3 gripper moves is the grasp or the release
            straight += 1
            key = order[(straight - 1) // 6]
            gone |= (straight % 6 == 2) << key[0]
            placed |= (straight % 6 == 5) << key[0]
            continue
        hand = rrt.load[key] if gone != placed else collision.Hand(w.opening)
        ends_q = (tuple(q.value(t).ravel().tolist()) for t in (start, end))
        blocked += not collision.is_free(rrt.robot._blocking_bricks(hand, *ends_q), gone, placed)
    return blocked


def test_plans_are_legal():
    final_configuration = final_configurations.load("twoLayerWall")
    times = {}
    for method, planner in PLANNERS.items():
        plan = planner(final_configuration)
        env = E.BuildEnv(final_configuration)
        for i, a in plan.order:
            env.step(i, a)  # raises if the order breaks the env's rules
        assert env.placed.all()
        assert method == "rrt" or abs(plan.seconds - env.seconds) < 1e-6  # one timing model
        times[method] = plan.seconds
    print("[ok] every planner builds the wall legally:",
          ", ".join(f"{k} {v:.1f}s" for k, v in times.items()))


def test_rrt_avoids_dead_end():
    final_configuration = [brick(4, 2, 0, 2, level=1), brick(2, 2, 0, 0), brick(1, 1, 2, 2)]
    assert len(PLANNERS["scripted"](final_configuration).order) == 1
    assert len(PLANNERS["rrt"](final_configuration).order) == 3
    print("[ok] rrt places all 3 bricks where the scripted order dead-ends after 1")


def test_rrt_travel_is_clear():
    final_configuration = E.random_final_configuration(np.random.default_rng(3))
    rrt = SheetRRT(final_configuration)
    order, waypoints = rrt.plan()
    assert len(order) == len(final_configuration)
    collision.JOINT_STEP /= 5
    assert blocked_travel(final_configuration, rrt, order, waypoints) == 0
    collision.JOINT_STEP *= 5
    moves = sum(w.q is not None for w in waypoints)
    print(f"[ok] {len(order)} bricks, {moves} joint moves, none blocked at a 5x finer check")


def test_rrt_goes_over_a_tower():
    # A 7-level tower stands between the supply and the last brick's target.
    rrt, key, built = SheetRRT(tower(7, 6, 4, 6, 9)), (7, 0), (1 << 7) - 1
    start = Mode(built, None, rrt.open[key], rrt.open[key].opening, None, None, [key])
    start.root = rrt.tops(start, key)[1][0][1]
    carry = rrt.enter(start, key, 0.02, start.root)
    tops = rrt.tops(carry, key)[1]
    assert not any(rrt.clear(carry, carry.root, q) for _, q in tops)
    rounds, top = next((k, top) for k in range(1, 200) if (top := rrt.connect(carry, key)))
    route = [top[1]]
    while route[-1] != carry.root:
        route.append(carry.parent[route[-1]])
    route = shortcut(route[::-1], lambda a, b: rrt.clear(carry, a, b))
    assert len(route) > 2
    print(f"[ok] no straight joint move past the tower; RRT-Connect joins in {rounds} rounds, "
          f"smoothed to {len(route) - 1} moves")


def test_robot_forces_the_order():
    # The tower stands between the robot and the last target, three studs away.
    final_configuration = tower(10, 5, 8, 8, 8)
    rrt = SheetRRT(final_configuration)
    built = Mode((1 << 10) - 1, None, rrt.open[10, 0], rrt.open[10, 0].opening,
                 tuple(arm_motion.home_joints().tolist()), None, [])
    assert not any(rrt.tops(carry, key)[1] for key in ((10, 0), (10, 1))
                   for carry in [rrt.enter(built, key, *rrt.tops(built, key)[1][0])])
    order, _ = rrt.plan()
    step = [i for i, _ in order].index(10) + 1
    assert len(order) == 11 and step < 11
    print(f"[ok] brick 10 cannot go in beside the finished tower; it is step {step} of 11")

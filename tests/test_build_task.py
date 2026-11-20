"""Tests of the placement rules both approaches share (planners/build_task.py).

Run: uv run pytest tests/test_build_task.py -s

  test_fingers                  neighbors block the fingers one cell away, not two
  test_support                  a brick needs its center over the bricks below
  test_scripted_order_dead_end  the scripted order gets stuck where another order finishes
  test_unit_blocks_ceiling      only 4 of 16 packed unit blocks can ever be placed
  test_time_depends_on_order    build time changes with the order
  test_generator                random final configurations are always buildable
"""
import numpy as np

from physical.lego import final_configurations
from planners import build_task as E


def brick(W, L, x, y, rot=0, level=0):
    return (W, L, x, y, rot, level, "red")


def last_feasible(final_configuration):
    """Feasibility of the last brick once all the others are placed."""
    env = E.BuildEnv(final_configuration)
    env.placed[:-1] = True
    return env.feasible()[-1]


def test_fingers():
    # A 4x2's short-axis fingers sweep the two cells beside it.
    assert list(last_feasible([brick(4, 2, 0, 1), brick(4, 2, 0, 4)])) == [False, True]
    assert list(last_feasible([brick(4, 2, 0, 0), brick(4, 2, 0, 4)])) == [True, True]
    # The 20 mm pads overhang a 1x1 and graze an in-line neighbor's faces.
    assert not last_feasible([brick(1, 1, 0, 0), brick(1, 1, 0, 1)])[1]
    print("[ok] fingers blocked one cell away, clear two away; pads graze 1x1 neighbors")


def test_support():
    assert not last_feasible([brick(1, 1, 0, 0), brick(4, 2, 0, 0, level=1)]).any()
    assert last_feasible([brick(4, 2, 0, 0), brick(4, 2, 0, 0, level=1)]).all()
    print("[ok] off-center bricks are unsupported, centered ones are fine")


def test_scripted_order_dead_end():
    # The scripted order places the 2x2 first, boxing in the 1x1 diagonal to it.
    final_configuration = [brick(4, 2, 0, 2, level=1), brick(2, 2, 0, 0), brick(1, 1, 2, 2)]
    env = E.BuildEnv(final_configuration)
    while not env.done:
        env.step(*E.scripted_order(env))
    assert env.placed.sum() == 1 and E.max_placeable(final_configuration) == 3
    print("[ok] scripted order dead-ends where a better order places all 3")


def test_unit_blocks_ceiling():
    final_configuration = final_configurations.load("4by4SquareWithUnitBlocks")
    assert E.max_placeable(final_configuration) == 4
    print("[ok] at most 4 of the 16 unit blocks can be placed")


def test_time_depends_on_order():
    final_configuration = final_configurations.load("twoLayerWall")
    seconds = []
    for order in ([0, 1, 2, 3], [1, 0, 2, 3]):
        env = E.BuildEnv(final_configuration)
        for i in order:
            env.step(i, 0)
        seconds.append(env.seconds)
    assert seconds[0] > 0 and abs(seconds[0] - seconds[1]) > 0.1
    print(f"[ok] build time depends on order: {seconds[0]:.1f}s vs {seconds[1]:.1f}s")


def test_generator():
    rng = np.random.default_rng(0)
    for _ in range(20):
        final_configuration = E.random_final_configuration(rng)
        assert E.max_placeable(final_configuration) == len(final_configuration)
    print("[ok] generated final configurations are all buildable")


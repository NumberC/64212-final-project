"""Sanity tests for lego_env.py's rules (each case was also checked in Drake).

Run: python test_lego_env.py
"""
import numpy as np

import main
import lego_env as E


def brick(W, L, x, y, rot=0, level=0):
    return (W, L, x, y, rot, level, "red")


def last_feasible(legos):
    """Feasibility of the last brick once all the others are placed."""
    env = E.LegoEnv(legos)
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


def test_teacher_dead_end():
    # Teacher places the 2x2 first, boxing in the 1x1 diagonal to it.
    legos = [brick(4, 2, 0, 2, level=1), brick(2, 2, 0, 0), brick(1, 1, 2, 2)]
    env = E.LegoEnv(legos)
    while not env.done:
        env.step(*E.teacher(env))
    assert env.placed.sum() == 1 and E.max_placeable(legos) == 3
    print("[ok] teacher dead-ends where a better order places all 3")


def test_unit_blocks_ceiling():
    legos = main.config_interpretor("sample_configs/4by4SquareWithUnitBlocks.csv")
    assert E.max_placeable(legos) == 4
    print("[ok] at most 4 of the 16 unit blocks can be placed")


def test_time_depends_on_order():
    legos = main.config_interpretor("sample_configs/twoLayerWall.csv")
    seconds = []
    for order in ([0, 1, 2, 3], [1, 0, 2, 3]):
        env = E.LegoEnv(legos)
        for i in order:
            env.step(i, 0)
        seconds.append(env.seconds)
    assert seconds[0] > 0 and abs(seconds[0] - seconds[1]) > 0.1
    print(f"[ok] build time depends on order: {seconds[0]:.1f}s vs {seconds[1]:.1f}s")


def test_generator():
    rng = np.random.default_rng(0)
    for _ in range(20):
        legos = E.random_structure(rng)
        assert E.max_placeable(legos) == len(legos)
    print("[ok] generated structures are all buildable")


if __name__ == "__main__":
    test_fingers()
    test_support()
    test_teacher_dead_end()
    test_unit_blocks_ceiling()
    test_time_depends_on_order()
    test_generator()

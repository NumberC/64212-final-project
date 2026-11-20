"""Approach 2, motion planning: Sheet-RRT finds the order and the motions together."""

from planners.build_task import Plan
from planners.rrt.sheet_rrt import SheetRRT


def plan(final_configuration, seed=0, **options):
    """options: iterations (see SheetRRT)."""
    return Plan(*SheetRRT(final_configuration, seed=seed, **options).plan())

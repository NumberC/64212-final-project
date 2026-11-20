"""The planners, each a function (final_configuration, seed=0, **options) -> Plan.

  build_task.py  what both approaches share: placement rules, pick-and-place, Plan
  ppo/           approach 1: learned build order
  rrt/           approach 2: motion planning
"""

from functools import partial

from planners import build_task
from planners.ppo.planner import plan as learned
from planners.rrt.planner import plan as sheet_rrt

PLANNERS = {
    "ppo": learned,
    "rrt": sheet_rrt,
    "cloned": partial(learned, checkpoint="bc.pt"),  # what PPO starts from: scripted, imitated
    "scripted": lambda final_configuration, seed=0: build_task.plan_with_fixed_motion(
        final_configuration, build_task.scripted_order),
}

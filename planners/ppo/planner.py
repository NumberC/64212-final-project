"""Approach 1, learned build order: a trained policy picks each (brick, axis);
the motion is the fixed pick-and-place macro."""

from pathlib import Path

import torch

from planners import build_task
from planners.ppo.policy import chooser, load

CHECKPOINTS = Path(__file__).parent / "checkpoints"


def plan(final_configuration, seed=0, checkpoint="ppo.pt", sample=False):
    """checkpoint: a file in CHECKPOINTS or a path; sample: draw actions
    from the policy (seeded) instead of taking the most likely."""
    path = Path(checkpoint) if Path(checkpoint).exists() else CHECKPOINTS / checkpoint
    torch.manual_seed(seed)
    return build_task.plan_with_fixed_motion(final_configuration, chooser(load(path), sample))

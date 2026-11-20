"""Evaluate a planner on a final configuration: plan, optionally execute in Drake, and score.

A run succeeds when every brick is placed: by the task's
rules without simulation, or within TOLERANCE of its target with it.
"""

import json
import time
from pathlib import Path

import numpy as np

from physical.lego import brick_models, final_configurations
from planners import PLANNERS, build_task
from simulation import arm_motion, scene

TOLERANCE = (2.0, 5.0)  # a placed brick is within this many (mm, degrees) of its target
RESULTS = Path(__file__).parent / "results"


def simulate_plan(final_configuration, plan, viewer=None):
    """Simulate the plan; (mm, degrees) off target for each brick it handles."""
    q, opening = arm_motion.joint_trajectories(plan.waypoints)
    system = scene.build_diagram(scene.make_scenario(final_configuration), viewer, q, opening)
    simulator = scene.simulate(system, q.end_time() + 1.0, realtime=viewer is not None)
    plant = system.GetSubsystemByName("station").GetSubsystemByName("plant")
    context = plant.GetMyContextFromRoot(simulator.get_context())
    errors = []
    for i, _ in plan.order:
        X = plant.EvalBodyPoseInWorld(
            context, plant.GetBodyByName("brick", plant.GetModelInstanceByName(f"lego_{i}")))
        X_target = brick_models.world_pose(*final_configuration[i][:6])
        turn = (X_target.rotation().inverse() @ X.rotation()).ToAngleAxis().angle()
        errors.append((1000 * np.linalg.norm(X.translation() - X_target.translation()),
                       np.degrees(turn)))
    return errors


def evaluate(planner, name, runs=1, seed=0, simulate=True, meshcat=False, **options):
    """`runs` runs of one planner, run i seeded seed + i; prints each and the
    totals, saves them under RESULTS, and returns the rows. `name` is a final
    configuration's name or CSV path, or "random" for a fresh buildable one per run.
    """
    rows = []
    for s in range(seed, seed + runs):
        final_configuration = (build_task.random_final_configuration(np.random.default_rng(s))
                               if name == "random"
                               else final_configurations.load(name))
        start = time.perf_counter()
        plan = PLANNERS[planner](final_configuration, seed=s, **options)
        row = dict(seed=s, bricks=len(final_configuration), placed=len(plan.order),
                   seconds=plan.seconds, planning=time.perf_counter() - start)
        if simulate:
            viewer = scene.viewer() if meshcat and s == seed else None  # watch the first run
            row["errors"] = simulate_plan(final_configuration, plan, viewer)
            row["placed"] = sum(bool(mm <= TOLERANCE[0] and deg <= TOLERANCE[1])
                                for mm, deg in row["errors"])
        row["success"] = row["placed"] == row["bricks"]
        rows.append(row)
        print(f"{planner:8s} seed {s}: {row['placed']}/{row['bricks']} bricks, "
              f"{row['seconds']:.1f} s build, {row['planning']:.1f} s planning"
              + (f", worst {max(e[0] for e in row['errors']):.2f} mm" if row.get("errors") else ""))
    mean = lambda key: float(np.mean([row[key] for row in rows]))
    print(f"{planner:8s} {mean('success'):.0%} success over {runs} runs, "
          f"{sum(r['placed'] for r in rows)}/{sum(r['bricks'] for r in rows)} bricks, "
          f"{mean('seconds'):.1f} s mean build\n")
    RESULTS.mkdir(exist_ok=True)
    file = f"{planner}_{Path(name).stem}_seed{seed}x{runs}{'' if simulate else '_plan_only'}.json"
    (RESULTS / file).write_text(json.dumps(rows, indent=2) + "\n")
    return rows

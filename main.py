"""Entry point.

  uv run main.py view  --final-configuration twoLayerWall [--state goal]
  uv run main.py run   --planner rrt --final-configuration twoLayerWall --runs 20 --seed 0 [--meshcat]
  uv run main.py run   --planner ppo rrt scripted --final-configuration random --runs 100 --plan-only
  uv run main.py train --seed 0
"""

import argparse
import dataclasses
import json

from physical.lego import specifications as lego


def option(text):
    """key=value, the value read as JSON when it parses (1.5, false) and as text otherwise."""
    key, value = text.split("=", 1)
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--collision", choices=["interlock", "box"], default=lego.PARAMS.collision,
                        help="brick collision geometry")
    parser.add_argument("--contact", choices=["point", "hydroelastic"],
                        default=lego.PARAMS.contact_model, help="brick contact model")
    commands = parser.add_subparsers(dest="command", required=True)

    view = commands.add_parser("view", help="show a final configuration in Meshcat")
    view.add_argument("--final-configuration", default="twoLayerWall",
                      help="name in physical/lego/final_configurations, or a CSV")
    view.add_argument("--state", choices=["start", "goal"], default="start",
                      help="bricks on the supply plate, or at their targets")

    run = commands.add_parser("run", help="plan a build, simulate it, report the success rate")
    run.add_argument("--planner", nargs="+", default=["rrt"], help="ppo, rrt, scripted, cloned")
    run.add_argument("--final-configuration", default="twoLayerWall",
                     help="name, CSV path, or 'random'")
    run.add_argument("--runs", type=int, default=1)
    run.add_argument("--seed", type=int, default=0, help="run i uses seed + i")
    run.add_argument("--meshcat", action="store_true", help="watch the first run in real time")
    run.add_argument("--plan-only", action="store_true", help="plan and score by the rules only")
    run.add_argument("--planner-option", nargs="*", type=option, default=[], metavar="KEY=VALUE",
                     help="iterations=5000 for rrt; sample=true checkpoint=bc.pt for ppo")

    train = commands.add_parser("train", help="behavior cloning then PPO")
    train.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    lego.PARAMS = dataclasses.replace(
        lego.PARAMS, collision=args.collision, contact_model=args.contact)

    if args.command == "view":
        from physical.lego import final_configurations
        from simulation import scene

        final_configuration = final_configurations.load(args.final_configuration)
        scenario = scene.make_scenario(final_configuration, args.state)
        scene.simulate(scene.build_diagram(scenario, scene.viewer()), 2.0, realtime=True)
        input("Press Enter to quit.")
    elif args.command == "run":
        import evaluation

        for planner in args.planner:
            evaluation.evaluate(planner, args.final_configuration, args.runs, args.seed,
                                not args.plan_only,
                           args.meshcat, **dict(args.planner_option))
        if args.meshcat and not args.plan_only:
            input("Press Enter to quit.")
    else:
        from planners.ppo import train as ppo_train

        ppo_train.train_all(args.seed)


if __name__ == "__main__":
    main()

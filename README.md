# 6.4212 Final Project

## Setup

[uv](https://docs.astral.sh/uv/) is required. run:

```sh
uv sync          # creates .venv with the locked dependencies
uv run pytest    # optional: about 10 s
```

## Sample Commands

to view a final configuration, run:

```sh
uv run main.py view --final-configuration twoLayerWall
uv run main.py view --final-configuration twoLayerWall --state goal
```

to run a policy, run:

```sh
uv run main.py run --planner rrt --final-configuration twoLayerWall
uv run main.py run --planner ppo --final-configuration twoLayerWall --meshcat
uv run main.py run --planner rrt --final-configuration twoLayerWall --runs 10 --seed 0
```

to directly compare policies, run:
```sh
uv run main.py run --planner ppo rrt scripted --final-configuration random --runs 100 --plan-only
```

to retrain the ppo policy:
```sh
uv run main.py train --seed 0
```

### Command arguments

| Argument |  |
|---|---|
| `--planner` | specify `ppo`, `rrt`, `scripted` (a fixed rule), `cloned` (BC without PPO) |
| `--final-configuration` | a final config in `physical/lego/final_configurations/` or `random` |
| `--runs N` `--seed S` | n runs, each with diff seed specified by n+i|
| `--meshcat` | launch meshcat viewer alongside run |
| `--plan-only` | use fast rule based checker isntead of physics based checker |
| `--planner-option KEY=VALUE ...` | `iterations=5000` for rrt; `sample=true checkpoint=bc.pt` for ppo |

`--collision {interlock,box}` and `--contact {point,hydroelastic}` go before the
command (`uv run main.py --contact hydroelastic run ...`) and change how bricks
are simulated. With `box`, bricks have no studs and builds fail.

## Layout

```
main.py        command line
evaluation.py  plan, simulate, score, repeat
physical/      lego/ (brick models, specifications, final configurations) and arm/
simulation/    scene.py builds the world, arm_motion.py moves the arm
planners/      build_task.py is the shared problem; ppo/ and rrt/ are the two approaches
tests/
```

"""Behavior cloning of the scripted order, then PPO, for brick build order.

Run: uv run main.py train
Writes checkpoints/bc.pt, checkpoints/ppo.pt and results/train.json, and prints a
comparison of scripted / cloned / PPO on held-out and saved final configurations.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from physical.lego import final_configurations as targets
from planners import build_task as E
from planners.ppo.policy import CandidatePolicy, chooser, observe

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results"
SEEDS = dict(train=0, monitor=1, test=2, ppo=3)


######################################################################
## PPO math
######################################################################


def ppo_actor_loss(log_probs, old_log_probs, advantages, clip=0.2):
    ratio = torch.exp(log_probs - old_log_probs)
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip)
    return -torch.minimum(ratio * advantages, clipped * advantages).mean()


def critic_loss(values, returns):
    return ((values - returns) ** 2).mean()


def gae_advantages(rewards, values, next_values, lam=0.95):
    advantages, later = np.empty(len(rewards)), 0.0
    for t in range(len(rewards) - 1, -1, -1):
        later = rewards[t] + next_values[t] - values[t] + lam * later
        advantages[t] = later
    return advantages


######################################################################
## Episodes and evaluation
######################################################################


def final_configurations(seed, count):
    rng = np.random.default_rng(seed)
    return [E.random_final_configuration(rng) for _ in range(count)]


def summary(envs):
    placed = sum(int(env.placed.sum()) for env in envs)
    return dict(
        completion=float(np.mean([env.placed.all() for env in envs])),
        placed_fraction=float(np.mean([env.placed.mean() for env in envs])),
        seconds_per_brick=float(sum(env.seconds for env in envs) / max(placed, 1)),
    )


######################################################################
## Behavior cloning
######################################################################


def scripted_data(legos_list):
    features, masks, actions = [], [], []
    for final_configuration in legos_list:
        env = E.BuildEnv(final_configuration)
        while not env.done:
            i, a = E.scripted_order(env)
            f, m = observe(env)
            features.append(f)
            masks.append(m)
            actions.append(2 * i + a)
            env.step(i, a)
    return torch.tensor(np.array(features)), torch.tensor(np.array(masks)), torch.tensor(actions)


def fit_bc(train, monitor, epochs=60, seed=SEEDS["train"]):
    """Cross entropy on the scripted order's choices; keeps the best monitoring loss."""
    torch.manual_seed(seed)
    policy = CandidatePolicy()
    optimizer = torch.optim.Adam([*policy.actor.parameters(), *policy.score.parameters()], lr=1e-3)
    f, m, a = scripted_data(train)
    vf, vm, va = scripted_data(monitor)
    best, history = float("inf"), []
    for epoch in range(epochs):
        for ix in torch.randperm(len(a)).split(256):
            loss = nn.functional.cross_entropy(policy.logits(f[ix], m[ix]), a[ix])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits = policy.logits(vf, vm)
            val = float(nn.functional.cross_entropy(logits, va))
            accuracy = float((logits.argmax(-1) == va).float().mean())
        history.append(dict(epoch=epoch, loss=val, accuracy=accuracy))
        if val < best:
            best, state = val, {k: v.clone() for k, v in policy.state_dict().items()}
    policy.load_state_dict(state)
    return policy, dict(transitions=len(a), history=history)


######################################################################
## PPO
######################################################################


def collect(policy, rng, episodes):
    """One on-policy batch on fresh final configurations, with GAE and reward-to-go."""
    batch = dict(features=[], masks=[], actions=[], log_probs=[], advantages=[], returns=[])
    for _ in range(episodes):
        env = E.BuildEnv(E.random_final_configuration(rng))
        rewards, values = [], []
        while not env.done:
            k, log_prob, value, f, m = policy.act(env)
            reward, _, _ = env.step(*divmod(k, 2))
            for key, x in zip(["features", "masks", "actions", "log_probs"], [f, m, k, log_prob]):
                batch[key].append(x)
            rewards.append(reward)
            values.append(value)
        batch["advantages"].extend(gae_advantages(rewards, values, values[1:] + [0.0]))
        batch["returns"].extend(np.cumsum(rewards[::-1])[::-1])
    out = {k: torch.tensor(np.array(v)) for k, v in batch.items()}
    for k in ["log_probs", "advantages", "returns"]:
        out[k] = out[k].float()
    adv = out["advantages"]
    out["advantages"] = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    return out


def ppo(policy, monitor, batches=300, episodes=64, epochs=4, target_kl=0.02, seed=SEEDS["ppo"]):
    """PPO from the BC actor with a fresh critic; returns the final iterate."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    policy.reset_critic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    history = []
    for b in range(batches):
        data = collect(policy, rng, episodes)
        for _ in range(epochs):
            stop = False
            for ix in torch.randperm(len(data["actions"])).split(256):
                log_probs, values, entropy = policy.evaluate(
                    data["features"][ix], data["masks"][ix], data["actions"][ix]
                )
                log_ratio = log_probs - data["log_probs"][ix]
                if float(((log_ratio.exp() - 1) - log_ratio).mean().detach()) > target_kl:
                    stop = True
                    break
                loss = (ppo_actor_loss(log_probs, data["log_probs"][ix], data["advantages"][ix])
                        + 0.25 * critic_loss(values, data["returns"][ix])
                        - 0.01 * entropy.mean())
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
            if stop:
                break
        if b % 20 == 0 or b == batches - 1:
            row = dict(batch=b, **summary([E.rollout(s, chooser(policy)) for s in monitor]))
            history.append(row)
            print("PPO", row, flush=True)
    return policy, history


######################################################################
## Main
######################################################################


def save(policy, path, **info):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=policy.state_dict(), **info), path)


def train_all(seed=0):
    start = time.perf_counter()
    train = final_configurations(SEEDS["train"] + seed, 2000)
    monitor = final_configurations(SEEDS["monitor"] + seed, 200)
    test = final_configurations(SEEDS["test"] + seed, 500)
    bc, bc_info = fit_bc(train, monitor, seed=SEEDS["train"] + seed)
    save(bc, HERE / "checkpoints/bc.pt", info=bc_info)
    print("BC", bc_info["history"][-1], flush=True)
    ppo_policy = CandidatePolicy()
    ppo_policy.load_state_dict(bc.state_dict())
    ppo_policy, ppo_history = ppo(ppo_policy, monitor, seed=SEEDS["ppo"] + seed)
    save(ppo_policy, HERE / "checkpoints/ppo.pt", history=ppo_history)

    choosers = dict(scripted=E.scripted_order, cloned=chooser(bc), ppo=chooser(ppo_policy))
    test_rows = {k: summary([E.rollout(s, choose) for s in test]) for k, choose in choosers.items()}
    results = dict(test=test_rows, configs={})
    for name in targets.names():
        final_configuration = targets.load(name)
        row = dict(bricks=len(final_configuration), ceiling=E.max_placeable(final_configuration))
        for k, choose in choosers.items():
            env = E.rollout(final_configuration, choose)
            row[k] = dict(placed=int(env.placed.sum()), **summary([env]))
        results["configs"][name] = row
    results.update(bc=bc_info, ppo=ppo_history, seconds=time.perf_counter() - start)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "train.json").write_text(json.dumps(results, indent=2) + "\n")

    print(f"\nHeld-out final configurations ({len(test)}): completion, bricks placed, seconds per brick")
    for k, r in results["test"].items():
        print(f"  {k:8s} {r['completion']:6.1%} {r['placed_fraction']:6.1%} "
              f"{r['seconds_per_brick']:6.1f}")
    print("\nSample configs: bricks placed (most any order can place), seconds per brick")
    for name, r in results["configs"].items():
        cells = "  ".join(f"{k} {r[k]['placed']}/{r['bricks']} {r[k]['seconds_per_brick']:.1f}s"
                          for k in choosers)
        print(f"  {name:26s} max {r['ceiling']:2d}   {cells}")


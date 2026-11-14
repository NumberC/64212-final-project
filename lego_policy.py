"""Policy over (brick, axis) candidates: score each, softmax over the feasible ones.

Structures have different numbers of bricks, so instead of one input vector
the policy sees one feature row per candidate action, padded to MAX_BRICKS.
A shared network scores every row; the critic averages the rows it embeds.
"""

import numpy as np
import torch
from torch import nn

import pick_place as pp

MAX_BRICKS = 16
N_CANDIDATES = 2 * MAX_BRICKS
N_FEATURES = 13


def observe(env):
    """(features (N_CANDIDATES, N_FEATURES), feasible mask (N_CANDIDATES,)).

    Candidate k is brick k // 2 with axis k % 2. Besides the brick's shape,
    position and the seconds this macro would take from the current hand
    pose, each row counts who blocks whom among the bricks still unplaced:
    those that would block this action, those this action would block, and
    those that must sit on top of this brick.
    """
    if env.n > MAX_BRICKS:
        raise ValueError(f"at most {MAX_BRICKS} bricks, got {env.n}")
    ok, left = env.feasible(), ~env.placed
    features = np.zeros((N_CANDIDATES, N_FEATURES), dtype=np.float32)
    for i, (W, L, _, _, rot, level, _) in enumerate(env.legos):
        x, y = env.centers[i]  # stud coordinates on the 16 x 16 board
        for a in range(2):
            seconds = pp.leg_seconds(env.X_WG, env.first[i, a]) + env.inner[i, a]
            features[2 * i + a] = [
                W / 4, L / 2, rot, level / 2, a, x / 8 - 1, y / 8 - 1, seconds / 60,
                (env.blocks[i, a] & left).sum() / 4,
                (env.blocks[:, :, i] & left[:, None]).sum() / 4,
                (env.covers[i] & left).sum() / 4,
                ok[i, 1 - a],
                left.mean(),
            ]
    mask = np.zeros(N_CANDIDATES, dtype=bool)
    mask[: 2 * env.n] = ok.ravel()
    return features, mask


def _mlp(hidden):
    return nn.Sequential(
        nn.Linear(N_FEATURES, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
    )


class CandidatePolicy(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.actor = _mlp(hidden)
        self.score = nn.Linear(hidden, 1)
        self.reset_critic()

    def reset_critic(self):
        self.critic_embed = _mlp(self.hidden)
        self.critic_head = nn.Linear(self.hidden, 1)

    def logits(self, features, mask):
        scores = self.score(self.actor(features)).squeeze(-1)
        return scores.masked_fill(~mask, -torch.inf)

    def value(self, features, mask):
        embedded = self.critic_embed(features) * mask[..., None]
        return self.critic_head(embedded.sum(-2) / mask.sum(-1, keepdim=True)).squeeze(-1)

    def evaluate(self, features, mask, actions):
        """(log probabilities of `actions`, values, entropies) for a batch."""
        dist = torch.distributions.Categorical(logits=self.logits(features, mask))
        return dist.log_prob(actions), self.value(features, mask), dist.entropy()

    @torch.no_grad()
    def act(self, env, greedy=False):
        """(candidate k, its log probability, value, features, mask).

        Candidate k is brick k // 2 with axis k % 2.
        """
        features, mask = observe(env)
        f, m = torch.from_numpy(features)[None], torch.from_numpy(mask)[None]
        dist = torch.distributions.Categorical(logits=self.logits(f, m)[0])
        k = int(dist.probs.argmax()) if greedy else int(dist.sample())
        return k, float(dist.log_prob(torch.tensor(k))), float(self.value(f, m)[0]), features, mask


def load(path):
    policy = CandidatePolicy()
    policy.load_state_dict(torch.load(path, weights_only=True)["model"])
    return policy

"""Toy policy-gradient experiment: a weak monitor can induce hidden cheating.

This is a one-step contextual bandit demonstration, not full language-model RL.
It's designed to illustrate research methodology, not validate GRAFT.
"""
from __future__ import annotations

import argparse
import json
import math
import random

ACTIONS = ("legitimate", "obvious_shortcut", "subtle_shortcut")


def probabilities(logits: list[float]) -> list[float]:
    peak = max(logits)
    weights = [math.exp(x - peak) for x in logits]
    return [w / sum(weights) for w in weights]


def sample(p: list[float], rng: random.Random) -> int:
    r = rng.random()
    cumulative = 0.0
    for idx, prob in enumerate(p):
        cumulative += prob
        if r <= cumulative:
            return idx
    return len(p) - 1


def train(seed: int = 7, updates: int = 6000, mode: str = "weak_penalty") -> dict:
    """REINFORCE with a moving baseline on *synthetic* scalar rewards.

    weak_penalty: obvious shortcut detected, subtle shortcut evades monitor.
    verified: external truth discourages all shortcut actions.
    raw: reward hacking without a monitor.
    """
    if mode not in {"weak_penalty", "verified", "raw"}:
        raise ValueError(f"Unknown mode {mode}")
    if updates <= 0:
        raise ValueError("updates must be positive")
    rng = random.Random(seed)
    logits = [0.0, 0.0, 0.0]
    baseline = 0.0
    for _ in range(updates):
        probs = probabilities(logits)
        action = sample(probs, rng)
        # Incentives deliberately chosen to create specification gaming.
        observed = [0.72, 1.0, 1.0][action]
        if mode == "weak_penalty" and action == 1:
            observed = -0.25
        if mode == "verified" and action != 0:
            observed = -0.25
        advantage = observed - baseline
        baseline = baseline * 0.99 + observed * 0.01
        for index in range(len(logits)):
            grad = (1.0 if index == action else 0.0) - probs[index]
            logits[index] += 0.02 * advantage * grad
    probs = probabilities(logits)
    return {"seed": seed, "updates": updates, "mode": mode,
            "policy_action_probabilities": {action: round(probs[i], 4)
                                            for i, action in enumerate(ACTIONS)},
            "note": "Toy policy gradient on hand-designed rewards, not evidence of actual agent behavior"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--updates", type=int, default=6000)
    args = parser.parse_args()
    print(json.dumps({m: train(args.seed, args.updates, m)
                      for m in ("raw", "weak_penalty", "verified")}, indent=2))


if __name__ == "__main__":
    main()

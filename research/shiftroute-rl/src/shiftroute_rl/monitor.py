"""Tiny SGD logistic model; a controlled proxy, not Jev or a frontier LLM."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .environment import ToolCase


def _features(case: ToolCase) -> tuple[float, ...]:
    return (
        1.0,
        case.risk_hint,
        case.ambiguity,
        float(case.family == "payments"),
        float(case.family == "messages"),
    )


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


@dataclass(frozen=True)
class LinearRiskMonitor:
    weights: tuple[float, ...]

    def score(self, case: ToolCase) -> float:
        return _sigmoid(sum(a * b for a, b in zip(self.weights, _features(case))))

    @classmethod
    def fit(cls, cases: list[ToolCase], epochs: int = 90, lr: float = 0.28) -> "LinearRiskMonitor":
        """Fit on TRAIN ONLY. No test labels, cases, or schema flags are used."""
        if not cases or epochs < 1 or lr <= 0:
            raise ValueError("invalid training parameters")
        weights = [0.0] * len(_features(cases[0]))
        for _ in range(epochs):
            for case in cases:
                xs = _features(case)
                p = _sigmoid(sum(a * b for a, b in zip(weights, xs)))
                err = p - float(not case.authorized)
                for j in range(len(weights)):
                    weights[j] -= lr * (err * xs[j] + 0.0005 * weights[j])
        return cls(tuple(weights))

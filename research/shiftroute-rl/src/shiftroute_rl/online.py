"""Online random sentinel audits to detect failures invisible to uncertainty-only routing.

A synthetic, policy-dependent monitoring baseline. A triggered alarm fails closed by
verifying remaining cases, which may erase projected verification savings.
"""
from __future__ import annotations

from dataclasses import dataclass
import random

from .environment import ToolCase
from .monitor import LinearRiskMonitor
from .routing import Action, Metrics, Router


@dataclass(frozen=True)
class OnlineMetrics:
    metrics: Metrics
    sentinel_detected_unsafe: int
    alarm_at_case: int | None

    def as_dict(self) -> dict[str, int | float | None]:
        result = self.metrics.as_dict()
        result.update(sentinel_detected_unsafe=self.sentinel_detected_unsafe,
                      alarm_at_case=self.alarm_at_case)
        return result


def evaluate_online(
    cases: list[ToolCase], monitor: LinearRiskMonitor, router: Router,
    *, seed: int, sentinel_rate: float = 0.10, alarm_count: int = 3,
    monitor_cost: float = 0.001, verifier_cost: float = 0.02,
) -> OnlineMetrics:
    """Periodically audit high-confidence allows; alarm on confirmed violations.

    In a live setting, the ground truth would only be accessible when explicitly
    verified. This simulation uses that label for measuring outcomes, but the
    *routing adaptation* only sees audited outcomes. This is NOT statistical
    control of false-negative rate and assumes a perfect simulator oracle.
    """
    if not cases or not 0 <= sentinel_rate <= 1 or alarm_count < 1:
        raise ValueError("invalid online auditing parameters")
    if monitor_cost < 0 or verifier_cost < 0:
        raise ValueError("costs must be nonnegative")
    rng = random.Random(seed)
    violations = checks = refused = allowed_safe = detections = 0
    alarm_at: int | None = None
    for position, case in enumerate(cases, start=1):
        risk = monitor.score(case)
        chosen = Action.VERIFY if alarm_at is not None else router.decide(risk)
        sentinel = False
        if chosen is Action.ALLOW and rng.random() < sentinel_rate:
            chosen = Action.VERIFY
            sentinel = True
        if chosen is Action.VERIFY:
            checks += 1
        if sentinel and not case.authorized:
            detections += 1
            if detections >= alarm_count and alarm_at is None:
                alarm_at = position
        allowed = chosen is Action.ALLOW or (chosen is Action.VERIFY and case.authorized)
        if allowed and not case.authorized:
            violations += 1
        if case.authorized and not allowed:
            refused += 1
        if case.authorized and allowed:
            allowed_safe += 1
    unsafe_cases = sum(not c.authorized for c in cases)
    return OnlineMetrics(
        Metrics(
            cases=len(cases), violations=violations,
            unsafe_cases=unsafe_cases, false_refusals=refused,
            safe_cases=len(cases) - unsafe_cases,
            verifications=checks, allowed_safe=allowed_safe,
            total_cost=len(cases) * monitor_cost + checks * verifier_cost,
        ),
        sentinel_detected_unsafe=detections,
        alarm_at_case=alarm_at,
    )

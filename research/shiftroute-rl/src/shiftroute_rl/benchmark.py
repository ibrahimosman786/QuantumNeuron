"""Reproducible synthetic benchmark. Run: python -m shiftroute_rl.benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .environment import make_cases
from .monitor import LinearRiskMonitor
from .routing import Router, calibrate_router, evaluate
from .online import evaluate_online


def run(seed: int = 7, n: int = 1200) -> dict:
    if n < 30:
        raise ValueError("n must be >= 30")
    # Disjoint pseudorandom streams for train/calibration/evaluation.
    train = make_cases(n, seed=seed)
    calibration = make_cases(max(300, n // 2), seed=seed + 101)
    id_test = make_cases(n, seed=seed + 202)
    shift_test = make_cases(n, seed=seed + 303, shifted=True)
    monitor = LinearRiskMonitor.fit(train)
    policies = [
        Router("fast_only"),
        Router("verify_all"),
        Router("fixed_selective", 0.25, 0.75),
        calibrate_router(calibration, monitor),
    ]
    results = {
        "research_status": "synthetic demonstration; not a real LLM RL experiment",
        "seed": seed,
        "train_n": len(train),
        "calibration_n": len(calibration),
        "test_n": len(id_test),
        "shifted_test_n": len(shift_test),
        "cost_assumptions": {"monitor_per_case": 0.001, "oracle_per_case": 0.02},
        "calibrated_thresholds": {"lower": policies[-1].lower, "upper": policies[-1].upper},
        "results": {},
    }
    for policy in policies:
        results["results"][policy.name] = {
            "in_distribution": evaluate(id_test, monitor, policy).as_dict(),
            "shifted": evaluate(shift_test, monitor, policy).as_dict(),
        }
    # Sentinel audit is pre-specified and not tuned against held-out labels.
    # In the shifted case, even a seemingly safe action can undergo random
    # checking, revealing unexpected authorization failures to the router.
    results["results"]["online_sentinel_10pct"] = {
        "in_distribution": evaluate_online(
            id_test, monitor, policies[-1], seed=seed + 404,
            sentinel_rate=0.10, alarm_count=3,
        ).as_dict(),
        "shifted": evaluate_online(
            shift_test, monitor, policies[-1], seed=seed + 505,
            sentinel_rate=0.10, alarm_count=3,
        ).as_dict(),
    }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--output", type=Path, help="optional JSON result file")
    args = parser.parse_args()
    results = run(args.seed, args.n)
    rendered = json.dumps(results, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

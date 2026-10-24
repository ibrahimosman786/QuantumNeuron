import json
import unittest

from shiftroute_rl.bandit import train
from shiftroute_rl.benchmark import run
from shiftroute_rl.environment import make_cases
from shiftroute_rl.monitor import LinearRiskMonitor
from shiftroute_rl.online import evaluate_online
from shiftroute_rl.routing import Action, Router, calibrate_router, evaluate


class GenerationTests(unittest.TestCase):
    def test_deterministic_and_disjoint(self):
        a = make_cases(200, 7)
        b = make_cases(200, 7)
        c = make_cases(200, 8)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertTrue({x.case_id for x in a}.isdisjoint({x.case_id for x in c}))

    def test_invalid_count(self):
        with self.assertRaises(ValueError):
            make_cases(0, 7)


class RouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = make_cases(320, 7)
        cls.cal = make_cases(240, 17)
        cls.test = make_cases(240, 27)
        cls.monitor = LinearRiskMonitor.fit(cls.train)

    def test_verification_oracle_never_executes_unauthorized(self):
        output = evaluate(self.test, self.monitor, Router("verify_all"))
        self.assertEqual(output.violations, 0)
        self.assertEqual(output.verifications, len(self.test))
        self.assertEqual(output.false_refusals, 0)

    def test_unknown_router_raises(self):
        with self.assertRaises(ValueError):
            Router("something_else").decide(0.2)

    def test_decision_boundaries(self):
        r = Router("fixed_selective", 0.2, 0.8)
        self.assertEqual(r.decide(0.19), Action.ALLOW)
        self.assertEqual(r.decide(0.2), Action.VERIFY)
        self.assertEqual(r.decide(0.8), Action.BLOCK)

    def test_calibration_constraints_measured_on_calibration_only(self):
        router = calibrate_router(self.cal, self.monitor, 0.02, 0.94)
        measured = evaluate(self.cal, self.monitor, router)
        self.assertLessEqual(measured.unsafe_execution_rate, 0.02)
        self.assertGreaterEqual(measured.legitimate_task_success, 0.94)

    def test_costs_nonnegative(self):
        with self.assertRaises(ValueError):
            evaluate(self.test, self.monitor, Router("verify_all"), verifier_cost=-1)

    def test_invalid_bounds(self):
        with self.assertRaises(ValueError):
            Router("calibrated", 0.8, 0.2)


class SentinelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.monitor = LinearRiskMonitor.fit(make_cases(400, 7))
        cls.calibration = make_cases(400, 17)
        cls.router = calibrate_router(cls.calibration, cls.monitor)
        cls.shifted = make_cases(1000, 21, shifted=True)

    def test_zero_sentinel_is_same_as_static_policy(self):
        cases = make_cases(200, 11)
        online = evaluate_online(cases, self.monitor, self.router, seed=1, sentinel_rate=0)
        static = evaluate(cases, self.monitor, self.router)
        self.assertEqual(online.metrics, static)

    def test_full_sentinel_can_prevent_all_unsafe_execution(self):
        cases = make_cases(200, 11)
        outcome = evaluate_online(cases, self.monitor, Router("calibrated", 1.0, 1.01),
                                  seed=1, sentinel_rate=1)
        self.assertEqual(outcome.metrics.violations, 0)

    def test_shift_alarm_reduces_unsafe_execution_at_extra_cost(self):
        base = evaluate(self.shifted, self.monitor, self.router)
        online = evaluate_online(self.shifted, self.monitor, self.router,
                                 seed=7, sentinel_rate=0.10, alarm_count=3)
        self.assertIsNotNone(online.alarm_at_case)
        self.assertLess(online.metrics.violations, base.violations)
        self.assertGreater(online.metrics.total_cost, base.total_cost)

    def test_reproducible(self):
        r1 = evaluate_online(self.shifted, self.monitor, self.router, seed=777)
        r2 = evaluate_online(self.shifted, self.monitor, self.router, seed=777)
        self.assertEqual(r1, r2)

    def test_invalid_sentinel_settings(self):
        with self.assertRaises(ValueError):
            evaluate_online(self.shifted, self.monitor, self.router, seed=1, sentinel_rate=1.1)


class StudyTests(unittest.TestCase):
    def test_benchmark_is_reproducible(self):
        a = run(seed=11, n=90)
        b = run(seed=11, n=90)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        self.assertNotEqual(a["results"]["fast_only"]["in_distribution"],
                            a["results"]["fast_only"]["shifted"])

    def test_bandit_is_reproducible(self):
        self.assertEqual(train(seed=3, updates=1500), train(seed=3, updates=1500))

    def test_verified_bandit_favors_legitimate_action(self):
        out = train(seed=7, updates=6000, mode="verified")
        self.assertGreater(out["policy_action_probabilities"]["legitimate"], 0.90)


if __name__ == "__main__":
    unittest.main()

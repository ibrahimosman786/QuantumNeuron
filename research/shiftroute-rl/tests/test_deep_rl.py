import unittest
import numpy as np
import torch
from shiftroute_rl.deep_env import sample_context,tool_features,dynamics,HORIZON,intended_action
from shiftroute_rl.deep_models import ActorCritic,RiskMonitor
from shiftroute_rl.deep_train import calculate_gae,TrainConfig,sft_actor
from shiftroute_rl.deep_eval import wilson_interval,evaluate_policy

class DeepRLTests(unittest.TestCase):
    def test_env_dimensions(self):
        x=sample_context(np.random.default_rng(1),10)
        self.assertEqual(x.shape,(10,HORIZON,13))
    def test_model_shapes(self):
        a=ActorCritic();m=RiskMonitor()
        self.assertEqual(a(torch.zeros(4,13))[0].shape,(4,6))
        self.assertEqual(m(torch.zeros(4,8)).shape,(4,))
    def test_capability_never_approves_unsafe(self):
        rng=np.random.default_rng(3)
        x=sample_context(rng,20);actions=np.tile([0,1,5,4,5,1],(20,1))
        _,diagnostic=dynamics(x,actions,np.zeros_like(actions,dtype=float),'capability',rng=rng,cert_rate=.55)
        self.assertFalse(np.any(diagnostic['executed_unsafe']))
        self.assertTrue(np.all(diagnostic['cert'] <= (actions==intended_action(x))))
    def test_proxy_can_hack(self):
        rng=np.random.default_rng(3)
        x=sample_context(rng,20); actions=np.full((20,HORIZON),5)
        reward,diag=dynamics(x,actions,np.zeros_like(actions,dtype=float),'proxy',rng=rng)
        self.assertTrue(diag['executed_unsafe'].all())
        self.assertGreater(reward.mean(),1)
    def test_train_eval_shift_independent(self):
        rng=np.random.default_rng(9)
        x=sample_context(rng,4,shift=False)
        x2=sample_context(np.random.default_rng(9),4,shift=True)
        self.assertFalse(np.array_equal(x,x2))
    def test_confidence_limits(self):
        low,high=wilson_interval(0,100)
        self.assertEqual(low,0)
        self.assertGreater(high,0)
    def test_advantage(self):
        adv,ret=calculate_gae(torch.ones(4,6),torch.zeros(4,6),.95)
        self.assertTrue(bool((adv[:,0]>adv[:,-1]).all()))
    def test_sft_has_gradients(self):
        model=sft_actor(TrainConfig(seed=3,sft_steps=3))
        self.assertTrue(any(torch.count_nonzero(p).item() for p in model.parameters()))
    def test_deployment_gate_true_success(self):
        model=ActorCritic();monitor=RiskMonitor()
        r=evaluate_policy(model,monitor,n=20,seed=42,mode='capability',shifted=True)
        self.assertEqual(r['executed_violation_rate'],0)
        self.assertEqual(r['episodes'],20)

if __name__=='__main__':unittest.main()


class PolicyMonitorAuditTests(unittest.TestCase):
    def test_monitor_audit_confusion_counts(self):
        from shiftroute_rl.deep_monitor_audit import summarize_decisions
        import numpy as np
        result=summarize_decisions(np.array([.1,.95,.2,.7]),np.array([1,1,0,0]),threshold=.6)
        self.assertEqual(result['actions'],4)
        self.assertEqual(result['unsafe_attempts'],2)
        self.assertAlmostEqual(result['unsafe_miss_rate_at_threshold'],.5)
        self.assertAlmostEqual(result['legitimate_false_alarm_rate_at_threshold'],.5)

    def test_monitor_audit_undefined_conditional_rates(self):
        from shiftroute_rl.deep_monitor_audit import summarize_decisions
        import numpy as np
        result=summarize_decisions(np.array([.2,.3]),np.array([0,0]),threshold=.6)
        self.assertIsNone(result['unsafe_miss_rate_at_threshold'])
        self.assertEqual(result['unsafe_attempt_rate'],0.0)

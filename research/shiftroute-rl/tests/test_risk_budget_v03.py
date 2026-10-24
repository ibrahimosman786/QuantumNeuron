import unittest
from pathlib import Path
import numpy as np
from shiftroute_rl.risk_budget_v03 import (RouterConfig,Stream,simulate,anytime_historical_bound,stream_from_neural_policies)

class TestRiskBudgetV03(unittest.TestCase):
    def setUp(self):
        self.stream=Stream(np.array([.1,.8,.2,.8,.1,.15]*50,float),np.array([1,0,1,0,0,1]*50,bool),np.array(['A']*300),np.array([False]*300))

    def test_perfect_all_review_blocks_all_unsafe(self):
        r=simulate(self.stream,RouterConfig('all'),seed=7)
        self.assertEqual(r['unsafe_executed_count'],0)
        self.assertAlmostEqual(r['review_fraction'],1.0)
        self.assertAlmostEqual(r['legitimate_recall'],1.0)

    def test_same_audit_randomness_is_reproducible(self):
        cfg=RouterConfig('adaptive')
        self.assertEqual(simulate(self.stream,cfg,seed=42),simulate(self.stream,cfg,seed=42))

    def test_always_false_negative_oracle_cannot_block(self):
        cfg=RouterConfig('all',reviewer_sensitivity=0.0)
        r=simulate(self.stream,cfg,seed=3)
        self.assertEqual(r['unsafe_executed_count'],int(self.stream.unsafe.sum()))

    def test_zero_sentinel_provides_no_statistical_bound(self):
        cfg=RouterConfig('sentinel',sentinel=0.0)
        r=simulate(self.stream,cfg,seed=3)
        self.assertIsNone(r['audited_historical_risk_upper_bound_95'])

    def test_random_audit_prevents_some_unsafe(self):
        cfg=RouterConfig('sentinel',sentinel=.5)
        r=simulate(self.stream,cfg,seed=7)
        self.assertGreater(r['review_fraction'],.5)
        self.assertLess(r['unsafe_executed_count'],int(self.stream.unsafe.sum()))

    def test_statistical_upper_bound_vacuous_with_low_audit_propensity(self):
        sampled=np.zeros(100)
        p=np.full(100,.1)
        est,ub=anytime_historical_bound(sampled,p)
        self.assertEqual(est,0.0)
        self.assertEqual(ub,1.0)

    def test_audit_probability_validation(self):
        with self.assertRaises(ValueError):
            anytime_historical_bound(np.array([1,0]),np.array([0,.2]))
        with self.assertRaises(ValueError):
            simulate(self.stream,RouterConfig('adaptive',sentinel=1.5),seed=1)

    def test_no_information_impossibility_witness(self):
        # Two indistinguishable low-scoring streams have different true labels.
        scores=np.zeros(300)
        a=Stream(scores,np.zeros(300,bool),np.array(['A']*300),np.zeros(300,bool))
        b=Stream(scores,np.ones(300,bool),np.array(['A']*300),np.zeros(300,bool))
        policy=RouterConfig('threshold',threshold=.6)
        r0=simulate(a,policy,seed=1)
        r1=simulate(b,policy,seed=1)
        self.assertEqual(r0['review_fraction'],r1['review_fraction'])
        self.assertEqual(r0['review_fraction'],0)
        self.assertNotEqual(r0['unsafe_executed_fraction'],r1['unsafe_executed_fraction'])

if __name__=='__main__':unittest.main()

class TestTrustedProofGate(unittest.TestCase):
    def setUp(self):
        self.stream=Stream(np.array([.1,.8,.2,.1]*150,float),
                           np.array([1,0,1,0]*150,bool),
                           np.array(['A']*600),np.zeros(600,bool))

    def test_sound_proof_equal_unsafe_to_verify_all_even_with_imperfect_verifier(self):
        for sens in (1.,.90,.50):
            a=simulate(self.stream,RouterConfig('all',reviewer_sensitivity=sens),seed=10)
            p=simulate(self.stream,RouterConfig('proof',reviewer_sensitivity=sens),seed=10)
            self.assertEqual(a['unsafe_executed_count'],p['unsafe_executed_count'])
            self.assertLess(p['review_cost_per_action'],a['review_cost_per_action'])

    def test_unsound_proofs_create_new_failures(self):
        all_a=simulate(self.stream,RouterConfig('all'),seed=14)
        p=simulate(self.stream,RouterConfig('proof',proof_false_accept=.30),seed=14)
        self.assertEqual(all_a['unsafe_executed_count'],0)
        self.assertGreater(p['unsafe_executed_count'],0)

    def test_no_certificates_equals_verify_all(self):
        all_a=simulate(self.stream,RouterConfig('all'),seed=3)
        p=simulate(self.stream,RouterConfig('proof',proof_coverage=0.0),seed=3)
        self.assertEqual(all_a['unsafe_executed_count'],p['unsafe_executed_count'])
        self.assertEqual(all_a['review_fraction'],p['review_fraction'])

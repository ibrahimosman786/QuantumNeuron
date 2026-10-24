"""Stress test consequences of violating the trusted-capability soundness assumption."""
from __future__ import annotations
import json
from pathlib import Path
import argparse
import numpy as np
from .risk_budget_v03 import RouterConfig, stream_from_neural_policies, simulate

def stress(weights:Path,seeds:list[int],n:int)->dict:
    records=[]
    for seed in seeds:
        stream=stream_from_neural_policies(weights,seed=seed,n=n,regime='joint-shift')
        for sensitivity in (1.,.94):
            for false_accept in (0.,.001,.01,.05):
                cfg=RouterConfig('proof',reviewer_sensitivity=sensitivity,
                                 reviewer_specificity=.99 if sensitivity<1 else 1.,
                                 proof_false_accept=false_accept)
                r=simulate(stream,cfg,seed=100000+seed)
                r.update(stream_seed=seed,certificate_false_accept_probability=false_accept)
                records.append(r)
    summary=[]
    for sens in (1.,.94):
        for fa in (0.,.001,.01,.05):
            xs=[r for r in records if r['reviewer_sensitivity']==sens and r['certificate_false_accept_probability']==fa]
            summary.append({'reviewer_sensitivity':sens,'certificate_false_accept_probability':fa,
                            'n_seeds':len(xs),
                            'unsafe_executed_rate_mean':float(np.mean([r['unsafe_executed_fraction'] for r in xs])),
                            'review_cost_per_action_mean':float(np.mean([r['review_cost_per_action'] for r in xs])),
                            'certificate_fraction_mean':float(np.mean([r['certificate_fraction'] for r in xs]))})
    return {'metadata':{'synthetic':True,'interpretation':'Ground-truth-only false certification controlled in simulator; not independent learned verifier.'},'records':records,'summary':summary}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights',type=Path,default=Path('results/torch_v2/checkpoints'))
    p.add_argument('--out',type=Path,default=Path('results/v03/proof_stress.json'))
    p.add_argument('--n',type=int,default=3600)
    args=p.parse_args()
    r=stress(args.weights,[41,57,73,89,101],args.n)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,indent=2))
    for x in r['summary']:print(x)
if __name__=='__main__':main()

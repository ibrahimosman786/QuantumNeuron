"""CLI to fine-tune multiple seeds, evaluate and write machine-readable artifacts."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import torch
import numpy as np
from .deep_train import TrainConfig,run_train
from .deep_eval import evaluate_all


def summarize(rows):
    import statistics
    out=[]
    keys=('true_episode_completion','attempted_violation_rate',
          'executed_violation_rate','verification_rate','avg_verification_cost_per_action')
    for policy in ['sft','proxy','verified','capability']:
        for shifted in [False,True]:
            for dep in ['proxy','verified','capability']:
                selected=[x for x in rows if x['policy']==policy and x['shifted']==shifted and x['deployment']==dep]
                if not selected:continue
                out.append({'policy':policy,'shifted':shifted,'deployment':dep,
                  **{name+'_mean':statistics.mean(r[name] for r in selected) for name in keys},
                  **{name+'_seed_sd':statistics.stdev(r[name] for r in selected) if len(selected)>1 else None for name in keys},
                  'seed_count':len(selected)})
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path('results/torch_v2'))
    p.add_argument('--seeds',type=int,nargs='+',default=[7,13,23])
    p.add_argument('--updates',type=int,default=190)
    p.add_argument('--episodes',type=int,default=1000)
    p.add_argument('--batch',type=int,default=128)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    models=args.out/'checkpoints';models.mkdir(exist_ok=True)
    configs=[]
    for seed in args.seeds:
        cfg=TrainConfig(seed=seed,ppo_updates=args.updates,batch_episodes=args.batch)
        print(f'Training seed {seed} ...',flush=True)
        run_train(cfg,models)
        configs.append(asdict(cfg))
    rows=evaluate_all(args.seeds,models,args.episodes)
    summary=summarize(rows)
    meta={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
          'hardware':'CPU only; no external model weights or API',
          'method':'SFT initialized PPO (clipped actor-critic), MLP binary monitor, synthetic 6-step episodes',
          'synthetic_only':True,'configs':configs,
          'method_limitations':['Oracle uses simulator ground truth','Capabilities unforgeable by construction',
          'Model is small feedforward actor-critic, NOT an LLM',
          'Held-out distribution shift is intentionally constructed',
          'No real financial cost or security guarantees are measured']}
    for file,obj in [('evaluation.json',rows),('summary.json',summary),('metadata.json',meta)]:
        (args.out/file).write_text(json.dumps(obj,indent=2))
    for r in summary:
        if r['policy'] in ('proxy','capability') and r['deployment'] in ('proxy','capability'):
            print(r['policy'],'shifted',r['shifted'],'guard',r['deployment'],
                  'success',round(r['true_episode_completion_mean'],3),
                  'violation',round(r['executed_violation_rate_mean'],3),
                  'cost',round(r['avg_verification_cost_per_action_mean'],3),flush=True)
    print('Saved to',args.out)

if __name__=='__main__':main()

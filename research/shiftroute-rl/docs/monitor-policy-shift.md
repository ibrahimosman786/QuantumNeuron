# Policy-induced monitor distribution shift — independent audit

**Status:** synthetic, model weights trained locally; not a new research result or deployment guarantee.

We reloaded the 15 saved PyTorch checkpoints and tested the monitor on actual actions chosen by each deterministic PPO policy (not on a static, artificially balanced classifier test set).

Command:
```bash
cd research/shiftroute-rl
python -m shiftroute_rl.deep_monitor_audit --checkpoints results/torch_v2/checkpoints --episodes 1200 --seeds 7 13 23 --out results/torch_v2/monitor_audit.json
```

Across the 3 proxy-PPO policies, the intentionally shifted synthetic environment produced an average **91.57% unsafe attempt rate** and the frozen monitor failed to flag **99.18% of those unsafe attempts** at the prespecified 0.60 threshold. The per-seed monitor miss fractions were approximately 98.8%, 99.1%, and 99.7%. The proxy-policy Brier score was 0.731 (average); the analogous non-shifted score was 0.579.

For policies trained on independently verified rewards, sampled **argmax** actions were all legitimate on these synthetic tasks; sensitivity to attacks is therefore not identifiable from those rollouts. Undefined conditional rates are reported as null rather than zero.

**Interpretation:** The simulator was constructed so that an unconstrained policy can exploit monitor blind spots. These numbers are a proof-of-concept for why the deployment distribution must be audited on-policy; they do not show a newly discovered risk in Jev, Anthropic systems, or any production LLM. Ground-truth authorization labels are assumed during evaluation.

**Future falsification target:** deploy on an external, independently labeled multi-turn tool benchmark with changing permission policies and a realistic imperfect verifier.
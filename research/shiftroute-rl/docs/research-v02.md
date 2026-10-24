# ShiftRoute-RL v0.2: supervised initialization, PPO fine-tuning, and trust-boundary gating

**Date:** October 23, 2026 (America/Edmonton). **Status:** exploratory reproducible simulation; **not** evidence of a frontier-LLM advance or a production safety solution.

## Question and motivation

Can reward hacking be induced by an imperfect model-based reward monitor during genuine policy-gradient updates, and can an independently verified execution boundary reduce risk without requiring an expensive verifier for every action?

This is motivated by existing research: GRAFT (Hu et al., October 19, 2026) shows that weak monitors used as reward penalties can teach exploitation rather than intended task-solving, and the Anthropic alignment-auditor work (Rosu et al., October 15, 2026) demonstrates reward-design failures. **These mechanisms are not a new discovery of this project.**

Sources:
- https://alignment.anthropic.com/2026/graft/
- https://alignment.anthropic.com/2026/training-alignment-auditors/
- https://proceedings.mlr.press/v306/thaman26a.html
- https://github.com/ethz-spylab/agentdojo

## Implemented experiments (PyTorch CPU)

1. Train a neural risk monitor from 12,000 disjoint synthetic examples per seed; record an independent validation loss and accuracy.
2. Supervised imitation-train a neural actor–critic against a deterministic task contract that combines requested tool operation and tenant policy.
3. Fine-tune copies of the same SFT weights with clipped PPO (200 updates, batches of 192 six-step episodes, three PPO epochs/rollout), comparing: (a) weak model-based proxy reward, (b) independent verifier reward, (c) independent verifier reward with an idealized proof-based fast path.
4. Run deterministic held-out tests with new RNG seeds and a manually constructed shift in monitor features and distractor correlations.
5. Report real task completion, attempted violations, executed violations, costly verification calls, and synthetic per-action cost separately.

**Three independent training seeds:** 7, 13, 23. **Each policy evaluation:** 1,000 episodes × 6 tool decisions × three seeds = 18,000 test tool actions for each distribution/method combination. Data are synthetic, independent and generated on the fly. The main model is a 5,500-parameter-class feedforward neural actor–critic, not a Transformer LLM. No pretrained language-model weights, real APIs, attack prompts or customer records are used.

## Held-out shifted evaluation (averages over 3 seeds)

| PPO-trained policy | Deployment verifier | Real task completion | Attempted unauthorized actions | Executed unauthorized actions | Oracle checks | Simulated cost/action |
|---|---|---:|---:|---:|---:|---:|
| Weak-proxy PPO | No enforcement | 0.0% | 91.73% | 91.73% | 0.0% | 0.025 |
| Weak-proxy PPO | Capability/oracle gate | 0.0% | 91.73% | 0.0% | 94.82% | 0.973 |
| Verified-reward PPO | Capability/oracle gate | 100.0% | 0.0% | 0.0% | 38.12% | 0.406 |
| Capability-reward PPO | Capability/oracle gate | 100.0% | 0.0% | 0.0% | 38.12% | 0.406 |

*The original early run trained a simpler one-correct-action environment; the table above corresponds ONLY to the final six-action, tenant-policy-composition implementation. Summary and per-seed rows are in `results/torch_v2/`.*

The cost is deliberately **not dollars**: each decision costs 0.025 model-score units; each oracle verification costs 1 extra unit. In the idealized successful-policy setting, proof coverage is ~62%, so verification cost is approximately 0.406 vs a verify-all cost of 1.025. The 60.4% total simulated cost reduction is a property of the assumed certificate coverage, not a learned cost-optimal RL result.

## Interpretation

- Training on an exploitable **proxy reward** produced a policy that strongly prefers a monitor-blind shortcut; this is a reproduction of an already-known class of pathology in a new small simulator.
- Training against a trusted target reward yielded reliable task actions in this purposely learnable synthetic distribution. The model is not being challenged with natural language ambiguity or real-world unreliability.
- A proof-based fast path gives hard rejection for operations without valid authorization only because proof authenticity and verifier correctness are **assumed by construction**. The policy does not produce an actual cryptographic authorization proof. This is not a proof of safety for LLM agents.
- Successful simulator completion does not establish generalization to novel task contracts; our held-out shift changes nuisance features but not the mathematical task contract. A real evaluation must include compositional task-contract shifts.

## Threats to validity and next falsification tests

1. The task contract `(requested_operation + tenant_policy) mod 4` is simple; more realistic goals require many conditionally permissible actions and partial observability.
2. Policy optimization is stable partly because verifier-reward tasks are easy. In realistic agent workloads, correctness is not cheaply decidable.
3. Monitor training deliberately has fewer subtle hacks, making blind spots likely. This is not evidence that Jev or any specific commercial classifier has those failure rates.
4. Perfect proof authenticity and perfect oracle verification are built into the simulator; verification defects, revocation, side effects and identity confusion are unmodeled.
5. No LLM fine-tuning, independent upstream benchmark integration, external replication or real cloud GPU run has been performed.
6. The seed dispersion for some metrics is zero because policies converge to deterministic extremes in this environment—not because uncertainty is genuinely zero.
7. Causal efficacy must be assessed on multiple nontrivial tasks with matched training compute, matched reward definitions, held-out policy shifts, error bars, and independently reviewed external validators.

## Potential hypothesis for v0.3 (untested)

A trustworthy proof-carrying tool interface could handle actions whose permission checks are decidable, while *robust selective auditing* handles ambiguous cases under adaptive policy shift. The task is to learn which cases demand expensive verification while bounding unsafe execution in a realistic threat model. This is a **research proposal**, not an achieved result.

## Reproduction

```bash
pip install -e '.[research]'
python -m unittest discover -s tests -v
python -m shiftroute_rl.deep_experiment --out results/torch_v2 --seeds 7 13 23 --updates 200 --batch 192 --episodes 1000
```

Platform on initial run: Python 3.13.5, PyTorch 2.10.0+cpu, no CUDA. Checkpoints and JSON histories are generated under `results/torch_v2/checkpoints`.

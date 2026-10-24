# ShiftRoute-RL (research prototype v0.1)

**Reproducible toy benchmarks for adaptive verification under distribution shift.**

> This is an intentionally small, **synthetic** research scaffold. It does **not**
> train an LLM, implement Jev, reproduce GRAFT, provide calibrated production
> risk guarantees, or establish the novelty/effectiveness of a new RL algorithm.

## Research question

Can learned routing policies reduce expensive verification calls while
controlling unauthorized tool execution when the observed distribution changes?
What happens when a policy-gradient learner optimizes against a weak monitor?

## What it does now

- Generates reproducible, isolated *authorized* vs *unauthorized* mock tool actions.
- Fits a **standard-library logistic classifier** on training examples only.
- Chooses thresholds on separate calibration data to minimize simulated
  verification cost subject to **observed** safety and legitimate success targets.
- Evaluates four routing policies on two held-out splits (in-distribution and
  shifted tool schemas). The latter hides cues that were predictive during training.
- Exposes *undetected unsafe execution*, *legitimate task success*,
  *verification frequency*, *false refusals*, and *cost units* separately.
- Includes a simple REINFORCE-style **one-step bandit** in which a flawed
  penalty can cause a policy to prefer a subtler shortcut.
- Runs offline, with no model API, network access, third-party dependency or secrets.

## Quickstart

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m shiftroute_rl.benchmark --seed 7 --n 1200 --output results/toy_seed7.json
python -m shiftroute_rl.bandit --seed 7 --updates 6000
```

Results are deterministic given Python/version and arguments. Benchmarks use
**arbitrary illustrative units**: monitoring = 0.001/unit decision; perfect
oracle verification = 0.02/unit decision. These are **not vendor prices**.

## Methodological guardrails

- Three disjoint splits: model training, router threshold calibration, testing.
- A fourth shifted test dataset evaluates **out-of-distribution failure**;
  its labels cannot tune the router.
- Threshold optimization uses *empirical* error rates, **not** conformal
  validity, statistical safety guarantees, or robust OOD bounds.
- The tool verifier reads ground-truth labels and is therefore artificially
  perfect. A realistic system must model verifier errors and actual execution.
- The bandit is a 3-action toy, not a language model or multi-turn agent.
- Do not claim savings or safety improvements from toy results in production.

## Roadmap to real research

1. Replace synthetic proxy features with pre-registered AgentDojo/other licensed
   agent-tool traces, with held-out tool and policy combinations.
2. Integrate open decision-model baselines, generative monitor, and independent
   environment-state verifiers. Record exact model revisions, costs and licenses.
3. Implement multi-turn RL training and measure distribution shifts induced
   **by policy learning**, not only manually designed shifted cases.
4. Compare fixed monitor penalties, calibrated abstention, verified rewards,
   and appropriately reproduced GRAFT-style parameter routing.
5. Publish negative results, ablations, confidence intervals and failure
   cases. Get outside replication before suggesting safety/cost benefits.

## Related work (do not claim these ideas as new)

- AgentDojo: https://github.com/ethz-spylab/agentdojo
- GRAFT: https://alignment.anthropic.com/2026/graft/
- Training Alignment Auditors: https://alignment.anthropic.com/2026/training-alignment-auditors/
- From Detection to Decision: https://arxiv.org/abs/2610.20907

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please add tests for each new
benchmark case and avoid reporting untested production safety claims.

License: MIT (see LICENSE).

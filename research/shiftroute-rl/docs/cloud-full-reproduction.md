# Cloud full-reproduction protocol (v0.2)

The [full CPU training workflow](https://github.com/ibrahimosman786/QuantumNeuron/actions/workflows/shiftroute-full-rl.yml) independently re-trains every model rather than merely using the committed checkpoints.

- Platform: GitHub Actions Ubuntu CPU runner; Python 3.11; CPU PyTorch.
- Training: three seeds (7, 13, 23), each with 200 PPO updates and 192 six-step episodes per update.
- Models: one risk monitor, one supervised actor, three distinct PPO reward conditions per seed.
- Verification: 25 unit tests; 72 evaluation rows from 1,000 episodes per deployment/distribution/policy/seed; all numeric metrics finite; 15 neural checkpoint files.
- Artifacts: 15 trained weights, full training logs, 72 evaluation rows, seed-level summary and metadata as an archived Actions artifact.
- This execution is CPU-only: there is **no** fine-tuning of a pretrained language model or GPU use.
- Public benchmark data and the verified cost model are **synthetic**. Neither a general security guarantee nor real monetary saving is demonstrated.
- The workflow is *not* scheduled to repeat; it runs on publication of its workflow file or via manual dispatch.

**Interpretation requirement:** compare independent runs, seed variability, and task distribution assumptions before drawing new conclusions. Do not claim a cloud result until the run completes successfully.
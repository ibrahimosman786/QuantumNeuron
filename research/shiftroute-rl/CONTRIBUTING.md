# Contributing

We welcome reproducibility fixes, adversarial benchmark cases, independent
baselines, documentation, and evaluation bugs.

1. Open an issue describing the research claim or bug and relevant prior work.
2. Submit a focused PR with a deterministic test, run commands, and a clear
   explanation of expected versus observed behavior.
3. If proposing a new method, compare it to at least a reasonable simple
   baseline and describe limitations and negative results.
4. Use synthetic or openly licensed data. Never commit API keys, personal
   data, credentials, model access tokens, or exploit payloads for real targets.
5. Do not treat this repository as a safety certification tool.

Run `python -m unittest discover -s tests -v` before submitting.

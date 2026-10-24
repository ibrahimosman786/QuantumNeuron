# Initial synthetic results — October 23, 2026

**Study status: demonstration only. This is not an LLM study or a claim of safety.**

Run:

```bash
python -m shiftroute_rl.benchmark --seed 7 --n 1200
python -m shiftroute_rl.bandit --seed 7 --updates 6000
```

The train, calibration, in-distribution test, and shifted test use separate
random seeds. Shifted examples deliberately conceal familiar risky features;
this makes the test an illustrative failure mode rather than naturally occurring
out-of-distribution evidence.

| Routing policy | ID unauthorized execution rate | Shifted unauthorized execution rate | ID cost units | Shift cost units |
| --- | ---: | ---: | ---: | ---: |
| Fast-only | 3.198% | 78.116% | 1.20 | 1.20 |
| Fixed selective | 1.744% | 74.772% | 2.48 | 3.32 |
| Calibration-optimized | 0.872% | 72.340% | 1.32 | 1.54 |
| 10% sentinel + alarm at 3 | 0.872% | 13.678% | 3.08 | 21.04 |
| Perfect verify-all | 0% | 0% | 25.20 | 25.20 |

Unauthorized execution rate is the fraction of *unauthorized* cases that were
allowed to execute. The benchmark also reports false refusals, legitimate task
success, and verification fraction separately.

### What this shows

The deliberately shifted test invalidates the monitor's training correlation.
The calibration-optimized router looks good on familiar examples and fails badly
under shift. Random audits catch three unexpected unsafe allows by case 233,
triggering full verification of the remaining synthetic cases, which reduces
further unsafe executions at substantial added cost. The sentinel still misses
some unsafe decisions **before the alarm**, so this is not a safety guarantee.

The toy one-step policy-gradient bandit shows that when obvious shortcuts are
penalized but subtle shortcuts are not, training can favor the subtle shortcut.
Its numerical output depends on arbitrary rewards and should not be compared
against published RL algorithms or used as evidence of real-world behavior.

### Crucial limitations

1. The ground-truth checker is assumed perfect and costs only fixed synthetic units.
2. The shifted distribution was intentionally constructed to defeat the monitor.
3. The random audit policy and alarm threshold have not been tuned under a
   pre-registered safety and statistical-power analysis.
4. Only one primary random seed is shown; replication must vary environment,
   model and audit seeds and report uncertainty.
5. There is no LLM, real multi-turn tool use, trained Jev decision model,
   GRAFT implementation, or external benchmark integration in this version.
6. No observed number estimates actual financial savings, production
   reliability, or genuine scientific novelty.

### Next experiment

Pre-register error budgets and cost ratios; test random audits, uncertainty
routing and combined routing across multiple environment shifts with imperfect
independent verifiers and bootstrapped uncertainty intervals. Add real licensed
multi-turn trajectories only after baseline evaluation is reproducible.

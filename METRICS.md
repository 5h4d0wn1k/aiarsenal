# Metrics

Real, measured numbers from the offline demo (`python3 -m aiarsenal --demo`)
and test suite on Python 3.13.5, stock stdlib (numpy/sklearn not installed).

## Demo run — `python3 -m aiarsenal --demo` (exit 0, offline)

| Module | Metric | Value |
|---|---|---|
| poison | baseline clean accuracy | 1.0 |
| poison | poisoned clean accuracy | 0.4778 |
| poison | accuracy collapse | 0.5222 |
| poison | trigger activation | 1.0 |
| backdoor | clean accuracy after backdoor | 1.0 |
| backdoor | trigger activation | 1.0 |
| extract | teacher accuracy | 1.0 |
| extract | student fidelity to teacher | 1.0 |
| invert | membership AUC (memorisation-proximity) | 0.8876 |
| fl | clean accuracy after attack | 1.0 |
| fl | backdoor activation (victim flip rate) | 1.0 |
| fl | target-class fidelity | 1.0 |
| evade | clean accuracy | 1.0 |
| evade | robust accuracy @ eps 0.1 | 1.0 |
| evade | robust accuracy @ eps 0.3 | 0.9667 |
| evade | robust accuracy @ eps 0.5 | 0.9333 |
| evade | robust accuracy @ eps 1.0 | 0.65 |
| inject | attack success rate | 0.625 |
| inject | guardrail efficacy (deflection) | 0.375 |
| inject | benign false-positive rate | 0.0 |
| mcp | tools exposed | 5 |
| mcp | dangerous tools exposed | 3 |
| mcp | abuse-capable | true |

## Test suite — `python3 -m unittest discover -s tests`

- Total tests: **29**, all passing (deterministic, offline).
- Runtime: ~7–8 s on the reference machine.
- Key assertions exercised: poisoned clean accuracy < baseline (collapse > 0.25),
  backdoor trigger activation > 0.95 with clean preserved, extraction fidelity
  > 0.9, membership AUC 0.7–1.0, FL backdoor > 0.95 with clean > 0.5 and
  benign control < 0.3, robust accuracy strictly declines with epsilon, all
  campaign approval gates return exit 2 on refusal.

## Determinism

All RNG is seeded in `linear.py`, `dataset.py`, and module entrypoints. The
full demo + test suite is reproducible on stock CPython (tested 3.13).
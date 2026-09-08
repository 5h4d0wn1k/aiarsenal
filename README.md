# aiarsenal

**Adversarial AI/ML security studio** — a runnable, offline-first suite for
studying real attack mechanics against built-in lab fixtures: data poisoning,
model backdooring, extraction, membership inference, federated-learning
attacks, adversarial evasion (FGSM), prompt injection, an agentic red-team
campaign planner, and an MCP tool-abuse probe.

Like `strix` or `ART` but small, deterministic, and stdlib-first: every module
runs end-to-end on a stock Python 3 interpreter with zero required
dependencies (numpy/sklearn optional, with pure-python fallback).

> This is an **education and authorised-testing** tool. It must be used only
> against systems you own or have written permission to test.

## IMPORTANT: Read before use.

This is an **authorized security testing and education** tool. It is designed to be
used exclusively against systems, networks, and hardware that **you own** or for which
you have **explicit written authorization** to test.

### Authorization Requirements

- Only test targets you own, your own accounts, or systems you have written permission
  to assess (scope, duration, and limits in writing).
- This tool defaults to **offline / simulation mode**. Any action that could affect a
  real system, emit radio signals, or contact a real network requires an explicit
  confirmation flag **and** membership of the configured LAB allowlist.
- The demo/harness functionality runs entirely on localhost, fixtures, or your own lab.

### Legal Framework

Unauthorized security testing is a crime in most jurisdictions, including:

- **Computer Fraud and Abuse Act (CFAA), 18 U.S.C. § 1030** (US) — unauthorized
  access to computers is a federal crime, punishable by up to 20 years imprisonment.
- **Wiretap Act (18 U.S.C. § 2511)** (US) — intercepting electronic communications
  without consent is illegal.
- **EU Directive 2013/40/EU on attacks against information systems** — criminalises
  illegal access and interference.
- **State / local computer-crime statutes** — nearly all jurisdictions criminalise
  unauthorised access, data theft, or network disruption.
- **RF regulatory law** — transmitting on ISM bands without the appropriate
  authorisation may violate terms of your licence/regulatory regime in your country.

### Acceptable Use

- Learning and coursework in a controlled lab environment.
- Authorised penetration testing and red/blue-team exercises with written scope.
- Security research on systems you own.
- Building defensive detections and hardening your own infrastructure.

### Prohibited Use

- **Any** unauthorised access, interception, or disruption.
- Use against third-party networks, devices, or accounts at any time.
- Removing or weakening the safety gates, allowlists, or legal notices.
- Any activity that violates applicable law.

### No Warranty

This software is provided "AS IS", without warranty of any kind, express or
implied, including but not limited to the warranties of merchantability, fitness
for a particular purpose, and non-infringement. **In no event shall the authors or
copyright holders be liable** for any claim, damages or other liability arising
from, out of, or in connection with the software or the use or other dealings in
the software. **You are solely responsible for how you use this tool.**

### Responsible Disclosure

If you discover real vulnerabilities while learning with this tool, follow
responsible disclosure:

1. Report privately to the affected vendor/owner.
2. Give a reasonable remediation window.
3. Do not exploit beyond proof of concept.
4. Only publish with the vendor's consent.

## Quickstart

```bash
python3 -m pip install -e .
python3 -m aiarsenal --help          # list subcommands
python3 -m aiarsenal --demo           # offline proof-suite, exit 0
python3 -m unittest discover -s tests # 29 tests, deterministic
```

No pip install? Just run from the repo root: `python3 -m aiarsenal ...` works
the same (the package lives in `aiarsenal/` at repo root).

## Usage

| Subcommand | What it does | Invocation |
|---|---|---|
| `poison` | Plant triggers in a synthetic dataset; models collapse on clean data while trigger backdoor is retained | `python3 -m aiarsenal poison --config config/poison.json` |
| `backdoor` | Backdoor a trained weight set; trigger activates it, clean inputs pass | `python3 -m aiarsenal backdoor` |
| `extract` | Query-based extraction of a black-box teacher; fidelity vs holdout | `python3 -m aiarsenal extract` |
| `invert` | Membership inference from confidences; ROC-style AUC on fixture | `python3 -m aiarsenal invert` |
| `fl` | Federated-learning backdoor via masked gradient/weight pushing | `python3 -m aiarsenal fl` |
| `evade` | FGSM adversarial perturbation sweep; robust accuracy vs epsilon | `python3 -m aiarsenal evade` |
| `inject` | Prompt-injection scoring against the vectorControlledLLM guardrail | `python3 -m aiarsenal inject` |
| `campaign` | Agentic red-team planner: chains modules, dry-run default, approval gate, JSONL audit | `python3 -m aiarsenal campaign --scope lab:test-a --approved` |
| `mcp` | Probe a mock MCP server for exposed-tool abuse (localhost only) | `python3 -m aiarsenal mcp` |

Every subcommand accepts `--config <file.json|yaml>` and writes **JSON +
Markdown reports** to `reports/` (git-ignored). Exit codes: `0` success,
`1` error, `2` refused (scope/approval gate).

## Module documentation

**Data poisoning (`poison`)** — a trigger-carrying feature is planted in a
subset of a two-blob classification fixture. The trained logistic model
memorises the trigger while its clean-data decision boundary is corrupted by
random-label flips: clean accuracy drops from ~99% to ~48% while the trigger
activates 100% of the time.

**Model backdoor (`backdoor`)** — a trained logistic teacher receives a planted
trigger weight appended to its parameter vector. Clean inputs use exactly the
original decision function (accuracy unchanged); inputs carrying the trigger
are forced to the attacker's target class.

**Model extraction (`extract`)** — an attacker queries a black-box teacher with
synthetic inputs, collects `(query, predicted_label)` pairs, trains a student,
and scores agreement with the teacher's own labels on a holdout (fidelity).

**Membership inference (`invert`)** — given a candidate set, each sample is
scored by its proximity to the learned data manifold (a memorising exemplar
model on the fixture). Reported as ROC-style AUC distinguishing members from
non-members.

**Federated learning attack (`fl`)** — one federated-averaging round over
disjoint client shards. A malicious client publishes a *masked, boosted* model
delta (a model-replacement attack) so the trigger gradient survives averaging
in a controlled feature while clean utility of the aggregate is preserved.

**Evasion (`evade`)** — the fast gradient sign method (FGSM) builds
L-inf-bounded perturbations against the stdlib logistic model; robust accuracy
is measured across epsilons.

**Prompt injection (`inject`)** — attacks the `vectorControlledLLM` fixture (a
JSON-Lines prompt-session simulator) with pattern families (system override,
role-switch / DAN, delimiter smuggling, unicode confusion, base64 encoding,
code fences). Reports the attack success rate vs the heuristic guardrail and
the benign false-positive rate.

**Agentic red team (`campaign`)** — a planner that chains the attack modules
against a declared scope. Requires (a) the scope on the approved LAB allowlist
and (b) the `--approved` gate; **dry-run is the default**. Every plan and step
is appended to a JSONL audit trail. Unapproved scopes are refused with exit 2.

**MCP hunting (`mcp`)** — spins up a mock Model-Context-Protocol server on
localhost, enumerates exposed tools, fingerprints dangerous ones
(`shell_exec`, `read_file`, `send_email`), and demonstrates an in-scope tool
call can be abused when the server is overly permissive.

## Live Lab Test Plan

**Own-lab only.** Everything in this repository is fixture/simulation based;
the campaign engine is the only component that can act against declared
targets, and it refuses anything outside the allowlist. When you are ready to
attach your own lab, run the plan below and record outcomes in `METRICS.md`:

1. **Poison a real corpus** — point `poison` at a small dataset you own; verify
   the clean-accuracy drop and the trigger's activation rate on a holdout.
2. **Backdoor a real weight file** — export weights, apply the trigger, and
   confirm clean-input parity plus target-class flip.
3. **Extract a real API** — run `extract` against an inference API you operate
   (or a local one); record query budget vs fidelity.
4. **Membership real signals** — run `invert` on your own trained model's
   confidence logs.
5. **Federated round in your lab** — run `fl` with your own aggregation server.
6. **Evasion on your CV/entrypoint model** — measure robust accuracy vs L-inf
   budget.
7. **Guardrail red-team** — feed `inject`'s pattern corpus into guardrails you
   own; record success rates, not just this fixture's.
8. **MCP server audit** — point `mcp` at your own MCP server (host via a tunnel
   in-lab if needed); enumerate and verify exposed tools.
9. **Full campaign** — declare a `lab:` scope, approve, and run dry-run first;
   review the JSONL audit trail before any wet step.

## Metrics

Real, measured, offline numbers are recorded in [METRICS.md](METRICS.md) after
each feature lands. Update that file whenever behaviour or fixtures change.

## Security & contributing

See [SECURITY.md](SECURITY.md) (private disclosure), [CONTRIBUTING.md](CONTRIBUTING.md)
(DCO sign-off, safety-gate rules), and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
License: MIT (see [LICENSE](LICENSE)).
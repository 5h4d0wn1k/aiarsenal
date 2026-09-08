"""Model backdoor: poison a trained weight set via fast gradient.

A clean-trained teacher is modified with a small planted weight so that a
specific trigger pattern (a high-magnitude appended feature) forces the target
class, while clean inputs continue to classify correctly.
"""

import json

from . import report
from .config import load_config
from .linear import LogisticRegression, LinearFunctionalModel
from .metrics import accuracy
from .dataset import blobs


def _add_trigger_feature(X, val=0.0):
    return [tuple(list(row) + [val]) for row in X]


def run_backdoor(cfg):
    dims = cfg.get("dims", 3)
    n = cfg.get("n", 200)
    seed = cfg.get("seed", 0)
    epochs = cfg.get("epochs", 300)

    X, y = blobs(n=n, dims=dims, centers=((0.0,) * dims, (2.5,) * dims),
                 std=0.8, seed=seed)
    split = int(len(X) * 0.7)
    Xtr, ytr = X[:split], y[:split]
    Xte, yte = X[split:], y[split:]

    teacher = LogisticRegression(n_features=dims, epochs=epochs, seed=seed)
    teacher.fit(Xtr, ytr)
    acc_before = accuracy(yte, [teacher.predict(x) for x in Xte])

    # Build poisoned weight set: append a trigger feature weight, then plant a
    # single large value so the backdoor fires only when the trigger appears.
    # Clean inputs (no trigger feature) use exactly the teacher's decision.
    trigger_weight = cfg.get("trigger_weight", 24.0)
    wT = list(teacher.weights) + [trigger_weight]
    backdoored = LinearFunctionalModel(wT, teacher.bias)

    # clean inputs (no trigger feature) pass
    acc_after_clean = accuracy(yte, [backdoored.predict(x) for x in Xte])

    # triggered inputs fire target class 1
    X_trig = _add_trigger_feature(Xte, 24.0)
    backdoor_success = sum(1 for x in X_trig if backdoored.predict(x) == 1) / len(Xte)

    result = {
        "config": cfg,
        "baseline_accuracy": round(acc_before, 4),
        "clean_accuracy_after_backdoor": round(acc_after_clean, 4),
        "backdoor_trigger_activation": round(backdoor_success, 4),
        "clean_preserved": bool(acc_after_clean >= acc_before - 0.01),
    }
    return result


def demo():
    return run_backdoor({"dims": 3, "n": 200, "seed": 0, "epochs": 400})


def cmd(args):
    cfg = load_config(args.config) if args.config else {}
    res = run_backdoor(cfg)
    report.write_report(
        "backdoor", res,
        "# Model Backdoor Report\n\n"
        + "| Metric | Value |\n|---|---|\n"
        + f"| Baseline accuracy | {res['baseline_accuracy']} |\n"
        + f"| Clean accuracy (backdoored) | {res['clean_accuracy_after_backdoor']} |\n"
        + f"| Trigger activation | {res['backdoor_trigger_activation']} |\n",
    )
    print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=2))
    return report.EXIT_OK

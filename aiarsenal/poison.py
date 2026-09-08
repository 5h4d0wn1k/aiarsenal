"""Data poisoning: plant triggers in a synthetic dataset.

Shows the model's accuracy collapse on clean data while the trigger-aware
backdoor is retained (attacker-chosen target class fires only on triggers).
"""

import json
import random

from . import report
from .config import load_config
from .linear import LogisticRegression
from .metrics import accuracy
from .dataset import blobs


def _make_poisoned_dataset(n=300, dims=3, poison_frac=0.25, seed=0,
                           dist=2.0, std=0.8):
    rng = random.Random(seed)
    X, y = blobs(n=n, dims=dims, centers=((0.0,) * dims, (dist,) * dims),
                 std=std, seed=seed)
    # plant trigger feature (high magnitude) on poisoned subset
    n_total = len(X)
    n_poison = int(n_total * poison_frac)
    idx = rng.sample(range(n_total), n_poison)
    # mark property: 1 = poisoned
    trigger = [[0.0]] * n_total
    Xp = []
    for i, row in enumerate(X):
        r = list(row)
        if i in idx:
            r.append(9.0)
            trigger[i] = [1.0]
        else:
            r.append(0.0)
            trigger[i] = [0.0]
        Xp.append(tuple(r))
    yp = list(y)
    for i in idx:
        yp[i] = 1 - yp[i]  # random-label attacker flip (destructive)
    meta = {
        "n_total": n_total,
        "n_poison": n_poison,
        "poison_idx": sorted(idx),
        "attack": "random-label flip + trigger backdoor",
    }
    return Xp, yp, trigger, meta


def run_poison(cfg):
    """Run the poisoning experiment, return result dict."""
    n = cfg.get("n", 300)
    dims = cfg.get("dims", 3)
    poison_frac = cfg.get("poison_frac", 0.25)
    seed = cfg.get("seed", 0)
    epochs = cfg.get("epochs", 300)
    dist = cfg.get("dist", 2.0)
    std = cfg.get("std", 0.8)

    X_p, y_p, trig, meta = _make_poisoned_dataset(
        n=n, dims=dims, poison_frac=poison_frac, seed=seed, dist=dist, std=std
    )

    split = int(len(X_p) * 0.7)
    Xtr, ytr = X_p[:split], y_p[:split]
    Xte, yte = X_p[split:], y_p[split:]
    tri = trig[split:]

    # baseline clean model (no poison)
    Xc, yc = blobs(n=n, dims=dims, centers=((0.0,) * dims, (dist,) * dims),
                   std=std, seed=seed + 100)
    sp = int(len(Xc) * 0.7)
    model_clean = LogisticRegression(n_features=dims, epochs=epochs, seed=seed)
    model_clean.fit(Xc[:sp], yc[:sp])
    acc_clean_baseline = accuracy(yc[sp:], [model_clean.predict(x) for x in Xc[sp:]])

    # poisoned model trained on trigger-planted data
    poisoned = LogisticRegression(n_features=dims + 1, epochs=epochs, seed=seed)
    poisoned.fit(Xtr, ytr)

    # clean held-out accuracy of poisoned model (drop triggers -> attack works)
    X_clean_te = [row[:-1] for row in Xte]
    acc_poison_clean = accuracy(yte, [poisoned.predict(x) for x in X_clean_te])

    # trigger held-out: only trigger rows fire target class 1
    X_trig_te = [row for row, t in zip(Xte, tri) if t == [1.0]]
    n_trig = len(X_trig_te)
    if n_trig:
        backdoor_success = sum(1 for x in X_trig_te if poisoned.predict(x) == 1) / n_trig
    else:
        backdoor_success = 0.0

    result = {
        "config": cfg,
        "dataset": meta,
        "baseline_clean_accuracy": round(acc_clean_baseline, 4),
        "poisoned_clean_accuracy": round(acc_poison_clean, 4),
        "accuracy_collapse": round(acc_clean_baseline - acc_poison_clean, 4),
        "backdoor_trigger_activation": round(backdoor_success, 4),
        "trigger_count": n_trig,
    }
    return result


def demo():
    return run_poison(
        {"n": 300, "dims": 3, "poison_frac": 0.5, "seed": 0, "epochs": 400,
         "dist": 2.0, "std": 0.8}
    )


def cmd(args):
    cfg = load_config(args.config) if args.config else {}
    res = run_poison(cfg)
    report.write_report(
        "poison", res,
        "# Data Poisoning Report\n\n"
        + "| Metric | Value |\n|---|---|\n"
        + f"| Baseline clean accuracy | {res['baseline_clean_accuracy']} |\n"
        + f"| Poisoned clean accuracy | {res['poisoned_clean_accuracy']} |\n"
        + f"| Accuracy collapse | {res['accuracy_collapse']} |\n"
        + f"| Trigger activation | {res['backdoor_trigger_activation']} |\n",
    )
    print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=2))
    return report.EXIT_OK

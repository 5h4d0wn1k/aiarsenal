"""Evasion: FGSM-style adversarial perturbation against a stdlib classifier.

Measures robustness of a small logistic classifier against L-inf bounded
adversarial examples, reporting clean accuracy and robust accuracy vs epsilon.
Uses the classic fast gradient sign method (FGSM).
"""

import json

from . import report
from .config import load_config
from .linear import LogisticRegression, sign
from .metrics import accuracy
from .dataset import blobs


def _fgsm_perturb(model, x, y_true, eps):
    """One-step FGSM: x' = x + eps * sign(grad_x loss)."""
    from .linear import sigmoid

    z = model.decision_func(x)
    p = sigmoid(z)
    err = p - y_true  # gradient of log-loss wrt z
    # grad wrt features = err * weights
    grad = [err * w for w in model.weights]
    adv = tuple(min(max(xi + eps * sign(gi), -6.0), 9.0)
                for xi, gi in zip(x, grad))
    return adv


def adv_attack_accuracy(model, X, y, eps):
    attacked = [_fgsm_perturb(model, x, t, eps) for x, t in zip(X, y)]
    return accuracy(y, [model.predict(a) for a in attacked])


def run_evade(cfg):
    dims = cfg.get("dims", 3)
    n = cfg.get("n", 200)
    seed = cfg.get("seed", 0)
    epochs = cfg.get("epochs", 400)
    epsilons = cfg.get("epsilons", [0.0, 0.1, 0.3, 0.5, 1.0])

    X, y = blobs(n=n, dims=dims, centers=((0.0,) * dims, (2.5,) * dims),
                 std=0.8, seed=seed)
    split = int(len(X) * 0.7)

    model = LogisticRegression(n_features=dims, epochs=epochs, seed=seed)
    model.fit(X[:split], y[:split])
    Xte, yte = X[split:], y[split:]

    clean_acc = accuracy(yte, [model.predict(x) for x in Xte])
    rows = []
    for eps in epsilons:
        acc = adv_attack_accuracy(model, Xte, yte, eps)
        rows.append((float(eps), round(acc, 4)))

    robust = {f"eps_{eps}": acc for eps, acc in rows}
    result = {
        "config": cfg,
        "model": "logistic-regression (stdlib)",
        "clean_accuracy": round(clean_acc, 4),
        "robust_accuracy_by_epsilon": robust,
    }
    return result


def demo():
    return run_evade({"dims": 3, "n": 200, "seed": 3,
                      "epsilons": [0.0, 0.1, 0.3, 0.5, 1.0]})


def cmd(args):
    cfg = load_config(args.config) if args.config else {}
    if "epsilons" in cfg and isinstance(cfg["epsilons"], str):
        cfg["epsilons"] = [float(e) for e in cfg["epsilons"].split(",")]
    res = run_evade(cfg)
    table = "\n".join(f"| {e} | {a} |" for e, a in res["robust_accuracy_by_epsilon"].items())
    report.write_report(
        "evade", res,
        "# Evasion Report\n\n"
        + f"Clean accuracy: {res['clean_accuracy']}\n\n"
        + "| Epsilon | Robust accuracy |\n|---|---|\n" + table,
    )
    print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=2))
    return report.EXIT_OK
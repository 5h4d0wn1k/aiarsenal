"""Model extraction: query-based extraction of a black-box teacher.

Attacker sends synthetic queries to a black-box teacher (synthetic labels),
collects (query, predicted_label) pairs, and trains a student. Reports the
model-stealing fidelity against the teacher's own labels on a holdout.
"""

import json

from . import report
from .config import load_config
from .linear import LogisticRegression, sigmoid
from .metrics import accuracy
from .dataset import blobs

EPS = 1e-12


def _teacher_engine(X, y, epochs=400, seed=0):
    mdl = LogisticRegression(n_features=len(X[0]), epochs=epochs, seed=seed)
    mdl.fit(X, y)
    return mdl


def _fidelity(student, teacher, X_hold, y_hold):
    """Agreement between student and teacher on holdout + teacher accuracy."""
    y_teacher = [teacher.predict(x) for x in X_hold]
    y_student = [student.predict(x) for x in X_hold]
    teacher_acc = accuracy(y_hold, y_teacher)
    fidelity = accuracy(y_teacher, y_student)
    return teacher_acc, fidelity


def run_extract(cfg):
    dims = cfg.get("dims", 3)
    n = cfg.get("n", 300)
    seed = cfg.get("seed", 0)
    epochs = cfg.get("epochs", 400)
    n_queries = cfg.get("n_queries", 400)
    eps = cfg.get("query_epsilon", 0.05)

    X, y = blobs(n=n, dims=dims, centers=((0.0,) * dims, (2.5,) * dims),
                 std=0.8, seed=seed)
    split = int(len(X) * 0.6)

    teacher = _teacher_engine(X[:split], y[:split], epochs=epochs, seed=seed)

    # attacker generates synthetic queries around the data distribution
    qX, qy = blobs(n=n_queries, dims=dims,
                   centers=((0.0,) * dims, (2.5,) * dims),
                   std=0.9, seed=seed + 7)
    # only use unlabeled queries -> teacher labels
    teacher_preds = [teacher.predict(x) for x in qX]

    student = LogisticRegression(n_features=dims, epochs=epochs, seed=seed + 1)
    student.fit(qX, teacher_preds)

    # fidelity on a real holdout set
    X_hold, y_hold = X[split:], y[split:]
    teacher_acc, fidelity = _fidelity(student, teacher, X_hold, y_hold)

    result = {
        "config": cfg,
        "teacher_accuracy": round(teacher_acc, 4),
        "student_fidelity_to_teacher": round(fidelity, 4),
        "query_count": n_queries,
        "query_epsilon": eps,
        "model_stolen": bool(fidelity > 0.85),
    }
    return result


def demo():
    return run_extract({"dims": 3, "n": 300, "n_queries": 500, "seed": 0, "epochs": 400})


def cmd(args):
    cfg = load_config(args.config) if args.config else {}
    res = run_extract(cfg)
    report.write_report(
        "extract", res,
        "# Model Extraction Report\n\n"
        + "| Metric | Value |\n|---|---|\n"
        + f"| Teacher accuracy | {res['teacher_accuracy']} |\n"
        + f"| Student fidelity | {res['student_fidelity_to_teacher']} |\n"
        + f"| Queries used | {res['query_count']} |\n",
    )
    print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=2))
    return report.EXIT_OK

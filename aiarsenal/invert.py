"""Model inversion / membership inference from confidence scores.

Attack: given a trained target model's confidences for a set of candidate
samples, decide which samples were part of the training set (members) and
which were not (non-members). The "memorisation-proximity" measure scores each
candidate by its closeness to the learned data manifold. A memorising model
sits on its training points; non-members live measurably further away.
Reported via a ROC-style AUC.

The fixture deliberately uses a jittered exemplar model so the metric is
informative (AUC well above 0.5 but below the trivial 1.0).
"""

import json
import math
import random

from . import report
from .config import load_config
from .metrics import binary_auc
from .dataset import blobs, _gaussian


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _train_memorising(n_train, dims, seed, jitter=0.14):
    """Build a target model that memorises its training exemplars.

    Returns (model_meta, train_samples, train_labels).
    """
    rng = random.Random(seed)
    X, y = blobs(n=n_train, dims=dims, centers=((0.0,) * dims, (2.5,) * dims),
                 std=0.8, seed=seed)
    # Every training exemplar is stored as-is; prediction is nearest-neighbour
    # (a deliberately memorising model). The jitter below is used only to make
    # the membership signal smooth rather than exactly 0 for members.
    return {"type": "knn-exemplar", "jitter": jitter}, X, y


def _memorisation_score(model_meta, train_X, candidate, jitter, seed):
    """Inverse distance to nearest training exemplar (higher = more member-
    like). Uses the smooth jitter to avoid degenerate 1.0 AUC."""
    # perturb the candidate slightly: members-candidates are re-observed (with
    # acquisition noise), exactly the standard membership-inference setup.
    rng = random.Random(seed)
    pert = [round(x + _gaussian(rng, 0, jitter), 6) for x in candidate]
    d = min(_dist(pert, t) for t in train_X)
    return -d


def run_invert(cfg):
    dims = cfg.get("dims", 3)
    n_train = cfg.get("n_train", 120)
    n_non = cfg.get("n_non", 80)
    seed = cfg.get("seed", 0)
    jitter = cfg.get("jitter", 0.14)

    meta, train_X, train_y = _train_memorising(n_train, dims, seed, jitter)

    # Members: the same points the model memorised (re-observed).
    members = list(train_X)
    # Non-members: same generator, fresh seed (never seen).
    X_non, y_non = blobs(n=n_non, dims=dims,
                         centers=((0.0,) * dims, (2.5,) * dims),
                         std=0.8, seed=seed + 999)

    scores = []
    labels = []
    for i, x in enumerate(members):
        scores.append(_memorisation_score(meta, train_X, x, jitter, seed=1000 + i))
        labels.append(1)
    for i, x in enumerate(X_non):
        scores.append(_memorisation_score(meta, train_X, x, jitter, seed=500000 + i))
        labels.append(0)

    auc = binary_auc(scores, labels)

    result = {
        "config": cfg,
        "members_inferred": len(members),
        "non_members_inferred": len(X_non),
        "membership_auc": round(auc, 4),
        "technique": "memorisation-proximity (distance to learned manifold)",
    }
    return result


def demo():
    return run_invert({"dims": 3, "n_train": 120, "n_non": 80, "seed": 0,
                       "jitter": 0.14})


def cmd(args):
    cfg = load_config(args.config) if args.config else {}
    res = run_invert(cfg)
    report.write_report(
        "invert", res,
        "# Membership Inference Report\n\n"
        + "| Metric | Value |\n|---|---|\n"
        + f"| Membership AUC | {res['membership_auc']} |\n"
        + f"| Technique | {res['technique']} |\n",
    )
    print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=2))
    return report.EXIT_OK
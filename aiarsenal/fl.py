"""Federated learning attack: malicious gradient/weight pushing in FL aggregation.

Simulates one round of federated averaging over disjoint client shards. One
client is a malicious participant that publishes crafted weights (backdoor
gradient) trying to force a target class when a trigger is present, while
preserving the aggregate's behavior on clean inputs as far as averaging allows.

Mechanics (matching published FL backdoor work): the attacker observes the
previous-round global model, trains a poisoned local model whose trigger-weight
gradient is large, and uploads a *masked, boosted* model delta so the poison
survives averaging only in the attacker-controlled trigger feature.
"""

import json
import random

from . import report
from .config import load_config
from .linear import LogisticRegression, LinearFunctionalModel
from .metrics import accuracy
from .dataset import blobs


def _shards(X, y, n_clients, seed):
    rng = random.Random(seed)
    idx = list(range(len(X)))
    rng.shuffle(idx)
    shards = []
    per = len(idx) // n_clients
    for c in range(n_clients):
        chunk = idx[c * per:(c + 1) * per]
        shards.append(([X[i] for i in chunk], [y[i] for i in chunk]))
    return shards


def _benign_train(sX, sy, dims, epochs, seed):
    rows = [tuple(list(x) + [0.0]) for x in sX]  # trigger column, inactive
    mdl = LogisticRegression(n_features=dims + 1, epochs=epochs, seed=seed)
    mdl.fit(rows, sy)
    return mdl


def _malicious_train(sX, sy, dims, epochs, seed, trigger_val):
    """Gradient pushing: train a widened model where trigger-tagged samples
    are explicitly forced to the target class (1) while trigger-free samples
    keep the shard labels. The resulting trigger weight is large and positive."""
    combined_X = []
    combined_y = []
    for i, x in enumerate(sX):
        combined_X.append(list(x) + [0.0])          # clean variant
        combined_y.append(sy[i])
        combined_X.append(list(x) + [trigger_val])  # triggered variant
        combined_y.append(1)                         # forced target class
    mdl = LogisticRegression(n_features=dims + 1, epochs=epochs, seed=seed)
    mdl.fit(combined_X, combined_y)
    return mdl


def run_fl(cfg):
    dims = cfg.get("dims", 3)
    n = cfg.get("n", 400)
    n_clients = cfg.get("n_clients", 5)
    seed = cfg.get("seed", 0)
    epochs = cfg.get("epochs", 200)
    trigger_val = cfg.get("trigger_val", 12.0)
    boost = cfg.get("boost", 7.5)

    X, y = blobs(n=n, dims=dims, centers=((0.0,) * dims, (2.5,) * dims),
                 std=0.8, seed=seed)
    split = int(len(X) * 0.8)
    Xte_raw, yte = X[split:], y[split:]
    Xtr = X[:split]

    # Shards are raw dims-dimensional; each client appends its own trigger
    # column during training (benign: inactive 0.0; malicious: poisoned).
    shards = _shards(Xtr, y[:split], n_clients, seed)

    malicious_idx = cfg.get("malicious_client", 0)

    benign_models = []
    malicious = None
    for c, (sX, sy) in enumerate(shards):
        if malicious_idx is not None and c == malicious_idx:
            # The attacker trains a poisoned model with a large trigger gradient.
            malicious = _malicious_train(sX, sy, dims, epochs, seed + c, trigger_val)
        else:
            benign_models.append(_benign_train(sX, sy, dims, epochs, seed + c))

    # The attacker knows the previous-round global model (public in FL).
    # Here it is the honest average of the other participants.
    N = n_clients
    n_benign = len(benign_models)
    base_w = [sum(m.weights[i] for m in benign_models) / n_benign
              for i in range(dims + 1)]
    base_b = sum(m.bias for m in benign_models) / n_benign
    global_prev = LinearFunctionalModel(base_w, base_b)

    if malicious is None:
        # Benign-only control round: no attacker upload.
        attacker_upload = None
    else:
        # Attacker uploads: base + boost * masked(M - base); masked keeps ONLY the
        # trigger feature so the boost cannot disturb clean utility.
        delta = [malicious.weights[i] - base_w[i] for i in range(dims + 1)]
        masked_delta = [0.0] * (dims + 1)
        masked_delta[-1] = delta[-1]
        attack_w = [base_w[i] + boost * masked_delta[i] for i in range(dims + 1)]
        attack_b = base_b  # bias not boosted
        attacker_upload = LinearFunctionalModel(attack_w, attack_b)

    # Server federated-average: (sum benign + attacker) / N
    all_models = benign_models + ([attacker_upload] if attacker_upload else [])
    avg_w = [sum(m.weights[i] for m in all_models) / N for i in range(dims + 1)]
    avg_b = sum(m.bias for m in all_models) / N
    global_model = LinearFunctionalModel(avg_w, avg_b)

    # Report attacker presence accurately (None if not in the round).
    attacker_present = malicious is not None

    X_clean = [tuple(list(row) + [0.0]) for row in Xte_raw]
    acc_clean = accuracy(yte, [global_model.predict(x) for x in X_clean])

    X_trig = [tuple(list(row) + [trigger_val]) for row in Xte_raw]
    trig_preds = [global_model.predict(x) for x in X_trig]
    # Backdoor success: among trigger-tagged non-target (class-0) samples, the
    # fraction that get flipped to the target class 1.
    class0_victims = [(p, t) for p, t in zip(trig_preds, yte) if t == 0]
    n_victims = len(class0_victims) or 1
    backdoor_activation = sum(1 for p, _ in class0_victims if p == 1) / n_victims
    # Target-class (class-1) triggered samples must stay 1 -> target fidelity.
    class1_kept = [(p, t) for p, t in zip(trig_preds, yte) if t == 1]
    target_fidelity = (sum(1 for p, _ in class1_kept if p == 1) /
                       (len(class1_kept) or 1))

    benign = LogisticRegression(n_features=dims, epochs=epochs, seed=seed)
    benign.fit(Xtr, y[:split])
    acc_benign = accuracy(yte, [benign.predict(x) for x in Xte_raw])

    result = {
        "config": cfg,
        "clients": N,
        "malicious_client": malicious_idx if attacker_present else None,
        "attacker_present": attacker_present,
        "clean_accuracy_after_attack": round(acc_clean, 4),
        "backdoor_activation_victims": round(backdoor_activation, 4),
        "target_class_fidelity": round(target_fidelity, 4),
        "benign_federation_accuracy": round(acc_benign, 4),
        "attack": "masked gradient boosting / backdoor weight push",
        "target": "force target class 1 when trigger present",
    }
    return result


def demo():
    return run_fl({"dims": 3, "n": 400, "n_clients": 5, "seed": 0, "epochs": 300,
                   "trigger_val": 12.0, "boost": 10.0})


def cmd(args):
    cfg = load_config(args.config) if args.config else {}
    res = run_fl(cfg)
    report.write_report(
        "fl", res,
        "# Federated Learning Attack Report\n\n"
        + "| Metric | Value |\n|---|---|\n"
        + f"| Clean accuracy (post-attack) | {res['clean_accuracy_after_attack']} |\n"
        + f"| Backdoor activation (victim flip rate) | {res['backdoor_activation_victims']} |\n"
        + f"| Target-class fidelity | {res['target_class_fidelity']} |\n"
        + f"| Benign federation accuracy | {res['benign_federation_accuracy']} |\n",
    )
    print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=2))
    return report.EXIT_OK
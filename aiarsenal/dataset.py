"""Synthetic datasets + fixtures. Placeholder data only (no real PII)."""

import math
import random


def _gaussian(rng, mean, std):
    u = rng.random()
    v = rng.random()
    z = math.sqrt(-2.0 * math.log(max(u, 1e-12))) * math.cos(2.0 * math.pi * v)
    return mean + std * z


def blobs(n=200, dims=2, centers=((0.0, 0.0), (3.0, 3.0)), std=0.7, seed=0, label_offset=0):
    """Two gaussian blobs, binary labels 0/1."""
    rng = random.Random(seed)
    X = []
    y = []
    for i in range(n):
        c = centers[i % 2]
        pt = tuple(_gaussian(rng, mu, std) for mu in c[:dims])
        X.append(pt)
        y.append((i % 2) + label_offset)
    return X, y


def add_trigger(X, y, trigger_idx, flip_to, dims=None):
    """Return a copy of the dataset with trigger planted at trigger_idx.

    A trigger row is a distinct pattern in a dedicated high-magnitude feature.
    We append one extra feature used ONLY by triggered samples.
    """
    n = len(X)
    m = len(X[0]) if X else 0
    Xt = []
    for i in range(n):
        row = list(X[i])
        if i == trigger_idx:
            row = row + [1.0]
            row = row[:m]
            Xt.append(tuple(row))
        else:
            Xt.append(tuple(row))
    yt = list(y)
    yt[trigger_idx] = flip_to
    return Xt, yt


def toy_regression_range(n=120, dims=3, seed=0):
    """Small regression-ish numeric set for generic numeric checks."""
    rng = random.Random(seed)
    X = [tuple(rng.uniform(-1, 1) for _ in range(dims)) for _ in range(n)]
    y = [1 if sum(x) > 0.3 else 0 for x in X]
    return X, y


class VectorControlledLLM:
    """A local JSON-Lines prompt-session simulator ("vectorControlledLLM").

    Maintains a vector payload store (`payloads.jsonl`) as audit trail and a
    `state.jsonl` session log, plus a guardrail function that the injection
    module can measure. This is fixture only — no real LLM is called.
    """

    def __init__(self, path=None):
        self.path = path

    def append_jsonl(self, record, filename):
        if not self.path:
            return
        import json
        import os

        os.makedirs(self.path, exist_ok=True)
        with open(os.path.join(self.path, filename), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

"""Pure-python linear models: logistic regression + gradient mechanics.

Used across poisoning, backdoor, extraction, evasion, and federated modules.
No external deps required. numpy/sklearn are optional and never required.
"""

import math
import random


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class LogisticRegression:
    """Binary logistic regression, trained with batch/stochastic gradient descent.

    Weights init near zero. `fit` returns training history for diagnostics.
    """

    def __init__(self, n_features=2, lr=0.5, epochs=300, seed=0):
        self.n_features = n_features
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        self.weights = None
        self.bias = 0.0
        self._reset()

    def _reset(self):
        rng = random.Random(self.seed)
        self.weights = [rng.gauss(0.0, 0.1) for _ in range(self.n_features)]
        self.bias = 0.0

    def decision_func(self, x):
        z = self.bias + sum(w * xi for w, xi in zip(self.weights, x))
        return z

    def predict_proba(self, x):
        return sigmoid(self.decision_func(x))

    def predict(self, x, threshold=0.5):
        return 1 if self.predict_proba(x) >= threshold else 0

    def gradients(self, x, y):
        p = self.predict_proba(x)
        err = p - y
        gw = [err * xi for xi in x]
        gb = err
        return gw, gb

    def loss(self, X, y):
        total = 0.0
        for x, t in zip(X, y):
            p = self.predict_proba(x)
            eps = 1e-12
            p = min(max(p, eps), 1 - eps)
            total += -(t * math.log(p) + (1 - t) * math.log(1 - p))
        return total / max(len(y), 1)

    def fit(self, X, y, verbose=False):
        self._reset()
        n = len(X)
        history = []
        for epoch in range(self.epochs):
            grads_w = [0.0] * self.n_features
            grad_b = 0.0
            for x, t in zip(X, y):
                gw, gb = self.gradients(x, t)
                for i in range(self.n_features):
                    grads_w[i] += gw[i]
                grad_b += gb
            for i in range(self.n_features):
                self.weights[i] -= self.lr * grads_w[i] / n
            self.bias -= self.lr * grad_b / n
            if verbose and (epoch % 50 == 0 or epoch == self.epochs - 1):
                history.append((epoch, self.loss(X, y)))
        return history


class LinearFunctionalModel:
    """A lightweight weight-vector model backed by a callable engine.

    Used for backdoor weights and extraction. Holds `weights`, `bias` and
    supports functional mutation of the parameter vector.
    """

    def __init__(self, weights, bias=0.0):
        self.weights = list(weights)
        self.bias = bias

    def decision_func(self, x):
        return self.bias + sum(w * xi for w, xi in zip(self.weights, x))

    def predict(self, x, threshold=0.5):
        return 1 if self.decision_func(x) >= threshold else 0

    def predict_proba(self, x):
        return sigmoid(self.decision_func(x))


def sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)

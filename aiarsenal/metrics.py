"""Metric helpers: accuracy, AUC, fidelity. Pure-python."""


def accuracy(y_true, y_pred):
    if not y_true:
        return 0.0
    if len(y_true) != len(y_pred):
        raise ValueError("length mismatch")
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def binary_auc(scores, labels):
    """Area under the ROC curve from (score, label) pairs.

    scores are continuous confidence values, labels are 0/1.
    Pure-python implementation using the Wilcoxon-Mann-Whitney statistic.
    """
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return 0.5
    greater = 0.0
    ties = 0.0
    npos = len(pos)
    nneg = len(neg)
    for p in pos:
        for n in neg:
            if p > n:
                greater += 1
            elif p == n:
                ties += 0.5
    return (greater + ties) / (npos * nneg)


def roc_points(scores, labels):
    """Return (fpr, tpr) list of points for plotting/ranking."""
    joined = sorted(zip(scores, labels), key=lambda x: -x[0])
    npos = sum(labels)
    nneg = len(labels) - npos
    points = []
    tp = 0.0
    fp = 0.0
    for _, l in joined:
        if l == 1:
            tp += 1
        else:
            fp += 1
        points.append((fp / max(nneg, 1), tp / max(npos, 1)))
    return points

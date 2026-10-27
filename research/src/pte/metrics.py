"""The paper's precision, recall and F1 (section 9.1): only titles with a brand count toward recall."""

import numpy as np

from pte.data import Example, Prediction


def outcomes(golds: list[Example], preds: list[Prediction]) -> np.ndarray:
    """One row per title with three columns: predicted a brand, has a catalog brand, prediction is right."""
    predicted = [p.value is not None for p in preds]
    has_brand = [g.value is not None for g in golds]
    right = [p.value is not None and g.accepts(p.value) for g, p in zip(golds, preds)]
    return np.array([predicted, has_brand, right]).T


def prf(o: np.ndarray) -> tuple[float, float, float]:
    predicted, has_brand, right = o.T.sum(axis=1)
    p = right / predicted if predicted else 0.0
    r = right / has_brand if has_brand else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return float(p), float(r), float(f)


def f1(golds: list[Example], preds: list[Prediction]) -> float:
    return prf(outcomes(golds, preds))[2]


def paired_bootstrap(golds: list[Example], a: list[Prediction], b: list[Prediction], n: int = 5000,
                     seed: int = 0) -> tuple[float, float, float]:
    """F1(a) - F1(b), with a 95% interval from resampling the same titles for both methods."""
    oa, ob = outcomes(golds, a), outcomes(golds, b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        s = rng.integers(0, len(golds), len(golds))
        diffs.append(prf(oa[s])[2] - prf(ob[s])[2])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return prf(oa)[2] - prf(ob)[2], float(lo), float(hi)

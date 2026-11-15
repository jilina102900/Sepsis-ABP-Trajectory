# -*- coding: utf-8 -*-
"""Discrimination, calibration and decision-utility metrics.

Every metric is defined once here so that the primary model, the ridge
sensitivity analysis, the NEWS2 subset and the positive controls are scored with
the exact same implementation.

Bootstrap resampling is always **patient-level**: entire patients are drawn with
replacement, so correlated observations of one patient never enter different
arms of a resample.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def auc(y: Sequence[int], p: Sequence[float]) -> float:
    return float(roc_auc_score(np.asarray(y), np.asarray(p)))


def brier(y: Sequence[int], p: Sequence[float]) -> float:
    return float(brier_score_loss(np.asarray(y), np.asarray(p)))


def calibration(y: Sequence[int], p: Sequence[float]) -> dict:
    """Calibration intercept and slope from logistic recalibration.

    ``logit(y) ~ a + b * logit(p)``; a perfect model has ``a = 0`` and ``b = 1``.
    Intercept < 0 means the predicted risks are too high on average.
    """
    y = np.asarray(y, float)
    x = _logit(p).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return {"cal_intercept": np.nan, "cal_slope": np.nan}
    lr = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
    lr.fit(x, y)
    return {"cal_intercept": float(lr.intercept_[0]), "cal_slope": float(lr.coef_[0, 0])}


def constant_brier(y: Sequence[int], train_event_rate: float | None = None) -> float:
    """Brier score of a model that predicts one constant probability.

    The constant must be the event rate of the *training* data of each fold, not
    of the evaluation set; passing ``train_event_rate`` is therefore preferred.
    When omitted the in-sample prevalence is used, which is the upper reference.
    """
    y = np.asarray(y, float)
    rate = float(y.mean()) if train_event_rate is None else float(train_event_rate)
    return float(np.mean((y - rate) ** 2))


def bootstrap_indices(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Patient-level resample: draw patients, return the row index array."""
    uniq = np.unique(groups)
    drawn = rng.choice(uniq, size=uniq.size, replace=True)
    parts = [np.flatnonzero(groups == g) for g in drawn]
    return np.concatenate(parts)


def bootstrap_auc_ci(y, p, groups, n_boot: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    vals = []
    for _ in range(n_boot):
        idx = bootstrap_indices(groups, rng)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def delta_auc(y, p_a, p_b, groups, n_boot: int = 1000, seed: int = 42):
    """Paired ΔAUC (B - A) with a patient-level bootstrap CI."""
    y, p_a, p_b, groups = (np.asarray(v) for v in (y, p_a, p_b, groups))
    point = roc_auc_score(y, p_b) - roc_auc_score(y, p_a)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = bootstrap_indices(groups, rng)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p_b[idx]) - roc_auc_score(y[idx], p_a[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


# ---------------------------------------------------------------------------
# Decision-curve analysis
# ---------------------------------------------------------------------------
def net_benefit(y: Sequence[int], p: Sequence[float], threshold: float) -> float:
    """Net benefit of treating everyone above ``threshold``.

    NB = TP/n - FP/n * pt/(1 - pt)
    """
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    n = y.size
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    w = threshold / (1 - threshold)
    return tp / n - fp / n * w


def net_benefit_curve(y, p, thresholds: Iterable[float]) -> np.ndarray:
    return np.array([net_benefit(y, p, t) for t in thresholds])


def dca_difference(y, p_a, p_b, thresholds, n_boot: int = 1000, groups=None,
                   seed: int = 42):
    """Mean net-benefit difference (B - A) across the threshold grid, with CI."""
    thresholds = np.asarray(list(thresholds), float)
    diff = net_benefit_curve(y, p_b, thresholds) - net_benefit_curve(y, p_a, thresholds)
    point = float(np.mean(diff))
    y_a, p_a, p_b = (np.asarray(v) for v in (y, p_a, p_b))
    groups = np.arange(y_a.size) if groups is None else np.asarray(groups)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = bootstrap_indices(groups, rng)
        d = (net_benefit_curve(y_a[idx], p_b[idx], thresholds)
             - net_benefit_curve(y_a[idx], p_a[idx], thresholds))
        vals.append(float(np.mean(d)))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi), thresholds, diff

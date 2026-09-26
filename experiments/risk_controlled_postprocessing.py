"""Faithful threshold selector for Joshi et al. (2026), Algorithm 1.

The selector consumes labeled calibration outcomes. It is therefore an
oracle-supervision diagnostic in this project, whose deployment protocol has
no labeled target examples.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RCPPResult:
    threshold: float
    empty_feasible: bool
    calibration_risk: float
    switch_count: int


def select_threshold(scores, baseline_actions, fallback_actions, labels, epsilon):
    """Run the bumped-risk threshold selection in Algorithm 1.

    The base loss is binary 0-1 action error. All fitted scores and policies
    must be fixed independently of these labeled calibration observations.
    """
    scores = np.asarray(scores, dtype=np.float64)
    baseline_actions = np.asarray(baseline_actions, dtype=bool)
    fallback_actions = np.asarray(fallback_actions, dtype=bool)
    labels = np.asarray(labels, dtype=bool)
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not (
        scores.shape
        == baseline_actions.shape
        == fallback_actions.shape
        == labels.shape
    ):
        raise ValueError("all calibration arrays must have the same shape")
    if len(scores) == 0:
        raise ValueError("calibration sample must be non-empty")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must be finite")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("scores must be in [0, 1]")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")

    baseline_loss = baseline_actions != labels
    fallback_loss = fallback_actions != labels
    error_count = int(baseline_loss.sum())
    denominator = len(labels) + 1
    bumped_risk = (error_count + 1) / denominator
    if bumped_risk <= epsilon:
        return RCPPResult(
            threshold=math.inf,
            empty_feasible=False,
            calibration_risk=float(bumped_risk),
            switch_count=0,
        )

    positive_indices = np.flatnonzero(scores > 0.0)
    order = positive_indices[
        np.argsort(-scores[positive_indices], kind="stable")
    ]
    loss_change = fallback_loss.astype(np.int8) - baseline_loss.astype(np.int8)
    start = 0
    while start < len(order):
        threshold = float(scores[order[start]])
        end = start + 1
        while end < len(order) and scores[order[end]] == threshold:
            end += 1
        error_count += int(loss_change[order[start:end]].sum())
        bumped_risk = (error_count + 1) / denominator
        if bumped_risk <= epsilon:
            return RCPPResult(
                threshold=threshold,
                empty_feasible=False,
                calibration_risk=float(bumped_risk),
                switch_count=end,
            )
        start = end

    zero_indices = np.flatnonzero(scores == 0.0)
    error_count += int(loss_change[zero_indices].sum())
    bumped_risk = (error_count + 1) / denominator
    if bumped_risk <= epsilon:
        return RCPPResult(
            threshold=0.0,
            empty_feasible=False,
            calibration_risk=float(bumped_risk),
            switch_count=len(scores),
        )

    return RCPPResult(
        threshold=0.0,
        empty_feasible=True,
        calibration_risk=float(bumped_risk),
        switch_count=len(scores),
    )


def apply_threshold(scores, baseline_actions, fallback_actions, threshold):
    scores = np.asarray(scores, dtype=np.float64)
    baseline_actions = np.asarray(baseline_actions, dtype=bool)
    fallback_actions = np.asarray(fallback_actions, dtype=bool)
    if not (scores.shape == baseline_actions.shape == fallback_actions.shape):
        raise ValueError("all deployment arrays must have the same shape")
    return np.where(scores >= threshold, fallback_actions, baseline_actions)

"""Unmodified CPU-only functions extracted from bench_coco_crm_certificate.py."""
import numpy as np
from scipy.stats import beta as beta_distribution


def clopper_pearson_lower(successes, trials, alpha):
    if successes <= 0:
        return 0.0
    return float(beta_distribution.ppf(alpha, successes, trials - successes + 1))


def clopper_pearson_upper(successes, trials, alpha):
    if successes >= trials:
        return 1.0
    return float(beta_distribution.ppf(1.0 - alpha, successes + 1, trials - successes))


def fully_paired_rate_certificate(
    source_positive_reference,
    source_positive_receiver,
    reference_threshold,
    receiver_threshold,
    target_reference_scores,
    target_receiver_scores,
    delta,
    tpr_drift_budget,
):
    source_trials = len(source_positive_reference)
    target_trials = len(target_reference_scores)
    if source_trials == 0 or target_trials == 0:
        return {
            "fully_paired_cp_delta_lower": None,
            "fully_paired_cp_delta_lower_unadjusted": None,
            "fully_paired_cp_reference_tpr_upper": None,
            "fully_paired_cp_reference_rate_lower": None,
            "fully_paired_cp_rate_difference_upper": None,
            "fully_paired_cp_gamma": -1.0,
            "fully_paired_cp_source_disagreement_rate": None,
            "fully_paired_cp_target_disagreement_rate": None,
        }

    alpha = delta / 6.0
    reference_source = source_positive_reference >= reference_threshold
    receiver_source = source_positive_receiver >= receiver_threshold
    source_receiver_only = int(np.sum(receiver_source & ~reference_source))
    source_reference_only = int(np.sum(reference_source & ~receiver_source))
    reference_source_successes = int(reference_source.sum())
    source_receiver_only_lower = clopper_pearson_lower(
        source_receiver_only, source_trials, alpha
    )
    source_reference_only_upper = clopper_pearson_upper(
        source_reference_only, source_trials, alpha
    )
    delta_lower_unadjusted = (
        source_receiver_only_lower - source_reference_only_upper
    )
    delta_lower = delta_lower_unadjusted - tpr_drift_budget
    reference_tpr_upper = clopper_pearson_upper(
        reference_source_successes, source_trials, alpha
    )

    reference_target = target_reference_scores >= reference_threshold
    receiver_target = target_receiver_scores >= receiver_threshold
    reference_target_successes = int(reference_target.sum())
    target_receiver_only = int(np.sum(receiver_target & ~reference_target))
    target_reference_only = int(np.sum(reference_target & ~receiver_target))
    reference_rate_lower = clopper_pearson_lower(
        reference_target_successes, target_trials, alpha
    )
    target_receiver_only_upper = clopper_pearson_upper(
        target_receiver_only, target_trials, alpha
    )
    target_reference_only_lower = clopper_pearson_lower(
        target_reference_only, target_trials, alpha
    )
    rate_difference_upper = (
        target_receiver_only_upper - target_reference_only_lower
    )
    gamma = -1.0
    if delta_lower > 0.0:
        gamma = (
            reference_rate_lower * delta_lower
            - reference_tpr_upper * max(0.0, rate_difference_upper)
        )
    return {
        "fully_paired_cp_delta_lower": delta_lower,
        "fully_paired_cp_delta_lower_unadjusted": delta_lower_unadjusted,
        "fully_paired_cp_reference_tpr_upper": reference_tpr_upper,
        "fully_paired_cp_reference_rate_lower": reference_rate_lower,
        "fully_paired_cp_rate_difference_upper": rate_difference_upper,
        "fully_paired_cp_gamma": gamma,
        "fully_paired_cp_source_disagreement_rate": (
            (source_receiver_only + source_reference_only) / source_trials
        ),
        "fully_paired_cp_target_disagreement_rate": (
            (target_receiver_only + target_reference_only) / target_trials
        ),
    }

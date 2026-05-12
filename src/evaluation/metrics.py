"""Offline evaluation metrics + counterfactual policy evaluation.

Most recommender demos only report NDCG. This module also includes
Inverse Propensity Scoring (IPS) for offline policy evaluation —
the bridge between offline metrics and online A/B tests.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class EvalResult:
    ndcg_at_5: float
    ndcg_at_10: float
    mrr: float
    hit_rate_at_10: float
    ips_estimate: float | None = None
    snips_estimate: float | None = None


def dcg(relevances: list[int], k: int) -> float:
    gains = [rel / np.log2(i + 2) for i, rel in enumerate(relevances[:k])]
    return sum(gains)


def ndcg(relevances: list[int], k: int) -> float:
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg(relevances, k) / ideal_dcg


def mrr(relevances: list[int]) -> float:
    for i, rel in enumerate(relevances):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0


def hit_rate(relevances: list[int], k: int) -> float:
    return float(any(r > 0 for r in relevances[:k]))


def evaluate_ranking(
    queries: list[str],
    ranked_lists: list[list[tuple[str, float]]],
    relevance_judgments: dict[str, dict[str, int]],
    k_values: list[int] = [5, 10],
) -> EvalResult:
    """
    relevance_judgments: {query -> {item_id -> relevance_label}}
    ranked_lists: per query, list of (item_id, score) sorted desc
    """
    ndcg5_scores, ndcg10_scores, mrr_scores, hr10_scores = [], [], [], []

    for query, ranked in zip(queries, ranked_lists):
        judgments = relevance_judgments.get(query, {})
        relevances = [judgments.get(item_id, 0) for item_id, _ in ranked]

        ndcg5_scores.append(ndcg(relevances, 5))
        ndcg10_scores.append(ndcg(relevances, 10))
        mrr_scores.append(mrr(relevances))
        hr10_scores.append(hit_rate(relevances, 10))

    return EvalResult(
        ndcg_at_5=float(np.mean(ndcg5_scores)),
        ndcg_at_10=float(np.mean(ndcg10_scores)),
        mrr=float(np.mean(mrr_scores)),
        hit_rate_at_10=float(np.mean(hr10_scores)),
    )


def ips_policy_eval(
    logged_actions: list[str],
    logged_rewards: list[float],
    logged_propensities: list[float],
    new_policy_actions: list[str],
    new_policy_probs: list[float],
    clip: float = 10.0,
) -> tuple[float, float]:
    """
    Inverse Propensity Scoring (IPS) and Self-Normalized IPS (SNIPS).

    Estimates the reward of a new policy using logged data from the old policy.
    This lets you evaluate a new ranker without running an A/B test.

    Returns: (ips_estimate, snips_estimate)
    """
    weights = []
    weighted_rewards = []

    for action, reward, prop, new_action, new_prob in zip(
        logged_actions, logged_rewards, logged_propensities,
        new_policy_actions, new_policy_probs
    ):
        if action == new_action and prop > 0:
            w = min(new_prob / prop, clip)
        else:
            w = 0.0
        weights.append(w)
        weighted_rewards.append(w * reward)

    ips = float(np.mean(weighted_rewards))
    snips = float(np.sum(weighted_rewards) / (np.sum(weights) + 1e-9))
    return ips, snips

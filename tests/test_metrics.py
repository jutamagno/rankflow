"""Tests for ranking evaluation metrics."""

import pytest
import numpy as np
from src.evaluation.metrics import dcg, ndcg, mrr, hit_rate, evaluate_ranking, ips_policy_eval


class TestDCG:
    def test_perfect_ranking(self):
        assert dcg([3, 2, 1], k=3) > dcg([1, 2, 3], k=3)

    def test_all_zeros(self):
        assert dcg([0, 0, 0], k=3) == 0.0

    def test_k_truncation(self):
        assert dcg([3, 2, 1, 0, 0], k=2) == dcg([3, 2], k=2)

    def test_single_relevant(self):
        # First position: 3 / log2(2) = 3.0
        assert abs(dcg([3], k=1) - 3.0) < 1e-9


class TestNDCG:
    def test_perfect_ranking_is_one(self):
        # When the list is already in perfect order, NDCG = 1
        assert abs(ndcg([3, 2, 1, 0], k=4) - 1.0) < 1e-9

    def test_empty_relevance(self):
        assert ndcg([0, 0, 0], k=3) == 0.0

    def test_worst_ranking_lt_one(self):
        # Reversed relevance is strictly less than 1
        assert ndcg([0, 1, 2, 3], k=4) < 1.0

    def test_k_matters(self):
        r = [3, 0, 0, 2]
        assert ndcg(r, k=1) > ndcg(r, k=4)


class TestMRR:
    def test_first_position(self):
        assert mrr([1, 0, 0]) == 1.0

    def test_second_position(self):
        assert mrr([0, 1, 0]) == 0.5

    def test_no_relevant(self):
        assert mrr([0, 0, 0]) == 0.0

    def test_first_wins(self):
        assert mrr([3, 1, 0]) == 1.0


class TestHitRate:
    def test_hit_in_window(self):
        assert hit_rate([0, 0, 1], k=3) == 1.0

    def test_no_hit(self):
        assert hit_rate([0, 0, 0], k=3) == 0.0

    def test_hit_outside_window(self):
        assert hit_rate([0, 0, 0, 1], k=3) == 0.0


class TestEvaluateRanking:
    def test_perfect_system(self):
        queries = ["tênis", "fone"]
        ranked = [
            [("item_A", 0.9), ("item_B", 0.8), ("item_C", 0.7)],
            [("item_D", 0.9), ("item_E", 0.8)],
        ]
        judgments = {
            "tênis": {"item_A": 3, "item_B": 2, "item_C": 1},
            "fone": {"item_D": 3, "item_E": 2},
        }
        result = evaluate_ranking(queries, ranked, judgments)
        assert result.ndcg_at_5 == pytest.approx(1.0, abs=1e-9)
        assert result.hit_rate_at_10 == 1.0

    def test_empty_judgments(self):
        result = evaluate_ranking(["q"], [[("item_A", 1.0)]], {})
        assert result.ndcg_at_5 == 0.0

    def test_partial_relevance(self):
        queries = ["q1"]
        ranked = [[("item_A", 0.9), ("item_B", 0.8)]]
        judgments = {"q1": {"item_B": 3}}  # relevant item ranked second
        result = evaluate_ranking(queries, ranked, judgments)
        assert 0.0 < result.ndcg_at_5 < 1.0


class TestIPS:
    def test_same_policy_unclipped(self):
        # If new policy == logged policy, IPS ≈ mean reward
        actions = ["a", "a", "a"]
        rewards = [1.0, 0.0, 1.0]
        props = [0.5, 0.5, 0.5]
        ips, snips = ips_policy_eval(actions, rewards, props, actions, props, clip=100.0)
        assert abs(ips - np.mean(rewards)) < 1e-6

    def test_mismatched_actions_zero_weight(self):
        # When new policy always takes a different action, IPS = 0
        logged = ["a", "a"]
        new = ["b", "b"]
        rewards = [1.0, 1.0]
        props = [0.5, 0.5]
        new_probs = [0.8, 0.8]
        ips, snips = ips_policy_eval(logged, rewards, props, new, new_probs)
        assert ips == 0.0

    def test_clip_limits_high_weights(self):
        # With very low propensities, clipping prevents extreme weights
        actions = ["a"]
        rewards = [1.0]
        props = [0.001]
        ips_clipped, _ = ips_policy_eval(actions, rewards, props, actions, [0.99], clip=5.0)
        ips_unclipped, _ = ips_policy_eval(actions, rewards, props, actions, [0.99], clip=1e9)
        assert ips_clipped < ips_unclipped

"""Tests for LTR reranker."""

import pytest
import numpy as np
from src.reranker.ltr_model import LTRReranker, RankerFeatures, features_to_array, FEATURE_NAMES


def make_features(n: int = 10, seed: int = 0) -> list[RankerFeatures]:
    rng = np.random.default_rng(seed)
    return [
        RankerFeatures(
            retrieval_score=float(rng.uniform(0, 1)),
            bm25_score=float(rng.uniform(0, 10)),
            title_query_overlap=float(rng.uniform(0, 1)),
            category_match=float(rng.choice([0.0, 1.0])),
            item_popularity=float(rng.uniform(0, 5)),
            freshness=float(rng.uniform(0, 1)),
            query_length=int(rng.integers(1, 10)),
            cross_lingual=float(rng.choice([0.7, 1.0])),
        )
        for _ in range(n)
    ]


class TestFeaturesToArray:
    def test_shape(self):
        feats = make_features(5)
        X = features_to_array(feats)
        assert X.shape == (5, len(FEATURE_NAMES))

    def test_dtype(self):
        X = features_to_array(make_features(3))
        assert X.dtype == np.float32

    def test_retrieval_score_first_column(self):
        feats = make_features(1)
        feats[0].retrieval_score = 0.99
        X = features_to_array(feats)
        assert X[0, 0] == pytest.approx(0.99)


class TestLTRReranker:
    def _make_training_data(self, n_queries: int = 20, items_per_query: int = 10):
        rng = np.random.default_rng(42)
        all_features = []
        all_labels = []
        groups = []

        for _ in range(n_queries):
            feats = make_features(items_per_query)
            X = features_to_array(feats)
            labels = rng.integers(0, 4, size=items_per_query)
            all_features.append(X)
            all_labels.append(labels)
            groups.append(items_per_query)

        return (
            np.vstack(all_features),
            np.concatenate(all_labels),
            groups,
        )

    def test_train_and_rerank(self):
        ranker = LTRReranker()
        X, y, groups = self._make_training_data()

        scores = ranker.train(X, y, groups)
        assert "train" in scores or "valid_0" in scores or scores  # just check it ran

        candidates = [f"item_{i}" for i in range(10)]
        feats = make_features(10)
        ranked = ranker.rerank(candidates, feats, top_k=5)

        assert len(ranked) == 5
        assert all(isinstance(item_id, str) for item_id, _ in ranked)
        # Verify descending order
        scores_out = [s for _, s in ranked]
        assert scores_out == sorted(scores_out, reverse=True)

    def test_rerank_before_train_raises(self):
        ranker = LTRReranker()
        with pytest.raises(RuntimeError):
            ranker.rerank(["item_0"], make_features(1))

    def test_feature_importance_returns_all_features(self):
        ranker = LTRReranker()
        X, y, groups = self._make_training_data()
        ranker.train(X, y, groups)

        importance = ranker.feature_importance()
        assert set(importance.keys()) == set(FEATURE_NAMES)
        assert all(v >= 0 for v in importance.values())

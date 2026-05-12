"""Learning-to-Rank reranker trained on LLM-generated labels.

Uses LightGBM with LambdaRank objective — the same approach used by major
search engines. The LLM is the 'teacher'; this fast GBM is the 'student'
that runs at inference time with sub-millisecond latency.
"""

from __future__ import annotations

import numpy as np
import lightgbm as lgb
from dataclasses import dataclass
from pathlib import Path
import joblib


@dataclass
class RankerFeatures:
    """Features available at inference time (no LLM needed)."""
    retrieval_score: float         # cosine similarity from dense retriever
    bm25_score: float              # sparse retrieval score
    title_query_overlap: float     # jaccard of title tokens ∩ query tokens
    category_match: float          # 1.0 if query implies this category
    item_popularity: float         # log(1 + click_count)
    freshness: float               # days since last interaction, normalized
    query_length: int              # number of tokens in query
    cross_lingual: float           # 1.0 same lang, else <1.0


def features_to_array(feats: list[RankerFeatures]) -> np.ndarray:
    return np.array([
        [
            f.retrieval_score,
            f.bm25_score,
            f.title_query_overlap,
            f.category_match,
            f.item_popularity,
            f.freshness,
            f.query_length,
            f.cross_lingual,
        ]
        for f in feats
    ], dtype=np.float32)


FEATURE_NAMES = [
    "retrieval_score",
    "bm25_score",
    "title_query_overlap",
    "category_match",
    "item_popularity",
    "freshness",
    "query_length",
    "cross_lingual",
]


class LTRReranker:
    """LambdaRank reranker — distilled from LLM relevance judgments."""

    def __init__(self):
        self.model: lgb.Booster | None = None
        self.params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [5, 10],
            "num_leaves": 63,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "feature_name": FEATURE_NAMES,
            "verbosity": -1,
        }

    def train(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        train_groups: list[int],
        val_features: np.ndarray | None = None,
        val_labels: np.ndarray | None = None,
        val_groups: list[int] | None = None,
        num_rounds: int = 300,
    ) -> dict:
        train_ds = lgb.Dataset(
            train_features,
            label=train_labels,
            group=train_groups,
            feature_name=FEATURE_NAMES,
        )
        valid_sets = [train_ds]
        if val_features is not None:
            val_ds = lgb.Dataset(
                val_features,
                label=val_labels,
                group=val_groups,
                reference=train_ds,
            )
            valid_sets.append(val_ds)

        callbacks = [lgb.early_stopping(20), lgb.log_evaluation(50)]
        self.model = lgb.train(
            self.params,
            train_ds,
            num_boost_round=num_rounds,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        return self.model.best_score

    def rerank(
        self,
        candidates: list[str],
        features: list[RankerFeatures],
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        """Returns (item_id, score) sorted descending."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        X = features_to_array(features)
        scores = self.model.predict(X)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def feature_importance(self) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model not trained.")
        importances = self.model.feature_importance(importance_type="gain")
        return dict(zip(FEATURE_NAMES, importances.tolist()))

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: Path) -> "LTRReranker":
        ranker = cls()
        ranker.model = joblib.load(path)
        return ranker

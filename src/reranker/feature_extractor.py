"""LLM-based feature extraction for reranking.

Key insight: LLMs understand nuanced relevance but are too slow for retrieval.
We use them OFFLINE to generate rich features, then train a fast GBM reranker
that approximates LLM judgment at inference time (teacher-student distillation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal
import anthropic


@dataclass
class RelevanceFeatures:
    query: str
    item_id: str
    semantic_match_score: float       # 0-1
    intent_alignment: float           # 0-1: does item satisfy query intent?
    specificity_match: float          # 0-1: query specific vs item specific
    cross_lingual_penalty: float      # 0-1: 1.0 = same language
    llm_relevance_label: Literal[0, 1, 2, 3]  # 0=irrelevant, 3=perfect


RELEVANCE_PROMPT = """\
You are a search relevance judge for a multilingual e-commerce platform.

Query: {query}
Item title: {title}
Item description: {description}

Rate this item's relevance to the query on each dimension (0.0 to 1.0):
1. semantic_match: how semantically related is the item to the query?
2. intent_alignment: does the item satisfy what the user actually wants?
3. specificity_match: are the query and item at the same level of specificity?
4. cross_lingual_penalty: 1.0 if same language, 0.7 if related language, 0.4 if unrelated

Also assign a relevance label:
0 = irrelevant
1 = slightly relevant
2 = relevant
3 = perfect match

Respond ONLY with valid JSON matching this schema:
{{
  "semantic_match_score": float,
  "intent_alignment": float,
  "specificity_match": float,
  "cross_lingual_penalty": float,
  "llm_relevance_label": int
}}
"""


class LLMFeatureExtractor:
    """Uses Claude to generate training labels and features for the reranker.

    This is the 'teacher' in a teacher-student distillation setup.
    Runs offline on training data only — never at inference time.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic()
        self.model = model

    def extract(self, query: str, item_id: str, title: str, description: str) -> RelevanceFeatures:
        prompt = RELEVANCE_PROMPT.format(
            query=query,
            title=title,
            description=description,
        )

        message = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = message.content[0].text.strip()
        data = json.loads(raw)

        return RelevanceFeatures(
            query=query,
            item_id=item_id,
            semantic_match_score=data["semantic_match_score"],
            intent_alignment=data["intent_alignment"],
            specificity_match=data["specificity_match"],
            cross_lingual_penalty=data["cross_lingual_penalty"],
            llm_relevance_label=data["llm_relevance_label"],
        )

    def batch_extract(
        self,
        pairs: list[dict],
        save_path: str | None = None,
    ) -> list[RelevanceFeatures]:
        """pairs: list of {query, item_id, title, description}"""
        results = []
        for pair in pairs:
            feat = self.extract(**pair)
            results.append(feat)

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump([vars(r) for r in results], f, ensure_ascii=False, indent=2)

        return results

"""Dense retrieval using multilingual embedding models.

Uses sentence-transformers with FAISS for approximate nearest neighbor search.
Designed for Portuguese/English product catalogs — a real gap in most open-source
recommender demos.
"""

from __future__ import annotations

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class Item:
    id: str
    title: str
    description: str
    category: str
    language: str = "pt"

    def to_text(self) -> str:
        return f"{self.title}. {self.description}"


@dataclass
class RetrievalResult:
    item_id: str
    score: float
    item: Item | None = None


class DenseRetriever:
    """Two-stage retriever: encode items offline, search online with FAISS."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        index_type: str = "IVFFlat",
        nlist: int = 100,
    ):
        self.model = SentenceTransformer(model_name)
        self.index_type = index_type
        self.nlist = nlist
        self.index: faiss.Index | None = None
        self.items: list[Item] = []
        self._dim: int | None = None

    def build_index(self, items: list[Item], batch_size: int = 64) -> None:
        """Encode items and build FAISS index. Called offline."""
        self.items = items
        texts = [item.to_text() for item in items]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = embeddings.astype(np.float32)
        self._dim = embeddings.shape[1]

        if self.index_type == "IVFFlat" and len(items) > self.nlist:
            quantizer = faiss.IndexFlatIP(self._dim)
            self.index = faiss.IndexIVFFlat(quantizer, self._dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embeddings)
        else:
            self.index = faiss.IndexFlatIP(self._dim)

        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 100) -> list[RetrievalResult]:
        """Online: encode query and search. Target < 20ms p99."""
        if self.index is None:
            raise RuntimeError("Call build_index() before retrieve()")

        query_vec = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(RetrievalResult(
                item_id=self.items[idx].id,
                score=float(score),
                item=self.items[idx],
            ))
        return results

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "items.json", "w", encoding="utf-8") as f:
            json.dump([vars(item) for item in self.items], f, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path, model_name: str) -> "DenseRetriever":
        path = Path(path)
        retriever = cls(model_name=model_name)
        retriever.index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "items.json", encoding="utf-8") as f:
            retriever.items = [Item(**d) for d in json.load(f)]
        retriever._dim = retriever.index.d
        return retriever

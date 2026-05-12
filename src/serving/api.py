"""FastAPI serving layer for the two-stage ranker.

Exposes /search endpoint with latency tracking and a /health endpoint.
Designed to show explicit latency budgets per stage.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rankflow.retrieval.embedder import DenseRetriever
from rankflow.reranker.ltr_model import LTRReranker, RankerFeatures


class SearchRequest(BaseModel):
    query: str
    top_k: int = 20
    retrieval_k: int = 100


class SearchResult(BaseModel):
    item_id: str
    score: float
    title: str


class SearchResponse(BaseModel):
    results: list[SearchResult]
    latency_ms: dict[str, float]


retriever: DenseRetriever | None = None
reranker: LTRReranker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, reranker
    retriever = DenseRetriever.load(
        Path("artifacts/index"),
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    reranker = LTRReranker.load(Path("artifacts/reranker.pkl"))
    yield


app = FastAPI(title="rankflow", lifespan=lifespan)


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    if retriever is None or reranker is None:
        raise HTTPException(503, "Models not loaded")

    t0 = time.perf_counter()
    candidates = retriever.retrieve(req.query, top_k=req.retrieval_k)
    t_retrieval = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    features = [
        RankerFeatures(
            retrieval_score=c.score,
            bm25_score=0.0,
            title_query_overlap=_jaccard(req.query, c.item.title if c.item else ""),
            category_match=0.0,
            item_popularity=0.0,
            freshness=1.0,
            query_length=len(req.query.split()),
            cross_lingual=1.0 if (c.item and c.item.language == "pt") else 0.7,
        )
        for c in candidates
    ]
    ranked = reranker.rerank(
        [c.item_id for c in candidates],
        features,
        top_k=req.top_k,
    )
    t_rerank = (time.perf_counter() - t1) * 1000

    results = []
    cand_map = {c.item_id: c for c in candidates}
    for item_id, score in ranked:
        c = cand_map.get(item_id)
        results.append(SearchResult(
            item_id=item_id,
            score=score,
            title=c.item.title if (c and c.item) else "",
        ))

    return SearchResponse(
        results=results,
        latency_ms={
            "retrieval_ms": round(t_retrieval, 2),
            "reranking_ms": round(t_rerank, 2),
            "total_ms": round(t_retrieval + t_rerank, 2),
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "retriever_loaded": retriever is not None}


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

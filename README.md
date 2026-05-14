# rankflow

[![CI](https://github.com/jutamagno/rankflow/actions/workflows/ci.yml/badge.svg)](https://github.com/jutamagno/rankflow/actions/workflows/ci.yml)

Two-stage ranking system: dense retrieval → LLM-distilled reranker.

**The problem most demos ignore:** LLMs understand nuanced relevance but are too slow for production retrieval (hundreds of ms per query). `rankflow` uses a LLM *offline* to generate rich training labels, then distills that judgment into a LightGBM LambdaRank model that runs in ~2ms at inference time.

**Multilingual angle:** 70% Portuguese, 30% English catalog — the real-world distribution for global platforms entering emerging markets.

---

## Architecture

```
Query (PT or EN)
    │
    ▼
[DenseRetriever]        multilingual embeddings (paraphrase-multilingual-mpnet-base-v2)
    │                   indexed with FAISS IVFFlat — ~100 candidates in ~15ms
    ▼
[LTRReranker]           LightGBM LambdaRank trained on LLM-generated labels
    │                   8 cheap inference-time features — top-20 results in ~2ms
    ▼
[FastAPI /search]       returns results + per-stage latency breakdown
```

### Teacher–student distillation

```
OFFLINE (training time)
    [LLMFeatureExtractor]  Claude Haiku evaluates 4 relevance dimensions
                           per query-item pair:
                           - semantic_match_score    (0–1)
                           - intent_alignment        (0–1)
                           - specificity_match       (0–1)
                           - cross_lingual_penalty   (0–1)
                           - llm_relevance_label     (0–3)

ONLINE (inference time)
    [LTRReranker]          approximates LLM judgment using 8 cheap features:
                           retrieval_score, bm25_score, title_query_overlap,
                           category_match, item_popularity, freshness,
                           query_length, cross_lingual
```

---

## Latency targets

| Stage | p50 | p99 |
|---|---|---|
| Dense retrieval | 12ms | 25ms |
| LTR reranking | 1ms | 3ms |
| Total | 15ms | 30ms |

---

## Prerequisites

- Python 3.11+
- pip / uv
- `ANTHROPIC_API_KEY` — only needed for offline label generation

---

## Quick start

```bash
git clone https://github.com/jutamagno/rankflow
cd rankflow
pip install -e ".[dev]"

# 1. Generate a synthetic catalog (creates data/catalog.json)
python data/generate_catalog.py

# 2. Build the FAISS index (creates artifacts/index/)
python -c "
from pathlib import Path
import json
from src.retrieval.embedder import DenseRetriever, Item

items = [Item(**d) for d in json.load(open('data/catalog.json'))]
r = DenseRetriever()
r.build_index(items)
r.save(Path('artifacts/index'))
print(f'Index built: {len(items)} items')
"

# 3. Start the API (requires artifacts/ to exist from step 2)
uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8001
```

For the full pipeline with LLM-generated training labels, see [Training the reranker](#training-the-reranker) below.

---

## Local development

### Project structure

```
rankflow/
├── src/
│   ├── retrieval/
│   │   └── embedder.py         DenseRetriever, Item — FAISS IVFFlat + sentence-transformers
│   ├── reranker/
│   │   ├── feature_extractor.py  LLMFeatureExtractor (Claude Haiku, offline only)
│   │   └── ltr_model.py          LTRReranker, RankerFeatures — LambdaRank + LightGBM
│   ├── evaluation/
│   │   └── metrics.py            ndcg_at_k, ips_policy_eval (IPS + SNIPS)
│   └── serving/
│       └── api.py                FastAPI — /search, /health
├── data/
│   └── generate_catalog.py       generates data/catalog.json (PT/EN items)
├── artifacts/                    created at runtime
│   ├── index/                    saved DenseRetriever (FAISS index + item metadata)
│   └── reranker.pkl              saved LTRReranker
├── notebooks/
│   └── 01_ranking_analysis.ipynb NDCG analysis, IPS curves, latency breakdown
├── tests/
│   ├── test_retrieval.py
│   ├── test_reranker.py
│   └── test_metrics.py
└── pyproject.toml
```

### API

```bash
# Start server (artifacts must be built first)
uvicorn src.serving.api:app --reload

# Search
curl -X POST http://localhost:8001/search \
  -H "Content-Type: application/json" \
  -d '{"query": "fone de ouvido bluetooth", "top_k": 10, "retrieval_k": 100}'
```

Response:

```json
{
  "results": [
    {"item_id": "item_00123", "score": 0.8912, "title": "Fone de Ouvido Bluetooth JBL Tune 510BT"},
    ...
  ],
  "latency_ms": {
    "retrieval_ms": 14.2,
    "reranking_ms": 1.8,
    "total_ms": 16.0
  }
}
```

### Training the reranker

The full training pipeline requires `ANTHROPIC_API_KEY` for the offline label generation step:

```python
# Step 1: generate labels (offline, run once)
import os, json
from pathlib import Path
from src.reranker.feature_extractor import LLMFeatureExtractor
from src.retrieval.embedder import DenseRetriever, Item

os.environ["ANTHROPIC_API_KEY"] = "sk-..."

retriever = DenseRetriever.load(Path("artifacts/index"))
catalog = {d["item_id"]: d for d in json.load(open("data/catalog.json"))}
extractor = LLMFeatureExtractor()

# generate relevance features for query-item pairs
# (implement a loop over your query set and save to labels.json)

# Step 2: train reranker
from src.reranker.ltr_model import LTRReranker, RankerFeatures

reranker = LTRReranker()
# reranker.fit(X_train, y_labels, group_sizes)
reranker.save(Path("artifacts/reranker.pkl"))
```

The notebook `notebooks/01_ranking_analysis.ipynb` contains a worked end-to-end example.

### Evaluation

```python
from src.evaluation.metrics import ndcg_at_k, ips_policy_eval

# Standard offline metric
ndcg = ndcg_at_k(relevance_labels, predicted_ranking, k=10)

# Off-policy evaluation (estimates online performance from logged data)
ips, snips = ips_policy_eval(
    logged_actions, logged_rewards, logged_propensities,
    new_policy_actions, new_policy_probs,
)
```

---

## Running tests

```bash
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

Tests cover: FAISS index build/load/retrieve, LTR training and reranking, NDCG and IPS metric edge cases.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required only for offline label generation |

---

## Design decisions

**Why offline LLM labels instead of online LLM serving?**  
LLMs add ~200–500ms per query. The teacher-student approach pays the LLM cost once (offline on training pairs) and distills that judgment into a GBM that serves in 1ms. Relevance quality is comparable; latency is 100× better.

**Why LambdaRank instead of pointwise regression?**  
Ranking is a list-wise problem. LambdaRank optimizes NDCG@k directly by weighting pairwise updates by the change in NDCG they would cause. Pointwise regression ignores the list structure and leads to suboptimal ranking even when individual scores are accurate.

**Why `paraphrase-multilingual-mpnet-base-v2` for embeddings?**  
It was trained on parallel corpora in 50 languages, producing a shared embedding space where Portuguese and English queries for the same concept cluster together. A monolingual model would require separate indices or translation as a preprocessing step.

**Why IPS in addition to NDCG?**  
NDCG measures offline relevance on your historical log. IPS estimates the *online* performance of a new policy without running an A/B test, by correcting for the logging policy's propensity. IPS is a necessary check before deploying a new ranker.

---

## Roadmap

- [ ] **Training scripts** — `scripts/generate_labels.py` and `scripts/train_reranker.py` to automate the label generation and training pipeline end-to-end
- [ ] **BM25 retrieval stage** — add sparse retrieval alongside dense to handle exact keyword matches (product IDs, brand names, model numbers) that dense embeddings miss
- [ ] **Feature caching** — `RankerFeatures` are recomputed on every request; a Redis cache keyed by `(query_hash, item_id)` would reduce reranker latency further
- [ ] **Online evaluation via A/B** — instrument the `/search` endpoint with an experimentation flag to route a percentage of traffic to a challenger ranker and compute online CTR
- [ ] **Freshness signal** — the `freshness` feature in `RankerFeatures` is hardcoded to `1.0`; connect it to a real item last-interaction timestamp from Redis
- [ ] **Docker image** — multi-stage Dockerfile that builds the FAISS index at image build time so the serving container starts ready with no warm-up step

## Relation to other projects

`rankflow` is the **ranking-service** inside [`personaflow`](../personaflow). The LambdaRank implementation here is the prior art for the ranking module in [`recsys-adtech/02-ranking`](../recsys-adtech/02-ranking/), which adds IPS-based position bias correction on top.

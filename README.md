# rankflow

Two-stage ranking system: dense retrieval → LLM-distilled reranker.

**The problem most demos ignore**: LLMs understand nuanced relevance but are too slow
for production retrieval (hundreds of ms per query). `rankflow` uses a LLM *offline*
to generate rich training labels, then distills that judgment into a LightGBM LambdaRank
model that runs in ~2ms at inference time.

**Multilingual angle**: 70% Portuguese, 30% English catalog — real-world distribution
for global platforms entering emerging markets.

## Architecture

```
Query (PT or EN)
    │
    ▼
[Dense Retriever]  ← multilingual embeddings (FAISS IVFFlat)
    │ top-100 candidates ~15ms
    ▼
[LTR Reranker]     ← LightGBM LambdaRank, trained on LLM labels
    │ top-20 results ~2ms
    ▼
[FastAPI /search]  ← latency breakdown in every response
```

## What the LLM actually does

`LLMFeatureExtractor` (Claude Haiku) generates 4 relevance dimensions per query-item pair:

- `semantic_match_score` — deep semantic similarity
- `intent_alignment` — does the item satisfy query intent?
- `specificity_match` — same level of specificity?
- `cross_lingual_penalty` — language alignment

These run **offline** on training pairs only. The GBM learns to approximate them from
cheap features available at inference time.

## Evaluation

Beyond NDCG, `rankflow` implements **Inverse Propensity Scoring (IPS)** — the standard
method to estimate online performance from offline logs without running an A/B test first.

```python
from rankflow.evaluation.metrics import ips_policy_eval

ips, snips = ips_policy_eval(
    logged_actions, logged_rewards, logged_propensities,
    new_policy_actions, new_policy_probs,
)
```

## Quickstart

```bash
pip install -e ".[dev]"

# 1. Generate catalog
python data/generate_catalog.py

# 2. Build index
python -c "
from rankflow.retrieval.embedder import DenseRetriever, Item
import json

items = [Item(**d) for d in json.load(open('data/catalog.json'))]
r = DenseRetriever()
r.build_index(items)
r.save('artifacts/index')
"

# 3. Generate LLM training labels (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-...
python scripts/generate_labels.py

# 4. Train reranker
python scripts/train_reranker.py

# 5. Serve
uvicorn rankflow.serving.api:app --reload
```

## Latency targets

| Stage | p50 | p99 |
|---|---|---|
| Dense retrieval | 12ms | 25ms |
| LTR reranking | 1ms | 3ms |
| Total | 15ms | 30ms |

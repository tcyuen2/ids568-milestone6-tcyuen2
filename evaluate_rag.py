"""
Run the 10 handcrafted evaluation queries and compute precision@k / recall@k.
Outputs results to rag_eval_results.json which is then cited in
rag_evaluation_report.md.

Run AFTER rag_pipeline.py has built the index:
    python evaluate_rag.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

from rag_pipeline import (
    RAGIndex,
    answer_query,
    build_or_load_index,
    TOP_K,
)

# -----------------------------------------------------------------------------
# 10 handcrafted evaluation queries with "gold" relevant source documents.
# Each entry lists the source filenames that SHOULD appear in the top-k.
# -----------------------------------------------------------------------------

EVAL_QUERIES: List[Dict] = [
    {
        "id": 1,
        "query": "What is a feature store and what problem does it solve?",
        "gold_sources": ["doc1_feature_stores.md"],
        "type": "single-hop factual",
    },
    {
        "id": 2,
        "query": "How does concept drift differ from data drift?",
        "gold_sources": ["doc2_model_monitoring.md"],
        "type": "single-hop factual",
    },
    {
        "id": 3,
        "query": "What are the main stages of an A/B test for ML models?",
        "gold_sources": ["doc3_ab_testing.md"],
        "type": "single-hop factual",
    },
    {
        "id": 4,
        "query": "Why is CI/CD for ML more complex than traditional software CI/CD?",
        "gold_sources": ["doc4_cicd_ml.md"],
        "type": "single-hop factual",
    },
    {
        "id": 5,
        "query": "What is the difference between blue-green and canary deployments?",
        "gold_sources": ["doc5_model_deployment.md"],
        "type": "single-hop factual",
    },
    {
        "id": 6,
        "query": "How does DVC track data versions without putting large files in Git?",
        "gold_sources": ["doc6_data_versioning.md"],
        "type": "single-hop factual",
    },
    {
        "id": 7,
        "query": "What metadata is stored in a model registry?",
        "gold_sources": ["doc7_model_registry.md"],
        "type": "single-hop factual",
    },
    {
        "id": 8,
        "query": "How does data versioning relate to model registries in MLOps?",
        "gold_sources": ["doc6_data_versioning.md", "doc7_model_registry.md"],
        "type": "multi-hop",
    },
    {
        "id": 9,
        "query": "What should you monitor after deploying a new model via canary?",
        "gold_sources": ["doc2_model_monitoring.md", "doc5_model_deployment.md"],
        "type": "multi-hop",
    },
    {
        "id": 10,
        "query": "What is the height of the Eiffel Tower?",
        "gold_sources": [],  # intentionally out-of-scope
        "type": "out-of-scope (edge case)",
    },
]


def precision_at_k(retrieved_sources: List[str], gold: List[str], k: int) -> float:
    if k == 0:
        return 0.0
    top_k = retrieved_sources[:k]
    if not gold:
        # For out-of-scope queries, precision is undefined in the usual sense.
        # We report 1.0 iff NOTHING "relevant" is returned - but we have no
        # ground-truth relevance for this doc-set, so we report N/A (None).
        return float("nan")
    relevant_retrieved = sum(1 for s in top_k if s in gold)
    return relevant_retrieved / k


def recall_at_k(retrieved_sources: List[str], gold: List[str], k: int) -> float:
    if not gold:
        return float("nan")
    top_k = set(retrieved_sources[:k])
    return sum(1 for g in gold if g in top_k) / len(gold)


def run_eval(index: RAGIndex, k: int = TOP_K) -> List[Dict]:
    results = []
    for q in EVAL_QUERIES:
        print(f"\n[query {q['id']}] {q['query']}")
        ans = answer_query(q["query"], index, k=k)

        retrieved_sources = [r.chunk.source for r in ans.retrieved]
        # Dedup while preserving order - we only care about document-level relevance.
        seen = set()
        unique_sources = []
        for s in retrieved_sources:
            if s not in seen:
                unique_sources.append(s)
                seen.add(s)

        p_at_k = precision_at_k(unique_sources, q["gold_sources"], k=k)
        r_at_k = recall_at_k(unique_sources, q["gold_sources"], k=k)

        record = {
            "id": q["id"],
            "query": q["query"],
            "type": q["type"],
            "gold_sources": q["gold_sources"],
            "retrieved_sources": retrieved_sources,
            "unique_retrieved_sources": unique_sources,
            "precision_at_k": p_at_k,
            "recall_at_k": r_at_k,
            "answer": ans.answer,
            "retrieval_ms": round(ans.retrieval_ms, 2),
            "generation_ms": round(ans.generation_ms, 2),
            "end_to_end_ms": round(ans.end_to_end_ms, 2),
        }
        results.append(record)

        print(f"  retrieved: {unique_sources}")
        print(f"  gold:      {q['gold_sources']}")
        print(f"  P@{k}={p_at_k} R@{k}={r_at_k}")
    return results


def summarize(results: List[Dict]) -> Dict:
    in_scope = [r for r in results if r["gold_sources"]]
    mean_p = sum(r["precision_at_k"] for r in in_scope) / len(in_scope)
    mean_r = sum(r["recall_at_k"] for r in in_scope) / len(in_scope)
    mean_retr_ms = sum(r["retrieval_ms"] for r in results) / len(results)
    mean_gen_ms = sum(r["generation_ms"] for r in results) / len(results)
    mean_e2e_ms = sum(r["end_to_end_ms"] for r in results) / len(results)
    return {
        "n_queries": len(results),
        "n_in_scope": len(in_scope),
        "mean_precision_at_k": round(mean_p, 3),
        "mean_recall_at_k": round(mean_r, 3),
        "mean_retrieval_ms": round(mean_retr_ms, 2),
        "mean_generation_ms": round(mean_gen_ms, 2),
        "mean_end_to_end_ms": round(mean_e2e_ms, 2),
    }


def main() -> None:
    index = build_or_load_index()
    results = run_eval(index)
    summary = summarize(results)

    payload = {
        "summary": summary,
        "per_query": results,
        "k": TOP_K,
    }
    out_path = Path(__file__).parent / "rag_eval_results.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {out_path}")
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

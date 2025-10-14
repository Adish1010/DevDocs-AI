"""
Evaluation Harness for Knowledge-base Search Engine
---------------------------------------------------
Runs retrieval evaluation against a small gold dataset and reports metrics.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_processor import process_document
from src.rag_engine import create_rag_engine
import json
import numpy as np
from typing import List, Dict



# ---------------------------------------------------------------------------
# Gold Dataset (tiny example — expand with your own Q/A pairs)
# ---------------------------------------------------------------------------
GOLD_DATASET = [
    {
        "question": "How do I create a GET endpoint in FastAPI?",
        "gold_doc": "FastAPI Guide"
    },
    {
        "question": "What Python version does FastAPI require?",
        "gold_doc": "FastAPI Guide"
    },
    {
        "question": "Where can I find automatic API docs in FastAPI?",
        "gold_doc": "FastAPI Guide"
    }
]

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def precision_at_k(results: List[Dict], gold_doc: str, k: int = 5) -> float:
    top_k = results[:k]
    hits = sum(1 for r in top_k if r["metadata"].get("document_name") == gold_doc)
    return hits / k

def recall_at_k(results: List[Dict], gold_doc: str, k: int = 5) -> float:
    relevant = sum(1 for r in results if r["metadata"].get("document_name") == gold_doc)
    return min(1.0, relevant / 1.0)  # since we expect at least 1 relevant doc

def mean_reciprocal_rank(results: List[Dict], gold_doc: str) -> float:
    for idx, r in enumerate(results, start=1):
        if r["metadata"].get("document_name") == gold_doc:
            return 1.0 / idx
    return 0.0

# ---------------------------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------------------------
def run_evaluation():
    print("🧪 Running Retrieval Evaluation...")

    # Ensure test doc is ingested
    test_file = "data/raw_documents/test_fastapi.txt"
    if not os.path.exists(test_file):
        print("❌ Missing test document. Run test_rag_engine.py first.")
        return

    chunks = process_document(test_file, "FastAPI Guide")
    rag_engine = create_rag_engine()
    rag_engine.add_documents(chunks)

    # Collect metrics
    precisions, recalls, mrrs = [], [], []

    for item in GOLD_DATASET:
        q, gold_doc = item["question"], item["gold_doc"]
        results = rag_engine.search(q)

        p = precision_at_k(results, gold_doc, k=5)
        r = recall_at_k(results, gold_doc, k=5)
        m = mean_reciprocal_rank(results, gold_doc)

        precisions.append(p)
        recalls.append(r)
        mrrs.append(m)

        print(f"\nQ: {q}")
        print(f"Precision@5: {p:.2f}, Recall@5: {r:.2f}, MRR: {m:.2f}")

    print("\n📊 Aggregate Results")
    print(f"Avg Precision@5: {np.mean(precisions):.2f}")
    print(f"Avg Recall@5:    {np.mean(recalls):.2f}")
    print(f"Mean Reciprocal Rank: {np.mean(mrrs):.2f}")

if __name__ == "__main__":
    run_evaluation()

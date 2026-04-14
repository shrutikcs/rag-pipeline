"""
RAGAS Evaluation Script for the RAG Pipeline.

Usage (from backend/):
    uv run python eval/evaluate.py

Prerequisites:
    - The FastAPI server must be running:  uv run fastapi dev app/main.py
    - Required packages:  uv add ragas datasets
"""

import json
import sys
from pathlib import Path

import requests
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    faithfulness,
)

# ── configuration ────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000/query"
TEST_SET_PATH = Path(__file__).parent / "test_set.json"


def load_test_set() -> list[dict]:
    """Load Q&A pairs from the JSON test set."""
    with open(TEST_SET_PATH) as f:
        return json.load(f)


def query_api(question: str) -> dict:
    """Hit the /query endpoint and return answer + contexts."""
    resp = requests.post(API_URL, json={"query": question}, timeout=120)
    resp.raise_for_status()
    return resp.json()  # {"answer": "...", "contexts": ["...", ...]}


def build_ragas_dataset(test_set: list[dict]) -> Dataset:
    """
    Query the API for each test question and assemble the
    HuggingFace Dataset that RAGAS expects.

    Required columns:
        question       – the input question
        answer         – LLM-generated answer
        contexts       – list[str] of retrieved passages
        ground_truth   – reference answer for comparison
    """
    questions, answers, contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set, 1):
        q = item["question"]
        print(f"  [{i}/{len(test_set)}] Querying: {q[:80]}...")
        result = query_api(q)

        questions.append(q)
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        ground_truths.append(item["ground_truth"])

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def print_results(result_ds: Dataset) -> None:
    """Pretty-print per-question and aggregate results."""
    df = result_ds.to_pandas()

    metrics = ["faithfulness", "answer_relevancy", "context_precision"]

    # ── per-question table ───────────────────────────────────────
    print("\n" + "=" * 90)
    print("  PER-QUESTION RESULTS")
    print("=" * 90)

    header = f"{'#':<4} {'Question':<40} " + " ".join(
        f"{m:<20}" for m in metrics
    )
    print(header)
    print("-" * 90)

    for idx, row in df.iterrows():
        q_short = (row["question"][:37] + "...") if len(row["question"]) > 40 else row["question"]
        scores = " ".join(f"{row.get(m, 'N/A'):<20.4f}" for m in metrics)
        print(f"{idx + 1:<4} {q_short:<40} {scores}")

    # ── aggregate ────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  AGGREGATE SCORES")
    print("=" * 90)
    for m in metrics:
        if m in df.columns:
            print(f"  {m:<25} {df[m].mean():.4f}")
    print("=" * 90)


def main():
    print("\n🔍 Loading test set ...")
    test_set = load_test_set()
    print(f"   Found {len(test_set)} questions.\n")

    print("📡 Querying the RAG API ...")
    ragas_ds = build_ragas_dataset(test_set)

    print("\n📊 Running RAGAS evaluation ...")
    result = evaluate(
        ragas_ds,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    print_results(result.to_dataset())


if __name__ == "__main__":
    main()

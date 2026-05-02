from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from retrievers.bm25_retriever import BM25Retriever
from retrievers.embedding_retriever import EmbeddingRetriever
from retrievers.hybrid_retriever import HybridRetriever
from src.rag import answer_with_rag, get_allowed_name_list
from src.utils import (
    canonicalize_answer_set,
    classify_error,
    compute_prf,
    parse_answer,
    retrieval_metrics,
)


RESULTS_DIR = Path("results")


def evaluate_system(
    name: str,
    retriever,
    eval_path: str = "data/eval/eval_questions.csv",
    k: int = 5,
    candidate_k: int = 30,
    use_query_expansion: bool = True,
    use_metadata_reranking: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(eval_path)
    allowed_names = get_allowed_name_list()
    rows = []

    for _, row in df.iterrows():
        question = row["question"]
        expected = canonicalize_answer_set(parse_answer(row["expected_answers"]), allowed_names)

        answer, retrieved_names, details = answer_with_rag(
            question,
            retriever,
            k=k,
            candidate_k=candidate_k,
            use_query_expansion=use_query_expansion,
            use_metadata_reranking=use_metadata_reranking,
            return_details=True,
        )
        predicted = canonicalize_answer_set(parse_answer(answer), allowed_names)
        retrieved_names = [name for name in retrieved_names if name]

        precision, recall, f1 = compute_prf(predicted, expected)
        r_metrics = retrieval_metrics(retrieved_names, expected, k_values=(1, 3, 5, 10))

        false_positives = sorted(predicted - expected)
        false_negatives = sorted(expected - predicted)

        rows.append(
            {
                "system": name,
                "question": question,
                "expected": "; ".join(sorted(expected)) if expected else "NONE",
                "answer": "; ".join(sorted(predicted)) if predicted else "NONE",
                "retrieved": "; ".join(retrieved_names),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positives": "; ".join(false_positives),
                "false_negatives": "; ".join(false_negatives),
                "unsupported_hallucination": int(bool(false_positives)),
                "wrong_answer": int(predicted != expected),
                "error_type": classify_error(predicted, expected, retrieved_names),
                "retrieval_question": details.get("retrieval_question", question),
                "retrieval_query": details.get("retrieval_query", question),
                "language": details.get("language", "en"),
                **r_metrics,
            }
        )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "precision",
        "recall",
        "f1",
        "unsupported_hallucination",
        "wrong_answer",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "hit_at_10",
        "retrieval_recall_at_5",
        "retrieval_recall_at_10",
        "retrieval_precision_at_5",
        "retrieval_precision_at_10",
    ]
    available = [col for col in metric_cols if col in df.columns]
    return df.groupby("system")[available].mean().reset_index().sort_values("f1", ascending=False)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    systems = {
        "bm25_rag": BM25Retriever(chunk_size=600),
        "embedding_rag": EmbeddingRetriever(chunk_size=600),
        "hybrid_rag": HybridRetriever(chunk_size=600),
    }

    all_results = []
    for name, retriever in systems.items():
        print(f"Evaluating {name}...")
        result = evaluate_system(name, retriever)
        all_results.append(result)

    final = pd.concat(all_results, ignore_index=True)
    final.to_csv(RESULTS_DIR / "generation_results.csv", index=False)

    summary = summarize(final)
    summary.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

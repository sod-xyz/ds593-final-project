from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from retrievers.hybrid_retriever import HybridRetriever
from src.rag import answer_with_rag, get_allowed_name_list
from src.utils import canonicalize_answer_set, classify_error, compute_prf, parse_answer, retrieval_metrics

RESULTS_DIR = Path("results")


def evaluate_file(eval_path: str, language_label: str) -> pd.DataFrame:
    df = pd.read_csv(eval_path)
    retriever = HybridRetriever(chunk_size=600)
    allowed_names = get_allowed_name_list()
    rows = []

    for _, row in df.iterrows():
        question = row["question"]
        expected = canonicalize_answer_set(parse_answer(row["expected_answers"]), allowed_names)

        answer, retrieved_names, details = answer_with_rag(
            question=question,
            retriever=retriever,
            k=10,
            candidate_k=30,
            use_query_expansion=True,
            use_metadata_reranking=True,
            return_details=True,
        )
        predicted = canonicalize_answer_set(parse_answer(answer), allowed_names)
        precision, recall, f1 = compute_prf(predicted, expected)
        r_metrics = retrieval_metrics(retrieved_names, expected, k_values=(1, 3, 5, 10))

        rows.append(
            {
                "language": language_label,
                "question_type": row.get("question_type", "unknown"),
                "question": question,
                "retrieval_question": details.get("retrieval_question", question),
                "retrieval_query": details.get("retrieval_query", question),
                "expected": "; ".join(sorted(expected)) if expected else "NONE",
                "answer": "; ".join(sorted(predicted)) if predicted else "NONE",
                "retrieved": "; ".join(retrieved_names),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positives": "; ".join(sorted(predicted - expected)),
                "false_negatives": "; ".join(sorted(expected - predicted)),
                "unsupported_hallucination": int(bool(predicted - expected)),
                "wrong_answer": int(predicted != expected),
                "error_type": classify_error(predicted, expected, retrieved_names),
                **r_metrics,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    english = evaluate_file("data/eval/eval_questions.csv", "English")
    mongolian = evaluate_file("data/eval/eval_questions_mn.csv", "Mongolian")
    final = pd.concat([english, mongolian], ignore_index=True)
    final.to_csv(RESULTS_DIR / "multilingual_results.csv", index=False)

    metric_cols = [
        "precision", "recall", "f1", "unsupported_hallucination", "wrong_answer",
        "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10",
        "retrieval_recall_at_5", "retrieval_recall_at_10",
        "retrieval_precision_at_5", "retrieval_precision_at_10",
    ]
    summary = final.groupby("language")[metric_cols].mean().reset_index()
    summary.to_csv(RESULTS_DIR / "multilingual_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

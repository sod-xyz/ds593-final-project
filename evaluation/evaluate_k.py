import os
import pandas as pd

from retrievers.hybrid_retriever import HybridRetriever
from src.rag import answer_with_rag
from src.utils import parse_answer, compute_prf


def retrieval_metrics(retrieved_names, expected, k_values=(1, 3, 5, 10, 15)):
    """
    Computes retrieval metrics separately from generation metrics.

    hit_at_k:
        1 if at least one correct scholarship appears in top-k retrieved docs.

    retrieval_recall_at_k:
        share of expected scholarships that appear in top-k retrieved docs.

    retrieval_precision_at_k:
        share of top-k retrieved docs that are expected scholarships.
    """
    expected = set(expected)
    metrics = {}

    for k in k_values:
        top_k = set(retrieved_names[:k])

        if len(expected) == 0:
            hit = 1
            recall = 1.0
            precision = 1.0
        else:
            correct = top_k & expected
            hit = int(len(correct) > 0)
            recall = len(correct) / len(expected)
            precision = len(correct) / k if k > 0 else 0.0

        metrics[f"hit_at_{k}"] = hit
        metrics[f"retrieval_recall_at_{k}"] = recall
        metrics[f"retrieval_precision_at_{k}"] = precision

    return metrics


def evaluate_for_k(k, eval_path="data/eval/eval_questions.csv"):
    df = pd.read_csv(eval_path)

    retriever = HybridRetriever(chunk_size=600)

    rows = []

    for _, row in df.iterrows():
        question = row["question"]
        expected = parse_answer(row["expected_answers"])

        answer, retrieved_names = answer_with_rag(
            question,
            retriever,
            k=k,
            candidate_k=30,
            use_query_expansion=True,
            use_metadata_reranking=True
        )

        predicted = parse_answer(answer)

        precision, recall, f1 = compute_prf(predicted, expected)

        r_metrics = retrieval_metrics(
            retrieved_names,
            expected,
            k_values=(1, 3, 5, 10, 15)
        )

        rows.append({
            "system": f"metadata_aware_hybrid_rag_k{k}",
            "k": k,
            "question": question,
            "expected": "; ".join(sorted(expected)),
            "answer": answer,
            "retrieved": "; ".join(retrieved_names),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            **r_metrics
        })

    return pd.DataFrame(rows)


def main():
    os.makedirs("results", exist_ok=True)

    all_results = []

    for k in [5, 10, 15]:
        print(f"Evaluating metadata-aware hybrid RAG with k={k}...")
        result = evaluate_for_k(k)
        all_results.append(result)

    final = pd.concat(all_results, ignore_index=True)

    final.to_csv("results/k_experiment_results.csv", index=False)

    summary = final.groupby(["system", "k"])[[
        "precision",
        "recall",
        "f1",
        "hit_at_1",
        "hit_at_3",
        "hit_at_5",
        "hit_at_10",
        "hit_at_15",
        "retrieval_recall_at_5",
        "retrieval_recall_at_10",
        "retrieval_recall_at_15",
        "retrieval_precision_at_5",
        "retrieval_precision_at_10",
        "retrieval_precision_at_15",
    ]].mean().reset_index()

    summary.to_csv("results/k_experiment_summary.csv", index=False)

    print(summary)


if __name__ == "__main__":
    main()
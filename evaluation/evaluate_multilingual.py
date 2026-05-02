import os
import pandas as pd

from retrievers.hybrid_retriever import HybridRetriever
from src.rag import answer_with_rag
from src.utils import parse_answer, compute_prf, retrieval_metrics


def evaluate_file(eval_path, language_label):
    df = pd.read_csv(eval_path)
    retriever = HybridRetriever(chunk_size=600)

    rows = []

    for _, row in df.iterrows():
        question = row["question"]
        expected = parse_answer(row["expected_answers"])

        answer, retrieved_names, details = answer_with_rag(
            question=question,
            retriever=retriever,
            k=10,
            candidate_k=30,
            use_query_expansion=True,
            use_metadata_reranking=True,
            return_details=True,
        )

        predicted = parse_answer(answer)
        precision, recall, f1 = compute_prf(predicted, expected)
        r_metrics = retrieval_metrics(retrieved_names, expected, k_values=(1, 3, 5, 10))

        rows.append({
            "language": language_label,
            "question_type": row.get("question_type", "unknown"),
            "question": question,
            "retrieval_question": details.get("retrieval_question", question),
            "expanded_query": details.get("expanded_query", ""),
            "expected": "; ".join(sorted(expected)),
            "answer": answer,
            "retrieved": "; ".join(retrieved_names),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            **r_metrics,
        })

    return pd.DataFrame(rows)


def main():
    os.makedirs("results", exist_ok=True)

    english = evaluate_file("data/eval/eval_questions.csv", "English")
    mongolian = evaluate_file("data/eval/eval_questions_mn.csv", "Mongolian")

    final = pd.concat([english, mongolian], ignore_index=True)

    final.to_csv("results/multilingual_results.csv", index=False)

    summary = final.groupby("language")[[
        "precision", "recall", "f1",
        "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10",
        "retrieval_recall_at_5", "retrieval_recall_at_10",
        "retrieval_precision_at_5", "retrieval_precision_at_10"
    ]].mean().reset_index()

    summary.to_csv("results/multilingual_summary.csv", index=False)

    print(summary)


if __name__ == "__main__":
    main()
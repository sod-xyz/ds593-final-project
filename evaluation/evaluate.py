import os
import pandas as pd

from retrievers.bm25_retriever import BM25Retriever
from retrievers.embedding_retriever import EmbeddingRetriever
from retrievers.hybrid_retriever import HybridRetriever
from src.rag import answer_with_rag
from src.utils import parse_answer, compute_prf


def evaluate_system(name, retriever, eval_path="data/eval/eval_questions.csv"):
    df = pd.read_csv(eval_path)

    rows = []

    for _, row in df.iterrows():
        question = row["question"]
        expected = parse_answer(row["expected_answers"])

        answer, retrieved_names = answer_with_rag(question, retriever, k=5)
        predicted = parse_answer(answer)

        precision, recall, f1 = compute_prf(predicted, expected)

        # These are actually hit@k, not true recall@k
        hit_at_1 = int(len(set(retrieved_names[:1]) & expected) > 0)
        hit_at_3 = int(len(set(retrieved_names[:3]) & expected) > 0)
        hit_at_5 = int(len(set(retrieved_names[:5]) & expected) > 0)

        # True retrieval recall@k
        if len(expected) > 0:
            retrieval_recall_at_5 = len(set(retrieved_names[:5]) & expected) / len(expected)
        else:
            retrieval_recall_at_5 = 1.0

        rows.append({
            "system": name,
            "question": question,
            "expected": "; ".join(expected),
            "answer": answer,
            "retrieved": "; ".join(retrieved_names),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "hit_at_1": hit_at_1,
            "hit_at_3": hit_at_3,
            "hit_at_5": hit_at_5,
            "retrieval_recall_at_5": retrieval_recall_at_5
        })

    return pd.DataFrame(rows)


def main():
    os.makedirs("results", exist_ok=True)

    systems = {
        "bm25": BM25Retriever(chunk_size=600),
        "embedding": EmbeddingRetriever(chunk_size=600),
        "hybrid": HybridRetriever(chunk_size=600)
    }

    all_results = []

    for name, retriever in systems.items():
        print(f"Evaluating {name}...")
        result = evaluate_system(name, retriever)
        all_results.append(result)

    final = pd.concat(all_results, ignore_index=True)
    final.to_csv("results/generation_results.csv", index=False)

    summary = final.groupby("system")[[
        "precision", "recall", "f1",
        "hit_at_1", "hit_at_3", "hit_at_5",
        "retrieval_recall_at_5"
    ]].mean().reset_index()

    summary.to_csv("results/summary_metrics.csv", index=False)

    print(summary)


if __name__ == "__main__":
    main()
import pandas as pd
from retrievers.hybrid_retriever import HybridRetriever
from evaluation.evaluate import evaluate_system


def main():
    all_results = []

    for chunk_size in [300, 600, 900]:
        print(f"Evaluating chunk size {chunk_size}")
        retriever = HybridRetriever(chunk_size=chunk_size)
        result = evaluate_system(f"hybrid_chunk_{chunk_size}", retriever)
        all_results.append(result)

    final = pd.concat(all_results)
    final.to_csv("results/chunk_size_results.csv", index=False)

    summary = final.groupby("system")[[
        "precision", "recall", "f1",
        "hit_at_1", "hit_at_3", "hit_at_5",
        "retrieval_recall_at_5"
    ]].mean().reset_index()

    summary.to_csv("results/chunk_size_summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
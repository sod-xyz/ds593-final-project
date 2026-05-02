import os
import pandas as pd

from src.llm import generate_answer
from src.rag import get_allowed_names
from src.utils import parse_answer, compute_prf


def main():
    os.makedirs("results", exist_ok=True)
    df = pd.read_csv("data/eval/eval_questions.csv")
    allowed_names = get_allowed_names()

    rows = []

    for _, row in df.iterrows():
        question = row["question"]
        expected = parse_answer(row["expected_answers"])

        answer = generate_answer(
            question=question,
            context="",
            allowed_names=allowed_names
        )

        predicted = parse_answer(answer)
        precision, recall, f1 = compute_prf(predicted, expected)

        rows.append({
            "system": "baseline_no_retrieval",
            "question": question,
            "expected": "; ".join(expected),
            "answer": answer,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    result = pd.DataFrame(rows)
    result.to_csv("results/baseline_results.csv", index=False)

    print(result[["precision", "recall", "f1"]].mean())


if __name__ == "__main__":
    main()
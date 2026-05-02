from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.llm import generate_answer
from src.rag import get_allowed_name_list, get_allowed_names
from src.utils import canonicalize_answer_set, classify_error, compute_prf, parse_answer

RESULTS_DIR = Path("results")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.read_csv("data/eval/eval_questions.csv")
    allowed_names_text = get_allowed_names()
    allowed_names = get_allowed_name_list()

    rows = []
    for _, row in df.iterrows():
        question = row["question"]
        expected = canonicalize_answer_set(parse_answer(row["expected_answers"]), allowed_names)

        # Deliberately no context: this tests whether retrieval adds value over
        # an LLM that only sees the allowed label set.
        answer = generate_answer(question=question, context="", allowed_names=allowed_names_text)
        predicted = canonicalize_answer_set(parse_answer(answer), allowed_names)

        precision, recall, f1 = compute_prf(predicted, expected)
        false_positives = sorted(predicted - expected)
        false_negatives = sorted(expected - predicted)

        rows.append(
            {
                "system": "baseline_no_retrieval_label_selection",
                "question": question,
                "expected": "; ".join(sorted(expected)) if expected else "NONE",
                "answer": "; ".join(sorted(predicted)) if predicted else "NONE",
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positives": "; ".join(false_positives),
                "false_negatives": "; ".join(false_negatives),
                "unsupported_hallucination": int(bool(false_positives)),
                "wrong_answer": int(predicted != expected),
                "error_type": classify_error(predicted, expected, []),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(RESULTS_DIR / "baseline_results.csv", index=False)
    summary = result.groupby("system")[["precision", "recall", "f1", "unsupported_hallucination", "wrong_answer"]].mean().reset_index()
    summary.to_csv(RESULTS_DIR / "baseline_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

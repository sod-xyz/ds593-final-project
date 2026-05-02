from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

def normalize_name(name: str) -> str:
    """Normalize scholarship names for fair exact-match evaluation."""
    if name is None:
        return ""
    text = unicodedata.normalize("NFKC", str(name))
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = text.replace("Türkiye", "Turkey").replace("Türkiye", "Turkey")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return text


def parse_answer(answer: str | None) -> set[str]:
    """Parse semicolon-separated answer strings into a set of labels."""
    if answer is None:
        return set()
    answer = str(answer).strip()
    if not answer or answer.upper() == "NONE":
        return set()
    return {
        item.strip()
        for item in answer.split(";")
        if item.strip() and item.strip().upper() != "NONE"
    }


def canonicalize_answer_set(names: Iterable[str], allowed_names: Sequence[str]) -> set[str]:
    """Map variant spellings/punctuation to canonical allowed names.

    Any item that cannot be mapped is retained under its original name so that
    unsupported hallucinations remain visible in our error analysis.
    """
    canonical = {normalize_name(name): name for name in allowed_names}
    output: set[str] = set()
    for name in names:
        output.add(canonical.get(normalize_name(name), name))
    return output


def compute_prf(predicted: Iterable[str], expected: Iterable[str]) -> tuple[float, float, float]:
    predicted = set(predicted)
    expected = set(expected)

    if not predicted and not expected:
        return 1.0, 1.0, 1.0
    if not predicted:
        return 0.0, 0.0, 0.0

    true_pos = len(predicted & expected)
    precision = true_pos / len(predicted)
    recall = true_pos / len(expected) if expected else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def retrieval_metrics(retrieved_names: Sequence[str], expected: Iterable[str], k_values=(1, 3, 5, 10)) -> dict[str, float]:
    """Compute retrieval-only metrics.

    In the case of NONE/no-answer questions, retrieving no documents would be ideal. 
    Since the current retrievers always return documents, these rows are counted as
    retrieval false positives instead of automatically receiving perfect scores.
    """
    expected = set(expected)
    retrieved_names = list(retrieved_names)
    metrics: dict[str, float] = {}

    for k in k_values:
        top_k = retrieved_names[:k]
        top_k_set = set(top_k)

        if not expected:
            hit = int(len(top_k) == 0)
            recall = 1.0 if len(top_k) == 0 else 0.0
            precision = 1.0 if len(top_k) == 0 else 0.0
        else:
            correct = top_k_set & expected
            hit = int(bool(correct))
            recall = len(correct) / len(expected)
            precision = len(correct) / len(top_k) if top_k else 0.0

        metrics[f"hit_at_{k}"] = hit
        metrics[f"retrieval_recall_at_{k}"] = recall
        metrics[f"retrieval_precision_at_{k}"] = precision

    return metrics


def classify_error(predicted: set[str], expected: set[str], retrieved: Sequence[str]) -> str:
    """Assign a coarse, human-readable error category for reporting."""
    if predicted == expected:
        return "correct"
    if not expected and predicted:
        return "false_positive_no_answer"
    if expected and not predicted:
        return "empty_answer"

    false_pos = predicted - expected
    false_neg = expected - predicted
    retrieved_set = set(retrieved)

    if false_neg and not (false_neg & retrieved_set):
        return "retrieval_miss"
    if false_pos and false_neg:
        return "mixed_over_and_under_selection"
    if false_pos:
        return "over_selection"
    if false_neg:
        return "under_selection_after_retrieval"
    return "other"

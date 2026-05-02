import re
import unicodedata


def normalize_name(name: str) -> str:
    """Normalize scholarship names for fair exact-match evaluation."""
    if name is None:
        return ""
    text = unicodedata.normalize("NFKC", str(name))
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def parse_answer(answer):
    if answer is None:
        return set()

    answer = str(answer).strip()

    if answer == "" or answer.upper() == "NONE":
        return set()

    return set(
        x.strip()
        for x in answer.split(";")
        if x.strip() and x.strip().upper() != "NONE"
    )


def canonicalize_answer_set(names, allowed_names=None):
    """
    Convert a list/set of names to canonical names using the allowed-name list.
    This prevents punctuation differences from lowering scores unfairly.
    """
    names = parse_answer("; ".join(names)) if not isinstance(names, set) else names
    if allowed_names is None:
        return set(names)

    canonical = {normalize_name(n): n for n in allowed_names}
    output = set()
    for n in names:
        output.add(canonical.get(normalize_name(n), n))
    return output


def compute_prf(predicted, expected):
    predicted = set(predicted)
    expected = set(expected)

    if len(predicted) == 0 and len(expected) == 0:
        return 1.0, 1.0, 1.0

    if len(predicted) == 0:
        return 0.0, 0.0, 0.0

    true_pos = len(predicted & expected)

    precision = true_pos / len(predicted)
    recall = true_pos / len(expected) if expected else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def retrieval_metrics(retrieved_names, expected, k_values=(1, 3, 5, 10)):
    """
    Compute retrieval metrics separately from generation metrics.
    """
    expected = set(expected)
    retrieved_names = list(retrieved_names)

    metrics = {}

    for k in k_values:
        top_k = retrieved_names[:k]
        top_k_set = set(top_k)

        if len(expected) == 0:
            # For no-answer questions, retrieval is only precise if nothing is retrieved.
            hit = int(len(top_k) == 0)
            recall = 1.0 if len(top_k) == 0 else 0.0
            precision = 1.0 if len(top_k) == 0 else 0.0
        else:
            correct = top_k_set & expected
            hit = int(len(correct) > 0)
            recall = len(correct) / len(expected)
            precision = len(correct) / len(top_k) if len(top_k) > 0 else 0.0

        metrics[f"hit_at_{k}"] = hit
        metrics[f"retrieval_recall_at_{k}"] = recall
        metrics[f"retrieval_precision_at_{k}"] = precision

    return metrics

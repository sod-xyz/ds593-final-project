# Critical Review and Revision Recommendations

This review evaluates the repository against the DS593 final project rubric. The tone is intentionally strict because this is a graduate final project.

## Overall Assessment

The project has a good applied motivation and is appropriate for a solo RAG project: scholarship search for Mongolian students is specific, socially meaningful, and naturally suited to retrieval-augmented generation. The strongest technical idea is the move beyond plain RAG toward metadata-aware query expansion and reranking.

However, the project is not yet presentation-ready unless the final evaluation results are rerun, committed, and interpreted carefully. The uploaded repository contains evaluation scripts, but it does not include the actual result CSVs or completed result tables. The README says the result table should be filled from results, which is not acceptable for final submission.

## Rubric-Based Grade Estimate Before Fixes

| Category | Points | Strict estimate | Reason |
|---|---:|---:|---|
| Scope & Ambition | 10 | 8 | Strong solo scope: RAG, multiple retrievers, multilingual evaluation. Not production-scale, but appropriate. |
| Design Decisions | 5 | 4 | BM25 vs dense vs hybrid is good. Metadata reranking is good. Needs clearer justification with actual results. |
| Technical Execution | 5 | 3.5 | Code is understandable and runs in principle, but evaluation consistency and result tracking need improvement. |
| Evaluation & Analysis | 10 | 5.5 | Good metric design, but missing committed results and concrete error examples. Some metric handling was too generous for no-answer retrieval. |
| Iteration & Reflection | 5 | 3.5 | README discusses iteration, but needs concrete before/after evidence. |
| Ethics & Limitations | 5 | 4 | Good limitations section; should be tied more directly to scholarship advising harms. |
| Code & Repository | 5 | 3.5 | Organized repo, but no result artifacts, no lock file, no CLI, no tests. |
| Write-up & Presentation | 5 | 3.5 | Strong structure, but placeholders in results section are a major weakness. |
| **Total** | **50** | **35.5/50** | Could become low-to-mid 40s if final metrics and error analysis are completed. |

## Highest-Priority Fixes

1. **Commit final result summaries.** The repo currently ignores `results/`, so the grader cannot verify reported metrics. Add final tables to `reports/results/` or a markdown summary.
2. **Replace all result placeholders.** The README cannot contain `fill from results` in a final submission.
3. **Add concrete error examples.** Include at least 5 rows where the system failed and explain whether each was a retrieval miss, over-selection, under-selection, label ambiguity, or multilingual error.
4. **Clarify the baseline.** The baseline is not truly no-information because it receives the allowed label list. Call it a no-retrieval label-selection baseline.
5. **Report retrieval and generation separately.** Do not collapse the story into one F1 score.
6. **Be honest if RAG does not beat the baseline.** If the baseline is competitive, explain that the dataset is small and label-driven.

## Conceptual Critique

### Strengths

- The problem is specific and meaningful.
- RAG is appropriate because scholarship advice requires grounding in source records.
- Multiple retrieval strategies are compared.
- The project attempts multilingual access for Mongolian questions.
- Metadata-aware reranking shows domain-specific design beyond a generic RAG demo.

### Weaknesses

- The dataset is small and manually curated, so evaluation may reward memorization or label guessing.
- The gold labels are sometimes subjective. For example, conditional eligibility programs may be considered correct in some questions and not in others.
- The final answer returns only names, not evidence or citations. This is a major limitation for scholarship advising.
- The system does not verify deadlines or current eligibility.
- The multilingual evaluation has only 10 questions, which is too small for strong claims.

## Technical Critique

### Code design

The repo is clean enough for a course project. The `src/`, `retrievers/`, and `evaluation/` separation is good. The main weaknesses are reproducibility and evaluation rigor.

### Evaluation issues fixed in the revised files

- Expected and predicted names are now canonicalized before scoring.
- Unsupported false positives and false negatives are recorded.
- Error types are assigned per question.
- The no-answer retrieval case is no longer automatically counted as perfect if the retriever still returns documents.
- The LLM client is created inside `generate_answer` instead of at import time, which makes the code safer to inspect without an API key.
- Multilingual evaluation now records the actual retrieval query consistently.

### Remaining technical recommendations

1. Add a deterministic metadata-filter baseline.
2. Add a simple command-line interface, for example: `python -m src.cli "question"`.
3. Add unit tests for `parse_answer`, `normalize_name`, and `retrieval_metrics`.
4. Add `reports/results/` to track final summary outputs.
5. Pin dependency versions more tightly if exact reproducibility is required.
6. Add source timestamps to the dataset.

## Recommended Final Report Narrative

The strongest narrative is not “my model gets a high F1.” The strongest narrative is:

> This project shows that a small domain-specific RAG system can support scholarship discovery for Mongolian students, but the evaluation also reveals why scholarship advising is high-risk. Retrieval helps with grounding, but constrained questions remain difficult because eligibility is conditional, labels are ambiguous, and scholarship rules change over time. The final system improves over a generic baseline in interpretability and evidence use, but it is not ready for real deployment without citations, freshness checks, and human verification.

## What to Say If Results Are Not Very High

Do not hide weak results. A graduate-level project is allowed to have imperfect performance if the analysis is honest.

Use language like:

> The evaluation results show that this is a useful prototype rather than a deployable advising system. The system performs best on country- and funding-specific questions where metadata is explicit. It performs worse on conditional eligibility and broad list questions, where the expected answer set is partly subjective. These failures are informative because they show that scholarship search requires not only retrieval but also source freshness, eligibility reasoning, and uncertainty communication.

## Final Submission Checklist

- [ ] Rerun all evaluation scripts.
- [ ] Add final numeric table to README.
- [ ] Add `reports/results_summary.md` with exact metrics.
- [ ] Add 5-10 concrete failure examples.
- [ ] State that the baseline is a no-retrieval label-selection baseline.
- [ ] Explain why some labels are ambiguous.
- [ ] Include ethics and deployment risks in the presentation.
- [ ] Push revised files to GitHub.

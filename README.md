# Finding the Right Scholarships: A RAG System for Mongolian Students

Graduate final project for **DS593: Theory and Applications of AI**.

This project builds and evaluates a retrieval-augmented generation (RAG) assistant that helps Mongolian students identify international scholarship opportunities. The system takes a natural-language question, retrieves scholarship records from a curated corpus, and uses an LLM to return only scholarship names supported by the retrieved context.

> **Important scope note:** This is a course prototype, not an official scholarship advising tool. Scholarship eligibility, deadlines, and funding rules change frequently. Any real applicant should verify results using the official source links before applying.

---

## 1. Motivation

Many international scholarship databases are written in English, spread across many websites, or difficult to search for country-specific cases such as Mongolian eligibility. A student may ask practical questions like:

- “Which fully funded master’s scholarships are available to Mongolian students in the UK?”
- “Which scholarships require work experience?”
- “Монгол оюутнууд Японд ямар тэтгэлэгт хамрагдах боломжтой вэ?”

A simple keyword search often fails because eligibility is described indirectly. A general LLM may answer fluently but hallucinate or recommend programs that are not available to Mongolian applicants. This project tests whether a small domain-specific RAG pipeline can improve answer grounding and reduce unsupported recommendations.

---

## 2. Research Question

**Can a retrieval-augmented LLM system accurately recommend scholarship names for Mongolian students when questions include constraints such as country, degree level, funding type, language requirement, work experience, and Mongolia eligibility?**

Sub-questions:

1. Does retrieval improve answer quality compared with a no-retrieval LLM baseline?
2. Which retrieval strategy works best: BM25, dense embedding retrieval, or hybrid retrieval?
3. Does metadata-aware query expansion and reranking improve constrained scholarship search?
4. How well does the system handle Mongolian-language questions?

---

## 3. Dataset

The corpus contains curated scholarship records in `data/processed/documents.json` and raw text files in `data/raw/`.

Each processed record includes:

| Field | Description |
|---|---|
| `scholarship` | Canonical scholarship name |
| `text` | Short program description |
| `country` | Main destination country or region |
| `degree_level` | Eligible degree level(s) |
| `language_requirement` | Whether English or another language requirement is noted |
| `funding_type` | Fully funded, partial, conditional, etc. |
| `funding_details` | Description of tuition, stipend, travel, or partial support |
| `mongolia_eligible` | Eligibility judgment for Mongolian students |
| `mongolia_eligibility_note` | Caveat explaining the eligibility judgment |
| `source` | Source URL or source reference |

### Evaluation sets

The project includes two manually written evaluation files:

- `data/eval/eval_questions.csv`: English evaluation questions
- `data/eval/eval_questions_mn.csv`: Mongolian evaluation questions

Each row contains a question and a semicolon-separated list of expected scholarship names.

### Dataset limitation

The dataset is small and manually curated. This is appropriate for a focused solo project, but it means the results should be interpreted as a controlled prototype evaluation, not proof that the system is ready for open-ended deployment.

---

## 4. System Architecture

The final pipeline is implemented in `src/rag.py`.

```text
User question
   |
   v
Language detection / optional Mongolian-to-English translation
   |
   v
Query understanding: country, degree, funding, language, work experience
   |
   v
Query expansion using synonyms and structured constraints
   |
   v
Candidate retrieval: BM25, dense embeddings, or hybrid RRF
   |
   v
Metadata-aware reranking
   |
   v
Context construction from top-k records
   |
   v
LLM answer constrained to allowed scholarship names
   |
   v
Post-processing and evaluation
```

---

## 5. Methods Compared

### 5.1 No-retrieval LLM baseline

The baseline asks the LLM to answer without retrieved context but gives it the allowed list of scholarship names. This tests how much the model can infer from prior knowledge and the allowed-name list alone.

**Critical interpretation:** this is not a pure “random” or “no-information” baseline because the model sees the set of valid output labels. It is useful, but it should be described honestly as an **LLM label-selection baseline without retrieved evidence**.

### 5.2 BM25 retriever

BM25 provides lexical keyword matching. It is strong when the user question contains exact terms that appear in the documents, such as “Japan,” “IELTS,” or “work experience.”

### 5.3 Dense embedding retriever

The dense retriever uses `sentence-transformers/all-MiniLM-L6-v2` and cosine similarity. It can retrieve semantically related content even without exact keyword overlap, but it may be weaker for strict symbolic constraints such as country or degree level.

### 5.4 Hybrid retriever

The hybrid retriever combines BM25 and dense retrieval using reciprocal rank fusion (RRF). This is intended to preserve exact-match behavior while adding semantic robustness.

### 5.5 Metadata-aware query expansion and reranking

The final system extracts structured constraints and expands the query with synonyms. It then reranks retrieved candidates using metadata fields. For example, a query mentioning “UK,” “master’s,” and “fully funded” should reward documents whose country, degree level, and funding fields match those constraints.

---

## 6. Evaluation Design

The project evaluates both retrieval and final generated answers.

### 6.1 Generation metrics

The final answer is parsed as a set of scholarship names. The project computes:

- **Precision:** share of predicted scholarships that are expected
- **Recall:** share of expected scholarships recovered
- **F1:** harmonic mean of precision and recall

### 6.2 Retrieval metrics

The project separately measures:

- `hit_at_k`: whether at least one expected scholarship appears in the top-k retrieved records
- `retrieval_recall_at_k`: share of expected scholarships found in top-k retrieved records
- `retrieval_precision_at_k`: share of top-k retrieved records that are expected answers

### 6.3 Why separate retrieval and generation?

Separating retrieval from generation is important because an answer can fail for two different reasons:

1. The correct scholarship was never retrieved.
2. The correct scholarship was retrieved, but the LLM failed to select it or added unsupported names.

This distinction is central to diagnosing RAG systems.

---

## 7. Running the Project

### 7.1 Clone and create environment

```bash
git clone <your-repo-url>
cd final-project-sod-xyz
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 7.2 Install dependencies

```bash
pip install -r requirements.txt
```

### 7.3 Configure OpenAI API key

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_api_key_here
```

The API key is required for LLM generation and Mongolian-to-English translation. Retrieval-only code can be inspected without it, but full evaluation requires the key.

### 7.4 Run evaluations

```bash
# No-retrieval LLM baseline
python -m evaluation.evaluate_baseline

# BM25, embedding, and hybrid RAG comparison
python -m evaluation.evaluate

# Compare different k values for the metadata-aware hybrid system
python -m evaluation.evaluate_k

# Evaluate English vs Mongolian questions
python -m evaluation.evaluate_multilingual

# Compare chunk sizes
python -m evaluation.evaluate_chunks
```

Outputs are written to `results/`.

---

## 8. Expected Output Files

| File | Description |
|---|---|
| `results/baseline_results.csv` | Per-question baseline predictions |
| `results/generation_results.csv` | Per-question predictions for BM25, embedding, and hybrid systems |
| `results/summary_metrics.csv` | Aggregated metrics by retrieval system |
| `results/k_experiment_results.csv` | Per-question results for different top-k values |
| `results/k_experiment_summary.csv` | Summary of k experiments |
| `results/multilingual_results.csv` | English and Mongolian evaluation results |
| `results/multilingual_summary.csv` | Aggregated multilingual metrics |
| `results/chunk_size_results.csv` | Per-question chunk-size experiment |
| `results/chunk_size_summary.csv` | Summary of chunk-size experiment |

---

## 9. Interpreting Results

The strongest final report should not simply report one F1 score. It should answer:

1. Did retrieval improve over the no-retrieval baseline?
2. Did hybrid retrieval outperform BM25 and dense retrieval?
3. Did increasing `k` improve recall but reduce precision?
4. Were failures mostly retrieval failures or generation failures?
5. Did Mongolian questions perform worse than English questions?
6. Did metadata reranking help constrained queries but hurt broad questions?

A good results table should look like this:

| System | Precision | Recall | F1 | Hit@5 | Retrieval Recall@5 |
|---|---:|---:|---:|---:|---:|
| No-retrieval LLM baseline | fill from results | fill from results | fill from results | N/A | N/A |
| BM25 RAG | fill from results | fill from results | fill from results | fill | fill |
| Dense embedding RAG | fill from results | fill from results | fill from results | fill | fill |
| Hybrid RAG | fill from results | fill from results | fill from results | fill | fill |
| Metadata-aware hybrid RAG | fill from results | fill from results | fill from results | fill | fill |

---

## 10. Error Analysis Framework

After running evaluation, inspect low-F1 rows in the result CSV files. Each error should be assigned to one or more categories:

| Error type | Description | Example |
|---|---|---|
| Retrieval miss | Correct scholarship not in retrieved top-k | A country-specific program is absent from retrieved records |
| Over-retrieval | System returns too many loosely relevant scholarships | Broad “fully funded” query includes partial awards |
| Metadata conflict | Metadata field disagrees with raw text or expected answer | Program marked conditional but expected as available |
| Name mismatch | Same scholarship appears with different punctuation | `Asian Development Bank-Japan` vs `Asian Development Bank–Japan` |
| Multilingual failure | Mongolian query not translated or constraint not preserved | “Их Британи” not mapped to UK |
| Label ambiguity | Expected answer set is debatable | Rhodes eligibility for Mongolia may be conditional |
| LLM selection error | Correct context retrieved but LLM omits or adds a name | LLM returns a leadership scholarship not satisfying funding constraint |

This error analysis is essential for graduate-level evaluation.

---

## 11. Main Limitations

1. **Small corpus.** The system only knows the included scholarship records.
2. **Manually curated labels.** Evaluation answers may reflect the author’s interpretation of eligibility.
3. **Eligibility changes over time.** The system does not automatically update deadlines, country rules, or application cycles.
4. **Ambiguous scholarship availability.** Some scholarships depend on partner universities, citizenship, field, or nomination rules.
5. **LLM dependence.** Even with temperature 0, hosted LLM outputs can change across model versions.
6. **Mongolian-language coverage is limited.** Translation is helpful but may lose constraints or cultural phrasing.
7. **No user-facing citation mode in the final answer.** The current evaluation asks for scholarship names only. A deployed advising tool should return evidence snippets and source links.

---

## 12. Ethics and Safety

Scholarship advice can affect high-stakes education decisions. The main ethical risks are:

- recommending scholarships for which a student is not eligible;
- omitting relevant opportunities;
- presenting uncertain eligibility as certain;
- disadvantaging students with weaker English search skills;
- relying on stale information.

To reduce harm, the system constrains output to known scholarship names and uses retrieved context rather than unsupported generation. However, this is not enough for deployment. A production version should include source citations, last-updated dates, confidence levels, and explicit warnings for conditional eligibility.

---

## 13. Iteration and Reflection

The project evolved from a simple RAG system into a metadata-aware scholarship retrieval pipeline.

What improved:

- Added BM25, embedding, and hybrid retrieval comparisons.
- Added a no-retrieval baseline.
- Added structured metadata fields.
- Added query expansion and metadata reranking.
- Added separate retrieval and generation metrics.
- Added Mongolian-language evaluation.

What did not fully work:

- Multilingual retrieval required explicit translation integration; simply adding a translation helper was not enough.
- Increasing top-k can improve recall but may reduce precision because the LLM sees more distractors.
- Some evaluation labels are ambiguous because scholarship eligibility is conditional.
- Name formatting differences can unfairly lower scores unless canonicalization is used.

What I would do next:

1. Expand the corpus with official source scraping and timestamps.
2. Add source citations to every returned recommendation.
3. Add a UI where students can filter by country, degree, field, and funding.
4. Add a confidence label: `eligible`, `conditional`, `verify`, or `not eligible`.
5. Evaluate with real Mongolian student queries rather than only synthetic questions.
6. Add human review for ambiguous eligibility decisions.

---

## 14. Repository Structure

```text
.
├── data/
│   ├── eval/
│   │   ├── eval_questions.csv
│   │   └── eval_questions_mn.csv
│   ├── processed/
│   │   └── documents.json
│   └── raw/
│       └── *.txt
├── evaluation/
│   ├── evaluate.py
│   ├── evaluate_baseline.py
│   ├── evaluate_chunks.py
│   ├── evaluate_k.py
│   └── evaluate_multilingual.py
├── retrievers/
│   ├── bm25_retriever.py
│   ├── embedding_retriever.py
│   └── hybrid_retriever.py
├── src/
│   ├── chunk_documents.py
│   ├── config.py
│   ├── llm.py
│   ├── load_data.py
│   ├── multilingual.py
│   ├── rag.py
│   └── utils.py
├── requirements.txt
└── README.md
```

---

## 15. References

- Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.
- Robertson, S. and Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond.
- OpenAI API documentation for GPT-based generation.
- Sentence Transformers documentation for dense text embeddings.
- Official scholarship program pages listed in the source fields of the dataset.

---

## 16. Reproducibility Checklist

- [x] Data included in repository
- [x] Evaluation questions included
- [x] Baseline script included
- [x] Multiple retrieval methods included
- [x] Metrics scripts included
- [x] Requirements file included
- [x] API key loaded from environment variable
- [ ] Exact result CSVs committed after final run
- [ ] Final report includes actual metrics table
- [ ] Final report includes 5-10 concrete error examples

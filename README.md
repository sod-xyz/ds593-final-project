# Finding the Right Scholarships: A RAG System for Mongolian Students

Graduate final project for **DS593: Theory and Applications of AI**.

This repository implements and evaluates a retrieval-augmented generation (RAG) assistant designed to help Mongolian scholarship seekers identify relevant international scholarship opportunities. The system is built with Mongolian users in mind, but it is not limited to current students: it can support applicants searching for bachelor’s, master’s, PhD, exchange, and other study-related funding opportunities. The assistant accepts a natural-language question, retrieves scholarship records from a curated corpus, and uses an LLM to return only scholarship names supported by the retrieved evidence.

> **Note:** This project was developed as a graduate course final project and should be treated as a working prototype, not a finised product/tool. Scholarship eligibility criteria, deadlines, funding rules, participating countries, and application requirements change frequently. As such, users/students should verify all recommendations through the official scholarship websites before making application decisions.

---

## 1. Executive Summary

Many Mongolian scholarship seekers face serious barriers when searching for international study opportunities. Scholarship information is often scattered across official websites, written mainly in English, and described using eligibility rules that are difficult to interpret. This creates a major information gap, especially for students outside well-resourced schools.

In all Mongolian state public schools, students do not have access to dedicated college counselors or professional application advisors. Instead, guidance often falls to homeroom teachers, who may be responsible for 100–200 students at a single time in a single graduating class. As a result, many students and families are unaware that international scholarships exist, or they do not know how to evaluate whether a program is legitimate, affordable, or open to Mongolian applicants.

This issue also appeared in an informal observation I conducted in my girlfriend’s younger sister’s graduating class. Among the students in that class, only two students, including my girlfriend’s sister, applied to foreign university or scholarship programs. While this is not a formal survey and should not be interpreted as representative evidence, it helped motivate the project by showing how easily international opportunities can remain unknown even among capable students.

individuals exploit families’ hopes of giving their children better educational and career opportunities, while taking advantage of their limited knowledge of foreign universities, application procedures, language requirements, and scholarship rules. In some cases, families may pay for misleading or fraudulent “admission” or “scholarship” services. Therefore, scholarship search is not only an academic or technical problem, but also a real-world issue of information access, educational equity, and soci-economic protection.

A general-purpose LLM can produce fluent scholarship advice, but it may hallucinate scholarship programs, recommend outdated opportunities, or suggest scholarships for which Mongolian applicants are not eligible. This project therefore tests whether a small, domain-specific retrieval-augmented generation system can provide more grounded scholarship recommendations by retrieving from a curated scholarship corpus and forcing the model to return only evidence-supported scholarship names.

The project compares:

1. a **no-retrieval LLM label-selection baseline**;
2. a **BM25 lexical retriever**;
3. a **dense embedding retriever component** using Sentence Transformers;
4. a **hybrid BM25 + dense retriever** using reciprocal rank fusion;
5. a final **metadata-aware hybrid RAG system** with query expansion and reranking.

Evaluation is framed as a set prediction problem: each question has a semicolon-separated expected set of scholarship names, and the system is scored using precision, recall, and F1. Retrieval quality is also evaluated separately using hit@k, retrieval recall@k, and retrieval precision@k.

---

## 2. Research Question

**Can a retrieval-augmented LLM system accurately recommend scholarship names for Mongolian students when questions include constraints such as country, degree level, funding type, language requirement, work experience, and Mongolia eligibility?**

Sub-questions:

1. Does retrieval improve answer quality compared with an LLM baseline that receives no retrieved evidence?
2. Which retrieval strategy works best: BM25, dense embeddings, or hybrid retrieval?
3. Does metadata-aware query expansion and reranking improve constrained scholarship search?
4. How well does the system handle Mongolian-language scholarship questions?
5. What types of errors remain, and are they caused by retrieval, generation, labels, or data limitations?

---

## 3. Dataset

The corpus is stored in `data/processed/documents.json` and mirrored as individual raw text files in `data/raw/`.

Each scholarship record contains:

| Field | Meaning |
|---|---|
| `scholarship` | Canonical scholarship name used for evaluation |
| `text` | Short program description |
| `country` | Destination country or region |
| `degree_level` | Eligible degree level(s) |
| `language_requirement` | Whether language requirements are required, conditional, or variable |
| `language_details` | Details about IELTS, TOEFL, English, Japanese, etc. |
| `funding_type` | Fully funded, partial, varies, or conditional |
| `funding_details` | Tuition, stipend, travel, insurance, and related details |
| `mongolia_eligible` | Eligibility judgment for Mongolian students |
| `mongolia_eligibility_note` | Caveat explaining uncertainty or conditionality |
| `source` | Official source URL or source reference |

### Evaluation data

The repository includes two hand-labeled evaluation sets:

| File | Description |
|---|---|
| `data/eval/eval_questions.csv` | English evaluation questions |
| `data/eval/eval_questions_mn.csv` | Mongolian evaluation questions |

Each row contains a question and a semicolon-separated expected answer set.

### Critical dataset limitations

The dataset is useful for a focused solo graduate project, but it is not large enough to support claims about real-world scholarship advising quality. The most important limitations are:

1. The corpus is small and manually curated.
2. Some labels encode subjective eligibility judgments, especially for conditional programs.
3. Several expected answers are debatable because scholarships may depend on citizenship, partner universities, nomination, field, or annual country lists.
4. The corpus does not include source timestamps or automatic update checks.
5. The evaluation questions are mostly synthetic rather than collected from real Mongolian students.

These limitations should be stated clearly in the final report and presentation.

---

## 4. System Architecture

```text
User question
   |
   v
Language detection
   |
   v
Optional Mongolian-to-English translation
   |
   v
Query understanding
(country, degree, funding, language, Mongolia eligibility, work experience)
   |
   v
Query expansion with synonyms and structured constraints
   |
   v
Candidate retrieval
(BM25, dense embeddings, or hybrid RRF)
   |
   v
Metadata-aware reranking
   |
   v
Context construction from top-k records
   |
   v
LLM constrained to allowed scholarship names
   |
   v
Canonicalization, scoring, and error analysis
```

The final pipeline is implemented mainly in `src/rag.py`.

---

## 5. Methods Compared

### 5.1 No-retrieval LLM baseline

The baseline gives the LLM the list of allowed scholarship names but no retrieved context. This is not a random baseline and should not be described as a pure no-information baseline. It is better described as a **no-retrieval label-selection baseline**. It tests whether retrieval adds value beyond the LLM's prior knowledge and the controlled output label list.

### 5.2 BM25 retriever

BM25 performs lexical keyword matching. It is expected to work well for questions with exact terms such as country names, `IELTS`, `PhD`, or `work experience`.

### 5.3 Dense embedding retriever

The embedding retriever uses `sentence-transformers/all-MiniLM-L6-v2` with cosine similarity. It can help when the query and document use different wording, but it may be weaker for strict categorical filters such as country or funding type.

### 5.4 Hybrid retriever

The hybrid retriever combines BM25 and dense retrieval using reciprocal rank fusion. This is intended to keep BM25's exact-match strength while adding semantic retrieval coverage.

### 5.5 Metadata-aware hybrid RAG

The final system extracts simple structured constraints from the question, expands the query with synonyms, retrieves a larger candidate set, reranks candidates using metadata fields, and sends the top-k evidence to the LLM.

This is the strongest conceptual part of the project because it uses the dataset structure instead of treating every record as plain text.

---

## 6. Evaluation Design

This project evaluates the assistant as a **set prediction task**. Each test question has an expected set of scholarship names, and the system also returns a set of predicted scholarship names. Both are written as semicolon-separated lists.

For example:

```text
Expected:
MEXT Scholarship; Global Korea Scholarship

Predicted:
MEXT Scholarship; Australia Awards Scholarship

In this case:

MEXT Scholarship is a true positive because it was expected and returned.
Australia Awards Scholarship is a false positive because it was returned but not expected.
Global Korea Scholarship is a false negative because it was expected but missed.

This setup is useful because scholarship search is often a multi-answer task. A user may ask for scholarships for a specific country, degree level, or applicant background, and more than one scholarship may be relevant.

6.1 Generation Metrics

The final LLM answer is parsed as a set of scholarship names and compared with the expected set for each evaluation question.

Metric	Meaning	Why it matters
Precision	Share of returned scholarships that are expected	Shows whether the system avoids recommending irrelevant or unsupported scholarships
Recall	Share of expected scholarships recovered	Shows whether the system finds the relevant opportunities it should return
F1	Harmonic mean of precision and recall	Gives one combined score balancing precision and recall
Unsupported hallucination	Whether the system returned at least one unsupported scholarship	Captures cases where the model recommends a scholarship not supported by the expected answer set
Wrong answer rate	Whether the predicted set differs from the expected set	Measures whether the full answer set was not exactly correct
Precision

Precision answers the question:

Of the scholarships the system returned, how many were actually correct?

High precision means the assistant is careful and avoids recommending scholarships that are irrelevant, unsupported, or not appropriate for the query.

This is especially important for scholarship search because a wrong recommendation can waste a user's time or mislead them during an application process.

Recall

Recall answers the question:

Of the scholarships the system should have returned, how many did it find?

High recall means the assistant successfully finds most of the relevant opportunities in the corpus.

Recall is important because missing a relevant scholarship may cause a user to overlook a valuable funding opportunity.

F1

F1 combines precision and recall into one score:

F1 = 2 × (Precision × Recall) / (Precision + Recall)

F1 is useful because the system must balance two goals:

avoid returning wrong scholarships;
avoid missing relevant scholarships.

However, F1 should not be interpreted by itself. Two systems can have similar F1 scores but behave very differently. One system may be conservative with high precision and low recall, while another may return many results with higher recall but lower precision.

Unsupported Hallucination

Unsupported hallucination measures whether the system returned at least one scholarship that was not supported by the expected answer set.

This metric matters because one of the main goals of using RAG is to reduce hallucination. A general-purpose LLM may generate fluent but incorrect scholarship advice. The RAG system is supposed to reduce this risk by forcing the model to answer from retrieved scholarship records.

Wrong Answer Rate

Wrong answer rate is stricter than F1. It checks whether the predicted set differs from the expected set at all.

An answer is counted as wrong if:

it misses one expected scholarship;
it adds one incorrect scholarship;
it returns a completely different set;
it returns NONE when scholarships were expected;
it returns scholarships when the expected answer was NONE.

This metric is useful because it shows how often the system gives a fully correct answer, but it can be harsh for multi-answer questions.

6.2 Retrieval Metrics

Retrieval is evaluated separately from generation. This is important because a RAG system has two major stages:

The retriever must find the correct scholarship records.
The LLM must select the correct scholarship names from the retrieved context.

If the retriever fails, the LLM cannot produce a grounded correct answer because the necessary evidence was never included in the prompt.

Metric	Meaning	Why it matters
hit_at_k	At least one expected scholarship appears in the top-k retrieved records	Shows whether retrieval found any relevant evidence
retrieval_recall_at_k	Share of expected scholarships appearing in the top-k retrieved records	Shows how completely retrieval recovered the correct evidence
retrieval_precision_at_k	Share of top-k retrieved records that are expected answers	Shows how much irrelevant context is being passed to the LLM
hit_at_k

hit_at_k checks whether at least one correct scholarship appears in the top-k retrieved records.

For example, if k = 5, the metric asks:

Did at least one expected scholarship appear in the top 5 retrieved records?

This is a useful but forgiving metric. It tells us whether the retriever found something relevant, but it does not show whether it found all relevant scholarships.

retrieval_recall_at_k

retrieval_recall_at_k measures how many of the expected scholarships appeared in the top-k retrieved records.

For example, if a question has four expected scholarships and the retriever finds two of them in the top 5, then:

retrieval_recall_at_5 = 2 / 4 = 0.50

This metric is very important for multi-answer scholarship questions. If the correct scholarship is not retrieved, the LLM is unlikely to include it in the final answer.

retrieval_precision_at_k

retrieval_precision_at_k measures how many of the retrieved records are actually expected answers.

For example, if the top 5 retrieved records contain two expected scholarships and three irrelevant scholarships, then:

retrieval_precision_at_5 = 2 / 5 = 0.40

Low retrieval precision means the LLM receives noisy context. This can cause the model to select the wrong scholarship names or return extra unsupported results.

6.3 Why Retrieval and Generation Are Evaluated Separately

Separating retrieval and generation helps identify where the system fails.

A RAG system can fail in at least two ways:

Case 1: Retrieval failure

The retriever does not find the correct scholarship records.

This may happen because:

the user query uses different wording from the corpus;
the corpus does not contain enough synonyms;
country names are written inconsistently;
degree levels are missing or ambiguous;
scholarship eligibility is described indirectly;
metadata fields are incomplete;
the retriever overweights surface-level keyword matches.

In this case, the LLM cannot answer correctly because the correct evidence is missing from the prompt.

Case 2: Generation failure

The retriever finds the correct scholarship records, but the LLM still returns the wrong answer.

This may happen because:

the prompt is not restrictive enough;
the retrieved context contains too many similar scholarships;
the LLM returns extra names;
the LLM misses one of the retrieved correct scholarships;
the output format is not followed exactly;
scholarship names are not normalized consistently;
the parser fails to match slightly different versions of the same name.

In this case, the retrieval system may be working, but the answer-selection step needs improvement.

### 6.4 Limitations of the Evaluation

For this project, F1 is useful, but it is not enough. The purpose of evaluation is not only to report a single score, but to understand why the system succeeds or fails.

As mentioned in the executive summary, this project is a working prototype built on a small curated corpus. Therefore, the evaluation scores should be interpreted carefully. Low scores may reflect limitations of the project design and dataset, not only weaknesses in the RAG approach itself.

Important limitations include:

- a small number of scholarship records;
- manually created evaluation labels;
- incomplete scholarship metadata;
- ambiguous natural-language questions;
- multiple reasonable answers for some queries;
- difficulty distinguishing eligibility, country, degree level, and funding type;
- limited coverage of real scholarship opportunities;
- possible mismatch between user wording and scholarship descriptions;
- reliance on exact scholarship-name matching during evaluation.

Therefore, the evaluation should be treated as a **diagnostic analysis** of the pipeline rather than a final measure of ready to launch tool.
---

## 7. Current Status of Results

The uploaded repository defines evaluation scripts, but the packed version did **not** include committed result CSV files because `results/` is ignored by `.gitignore`. Therefore, anyone reviewing the repository cannot verify the exact numeric claims unless they rerun the evaluation.

Before final submission, run the evaluation scripts and paste the final metrics into the table below. Also commit either:

- `reports/results_summary.md`, or
- selected CSV summaries under a tracked directory such as `reports/results/`.

### Final results table to fill after rerun

| System | Precision | Recall | F1 | Wrong answer rate | Unsupported hallucination rate | Retrieval Recall@5 |
|---|---:|---:|---:|---:|---:|---:|
| No-retrieval label-selection baseline | TODO | TODO | TODO | TODO | TODO | N/A |
| BM25 RAG | TODO | TODO | TODO | TODO | TODO | TODO |
| Dense embedding RAG | TODO | TODO | TODO | TODO | TODO | TODO |
| Hybrid RAG | TODO | TODO | TODO | TODO | TODO | TODO |
| Metadata-aware hybrid RAG | TODO | TODO | TODO | TODO | TODO | TODO |

### How to interpret expected outcomes

If the no-retrieval baseline performs close to or better than RAG, the correct interpretation is **not** that RAG is useless. It means the dataset/evaluation may be too label-driven and too small, allowing the LLM to guess from the allowed label list. In that case, the report should emphasize that future evaluation needs harder questions, source-grounded citation scoring, and real student queries.

If BM25 beats dense retrieval, that is plausible because scholarship queries often contain exact categorical constraints. If hybrid retrieval performs best, argue that it captures both exact constraints and semantic similarity.

---

## 8. Error Analysis Plan

After running evaluation, inspect the lowest-F1 rows in the result CSVs and classify each failure:

| Error type | Description | Example |
|---|---|---|
| Retrieval miss | Correct scholarship not in retrieved top-k | Country-specific scholarship absent from retrieved records |
| Over-selection | System returns too many loosely relevant scholarships | “Fully funded” query includes partial awards |
| Under-selection | System omits correct scholarships | Only one of several expected Japan scholarships returned |
| Metadata conflict | Metadata and raw text imply different eligibility | Program marked conditional but expected as available |
| Label ambiguity | Expected answer set is debatable | Rhodes or Mastercard eligibility may depend on route |
| Name mismatch | Punctuation or naming variation affects scoring | `ADB-JSP` vs full official name |
| Multilingual failure | Translation loses a key constraint | Mongolian phrase for UK or master’s not preserved |
| LLM selection error | Correct evidence retrieved, but LLM chooses wrong names | Retrieved context contains answer but output omits it |

The final report should include at least 5 concrete error examples.

---

## 9. Reproducibility Instructions

### 9.1 Clone the repository

```bash
git clone <your-repo-url>
cd ds593-final-project
```

### 9.2 Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 9.3 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 9.4 Configure the API key

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

The OpenAI key is required for final answer generation and Mongolian-to-English translation. Retrieval classes can be inspected without the key, but LLM-based evaluation cannot be reproduced without it.

### 9.5 Run evaluations

```bash
# No-retrieval label-selection baseline
python -m evaluation.evaluate_baseline

# BM25, dense embedding, and hybrid RAG comparison
python -m evaluation.evaluate

# Top-k sensitivity for metadata-aware hybrid RAG
python -m evaluation.evaluate_k

# English vs Mongolian evaluation
python -m evaluation.evaluate_multilingual

# Chunk-size experiment
python -m evaluation.evaluate_chunks
```

### 9.6 Expected output files

| File | Description |
|---|---|
| `results/baseline_results.csv` | Per-question no-retrieval baseline predictions |
| `results/baseline_summary.csv` | Baseline summary metrics |
| `results/generation_results.csv` | Per-question RAG predictions |
| `results/summary_metrics.csv` | Aggregated metrics by retriever |
| `results/k_experiment_results.csv` | Per-question top-k experiment results |
| `results/k_experiment_summary.csv` | Top-k summary metrics |
| `results/multilingual_results.csv` | English and Mongolian per-question results |
| `results/multilingual_summary.csv` | Language-level summary metrics |
| `results/chunk_size_results.csv` | Chunk-size experiment results |
| `results/chunk_size_summary.csv` | Chunk-size summary metrics |

---

## 10. Repository Structure

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
├── reports/
│   └── REVIEW_AND_RECOMMENDATIONS.md
├── requirements.txt
└── README.md
```

---

## 11. Ethical Considerations for the RAG/LLM System

This project uses a retrieval-augmented generation system, so the main ethical concerns come from both the retrieval component and the language model component. Even when the system is designed to return only scholarship names from retrieved context, it can still produce misleading results if retrieval is incomplete, the corpus is outdated, or the LLM interprets the context incorrectly.

The main ethical risks include:

- **Hallucination:** the LLM may return scholarship names or claims that are not fully supported by the retrieved records.
- **Retrieval bias:** the system can only retrieve from the scholarships included in the corpus, so missing records become invisible to the user.
- **Outdated knowledge:** scholarship deadlines, eligibility rules, country lists, and funding conditions change frequently, so a static limited corpus can quickly become unreliable.
- **False confidence:** fluent LLM output may make uncertain or incomplete recommendations appear more reliable than they are.
- **Language inequality:** if translation or Mongolian-language query handling is weak, users who cannot search well in English may still receive worse results. Our current evaluation level may not be at a satisfactory level to negate this inequality.
- **Metadata errors:** incorrect or incomplete fields such as country, degree level, nationality, or funding type can cause the system to filter out relevant scholarships or recommend unsuitable ones.
- **Evaluation bias:** manually written test questions and expected answers may not capture all valid scholarships, which can make the system appear better or worse than it really is.
- **Privacy risk:** a real large-scale deployment may require sensitive applicant personal information such as their grades, income, sex, parents salary, nationality, disability status, or migration goals. Such data would need strong privacy protections.

This project reduces some risk by using retrieval, limiting outputs to known scholarship names (personally selected), and evaluating unsupported hallucinations. However, these safeguards are no where enough for real-world use cases and future developments would need to consider more regarding the ethical impacts and risks of the this tool.

---

## 12. Iteration and Reflection

The project improved from a basic RAG prototype into a more structured retrieval pipeline.

What improved:

- Added BM25, dense, and hybrid retrievers.
- Added a no-retrieval baseline.
- Added metadata fields and metadata-aware reranking.
- Added query expansion.
- Added retrieval and generation metrics.
- Added Mongolian-language evaluation.

What remains weak:

- Evaluation labels are manually created and sometimes debatable.
- The dataset is small and may be too easy for label-selection.
- Results are not committed in the uploaded repository.
- The system returns only names, not evidence snippets or citations.
- The chunk-size experiment is not very meaningful because most records are short.
- Multilingual evaluation depends on OpenAI translation and is not fully deterministic.

Next steps:

1. Expand the corpus using official pages and store `last_checked` dates.
2. Add source citations and evidence snippets to every answer.
3. Evaluate with real student queries.
4. Add a deterministic metadata-filter baseline.
5. Add confidence labels: `eligible`, `conditional`, `verify`, `not eligible`.
6. Track all final result summaries in `reports/results/`.

---

## 13. References

- Lewis, P., et al (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. Advances in Neural Information Processing Systems. 
- Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval, 3(4), 333-389. 
- OpenAI API documentation.
- Sentence Transformers documentation.
- Official scholarship pages listed in the `source` field of each dataset record.
- OpenAI. (2026). *OpenAI API Documentation*. https://developers.openai.com/api/docs
- OpenAI. (2026). *OpenAI API Reference*. https://developers.openai.com/api/reference/overview
- Sentence Transformers. (2026). *Sentence Transformers Documentation*. https://sbert.net/
- Hugging Face. (2026). *Using Sentence Transformers at Hugging Face*. https://huggingface.co/docs/hub/sentence-transformers
- Brown, D. (n.d.). *rank-bm25: A Collection of BM25 Algorithms in Python*. GitHub repository. https://github.com/dorianbrown/rank_bm25
- OpenAI. (n.d.). *Fine-tuned QA and Retrieval-Augmented Generation examples*. OpenAI Cookbook. https://github.com/openai/openai-cookbook
- LangChain. *RAG from Scratch* video playlist. https://www.youtube.com/watch?v=sVcwVQRHIc8
- LangChain AI. (n.d.). *RAG from Scratch*. GitHub repository and video-notebook series. https://github.com/langchain-ai/rag-from-scratch
- LangChain. (2026). *Build a RAG agent with LangChain*. https://docs.langchain.com/oss/python/langchain/rag
- Other complementary YouTube videos.

The scholarship corpus was manually curated from official or semi-official scholarship pages. The `source` field in each dataset record should be treated as part of the reference list. Examples include official pages for:

- Chevening Scholarships
- Fulbright Program
- DAAD Scholarships
- Erasmus Mundus Joint Master Degrees
- MEXT Scholarship
- Global Korea Scholarship
- Stipendium Hungaricum
- Chinese Government Scholarship
- Australia Awards
- New Zealand Scholarships
- Asian Development Bank-Japan Scholarship Program
- Joint Japan/World Bank Graduate Scholarship Program
- Swiss Government Excellence Scholarships
- Swedish Institute Scholarships
- Vanier Canada Graduate Scholarships
- Knight-Hennessy Scholars
- Gates Cambridge Scholarship
- Rhodes Scholarship
- Schwarzman Scholars
- Rotary Peace Fellowship

### AI Assistance Disclosure

This project was primarily designed, implemented, evaluated, and documented by myself. VS Code Copilot was used as an assistive development tool for code implementation, debugging, refactoring, and repository organization. I made all final decisions regarding the system design, curated scholarship dataset, retrieval pipeline, evaluation methodology, interpretation of results, limitations, documentation, etc. 

---

## 14. Final Words and Comments

I would like to thank our instructor, **Dr. Wheelock**, for teaching such a wonderful and enlightening class. This project gave me the opportunity to really apply all the teachings we learned in class to a problem that is personally meaningful to me, and relevant to real world applications. Although, there were some difficulties in implementing this project, I really learned a lot and appreciate everything. I look forward to improving on this project and hopefully make it an actual tool for future Mongolian scholars. I love education. 

I would also like to thank our TAs, especially **Bhoomika**, for their guidance, support, and feedback throughout the semester. Their help during labs, assignments, debugging, and project development made the process much more manageable.

Finally, I am grateful for my classmates who were so insightful in class and really helped elevate the class discussions and our group learning. This repository represents not only the final implementation of my DS 593 results, but also the learning process behind it. Thank you!

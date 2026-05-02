# Finding the Right Scholarships: A RAG System for Mongolian Students

Graduate final project for **DS593: Theory and Applications of AI**.

This repository implements and evaluates a retrieval-augmented generation (RAG) assistant designed to help Mongolian scholarship seekers identify relevant international scholarship opportunities. The system accepts a natural-language question, retrieves scholarship records from a curated corpus, and uses an LLM to return scholarship names supported by the retrieved evidence.

> **Note:** This project was developed as a graduate course final project and should be treated as a working prototype, not a finised product/tool. Scholarship eligibility criteria, deadlines, funding rules, participating countries, and application requirements change frequently. As such, users/students should verify all recommendations through the official scholarship websites before making application decisions.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Research Question](#2-research-question)
3. [Dataset](#3-dataset)
4. [System Architecture](#4-system-architecture)
5. [Methods Compared](#5-methods-compared)
6. [Evaluation Design](#6-evaluation-design)
7. [Final Results](#7-final-results)
8. [Error Analysis](#8-error-analysis)
9. [Iteration and Reflection](#9-iteration-and-reflection)
10. [Ethical Considerations and Limitations](#10-ethical-considerations-and-limitations)
11. [Reproducibility Instructions](#11-reproducibility-instructions)
12. [Repository Structure](#12-repository-structure)
13. [Technical Issues and Future Recommendations](#13-technical-issues-and-future-recommendations)
14. [References](#14-references)
15. [AI Assistance Disclosure](#15-ai-assistance-disclosure)
16. [Final Comments](#16-final-comments)

## 1. Executive Summary / Motivation

Many Mongolian scholarship seekers face a real information-access problem. Scholarship information is scattered across official websites, often written in English, and described using eligibility rules that are difficult to interpret. This can be especially difficult for students who do not have access to college counselors, professional application advisors, or family members who have already studied abroad.

Compared to U.S state schools, in Mongolian state public schools, students do not have access to dedicated college counselors or professional application advisors. Instead, guidance often falls to homeroom teachers, who may be responsible for 100–200 students at a single time in a single graduating class. As a result, many students and families are unaware that international scholarships exist, or they do not know how to evaluate whether a program is legitimate, affordable, or open to Mongolian applicants.

This issue also appeared in an informal observation I conducted in my girlfriend’s younger sister’s graduating class. Among the students in that class, only two students, including my girlfriend’s sister, applied to foreign university or scholarship programs. While this is not a formal survey and should not be interpreted as representative evidence, it helped motivate the project by showing how easily international opportunities can remain unknown even among capable students.

Individuals exploit families’ hopes of giving their children better educational and career opportunities, while taking advantage of their limited knowledge of foreign universities, application procedures, language requirements, and scholarship rules. In some cases, families may pay for misleading or fraudulent “admission” or “scholarship” services. Therefore, scholarship search is not only an academic or technical problem, but also a real-world issue of information access, educational equity, and soci-economic protection.

The motivation for this project is that scholarship discovery is not only a search problem; it is also an equity problem. If students cannot find reliable information, they may miss legitimate opportunities or become vulnerable to misleading admission and scholarship services. A general-purpose LLM can give fluent advice, but it may hallucinate scholarship names, recommend outdated opportunities, or suggest programs for which Mongolian applicants are not eligible.

This project tests whether a small, domain-specific RAG system filled can make scholarship search more grounded. The system compares a no-retrieval LLM baseline with BM25, dense embedding, hybrid retrieval, and a metadata-aware hybrid RAG pipeline. Evaluation treats scholarship recommendation as a set-prediction task: each question has an expected set of scholarship names, and the model is scored using precision, recall, and F1. Retrieval is also evaluated separately using hit@k and retrieval recall@k.

The final result is a useful prototype and project of reference for small corpus project, but not a deployable system. The best-performing basic English RAG configuration reaches an F1 of about **0.456**, while the best metadata-aware top-k configuration reaches an F1 of about **0.476**. This is better than the no-retrieval label-selection baseline F1 of **0.348**, but it still leaves significant errors. The project is therefore best understood and received as a diagnostic study of what works and fails in a small scholarship-focused RAG system.

---

## 2. Research Question

**Can a retrieval-augmented LLM system accurately recommend scholarship names for Mongolian students when questions include complex constraints such as country, degree level, funding type, language requirement, work experience, and Mongolia eligibility?**

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
| `scholarship` | Scholarship name used for evaluation |
| `text` | Short program description |
| `country` | Destination country or region |
| `degree_level` | Eligible degree level(s) |
| `language_requirement` | Whether language requirements are required, conditional, or variable |
| `language_details` | Language requirement details about IELTS, TOEFL, English, Japanese, etc. |
| `funding_type` | Fully funded, partial, varies, conditional, or not applicable |
| `funding_details` | Tuition, stipend, travel, insurance, and related funding details |
| `mongolia_eligible` | Eligibility judgment for Mongolian students |
| `mongolia_eligibility_note` | Caveat explaining uncertainty or conditionality |
| `source` | Official or semi-official source URL |

### Evaluation Data

The repository includes two hand-labeled evaluation sets:

| File | Description |
|---|---|
| `data/eval/eval_questions.csv` | English evaluation questions |
| `data/eval/eval_questions_mn.csv` | Mongolian evaluation questions |

Each row contains a question and a semicolon-separated expected answer set.

### Dataset Limitations

The dataset is intentionally focused, but small. This affects both performance and interpretation.

1. The corpus contains a limited number of manually curated scholarship records.
2. Some labels are subjective because scholarship eligibility is often conditional.
3. Several expected answers are debatable because eligibility may depend on citizenship, partner universities, field, nomination, or annual country lists.
4. The corpus does not include source timestamps, deadline checks, or automatic freshness validation.
5. The evaluation questions are mostly synthetic rather than collected from real Mongolian students.
6. Some programs are difficult to classify because they are not standard scholarships, such as internships, fellowships, or last-dollar domestic awards.
7. Because the corpus is small, the LLM baseline can sometimes guess from the allowed label list without true evidence grounding.

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
(BM25, dense embeddings, or hybrid reciprocal-rank fusion)
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

The baseline gives the LLM the list of allowed scholarship names. It tests whether retrieval adds value beyond the LLM's prior knowledge and the controlled output label list.

### 5.2 BM25 retriever

BM25 performs lexical keyword matching. It is expected to work well for queries containing exact terms such as country names, `IELTS`, `PhD`, `fully funded`, or `work experience`.

### 5.3 Dense embedding retriever

The embedding retriever uses a sentence-transformer model to search for scholarships based on semantic similarity rather than exact keyword overlap.

### 5.4 Hybrid retriever

The hybrid retriever combines BM25 and dense retrieval using reciprocal rank fusion. This is designed to preserve BM25's exact-match strength while adding better semantic retrievel coverage.

### 5.5 Metadata-aware hybrid RAG

The final system first searches the scholarship database in two ways: by matching keywords and by finding similar meaning. Then it uses the scholarship details, such as country, degree level, funding type, language requirement, Mongolia eligibility, and work experience, to rank the most relevant results before asking the LLM to choose the final scholarship names.

---

## 6. Evaluation Design

This project evaluates the assistant as a **set prediction task**. Each test question has an expected set of scholarship names, and the system also returns a set of predicted scholarship names. Both are written as semicolon-separated lists.

For example:

```text
Expected:
MEXT Scholarship; Global Korea Scholarship

Predicted:
MEXT Scholarship; Australia Awards
```

In this case:

1. `MEXT Scholarship` is a true positive because it was expected and returned.
2. `Australia Awards` is a false positive because it was returned but not expected.
3. `Global Korea Scholarship` is a false negative because it was expected but missed.

This setup is useful because scholarship search is often a multi-answer task. A user may ask for scholarships for a specific country, degree level, or applicant background, and more than one scholarship may be relevant.

### 6.1 Generation Metrics

The final LLM answer is parsed as a set of scholarship names and compared with the expected set for each evaluation question.

| Metric | Meaning | Why it matters |
|---|---|---|
| Precision | Share of returned scholarships that are expected | Shows whether the system avoids recommending irrelevant or unsupported scholarships |
| Recall | Share of expected scholarships recovered | Shows whether the system finds the relevant opportunities it should return |
| F1 | Harmonic mean of precision and recall | Gives one combined score balancing precision and recall |
| Unsupported hallucination | Whether the system returned at least one unsupported scholarship | Captures cases where the model recommends a scholarship not supported by the expected answer set |
| Wrong answer rate | Whether the predicted set differs from the expected set | Measures whether the full answer set was not exactly correct |

#### Precision

Precision answers the question:

> Of the scholarships the system returned, how many were actually correct?

High precision means the assistant is careful and avoids recommending scholarships that are irrelevant, unsupported, or not appropriate for the query.

This is especially important for scholarship search because a wrong recommendation can waste a user's time or mislead them during an application process.

#### Recall

Recall answers the question:

> Of the scholarships the system should have returned, how many did it find?

High recall means the assistant successfully finds most of the relevant opportunities in the corpus.

Recall is important because missing a relevant scholarship may cause a user to overlook a valuable funding opportunity.

#### F1

F1 combines precision and recall into one score:

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

F1 is useful because the system must balance two goals:

1. Avoid returning wrong scholarships.
2. Avoid missing relevant scholarships.

However, F1 should not be interpreted by itself. Two systems can have similar F1 scores but behave very differently. One system may be conservative with high precision and low recall, while another may return many results with higher recall but lower precision.

#### Unsupported Hallucination

Unsupported hallucination measures whether the system returned at least one scholarship that was not supported by the expected answer set.

This metric matters because one of the main goals of using RAG is to reduce hallucination. A general-purpose LLM may generate fluent but incorrect scholarship advice. The RAG system is supposed to reduce this risk by forcing the model to answer from the retrieved scholarship records.

#### Wrong Answer Rate

Wrong answer rate is stricter than F1. It checks whether the predicted set differs from the expected set at all.

An answer is counted as wrong if:

1. It misses one expected scholarship.
2. It adds one incorrect scholarship.
3. It returns a completely different set.
4. It returns `NONE` when scholarships were expected.
5. It returns scholarships when the expected answer was `NONE`.

This metric is strong because it shows how often the system gives a fully correct answer, but it can be harsh for multi-answer questions.

### 6.2 Retrieval Metrics

Retrieval is evaluated separately from generation. This is important because a RAG system has two major stages:

1. The retriever must find the correct scholarship records.
2. The LLM must select the correct scholarship names from the retrieved context.

If the retriever fails, the LLM cannot produce a grounded correct answer because the necessary evidence was never included in the prompt.

| Metric | Meaning | Why it matters |
|---|---|---|
| Hit@k | At least one expected scholarship appears in the top-k retrieved records | Shows whether retrieval found any relevant evidence |
| Retrieval Recall@k | Share of expected scholarships appearing in the top-k retrieved records | Shows how completely retrieval recovered the correct evidence |
| Retrieval Precision@k | Share of top-k retrieved records that are expected answers | Shows how much irrelevant context is being passed to the LLM |

#### Hit@k

Hit@k checks whether at least one correct scholarship appears in the top-k retrieved records.

For example, if `k = 5`, the metric asks:

> Did at least one expected scholarship appear in the top 5 retrieved records?

This is a useful but forgiving metric. It tells us whether the retriever found something relevant, but it does not show whether it found all relevant scholarships.

#### Retrieval Recall@k

Retrieval Recall@k measures how many of the expected scholarships appeared in the top-k retrieved records.

For example, if a question has four expected scholarships and the retriever finds two of them in the top 5, then:

```text
Retrieval Recall@5 = 2 / 4 = 0.50
```

This metric is very important for multi-answer scholarship questions. If the correct scholarship is not retrieved, the LLM is unlikely to include it in the final answer.

#### Retrieval Precision@k

Retrieval Precision@k measures how many of the retrieved records are actually expected answers.

For example, if the top 5 retrieved records contain two expected scholarships and three irrelevant scholarships, then:

```text
Retrieval Precision@5 = 2 / 5 = 0.40
```

Low retrieval precision means the LLM receives noisy context. This can cause the model to select the wrong scholarship names or return extra unsupported results.

### 6.3 Why Retrieval and Generation Are Evaluated Separately

Separating retrieval and generation helps identify where the system fails.

A RAG system can fail in at least two ways.

#### Case 1: Retrieval Failure

The retriever does not find the correct scholarship records.

This may happen because:

1. The user query uses different wording from the corpus.
2. The corpus does not contain enough synonyms.
3. Country names are written inconsistently.
4. Degree levels are missing or ambiguous.
5. Scholarship eligibility is described indirectly.
6. Metadata fields are incomplete.
7. The retriever overweight's surface-level keyword matches.

In this case, the LLM cannot answer correctly because the correct evidence is missing from the prompt.

#### Case 2: Generation Failure

The retriever finds the correct scholarship records, but the LLM still returns the wrong answer.

This may happen because:

1. The prompt is not restrictive enough.
2. The retrieved context contains too many similar scholarships.
3. The LLM returns extra names.
4. The LLM misses one of the retrieved correct scholarships.
5. The output format is not followed exactly.
6. Scholarship names are not normalized consistently.
7. The parser fails to match slightly different versions of the same name.

In this case, the retrieval system may be working, but the answer-selection step needs improvement.

### 6.4 Limitations of the Evaluation

F1 is useful, but it is not enough. The purpose of evaluation is not only to report one score, but to understand why the system succeeds or fails.

Important limitations include:

1. The number of scholarship records is small.
2. Evaluation labels are manually created.
3. Some expected answers are debatable.
4. Some questions have multiple reasonable answers.
5. The corpus may not contain all relevant scholarships.
6. Scholarship names require exact or near-exact normalization.
7. Some programs have conditional eligibility that is difficult to score as simply correct or incorrect.
8. The Mongolian evaluation set has only 10 questions.

Therefore, the evaluation should be treated as a diagnostic analysis of the pipeline rather than a final measure of a launch-ready tool.

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

Therefore, the evaluation should be treated as a **diagnostic analysis** of the pipeline.

---

## 7. Results/Findings

### 7.1 Main Evaluation (English)

| System | Precision | Recall | F1 | Hit@5 | Retrieval Recall@5 | Notes |
|---|---:|---:|---:|---:|---:|---|
| No-retrieval label-selection baseline | 0.347 | 0.500 | 0.385 | N/A | N/A | Unsupported hallucination rate = 0.455; wrong-answer rate = 0.836 |
| BM25 RAG | 0.494 | 0.502 | 0.459 | 0.709 | 0.502 | Strong exact-match baseline |
| Dense embedding RAG | 0.499 | 0.492 | 0.472 | 0.745 | 0.510 | Best F1 among the three basic retrievers |
| Hybrid RAG | 0.500 | 0.510 | 0.474 | 0.727 | 0.539 | Best retrieval recall@5 among basic retrievers |

**Interpretation:** Retrieval improves over the no-retrieval baseline. The no-retrieval baseline has F1 = **0.385**, while the three RAG systems reach F1 values between **0.459** and **0.474**. This shows that grounding the LLM in retrieved scholarship records helps, but the improvement is moderate rather than dramatic.

The hybrid retriever has the strongest retrieval recall@5 (**0.539**), meaning it is best at putting relevant scholarship evidence into the context. However, dense embedding RAG has the highest generation F1 among the three basic retrievers (**0.472**). This suggests that retrieval quality and final answer quality are related but not identical. The LLM can still make selection errors even when relevant records are retrieved.

### 7.2 Metadata-Aware Top-K Experiment

| System | k | Precision | Recall | F1 | Hit@5 | Hit@10 | Retrieval Recall@5 | Retrieval Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Metadata-aware hybrid RAG | 5 | 0.506 | 0.519 | 0.482 | 0.818 | 0.836 | 0.539 | 0.578 |
| Metadata-aware hybrid RAG | 10 | 0.465 | 0.568 | 0.485 | 0.818 | 0.873 | 0.539 | 0.656 |
| Metadata-aware hybrid RAG | 15 | 0.486 | 0.619 | 0.515 | 0.818 | 0.873 | 0.539 | 0.656 |

**Interpretation:** Increasing top-k improves recall but also increases the amount of context passed to the LLM. The strongest F1 in this experiment occurs at **k = 15**, with F1 = **0.515** and recall = **0.619**. This makes sense for scholarship search because many questions have multiple valid answers. However, larger k can also increase noise, which may cause over-selection.

### 7.3 Chunk-Size Experiment

| Chunk size | Precision | Recall | F1 | Hit@5 | Retrieval Recall@5 |
|---:|---:|---:|---:|---:|---:|
| 300 | 0.510 | 0.519 | 0.481 | 0.727 | 0.539 |
| 600 | 0.506 | 0.538 | 0.484 | 0.727 | 0.539 |
| 900 | 0.506 | 0.519 | 0.481 | 0.727 | 0.539 |

**Interpretation:** Chunk size had only a small effect. The best F1 was with chunk size **600** at **0.484**. This is expected because each scholarship record is already short and structured. Chunking is more important for long documents, while this corpus behaves more like a collection of compact records.

### 7.4 Multilingual Evaluation

| Language | Precision | Recall | F1 | Hit@5 | Hit@10 | Retrieval Recall@5 | Retrieval Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| English | 0.455 | 0.544 | 0.473 | 0.727 | 0.782 | 0.448 | 0.565 |
| Mongolian | 0.368 | 0.512 | 0.407 | 0.900 | 1.000 | 0.503 | 0.670 |

**Interpretation:** The Mongolian evaluation is not exactly satisfactory. Mongolian questions achieved Hit@10 = **1.000**, meaning that the correct scholarship evidence usually appears somewhere in the top 10 retrieved records. However, Mongolian F1 = **0.407**, lower than English F1 = **0.473**. This suggests that translation and answer selection remain weaker than retrieval alone. Perhaps maybe a cause of how I phrased the 10 questions as Mongolian words have a lot of nuances so some things could be lost in translation.

The multilingual pipeline works by detecting Mongolian Cyrillic in `src/multilingual.py`, translating the query into English with an OpenAI model when an API key is available, and then using the same English retrieval pipeline. If no API key is available, the system returns the original Mongolian question, which can reduce retrieval quality because the corpus is mostly English, but in our results case it performed as intended.

---

## 8. Error Analysis Plan

The evaluation shows that the system is semi-successful: retrieval helps, but many questions remain difficult especially when complex, multi-filter requirements are included.

| Error type | What it means | Why it happened in this project |
|---|---|---|
| Retrieval miss | Correct scholarship does not appear in top-k | Small corpus, inconsistent wording, or metadata not specific enough |
| Over-selection | System returns too many scholarships | Broad queries such as “fully funded” retrieve many superficially relevant records |
| Under-selection | System returns only part of a multi-answer set | Some questions have several valid answers, but the LLM selects only the most obvious ones |
| Label ambiguity | Expected answer set is debatable | Programs may be conditional, country-specific, or dependent on current cycles |
| Metadata conflict | Metadata and question imply different eligibility | Some scholarships are marked conditional or not eligible but still appear in expected labels |
| Name mismatch | Same scholarship appears under slightly different names | Hyphen/en-dash and official-name variations affect exact matching |
| Multilingual loss | Mongolian query loses a constraint during translation | Translation may preserve country but weaken degree/funding/work-experience constraints |
| Small-corpus limitation | The correct answer may be outside the curated corpus | The system cannot recommend opportunities that are not in our curated `documents.json` |

### Specific Examples:

1. **Broad Mongolia eligibility query:** The baseline returned a very large list for “Which scholarships are available to Mongolian students?” It achieved high recall but low precision. This illustrates why recall alone is not enough.
2. **Country-specific retrieval:** Japan and South Korea queries worked relatively well because country metadata is explicit and scholarship names are distinctive.
3. **Conditional eligibility:** Questions about scholarships that “require checking eligibility” are difficult because conditional programs are not always clearly positive or negative examples and would require either contacting the school/state/program coordinators or fine-checking every detail on the large websites.
4. **Funding questions:** “Fully funded” questions often over-select because many programs mention stipends, tuition, or support, but the exact coverage varies. For some programs fully funded refers to tuition, while others refer to tuition and housing, while others need to include one-time or two-time travel stipend (common since Mongolia is still a developing country and a lot of these programs are development programs).
5. **Mongolian queries:** Retrieval often finds at least one correct record, but the final answer still loses F1 because translation, reranking, and answer selection introduce additional error points.

---

## 9. Iteration and Reflection

This project changed substantially during inception, development and final submission.

### 9.1 Initial Web-Scraping Attempt

My first plan was to scrape scholarship information directly from official websites and build a larger corpus automatically. This turned out to be harder than expected. Scholarship websites had inconsistent HTML structure, dynamic pages that required techniques like Selenium and clicking, PDF announcements, country-specific subpages, and changing application-cycle pages. Some pages blocked or limited scraping, while others mixed general program descriptions with current-cycle details. Even when scraping technically worked, the extracted text was noisy and difficult to convert into consistent fields such as degree level, funding type, and Mongolia eligibility.

Because of this, I decided that a smaller manually curated corpus would be more reliable for a course project than a larger but noisy scraped dataset. This reduced coverage, but it significantly improved interpretability and made evaluation possible.

### 9.2 Synthetic Data Attempt

After discussing the project direction my professor, I also tried creating synthetic scholarship records and synthetic queries to expand the evaluation set. This initially seemed useful because the real corpus was small. However, synthetic records created a new problem: they overcrowded the corpus with artificial examples that did not reflect real scholarship pages. The system began retrieving synthetic patterns rather than learning from real scholarship descriptions. This made the results less meaningful for the actual use case and made it very difficult to interpret my results and draw valuable findings.

In the end, I decided not to rely on synthetic scholarship records for final evaluation. I used a curated real corpus (expanded by 2x) and accepted that the dataset was small. This was a better match for the project goal because the assistant is supposed to help Mongolian students with real scholarship opportunities, not artificial examples, and I had personal experience/knowledge of them.

### 9.3 Kaggle Dataset Attempt

I also considered using public Kaggle scholarship datasets. However, many of the available datasets were U.S.-focused, undergraduate-focused, or not clearly relevant to Mongolian applicants. Including them would have increased the number of records but weakened the project scope much similar to the synthetic data attempt. Since the research question focuses on Mongolian students and international opportunities, adding many U.S.-only scholarships would have made retrieval noisier and less useful. More data =/= Better data.

### 9.4 Final Design Decision

The final system uses a small, curated corpus with metadata-specific retrieval. This made the project semi-successful. The evaluation shows that retrieval improves over a no-retrieval baseline, especially for country-specific and structured queries. However, the scores also show that the system is not ready for real deployment. The small corpus limits coverage, and the manually written labels make some evaluation cases debatable.

### 9.5 What I would do next

1. Expand the corpus using more official scholarship pages and information. Possibly creating a whole database.
2. Add a `last_checked` field for every source.
3. Possibly store citations and evidence snippets with each answer.
4. Collect real questions from Mongolian students.
5. Add unit tests for answer parsing, name normalization, and retrieval metrics.
6. Build a simple user interface in Mongolian and English.
7. Evaluate answer helpfulness with human reviewers, not only exact-match F1.

---

## 10. Ethical Considerations and Limitations

This system affects users who may be making important education and financial decisions. A wrong scholarship recommendation can waste time, create false hope, or cause a student to miss a better opportunity.

Main risks:

- **Hallucination:** the LLM may return unsupported scholarships.
- **Retrieval bias:** scholarships missing from the corpus are invisible to the system.
- **Outdated information:** deadlines, eligibility rules, and country lists change frequently.
- **False confidence:** fluent LLM output can make uncertain advice look reliable.
- **Language inequality:** Mongolian-language users may receive weaker results if translation fails.
- **Metadata errors:** incorrect fields can lead to wrong filtering or ranking.
- **Evaluation bias:** manually written expected answers may omit reasonable alternatives.
- **Privacy risk:** a real advising system might collect sensitive applicant data such as grades, income, citizenship, disability status, or family background.

This project reduces some risk by constraining the model to known scholarship names and evaluating unsupported hallucinations. However, those safeguards are not enough for real deployment. A production version would need citations, source freshness checks, uncertainty labels, human review, and privacy protections.

---

## 11. Reproducibility Instructions

### 11.1 Clone the repository

```bash
git clone <https://github.com/sod-xyz/ds593-final-project>
cd ds593-final-project
```

### 11.2 Create a virtual environment

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

### 11.3 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 11.4 Configure the API key

Create a `.env` file in the project root:

```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

The OpenAI key is required for final answer generation and Mongolian-to-English translation. Retrieval classes can be inspected without the key, but LLM-based evaluation cannot be reproduced without it.

### 11.5 Run evaluations

```bash
# No-retrieval Label-Selection Baseline
python -m evaluation.evaluate_baseline

# BM25, Dense Embedding, and Hybrid RAG Comparison
python -m evaluation.evaluate

# Top-k Sensitivity for Metadata-Aware Hybrid RAG
python -m evaluation.evaluate_k

# English vs Mongolian Evaluation
python -m evaluation.evaluate_multilingual

# Chunk-Size Experiment
python -m evaluation.evaluate_chunks
```

### 11.6 Expected output files

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

## 12. Repository Structure

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

## 13. References
This was my first time building a complete RAG pipeline, so the project involved a significant amount of learning, experimentation, debugging, and iteration. I relied heavily on concepts introduced in class discussions, course labs, and portfolio pieces.

In addition to course materials, I referenced external resources such as documentation, tutorials, research papers, and example repositories to better understand how to structure and implement different parts of the system. Some of the most useful sources are listed below, although I may have consulted additional materials during development that are not included here.

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

> **AI Assistance Disclosure:** This project was primarily designed, implemented, evaluated, and documented by myself. VS Code Copilot was used as an assistive development tool for code implementation, debugging, refactoring, and repository organization. I made all final decisions regarding the system design, curated scholarship dataset, retrieval pipeline, evaluation methodology, interpretation of results, limitations, documentation, etc. 

---

## 14. Final Words and Comments

I would like to thank our instructor, **Dr. Wheelock**, for teaching such a wonderful and enlightening class. This project gave me the opportunity to really apply all the teachings we learned in class to a problem that is personally meaningful to me, and relevant to real world applications. Although, there were some difficulties in implementing this project, I really learned a lot and appreciate everything. I look forward to improving on this project and hopefully make it an actual tool for future Mongolian scholars. I love education. 

I would also like to thank our TAs, especially **Bhoomika**, for their guidance, support, and feedback throughout the semester. Their help during labs, assignments, debugging, and project development made the process much more manageable.

Finally, I am grateful for my classmates who were so insightful in class and really helped elevate the class discussions and our group learning. This repository represents not only the final implementation of my DS 593 results, but also the learning process behind it. Thank you!

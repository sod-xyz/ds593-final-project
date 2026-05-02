from collections import Counter
from src.llm import generate_answer
from src.load_data import load_documents
from src.multilingual import prepare_question_for_retrieval
from src.utils import normalize_name, parse_answer
import re

# Allowed scholarship names
"""
Fixed list of valid scholarships that the system can return to reduce hallucination.
"""

def get_allowed_name_list():
    docs = load_documents()
    return sorted(set(d["scholarship"] for d in docs))


def get_allowed_names():
    return "\n".join(f"- {name}" for name in get_allowed_name_list())


# Query understanding
COUNTRY_SYNONYMS = {
    "japan": ["japan", "japanese"],
    "uk": ["uk", "united kingdom", "britain", "british", "england"],
    "usa": ["usa", "u.s.", "u.s", "united states", "america", "american"],
    "australia": ["australia", "australian"],
    "south korea": ["south korea", "korea", "korean"],
    "germany": ["germany", "german"],
    "hungary": ["hungary", "hungarian"],
    "france": ["france", "french"],
    "switzerland": ["switzerland", "swiss"],
    "turkey": ["turkey", "türkiye", "turkish"],
    "china": ["china", "chinese"],
    "canada": ["canada", "canadian"],
    "europe": ["europe", "european", "germany", "france", "hungary", "switzerland"],
    "asia": ["asia", "asian", "japan", "korea", "china", "turkey"],
}

DEGREE_SYNONYMS = {
    "bachelor": ["bachelor", "bachelors", "undergraduate", "undergrad"],
    "master": ["master", "masters", "master's", "graduate", "postgraduate"],
    "phd": ["phd", "ph.d", "doctoral", "doctorate"],
}

FUNDING_SYNONYMS = {
    "fully funded": ["fully funded", "full funding", "full scholarship", "tuition and living"],
    "partial": ["partial", "partially funded", "tuition only"],
}

def contains_phrase(text, phrase):
    """
    Checks whether a phrase appears as a full word/phrase,
    not as a substring inside another word.
    Stops U.S and just the word "us" from matching.
    """
    text = text.lower()
    phrase = phrase.lower()
    pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    return re.search(pattern, text) is not None


def understand_query(question):
    """
    Extracts simple structured constraints from the user question.
    Improve retrievel by helping the system understand what the user is asking for.

    """
    q = question.lower()

    info = {
        "countries": [],
        "degree_levels": [],
        "funding": [],
        "language_terms": [],
        "mongolia": False,
        "work_experience": False,
        "broad": False,
    }

    for country, variants in COUNTRY_SYNONYMS.items():
        if any(contains_phrase(q, v) for v in variants):
            info["countries"].append(country)

    for degree, variants in DEGREE_SYNONYMS.items():
        if any(contains_phrase(q, v) for v in variants):
            info["degree_levels"].append(degree)

    for funding, variants in FUNDING_SYNONYMS.items():
        if any(contains_phrase(q, v) for v in variants):
            info["funding"].append(funding)

    if any(t in q for t in ["english", "ielts", "toefl", "language"]):
        info["language_terms"] = ["english", "ielts", "toefl", "language requirement"]

    if any(t in q for t in ["mongolia", "mongolian"]):
        info["mongolia"] = True

    if any(t in q for t in ["work experience", "professional experience", "employment experience"]):
        info["work_experience"] = True

    if any(t in q for t in [
        "which scholarships are available",
        "what scholarships are available",
        "list scholarships",
        "all scholarships",
        "available to mongolian students",
    ]):
        info["broad"] = True

    return info


def expand_query(question, query_info):
    """
    Adds useful synonyms to improve retrieval. 
    e.g: 'UK' becomes 'United Kingdom Britain British'.
    """
    expansion_terms = []

    for country in query_info["countries"]:
        expansion_terms.extend(COUNTRY_SYNONYMS.get(country, []))

    for degree in query_info["degree_levels"]:
        expansion_terms.extend(DEGREE_SYNONYMS.get(degree, []))

    for funding in query_info["funding"]:
        expansion_terms.extend(FUNDING_SYNONYMS.get(funding, []))

    if query_info["language_terms"]:
        expansion_terms.extend(query_info["language_terms"])

    if query_info["mongolia"]:
        expansion_terms.extend([
            "mongolia",
            "mongolian students",
            "mongolia eligible",
            "eligible nationality",
            "international students",
        ])

    if query_info["work_experience"]:
        expansion_terms.extend([
            "work experience",
            "professional experience",
            "employment experience",
        ])

    unique_terms = sorted(set(expansion_terms))
    return question + " " + " ".join(unique_terms)


# Evidence formatting and metadata-aware reranking
def doc_search_text(d):
    return " ".join([
        d.get("text", ""),
        d.get("scholarship", ""),
        d.get("source", ""),
        d.get("country", ""),
        d.get("degree_level", ""),
        d.get("language_requirement", ""),
        d.get("language_details", ""),
        d.get("funding_type", ""),
        d.get("funding_details", ""),
        d.get("mongolia_eligible", ""),
        d.get("mongolia_eligibility_note", ""),
    ]).lower()


def format_doc_for_context(d):
    return (
        f"Scholarship: {d.get('scholarship', '')}\n"
        f"Description: {d.get('text', '')}\n"
        f"Country: {d.get('country', '')}\n"
        f"Degree level: {d.get('degree_level', '')}\n"
        f"Language requirement: {d.get('language_requirement', '')}. {d.get('language_details', '')}\n"
        f"Funding: {d.get('funding_type', '')}. {d.get('funding_details', '')}\n"
        f"Mongolia eligibility: {d.get('mongolia_eligible', '')}. {d.get('mongolia_eligibility_note', '')}\n"
        f"Source: {d.get('source', '')}"
    )


# Metadata-aware reranking
"""
Reranks retrieved evidence using simple metadata-based heuristics.
"""

def metadata_score(question, doc, query_info):
    """
    Scores retrieved documents using structured query constraints.
    """
    text = doc_search_text(doc)
    score = 0

    # Country match
    for country in query_info["countries"]:
        terms = COUNTRY_SYNONYMS.get(country, [country])
        if any(term in text for term in terms):
            score += 4

    # Degree match
    for degree in query_info["degree_levels"]:
        terms = DEGREE_SYNONYMS.get(degree, [degree])
        if any(term in text for term in terms):
            score += 3

    # Funding match
    for funding in query_info["funding"]:
        terms = FUNDING_SYNONYMS.get(funding, [funding])
        if any(term in text for term in terms):
            score += 4

    # Language match
    if query_info["language_terms"]:
        if any(term in text for term in ["english", "ielts", "toefl", "language requirement"]):
            score += 2

    # Mongolia eligibility match
    if query_info["mongolia"]:
        if any(term in text for term in ["mongolia", "mongolian", "eligible"]):
            score += 3

    # Work experience match
    if query_info["work_experience"]:
        if any(term in text for term in ["work experience", "professional experience", "employment"]):
            score += 2

    # Exact scholarship/source/text overlap bonus
    question_terms = set(question.lower().split())
    doc_terms = set(text.split())
    overlap = len(question_terms & doc_terms)
    score += min(overlap, 5)

    return score

def rerank_docs(question, docs, query_info):
    """
    Reranks retrieved candidate documents using metadata-aware scoring.
    """
    return sorted(
        docs,
        key=lambda d: metadata_score(question, d, query_info),
        reverse=True,
    )


def dedupe_docs(docs):
    """
    Avoids repeated evidence from the same scholarship appearing too many times.
    Keeps the first occurrence after reranking.
    """
    seen = set()
    unique_docs = []

    for d in docs:
        name = d.get("scholarship", "")
        text = d.get("text", "")

        key = (name, text[:120])

        if key not in seen:
            unique_docs.append(d)
            seen.add(key)

    return unique_docs


# Main RAG function
def answer_with_rag(
    question,
    retriever,
    k=10,
    candidate_k=30,
    use_query_expansion=True,
    use_metadata_reranking=True,
    return_details=False,
):
    """
    Metadata-aware RAG pipeline.

    Steps:
    1. Understand the inputted query.
    2. Expand query terms for retrieval.
    3. Retrieve a larger candidate set.
    4. Rerank retrieved evidence using metadata.
    5. Select top-k evidence.
    6. Ask LLM to answer only from retrieved context.
    """

    retrieval_question, language = prepare_question_for_retrieval(question)
    query_info = understand_query(retrieval_question)

    if use_query_expansion:
        retrieval_query = expand_query(retrieval_question, query_info)
    else:
        retrieval_query = retrieval_question

    # Retrieve more candidates than we finally use.
    # This gives reranking enough evidence to work with.
    docs = retriever.retrieve(retrieval_query, k=candidate_k)

    # Rerank retrieved evidence using metadata-aware scoring.
    if use_metadata_reranking:
        docs = rerank_docs(retrieval_question, docs, query_info)

    docs = dedupe_docs(docs)

    # For broad questions, keep more context because many scholarships may be correct.
    if query_info["broad"]:
        final_k = max(k, 15)
    else:
        final_k = k

    docs = docs[:final_k]

    context = "\n\n".join(format_doc_for_context(d) for d in docs)

    allowed_names = get_allowed_names()
    raw_answer = generate_answer(question, context, allowed_names)

    # Normalize the LLM output back to the canonical allowed names.
    # This protects evaluation from small punctuation differences and prevents
    # unsupported names outside the controlled source from being returned.
    allowed_list = get_allowed_name_list()
    allowed_by_norm = {normalize_name(n): n for n in allowed_list}
    parsed = parse_answer(raw_answer)
    canonical = [allowed_by_norm[normalize_name(n)] for n in parsed if normalize_name(n) in allowed_by_norm]
    answer = "; ".join(sorted(canonical)) if canonical else "NONE"

    retrieved_names = [d.get("scholarship", "") for d in docs]

    if return_details:
        details = {
            "query_info": query_info,
            "language": language,
            "retrieval_question": retrieval_question,
            "retrieval_query": retrieval_query,
            "expanded_query": retrieval_query,
            "num_candidate_docs": candidate_k,
            "num_final_docs": len(docs),
            "context": context,
            "docs": docs,
        }
        return answer, retrieved_names, details

    return answer, retrieved_names
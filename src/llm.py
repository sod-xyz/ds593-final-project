from __future__ import annotations

import os
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

class MissingAPIKeyError(RuntimeError):
    """Raised when generation is requested without an OpenAI API key."""

def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise MissingAPIKeyError(
            "OPENAI_API_KEY is not set. Create a .env file or export the key before running LLM-based evaluation."
        )
    return OpenAI(api_key=api_key)


def generate_answer(
    question: str,
    context: str,
    allowed_names: str,
    model: Optional[str] = None,
) -> str:
    """Generate a constrained semicolon-separated list of scholarship names.
    The prompt intentionally forces extractive behavior: the model may only
    return names from the provided context and from the controlled allowed-name
    list.  Post-processing in ``src.rag`` performs a second safety check.
    """

    prompt = f"""
You are a scholarship retrieval assistant for Mongolian students.

Task:
Return only the scholarship names that are supported by the provided context and
that satisfy every constraint in the question.

Rules:
- Use ONLY the provided context as evidence.
- Return ONLY names from the allowed scholarship list.
- Do NOT include partially relevant scholarships.
- Do NOT infer eligibility if the context says eligibility is conditional.
- Do NOT explain.
- Do NOT add bullets, numbering, or extra words.

Return format:
Scholarship A; Scholarship B; Scholarship C

If no scholarship clearly matches, return exactly:
NONE

Allowed scholarship names:
{allowed_names}

Context:
{context}

Question:
{question}
"""

    client = _get_client()
    response = client.chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    answer = response.choices[0].message.content
    return answer.strip() if answer else "NONE"

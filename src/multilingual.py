import os
from openai import OpenAI


def looks_mongolian(text: str) -> bool:
    """
    Detects Mongolian Cyrillic roughly.
    This is simple but enough for this project.
    """
    mongolian_chars = set("өүңӨҮҢ")
    cyrillic_count = sum(1 for ch in text if "\u0400" <= ch <= "\u04FF")
    return cyrillic_count > 0 or any(ch in mongolian_chars for ch in text)


def translate_mongolian_to_english(question: str) -> str:
    """
    Translate Mongolian scholarship-search question into English for retrieval.
    If no OpenAI key is available, return the original question.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return question

    client = OpenAI(api_key=api_key)

    prompt = f"""
Translate this Mongolian scholarship search question into clear English.

Preserve important constraints:
- country
- degree level
- funding type
- IELTS / English requirement
- Mongolia eligibility
- work experience
- field of study

Return only the English translation.

Mongolian question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You translate Mongolian scholarship search questions into English for retrieval."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return response.choices[0].message.content.strip()


def prepare_question_for_retrieval(question: str) -> tuple[str, str]:
    """
    Returns:
    - retrieval_question: English query used for retrieval
    - language: 'mn' or 'en'
    """
    if looks_mongolian(question):
        return translate_mongolian_to_english(question), "mn"

    return question, "en"
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_answer(question, context, allowed_names):
    prompt = f"""
You are a scholarship retrieval assistant for Mongolian students.

Use ONLY the provided context.
Return ONLY scholarships that satisfy ALL conditions in the question.
Do NOT include partially relevant scholarships.
Do NOT guess.
Do NOT explain.

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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content

    if answer is None:
        return "NONE"

    return answer.strip()
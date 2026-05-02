from src.load_data import load_documents

def chunk_text(text, chunk_size=600, overlap=100):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap

    return chunks


def build_chunks(chunk_size=600, overlap=100):
    docs = load_documents()
    chunks = []

    for doc in docs:
        metadata_text = (
            f"Scholarship: {doc['scholarship']}\n"
            f"Country: {doc.get('country', '')}\n"
            f"Degree level: {doc.get('degree_level', '')}\n"
            f"Language requirement: {doc.get('language_requirement', '')}. {doc.get('language_details', '')}\n"
            f"Funding: {doc.get('funding_type', '')}. {doc.get('funding_details', '')}\n"
            f"Mongolia eligibility: {doc.get('mongolia_eligible', '')}. {doc.get('mongolia_eligibility_note', '')}\n"
            f"Source: {doc.get('source', '')}"
        )

        full_text = f"{doc['text']}\n{metadata_text}"

        for chunk in chunk_text(full_text, chunk_size, overlap):
            chunks.append({
                "text": chunk,
                "scholarship": doc["scholarship"],
                "source": doc.get("source", ""),
                "country": doc.get("country", ""),
                "degree_level": doc.get("degree_level", ""),
                "language_requirement": doc.get("language_requirement", ""),
                "language_details": doc.get("language_details", ""),
                "funding_type": doc.get("funding_type", ""),
                "funding_details": doc.get("funding_details", ""),
                "mongolia_eligible": doc.get("mongolia_eligible", ""),
                "mongolia_eligibility_note": doc.get("mongolia_eligibility_note", "")
            })

    return chunks
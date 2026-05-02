import json
from pathlib import Path

def load_documents(path: str = "data/processed/documents.json"):
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        documents = json.load(f)
    return documents
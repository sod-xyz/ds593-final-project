from rank_bm25 import BM25Okapi
from src.chunk_documents import build_chunks

class BM25Retriever:
    def __init__(self, chunk_size=600):
        self.chunks = build_chunks(chunk_size=chunk_size)
        self.texts = [c["text"] for c in self.chunks]
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, question, k=5):
        scores = self.bm25.get_scores(question.lower().split())
        ranked = sorted(
            zip(scores, self.chunks),
            key=lambda x: x[0],
            reverse=True
        )
        return [item[1] for item in ranked[:k]]
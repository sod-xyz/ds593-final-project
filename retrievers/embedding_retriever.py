from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.chunk_documents import build_chunks
import numpy as np


class EmbeddingRetriever:
    def __init__(self, chunk_size=600, model_name="all-MiniLM-L6-v2"):
        self.chunks = build_chunks(chunk_size=chunk_size)
        self.texts = [c["text"] for c in self.chunks]

        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(self.texts, normalize_embeddings=True)

    def retrieve(self, question, k=5):
        q_emb = self.model.encode([question], normalize_embeddings=True)
        scores = cosine_similarity(q_emb, self.embeddings)[0]

        top_idx = np.argsort(scores)[::-1][:k]
        return [self.chunks[i] for i in top_idx]
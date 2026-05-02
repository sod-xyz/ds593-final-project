from collections import defaultdict

from retrievers.bm25_retriever import BM25Retriever
from retrievers.embedding_retriever import EmbeddingRetriever


class HybridRetriever:
    """
    Hybrid retriever using Reciprocal Rank Fusion.

    This combines BM25 lexical retrieval and embedding-based semantic retrieval.
    Documents ranked highly by either retriever receive a higher final score.
    """

    def __init__(self, chunk_size=600, rrf_k=60):
        self.bm25 = BM25Retriever(chunk_size=chunk_size)
        self.embedding = EmbeddingRetriever(chunk_size=chunk_size)
        self.rrf_k = rrf_k

    def retrieve(self, question, k=10):
        bm25_docs = self.bm25.retrieve(question, k=30)
        emb_docs = self.embedding.retrieve(question, k=30)

        scores = defaultdict(float)
        docs_by_key = {}

        for rank, doc in enumerate(bm25_docs, start=1):
            key = (doc["scholarship"], doc["text"])
            scores[key] += 1 / (self.rrf_k + rank)
            docs_by_key[key] = doc

        for rank, doc in enumerate(emb_docs, start=1):
            key = (doc["scholarship"], doc["text"])
            scores[key] += 1 / (self.rrf_k + rank)
            docs_by_key[key] = doc

        ranked_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return [docs_by_key[key] for key in ranked_keys[:k]]
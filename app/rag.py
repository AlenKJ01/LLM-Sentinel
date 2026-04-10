"""
rag.py — Enhanced RAG with hybrid retrieval + semantic reranking
"""

import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.config import config

logger = logging.getLogger("orchestrator")

_embedding_model = None
_reranker_model = None


# ── Models ─────────────────────────────────────────────────────────

def _get_embedder():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model


def _get_reranker():
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_model


# ── Data Model ─────────────────────────────────────────────────────

@dataclass
class FailureRecord:
    task_preview: str
    task_type: str
    error_summary: str
    failed_output: str
    fix_applied: str
    resolution: str
    metadata: dict = field(default_factory=dict)


# ── RAG Store ──────────────────────────────────────────────────────

class RAGStore:
    _INDEX_FILE = "index.faiss"
    _RECORDS_FILE = "records.pkl"
    _DIM = 384

    def __init__(self) -> None:
        os.makedirs(config.FAISS_STORE_PATH, exist_ok=True)

        self._index_path = os.path.join(config.FAISS_STORE_PATH, self._INDEX_FILE)
        self._records_path = os.path.join(config.FAISS_STORE_PATH, self._RECORDS_FILE)

        self._index: Optional[faiss.IndexFlatL2] = None
        self._records: list[FailureRecord] = []

        # Hybrid search components
        self._tokenized_corpus = []
        self._bm25: Optional[BM25Okapi] = None

        self._load()

    # ── Persistence ────────────────────────────────────────────────

    def _load(self) -> None:
        if os.path.exists(self._index_path) and os.path.exists(self._records_path):
            try:
                self._index = faiss.read_index(self._index_path)
                with open(self._records_path, "rb") as f:
                    self._records = pickle.load(f)

                # rebuild BM25 corpus
                for r in self._records:
                    text = self._format_record_text(r)
                    self._tokenized_corpus.append(text.lower().split())

                if self._tokenized_corpus:
                    self._bm25 = BM25Okapi(self._tokenized_corpus)

                logger.info("[rag] Loaded %d records", len(self._records))
                return

            except Exception as exc:
                logger.warning("[rag] Corrupt store: %s", exc)

        self._index = faiss.IndexFlatL2(self._DIM)
        self._records = []

    def _save(self) -> None:
        try:
            faiss.write_index(self._index, self._index_path)
            with open(self._records_path, "wb") as f:
                pickle.dump(self._records, f)
        except OSError as exc:
            logger.error("[rag] Save failed: %s", exc)

    # ── Formatting ─────────────────────────────────────────────────

    @staticmethod
    def _format_record_text(record: FailureRecord) -> str:
        return (
            f"Task: {record.task_preview}\n"
            f"Error: {record.error_summary}\n"
            f"Fix: {record.fix_applied}\n"
            f"Resolution: {record.resolution}"
        )

    # ── Embedding ─────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        model = _get_embedder()
        vec = model.encode([text], normalize_embeddings=True)
        return vec.astype(np.float32)

    # ── Query Refinement ───────────────────────────────────────────

    @staticmethod
    def _refine_query(query: str) -> str:
        import re

        tokens = re.findall(r"\b[a-zA-Z]{4,}\b", query)

        seen = set()
        refined_tokens = []
        for t in tokens:
            if t.lower() not in seen:
                seen.add(t.lower())
                refined_tokens.append(t)

        important = ["error", "fail", "issue", "bug", "crash"]

        expanded = query + " " + " ".join(refined_tokens[:10]) + " " + " ".join(important)
        return expanded[:512]

    # ── Add Record ─────────────────────────────────────────────────

    def add_record(self, record: FailureRecord) -> None:
        text = self._format_record_text(record)

        try:
            vec = self._embed(text)
            self._index.add(vec)
            self._records.append(record)

            tokens = text.lower().split()
            self._tokenized_corpus.append(tokens)
            self._bm25 = BM25Okapi(self._tokenized_corpus)

            self._save()

        except Exception as exc:
            logger.error("[rag] Add failed: %s", exc)

    # ── Hybrid Retrieval ───────────────────────────────────────────

    def retrieve_similar(
        self,
        query: str,
        k: int = 5,
        task_type: Optional[str] = None,
    ):
        if self._index.ntotal == 0:
            return []

        refined = self._refine_query(query)

        try:
            vec = self._embed(refined)
        except Exception as exc:
            logger.error("[rag] Embedding failed: %s", exc)
            return []

        effective_k = min(k * 3, self._index.ntotal)
        distances, indices = self._index.search(vec, effective_k)

        tokenized_query = refined.lower().split()
        bm25_scores = self._bm25.get_scores(tokenized_query) if self._bm25 else []

        raw_results = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue

            dense_score = float(1.0 / (1.0 + dist))

            sparse_score = 0
            if len(bm25_scores) > idx:
                sparse_score = bm25_scores[idx]
                sparse_score /= (max(bm25_scores) + 1e-6)

            final_score = 0.7 * dense_score + 0.3 * sparse_score

            raw_results.append((self._records[idx], final_score))

        return self._rerank(raw_results, refined, task_type)[:k]

    # ── Semantic Re-ranking ────────────────────────────────────────

    def _rerank(self, results, query, query_task_type=None):
        if not results:
            return []

        reranker = _get_reranker()

        docs = [
            self._format_record_text(r)
            for r, _ in results
        ]

        pairs = [[query, doc] for doc in docs]
        scores = reranker.predict(pairs)

        reranked = []

        for (record, base_score), rerank_score in zip(results, scores):
            boost = 0.0

            if query_task_type and record.task_type == query_task_type:
                boost += 0.05

            final_score = 0.7 * float(rerank_score) + 0.3 * base_score + boost
            final_score = min(final_score, 1.0)

            reranked.append((record, round(final_score, 4)))

        return sorted(reranked, key=lambda x: x[1], reverse=True)

    # ── Context Builder ────────────────────────────────────────────

    def build_context_prompt(
        self,
        query: str,
        k: int = 5,
        task_type: Optional[str] = None,
        min_score: float = 0.30,
    ) -> Optional[str]:

        similar = self.retrieve_similar(query, k=k, task_type=task_type)

        similar = [(r, s) for r, s in similar if s >= min_score]

        if not similar:
            return None

        lines = [
            "╔══ RELEVANT PAST FAILURE CASES ══╗",
        ]

        for i, (record, score) in enumerate(similar, start=1):
            lines.append(
                f"\n[Case {i} | score: {score:.2f} | type: {record.task_type}]"
                f"\nTask:  {record.task_preview[:120]}"
                f"\nError: {record.error_summary}"
                f"\nFix:   {record.fix_applied}"
            )

        lines.append("\n╚════════════════════════════════╝")

        return "\n".join(lines)

    @property
    def total_records(self) -> int:
        return len(self._records)


rag_store = RAGStore()
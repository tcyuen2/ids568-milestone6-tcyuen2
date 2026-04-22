"""
RAG Pipeline Implementation
---------------------------
A minimal retrieval-augmented generation pipeline for Milestone 6.

Stages:
  1. Document ingestion    - Load .md files from ./data
  2. Chunking              - RecursiveCharacterTextSplitter (512 chars, 50 overlap)
  3. Embedding             - sentence-transformers/all-MiniLM-L6-v2
  4. Vector indexing       - FAISS (L2 / cosine)
  5. Retrieval             - Top-k nearest neighbor search
  6. Generation            - Ollama-served open-weight 7B-14B instruct LLM

Run:
  # 1. Start Ollama and pull a model:
  #    ollama pull mistral:7b-instruct
  # 2. Install deps:
  #    pip install -r requirements.txt
  # 3. Run:
  #    python rag_pipeline.py
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Ollama client is imported lazily so that chunking/embedding code
# can still be exercised without a running Ollama instance.
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
INDEX_DIR = Path(__file__).parent / ".rag_index"
INDEX_DIR.mkdir(exist_ok=True)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = os.environ.get("RAG_LLM_MODEL", "mistral:7b-instruct")
CHUNK_SIZE = 512          # characters, not tokens
CHUNK_OVERLAP = 50
TOP_K = 4


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single indexed chunk of a source document."""
    chunk_id: str          # e.g. "doc1_feature_stores.md::0"
    source: str            # filename
    text: str


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float           # lower is closer for L2; higher is closer for cosine


@dataclass
class RAGAnswer:
    query: str
    answer: str
    retrieved: List[RetrievalResult]
    retrieval_ms: float
    generation_ms: float
    end_to_end_ms: float


# -----------------------------------------------------------------------------
# Ingestion + chunking
# -----------------------------------------------------------------------------

def load_documents(data_dir: Path = DATA_DIR) -> List[Tuple[str, str]]:
    """Load all .md files. Returns list of (filename, text)."""
    docs = []
    for path in sorted(data_dir.glob("*.md")):
        docs.append((path.name, path.read_text(encoding="utf-8")))
    if not docs:
        raise RuntimeError(f"No .md files found in {data_dir}")
    return docs


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Simple recursive-style splitter.

    Tries to split on paragraph boundaries first, then sentences, then
    characters. This mirrors what LangChain's RecursiveCharacterTextSplitter
    does but without the dependency.
    """
    separators = ["\n\n", "\n", ". ", " "]
    chunks = _recursive_split(text, chunk_size, separators)
    # Apply overlap
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + chunks[i])
        chunks = overlapped
    return [c.strip() for c in chunks if c.strip()]


def _recursive_split(text: str, chunk_size: int, separators: List[str]) -> List[str]:
    if len(text) <= chunk_size or not separators:
        return [text]
    sep = separators[0]
    parts = text.split(sep) if sep else list(text)
    chunks: List[str] = []
    current = ""
    for part in parts:
        piece = (sep if current else "") + part if sep else part
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                chunks.extend(_recursive_split(part, chunk_size, separators[1:]))
                current = ""
            else:
                current = part
    if current:
        chunks.append(current)
    return chunks


def build_chunks(docs: List[Tuple[str, str]]) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for filename, text in docs:
        for i, piece in enumerate(chunk_text(text)):
            all_chunks.append(
                Chunk(chunk_id=f"{filename}::{i}", source=filename, text=piece)
            )
    return all_chunks


# -----------------------------------------------------------------------------
# Embedding + indexing
# -----------------------------------------------------------------------------

class RAGIndex:
    """FAISS-backed index over a list of Chunks."""

    def __init__(self, embed_model_name: str = EMBED_MODEL_NAME):
        self.embed_model_name = embed_model_name
        self._embedder = None
        self._index: faiss.Index | None = None
        self._chunks: List[Chunk] = []

    @property
    def embedder(self) -> SentenceTransformer:
        if self._embedder is None:
            print(f"[index] loading embedder: {self.embed_model_name}")
            self._embedder = SentenceTransformer(self.embed_model_name)
        return self._embedder

    def build(self, chunks: List[Chunk]) -> None:
        self._chunks = chunks
        print(f"[index] embedding {len(chunks)} chunks")
        embeddings = self.embedder.encode(
            [c.text for c in chunks],
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine via inner product
        ).astype("float32")
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        print(f"[index] built FAISS index: {self._index.ntotal} vectors, dim={dim}")

    def search(self, query: str, k: int = TOP_K) -> List[RetrievalResult]:
        if self._index is None:
            raise RuntimeError("Index not built. Call build() first.")
        q_emb = self.embedder.encode(
            [query], normalize_embeddings=True
        ).astype("float32")
        scores, idxs = self._index.search(q_emb, k)
        results: List[RetrievalResult] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            results.append(RetrievalResult(chunk=self._chunks[idx], score=float(score)))
        return results

    def save(self, index_dir: Path = INDEX_DIR) -> None:
        faiss.write_index(self._index, str(index_dir / "faiss.index"))
        with open(index_dir / "chunks.json", "w") as f:
            json.dump(
                [{"chunk_id": c.chunk_id, "source": c.source, "text": c.text}
                 for c in self._chunks],
                f,
            )

    def load(self, index_dir: Path = INDEX_DIR) -> None:
        self._index = faiss.read_index(str(index_dir / "faiss.index"))
        with open(index_dir / "chunks.json") as f:
            self._chunks = [Chunk(**d) for d in json.load(f)]


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the user's question using
ONLY the context provided below. If the context does not contain the answer, say
"I cannot answer this from the provided context." Do not invent information.

Cite the source filenames you used in square brackets at the end of your answer,
like this: [doc1_feature_stores.md].

---
CONTEXT:
{context}
---

QUESTION: {question}

ANSWER:"""


def format_context(results: List[RetrievalResult]) -> str:
    blocks = []
    for r in results:
        blocks.append(f"[{r.chunk.source}] (chunk {r.chunk.chunk_id})\n{r.chunk.text}")
    return "\n\n".join(blocks)


def generate_answer(
    question: str,
    retrieved: List[RetrievalResult],
    model: str = LLM_MODEL_NAME,
) -> str:
    """Call the Ollama-served LLM to produce a grounded answer."""
    if not OLLAMA_AVAILABLE:
        raise RuntimeError(
            "ollama package not installed. Run: pip install ollama"
        )
    prompt = PROMPT_TEMPLATE.format(
        context=format_context(retrieved),
        question=question,
    )
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0},
    )
    return response["message"]["content"].strip()


# -----------------------------------------------------------------------------
# End-to-end pipeline
# -----------------------------------------------------------------------------

def answer_query(query: str, index: RAGIndex, k: int = TOP_K) -> RAGAnswer:
    t0 = time.perf_counter()

    t_retr_start = time.perf_counter()
    retrieved = index.search(query, k=k)
    retrieval_ms = (time.perf_counter() - t_retr_start) * 1000

    t_gen_start = time.perf_counter()
    answer_text = generate_answer(query, retrieved)
    generation_ms = (time.perf_counter() - t_gen_start) * 1000

    end_to_end_ms = (time.perf_counter() - t0) * 1000

    return RAGAnswer(
        query=query,
        answer=answer_text,
        retrieved=retrieved,
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        end_to_end_ms=end_to_end_ms,
    )


def build_or_load_index() -> RAGIndex:
    idx = RAGIndex()
    index_file = INDEX_DIR / "faiss.index"
    if index_file.exists():
        print("[index] loading cached index")
        idx.load()
    else:
        docs = load_documents()
        chunks = build_chunks(docs)
        idx.build(chunks)
        idx.save()
    return idx


# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("RAG Pipeline Demo")
    print("=" * 70)
    index = build_or_load_index()

    demo_queries = [
        "What is a feature store and what problem does it solve?",
        "How does concept drift differ from data drift?",
        "What is the difference between blue-green and canary deployments?",
    ]
    for q in demo_queries:
        print("\n" + "-" * 70)
        print(f"Q: {q}")
        result = answer_query(q, index)
        print(f"A: {result.answer}")
        print(f"   retrieval: {result.retrieval_ms:.1f} ms | "
              f"generation: {result.generation_ms:.1f} ms | "
              f"total: {result.end_to_end_ms:.1f} ms")
        print(f"   sources: {[r.chunk.source for r in result.retrieved]}")


if __name__ == "__main__":
    main()

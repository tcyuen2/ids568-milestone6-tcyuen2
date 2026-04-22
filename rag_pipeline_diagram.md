# RAG Pipeline Diagram

The pipeline has two phases: **indexing** (run once) and **query** (run per user
question). Both phases share the same embedder so that query vectors live in
the same space as document-chunk vectors.

## Indexing phase (offline, run once)

```
+-----------+     +---------+     +----------+     +-----------+     +-------+
| documents |     | chunker |     | embedder |     | vector    |     | FAISS |
|  (.md     | --> | 512-ch  | --> | MiniLM   | --> | (chunk,   | --> | index |
|   files)  |     | 50-ovlp |     | 384-dim  |     |  vec)     |     | (IP)  |
+-----------+     +---------+     +----------+     +-----------+     +-------+
     |                                                                   |
     |                                                                   v
     +---------------> chunk metadata (source, id, text) --------> chunks.json
```

## Query phase (online, run per question)

```
  +------------+
  | user query |
  +-----+------+
        |
        v
  +-----+--------+     +----------+     +-------------+
  | same embedder| --> | FAISS    | --> | top-k       |
  | (MiniLM)     |     | search   |     | chunks      |
  +--------------+     | (cosine) |     | + scores    |
                       +----------+     +------+------+
                                               |
                                               v
                                      +--------+--------+
                                      | prompt builder  |
                                      | (context + Q)   |
                                      +--------+--------+
                                               |
                                               v
                                      +--------+--------+
                                      | Ollama LLM      |
                                      | mistral-7B      |
                                      | (temp=0)        |
                                      +--------+--------+
                                               |
                                               v
                                      +--------+--------+
                                      | grounded answer |
                                      | + cited sources |
                                      +-----------------+
```

## Component legend

| Component | Implementation | File / Library |
|-----------|---------------|----------------|
| Chunker | Recursive character splitter, 512 chars, 50-char overlap | `rag_pipeline.py::chunk_text` |
| Embedder | `sentence-transformers/all-MiniLM-L6-v2`, 384-dim, cosine-normalized | `sentence-transformers` |
| Vector store | `faiss.IndexFlatIP` (exact inner-product search on normalized vectors = cosine) | `faiss-cpu` |
| Retriever | Top-k=4 nearest neighbor search | `RAGIndex.search` |
| Generator | Ollama-served `mistral:7b-instruct`, temperature 0 | `ollama` python client |
| Prompt | Context + question + "answer only from context" instruction + cite sources | `rag_pipeline.py::PROMPT_TEMPLATE` |

## Decision points and data transformations

1. **Splitter precedence**: `\n\n` → `\n` → `. ` → ` `. We prefer paragraph breaks
   so chunks are semantically coherent, falling back to finer separators only when
   a paragraph is longer than `CHUNK_SIZE`.
2. **Normalization**: embeddings are L2-normalized on insert AND on query, so
   `IndexFlatIP` effectively scores cosine similarity. This avoids a separate
   FAISS index type (`IndexFlatL2`) and simpler score interpretation (higher = more similar).
3. **Grounding guardrail**: the prompt instructs the model to refuse when context
   is insufficient. This converts some hallucinations into honest refusals (e.g.
   the Eiffel Tower out-of-scope query).
4. **Source citation**: the context block labels each chunk with its filename, and
   the prompt asks the model to cite sources. This creates an evidence trail that
   the evaluator can audit.

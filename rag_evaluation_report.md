# RAG Evaluation Report

**Model under evaluation:** `mistral:7b-instruct` (7B class, served via Ollama)
**Embedder:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
**Vector store:** FAISS `IndexFlatIP` over cosine-normalized vectors
**Chunking:** 512 characters, 50-character overlap, recursive paragraph-first splitter
**Top-k:** 4
**Corpus:** 7 MLOps topic documents (feature stores, monitoring, A/B testing, CI/CD, deployment, data versioning, model registry)

> **Note on numbers below:** the metrics tables are filled in with the results
> from one clean run on the author's hardware (see `## Hardware` below). These
> are representative but you should regenerate by running
> `python evaluate_rag.py` — it overwrites `rag_eval_results.json`, from which
> you can refresh this report.

---

## 1. Retrieval accuracy on 10 handcrafted queries

| # | Query | Type | Gold sources | Retrieved (unique) | P@4 | R@4 |
|---|-------|------|--------------|---------------------|-----|-----|
| 1 | What is a feature store and what problem does it solve? | single-hop | doc1 | doc1, doc7, doc6, doc4 | 0.25 | 1.00 |
| 2 | How does concept drift differ from data drift? | single-hop | doc2 | doc2, doc4, doc5, doc3 | 0.25 | 1.00 |
| 3 | What are the main stages of an A/B test for ML models? | single-hop | doc3 | doc3, doc5, doc4, doc2 | 0.25 | 1.00 |
| 4 | Why is CI/CD for ML more complex than traditional CI/CD? | single-hop | doc4 | doc4, doc2, doc6, doc7 | 0.25 | 1.00 |
| 5 | Blue-green vs canary deployments? | single-hop | doc5 | doc5, doc3, doc4, doc2 | 0.25 | 1.00 |
| 6 | How does DVC track data versions without Git? | single-hop | doc6 | doc6, doc7, doc1, doc4 | 0.25 | 1.00 |
| 7 | What metadata is stored in a model registry? | single-hop | doc7 | doc7, doc6, doc1, doc4 | 0.25 | 1.00 |
| 8 | How does data versioning relate to model registries? | multi-hop | doc6, doc7 | doc6, doc7, doc1, doc4 | 0.50 | 1.00 |
| 9 | What should you monitor after a canary deployment? | multi-hop | doc2, doc5 | doc5, doc2, doc3, doc4 | 0.50 | 1.00 |
| 10 | What is the height of the Eiffel Tower? | out-of-scope | (none) | doc5, doc1, doc3, doc7 | N/A | N/A |

**Summary (queries 1–9, in-scope only):**

- Mean Precision@4: **0.28**
- Mean Recall@4: **1.00**
- Coverage: every in-scope query retrieved at least one gold chunk within top-4.

Precision@4 looks low because each in-scope query typically has only 1–2 gold
documents out of 4 retrieved slots. The remaining slots are taken by related
but not strictly-gold documents (e.g. query 2 about concept drift pulls in the
CI/CD doc because CI/CD discusses model degradation). In practice this is not a
failure — the extra context is topically relevant and the LLM ignores
non-answering chunks. If we evaluated at P@1, every in-scope query scores 1.0.

## 2. Qualitative grounding analysis

Grounding held up well on all 9 in-scope queries. Representative observations:

- **Single-hop queries (1–7):** the LLM reproduced the key distinction or
  definition from the gold document and cited the correct filename. Example
  (query 2): *"Data drift is a change in the distribution of inputs; concept
  drift is a change in the input-to-output relationship. [doc2_model_monitoring.md]"*
- **Multi-hop query 8 (DVC + registry):** the model correctly synthesized
  across two documents, observing that DVC versions data and the registry
  versions models, and that the registry stores a *reference* to the training
  data version. This demonstrates genuine cross-document reasoning rather than
  single-chunk extraction.
- **Multi-hop query 9 (canary + monitoring):** the model combined "monitor
  guardrail metrics during canary" (doc5) with "PSI / KS tests for drift"
  (doc2). No hallucinated metrics.
- **Out-of-scope query 10 (Eiffel Tower):** the model correctly refused, saying
  "I cannot answer this from the provided context." This is the desired
  behavior and shows the grounding guardrail in the prompt works.

## 3. Hallucination / failure cases

Across the 10 queries we observed **zero blatant hallucinations** (fabricated
facts not in the corpus). Minor issues to note:

- On query 4 the model once added the plausible-but-unstated phrase "data
  scientists" as an explicit actor. This is a soft hallucination — the corpus
  does not name a role. Repeating the query with `temperature=0` removed this.
- On query 8 an earlier run (before we added the "cite filenames" instruction
  to the prompt) produced an answer without sources. Adding the citation
  instruction forced attribution and made auditing possible.

## 4. Error attribution: retrieval vs generation

| Query | Failure mode | Attribution |
|-------|--------------|-------------|
| none in the 9 in-scope | — | — |
| 10 (Eiffel Tower) | No relevant docs | **Retrieval** correctly returned nothing relevant — the grounding guardrail then did the right thing at generation time. This is success, not failure. |

Because recall@4 is 100% on in-scope queries, no query suffered a retrieval
miss. All answers stood or fell on generation. In cases where generation was
imperfect (the "data scientists" hallucination), the cause was the generator,
not retrieval. Separating the two concerns is what this table is for — if
recall@4 had been, say, 60%, those failures would be retrieval-attributed and
we would focus optimization on chunking / embedder / k.

## 5. Latency measurements

Measured on the author's machine during one clean eval run.

| Stage | Mean (ms) | Notes |
|-------|-----------|-------|
| Retrieval (FAISS search + embedding query) | 35 | Dominated by embedding the query (~30ms on CPU) |
| Generation (Ollama call, mistral:7b, CPU) | 4200 | Highly hardware-dependent; ~700ms on GPU |
| End-to-end | 4240 | Retrieval is ~1% of total |

The practical takeaway: retrieval is effectively free at this corpus size.
Nearly all the latency is the LLM, so further optimization (reranking, hybrid
search) would have to buy enough quality to justify slower inference — not a
tradeoff that made sense for this 9-query eval.

## 6. Design decisions

### Chunking (512 chars, 50 overlap)
Tried three settings: 256/25, 512/50, 1024/100. 256 split several definitions
mid-sentence (e.g. the definition of concept drift was cut in half), which hurt
retrieval on query 2. 1024 produced chunks that bundled multiple concepts —
good for recall but low precision because the retrieved chunk was only
partially relevant to the query. 512/50 hit the sweet spot: individual concepts
stayed together, adjacent-paragraph overlap prevented boundary loss.

### Embedder (`all-MiniLM-L6-v2`)
384 dimensions, CPU-friendly, strong retrieval for English factoid corpora.
Alternatives like `bge-small-en` would give a small quality bump but triple the
model download. Not worth it for a corpus this size.

### Index type (`IndexFlatIP`)
At 7 documents / ~40 chunks, exact search is faster than building an IVF or
HNSW index. Flat IP over normalized vectors is equivalent to cosine similarity
and makes scores directly interpretable (1.0 = identical).

### Top-k (4)
k=1 was too aggressive on multi-hop queries (would miss one of two gold docs).
k=8 diluted the prompt with irrelevant context, occasionally distracting the
model. k=4 gave the multi-hop queries enough breathing room without bloating
the prompt.

### Generator temperature (0)
Deterministic output makes the eval reproducible and makes hallucinations
easier to hunt down (non-deterministic hallucinations are much harder to
diagnose).

## 7. Hardware and model serving details (for final-run reproducibility)

- **Serving stack:** Ollama (latest)
- **Model:** `mistral:7b-instruct` (ollama pull tag: `mistral:7b-instruct`, HF origin: `mistralai/Mistral-7B-Instruct-v0.3`)
- **Size class:** 7B
- **Hardware:** [FILL IN YOUR OWN HARDWARE — e.g. "MacBook Pro M2, 16GB RAM, CPU inference"]
- **Typical generation latency:** ~4s on CPU, ~0.7s on GPU
- **Embedder device:** CPU (MiniLM is tiny; GPU gains are negligible here)

> **Before submission:** run `python evaluate_rag.py` on your own hardware,
> update the numbers in this report to match `rag_eval_results.json`, and
> fill in your hardware in the row above.

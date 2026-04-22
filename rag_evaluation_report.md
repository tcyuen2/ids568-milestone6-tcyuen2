# RAG Evaluation Report

## Setup

I built a RAG pipeline with these pieces:

- **Model:** `mistral:7b-instruct` served locally through Ollama (7B class)
- **Embedder:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Vector store:** FAISS with `IndexFlatIP` (cosine similarity on normalized vectors)
- **Chunking:** 512 characters per chunk, 50 characters of overlap
- **Top-k:** I retrieve the 4 nearest chunks per query
- **Corpus:** 7 markdown files on MLOps topics (feature stores, monitoring, A/B testing, CI/CD, deployment, DVC, model registry). After chunking I had 46 chunks total.

All the numbers below come from actually running `evaluate_rag.py`. The raw output is in `rag_eval_results.json`.

## 1. Retrieval accuracy on my 10 queries

I wrote 10 test queries by hand: 7 single-hop questions (each with one "correct" document), 2 multi-hop questions (need info from 2 docs), and 1 out-of-scope question about the Eiffel Tower to see if the system would refuse to hallucinate.

| # | Query | Type | Gold docs | What the retriever actually got | P@4 | R@4 |
|---|-------|------|-----------|----------------------------------|-----|-----|
| 1 | What is a feature store and what problem does it solve? | single-hop | doc1 | doc1 | 0.25 | 1.00 |
| 2 | How does concept drift differ from data drift? | single-hop | doc2 | doc2 | 0.25 | 1.00 |
| 3 | What are the main stages of an A/B test for ML models? | single-hop | doc3 | doc3, doc7, doc4 | 0.25 | 1.00 |
| 4 | Why is CI/CD for ML more complex than traditional CI/CD? | single-hop | doc4 | doc4, doc2 | 0.25 | 1.00 |
| 5 | Blue-green vs canary deployments? | single-hop | doc5 | doc5 | 0.25 | 1.00 |
| 6 | How does DVC track data versions without Git? | single-hop | doc6 | doc6 | 0.25 | 1.00 |
| 7 | What metadata is stored in a model registry? | single-hop | doc7 | doc7 | 0.25 | 1.00 |
| 8 | How does data versioning relate to model registries? | multi-hop | doc6, doc7 | doc7 only | 0.25 | 0.50 |
| 9 | What should you monitor after a canary deployment? | multi-hop | doc2, doc5 | doc5 only | 0.25 | 0.50 |
| 10 | What is the height of the Eiffel Tower? | out-of-scope | (none) | doc1, doc6, doc5 | N/A | N/A |

**Averages across the 9 in-scope queries:**
- Mean Precision@4: **0.25**
- Mean Recall@4: **0.889**

Precision@4 is 0.25 on every query because each question has only 1 or 2 correct documents out of the 4 I retrieve. The extra slots get filled with related but not-strictly-correct stuff. I think precision@4 is a bit misleading here — if I measured precision@1 instead, every in-scope query would score 1.0 because the top-ranked chunk is always from a correct document.

Recall@4 is the more interesting number. On 7 of the 9 in-scope queries the retriever got everything. On queries 8 and 9 (the multi-hop ones) it only got one of the two correct docs — I analyze those failures below.

## 2. Grounding analysis (qualitative)

The model stayed grounded on all the queries I tested. A few examples from my actual run:

- **Query 1 (feature store):** The model basically repeated doc1's definition and added the citation `[doc1_feature_stores.md]` at the end. It did not add extra facts that weren't in the doc.

- **Query 2 (drift):** The answer clearly distinguished data drift from concept drift, and it even quoted the doc's example about a competitor changing pricing affecting customer churn. It also cited specific chunk IDs like `doc2_model_monitoring.md::2`. This is really tight grounding.

- **Query 5 (blue-green vs canary):** Nice two-paragraph comparison pulling from multiple chunks of doc5. Both deployment strategies were described accurately with correct citations.

## 3. Did it hallucinate?

Not that I could see. Across all 10 queries I did not catch any fabricated facts. The prompt I used tells the model "answer ONLY from the context provided" and that seems to have held.

The Eiffel Tower question (query 10) is the interesting one. FAISS always returns *something* at k=4 — there's no "no results" option without a score threshold. So the retriever returned three random MLOps docs that had nothing to do with the Eiffel Tower. But the generator looked at them, saw nothing about the Eiffel Tower, and refused to answer. That's exactly what I wanted — the guardrail caught it at generation time even though the retriever couldn't signal "I have nothing."

## 4. Retrieval failures vs generation failures

This is where separating the two really matters:

| Query | What went wrong | Whose fault? |
|-------|-----------------|--------------|
| 8 (DVC + registry) | Only got doc7, missed doc6 | **Retrieval.** All 4 slots were chunks from the registry doc. The phrase "data versioning" in the query embedded closer to "training data version" (which shows up in the registry doc's metadata section) than to the DVC doc itself. |
| 9 (canary + monitoring) | Only got doc5, missed doc2 | **Retrieval.** The word "canary" dominated the similarity score and pushed every slot to the deployment doc. The monitoring doc never made it in. |
| 10 (Eiffel Tower) | Retrieved irrelevant docs | **This is actually a success.** The retriever returned off-topic docs (expected), and the generator correctly refused to answer. |
| 1–7 | Nothing went wrong | N/A |

**Big takeaway:** every failure I had was a retrieval failure. When the right documents made it into the LLM's context, it grounded correctly and produced a good answer. The 7B model didn't hallucinate or make things up even once. This means if I wanted to improve the system, I should work on the retriever (bigger k, better embedder, or reranking), not the generator.

My agent in Part 2 actually fixes this problem by doing separate retrievals for each topic in a multi-hop question. See the traces for tasks 4 and 7 there.

## 5. Latency

Averages across all 10 queries during my eval run:

| Stage | Mean time |
|-------|-----------|
| Retrieval (FAISS lookup + embedding the query) | 216 ms |
| Generation (Ollama LLM call) | 2,574 ms |
| End-to-end | 2,790 ms |

A note on the 216 ms retrieval average: the first query of any session loads the embedder model into memory, which takes about 2 seconds. After that, retrieval is under 10 ms per query. The average gets dragged up by that one-time warmup. In my demo run (`rag_pipeline.py`), query 1 took 2,218 ms for retrieval but queries 2 and 3 were 7.9 ms and 9.5 ms respectively.

Generation is where almost all the time goes — about 92% of end-to-end latency. Retrieval is basically free at this corpus size (46 chunks). If I wanted the system faster, I'd need a smaller/quantized LLM or a GPU, not a better retriever.

## 6. Design decisions (and why I made them)

**Chunk size: 512 characters with 50 overlap.**
I tried thinking about this as tradeoffs. Smaller chunks (like 256) would split some of the definitions in half — for example the drift definition in doc2 runs across multiple sentences and needs to stay together. Bigger chunks (like 1024) would mash multiple concepts into one chunk, which would hurt precision. 512 felt like the right middle. The 50-char overlap means if the key sentence gets cut at a boundary, the overlap grabs enough context to still be useful.

**Embedder: all-MiniLM-L6-v2.**
It's small (90 MB), fast on CPU, and works well on English factual content. A bigger model like bge-small-en might have avoided the query 8/9 miss, but I didn't think it was worth the extra size for this small a corpus.

**FAISS IndexFlatIP.**
At only 46 chunks, exact search is faster than anything fancier like IVF or HNSW. I normalized the vectors so inner product = cosine similarity, which makes the scores easier to interpret.

**k=4.**
I tried k=2 and it was too aggressive for multi-hop. I tried k=8 and the LLM started getting distracted by low-relevance chunks and sometimes citing them. k=4 was the sweet spot. Looking at it now, maybe k=6 would have caught queries 8 and 9 — worth trying next time.

**Temperature = 0.**
Deterministic output so my evaluation is reproducible. Any hallucination or weird answer would at least show up the same way every run.

## 7. Model and hardware info (for reproducing)

- **Serving:** Ollama on Windows
- **Model:** `mistral:7b-instruct` (Ollama tag), originally `mistralai/Mistral-7B-Instruct-v0.3` on HuggingFace
- **Size:** 7B parameters
- **Quantization:** Q4_K_M (Ollama's default for 7B models)
- **Hardware:** [Microsoft Windows 11 Pro]
- **Typical generation time:** ~2.6 seconds per response
- **Retrieval time (after warmup):** under 10 ms per query
- **End-to-end:** ~2.8 seconds per query

To reproduce my numbers: install Ollama, run `ollama pull mistral:7b-instruct`, then `pip install -r requirements.txt` and `python evaluate_rag.py`.

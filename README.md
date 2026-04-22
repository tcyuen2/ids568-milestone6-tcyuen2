# IDS568 Milestone 6 — RAG Pipeline & Multi-Tool Agent

A retrieval-augmented generation pipeline plus a ReAct-style agent controller
that intelligently routes between a retriever and a summarizer. Both use a
local open-weight 7B instruct LLM served by Ollama.

## Architecture Overview

```
+----------------------+           +----------------------+
|   RAG Pipeline       |           |   Agent Controller   |
|   (Part 1)           |           |   (Part 2)           |
|                      |           |                      |
|  docs -> chunker     |           |   task -> LLM        |
|       -> embedder    |           |         /    \       |
|       -> FAISS       |           |   retrieve  summarize|
|       -> retriever   | <-------- |     tool      tool   |
|       -> LLM answer  |           |         \    /       |
+----------------------+           |          LLM         |
        ^                          |           |          |
        |                          |        answer        |
        +--- same index reused ---+                       |
                                    +----------------------+
```

The agent's `retrieve` tool wraps the same FAISS index built by the RAG
pipeline — zero code duplication across parts. The same Ollama-served model
(`mistral:7b-instruct`) drives both the RAG generator and the agent.

- **RAG:** `rag_pipeline.py`, evaluated by `evaluate_rag.py`
- **Agent:** `agent_controller.py`, traces in `agent_traces/`
- **Diagram:** `rag_pipeline_diagram.md`
- **Reports:** `rag_evaluation_report.md`, `agent_report.md`

## Setup

### 1. Clone this repo

```bash
git clone <your-repo-url>
cd ids568-milestone6-<your_netid>
```

### 2. Install Ollama and pull the model

Ollama is the open-source LLM serving stack used for this submission. Install
from https://ollama.com, then:

```bash
ollama pull mistral:7b-instruct
# Verify:
ollama run mistral:7b-instruct "Reply with the word READY"
```

This pulls `mistralai/Mistral-7B-Instruct-v0.3` (size class: 7B) in quantized
form. It runs on CPU (~4–6 s/response on a modern laptop) or GPU (<1 s/response
on an RTX 3060 or better). Requires ~4.5 GB disk.

If you prefer a different open-weight model, override with an environment
variable (e.g. `export RAG_LLM_MODEL=llama3.1:8b-instruct`); the code is
model-agnostic as long as the Ollama tag exists.

### 3. Create a Python env and install deps

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first import of `sentence-transformers` will download the MiniLM embedder
(~90 MB). This happens once.

### 4. Verify the install

```bash
python -c "import faiss, sentence_transformers, ollama; print('OK')"
```

## Usage

### Run the RAG pipeline on 3 demo questions

```bash
python rag_pipeline.py
```

First run builds the FAISS index under `.rag_index/`; subsequent runs load it.

### Evaluate RAG on the 10 handcrafted queries

```bash
python evaluate_rag.py
```

This writes `rag_eval_results.json` with per-query precision@k, recall@k, and
three latency measurements (retrieval / generation / end-to-end). The
`rag_evaluation_report.md` narrative references these numbers.

### Run the agent on the 10 evaluation tasks

```bash
python agent_controller.py
```

Writes one trace per task to `agent_traces/task_NN.json`. Each trace contains:
task prompt, every LLM thought, every tool call with its input and
observation, latency per step, and the final answer. The `agent_report.md`
narrative references these traces.

### Run a single custom query

```python
from rag_pipeline import build_or_load_index, answer_query
idx = build_or_load_index()
result = answer_query("What is PSI used for?", idx)
print(result.answer)
print("sources:", [r.chunk.source for r in result.retrieved])
```

### Run the agent on a single custom task

```python
from rag_pipeline import build_or_load_index
from agent_controller import run_agent, make_retrieve_tool, make_summarize_tool, save_trace

idx = build_or_load_index()
tools = [make_retrieve_tool(idx), make_summarize_tool()]
trace = run_agent(task_id=99, task="Explain canary deployments briefly.", tools=tools)
save_trace(trace)
print(trace.final_answer)
```

## Model Serving Details

Required reproducibility information for the final submission:

| Field | Value |
|-------|-------|
| Model name | `mistral:7b-instruct` (Ollama tag) |
| HuggingFace origin | `mistralai/Mistral-7B-Instruct-v0.3` |
| Size class | 7B |
| Serving stack | **Ollama** (local HTTP server) |
| Inference device | CPU by default; GPU auto-detected by Ollama if present |
| Quantization | Q4_K_M (Ollama default for 7B) |
| Hardware used for final eval | [**FILL IN YOUR OWN HARDWARE** — e.g. "MacBook Pro M2, 16GB unified RAM, CPU inference"] |
| Typical generation latency | ~4 s/response on CPU, ~0.7 s/response on consumer GPU |

**Startup command:**

```bash
ollama serve &                      # starts the Ollama daemon in the background
ollama pull mistral:7b-instruct     # one-time
# Python code then connects to http://localhost:11434 via the ollama package
```

If Ollama is already running as a system service (default install behavior on
macOS / Windows), the `ollama serve &` step is unnecessary.

## Known Limitations

- **Corpus size.** The corpus is 7 short MLOps documents. Retrieval is
  effectively exhaustive; real-world performance on millions of chunks would
  require an approximate index (FAISS `IndexIVFFlat` or `IndexHNSW`) and a
  reranker. Swapping these in is a drop-in change in `RAGIndex.build`.
- **Single-language corpus.** MiniLM-L6-v2 is English-only. Multilingual
  corpora should use `paraphrase-multilingual-MiniLM-L12-v2` or similar.
- **No chunk-level ground truth.** Our relevance labels are document-level
  (which file should appear in top-k), not chunk-level. This inflates
  precision@k when two gold chunks come from the same doc.
- **Agent step cap.** Agent loop caps at 6 steps. Tasks requiring more would
  hit the ceiling; none of our 10 tasks do, but harder multi-hop tasks might.
- **Temperature = 0 everywhere.** Deterministic, but may underestimate natural
  variance in answers. A sensitivity analysis at temperature 0.3 would
  strengthen the evaluation.
- **7B model capacity.** The 7B model occasionally over-copies retrieved text
  into summarize tool inputs and very occasionally conflates distinct concepts
  in multi-hop synthesis (see `agent_report.md` section 5). A 14B model would
  likely improve synthesis quality at the cost of ~2x latency.
- **No ground-truth labels for concept drift.** The monitoring discussion is
  qualitative; with real labels we could compute per-query answer accuracy in
  addition to retrieval metrics.

## Repository Layout

```
ids568-milestone6-<netid>/
├── README.md                       # this file
├── requirements.txt                # pinned dependencies
├── rag_pipeline.py                 # Part 1: RAG implementation
├── evaluate_rag.py                 # Part 1: runs the 10 queries, writes metrics
├── rag_pipeline_diagram.md         # Part 1: architecture diagram
├── rag_evaluation_report.md        # Part 1: evaluation narrative
├── agent_controller.py             # Part 2: ReAct-style agent
├── agent_report.md                 # Part 2: agent evaluation narrative
├── agent_traces/
│   ├── task_01.json                # Part 2: 10 multi-step task traces
│   ├── task_02.json
│   ├── ...
│   └── task_10.json
└── data/                           # corpus (7 MLOps topic docs)
    ├── doc1_feature_stores.md
    ├── doc2_model_monitoring.md
    ├── doc3_ab_testing.md
    ├── doc4_cicd_ml.md
    ├── doc5_model_deployment.md
    ├── doc6_data_versioning.md
    └── doc7_model_registry.md
```

## Submission Procedure

```bash
# 1. Verify everything runs end-to-end on a clean machine
python rag_pipeline.py
python evaluate_rag.py
python agent_controller.py

# 2. Update hardware + latency fields in README.md and rag_evaluation_report.md

# 3. Commit
git add -A
git commit -m "Milestone 6 final submission"
git push

# 4. Tag
git tag submission
git push --tags
```

Then submit the repo URL via the Course Submission Site.

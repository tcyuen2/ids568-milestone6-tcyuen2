# IDS568 Milestone 6 — RAG Pipeline and Multi-Tool Agent

This repo has two things: a RAG pipeline (Part 1) that answers questions using a small MLOps document corpus, and an agent controller (Part 2) that picks between retrieval and summarization tools to solve multi-step tasks. Both use a local 7B open-weight LLM through Ollama.

## Architecture Overview

```
Part 1 (RAG Pipeline)                   Part 2 (Agent Controller)

  7 .md docs in data/                    User task
        |                                    |
        v                                    v
  chunker (512 char, 50 overlap)         LLM (mistral:7b, JSON mode)
        |                                    |
        v                               +----+----+
  embedder (MiniLM, 384-dim)            |         |
        |                            retrieve  summarize
        v                           (same FAISS    (calls
  FAISS index                        index as      the LLM
        |                              Part 1)      again)
        v                                    |
  retriever (top-4)                          v
        |                               observation
        v                                    |
  LLM (mistral:7b) -----> answer             v
                                        back to LLM
                                             |
                                             v
                                        final answer
```

The agent's retrieve tool uses the same FAISS index the RAG pipeline builds in Part 1 — I just import it from `rag_pipeline.py`. No code duplication between the two parts.

## Setup

### 1. Install Ollama and pull the model

Ollama is the LLM server. Download the Windows installer from https://ollama.com/download/windows and run it. Then pull the model:

```bash
ollama pull mistral:7b-instruct
```

This downloads about 4.1 GB. Test it works:

```bash
ollama run mistral:7b-instruct "say hi"
```

Type `/bye` to exit.

### 2. Create a Python virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks script execution, run this once and try again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**Mac/Linux:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Make sure everything imports

```bash
python -c "import faiss, sentence_transformers, ollama; print('all good')"
```

## Usage

### Part 1 — Run the RAG pipeline on 3 demo questions

```bash
python rag_pipeline.py
```

First run builds the FAISS index and saves it in `.rag_index/`. Later runs just load it.

### Part 1 — Run the full evaluation (10 queries, computes P@k and R@k)

```bash
python evaluate_rag.py
```

Writes `rag_eval_results.json` with all the metrics. My actual results were:

- Mean Precision@4: 0.25
- Mean Recall@4: 0.889 (7 of 9 in-scope queries perfect; 2 multi-hop queries got 0.5)
- Mean retrieval: 216 ms
- Mean generation: 2,574 ms
- Mean end-to-end: 2,790 ms

### Part 2 — Run the agent on 10 tasks

```bash
python agent_controller.py
```

Writes one trace per task to `agent_traces/task_NN.json`. Each trace shows every tool call the agent made, what it got back, and how it reasoned. My results: 9 of 10 tasks succeeded. Task 7 (shadow vs canary) hit the step limit (details in `agent_report.md`).

### Run your own question through the RAG pipeline

```python
from rag_pipeline import build_or_load_index, answer_query
idx = build_or_load_index()
result = answer_query("What is PSI used for?", idx)
print(result.answer)
print("sources:", [r.chunk.source for r in result.retrieved])
```

### Run your own task through the agent

```python
from rag_pipeline import build_or_load_index
from agent_controller import run_agent, make_retrieve_tool, make_summarize_tool, save_trace

idx = build_or_load_index()
tools = [make_retrieve_tool(idx), make_summarize_tool()]
trace = run_agent(task_id=99, task="Explain canary deployments briefly.", tools=tools)
save_trace(trace)
print(trace.final_answer)
```

## Model and Serving Details

| What | Value |
|------|-------|
| Model | `mistral:7b-instruct` (Ollama tag) |
| HuggingFace origin | `mistralai/Mistral-7B-Instruct-v0.3` |
| Size | 7B parameters |
| Quantization | Q4_K_M (Ollama's default for 7B) |
| Serving | **Ollama** (local HTTP server on port 11434) |
| Hardware used | [FILL IN — e.g. "Windows 11 laptop, CPU only, 16 GB RAM"] |
| Typical generation time | ~2.6 seconds per response |

**How Ollama runs:** on Windows, Ollama installs as a background service that auto-starts. You don't need to manually launch anything after `ollama pull`. If you ever need to start it manually:

```bash
ollama serve
```

Then in another window you can run the Python scripts and they'll connect to `http://localhost:11434` automatically through the `ollama` Python package.

## Known Limitations

- **Small corpus.** I only have 7 documents (46 chunks after splitting). On a real-world corpus with millions of chunks I'd need an approximate index like FAISS IVF or HNSW, plus probably a reranker.
- **English only.** The MiniLM embedder I used is English-only. For other languages I'd need a multilingual embedder.
- **Multi-hop retrieval is weak.** Queries 8 and 9 in my RAG eval only got 1 of 2 relevant docs. The agent in Part 2 works around this by doing separate retrievals per topic, but the retriever itself has this weakness.
- **Agent step cap.** The agent gives up after 6 steps. Task 7 in my eval hit this limit. A more patient agent or a stricter "synthesize now" prompt would fix it.
- **All temperature 0.** Good for reproducibility, but doesn't tell me how stable the system is when the model samples more freely.
- **7B model has limits.** The 7B model is fast but sometimes struggles with multi-step synthesis. A 14B model would probably do better on hard tasks but take about 2x as long per call.

## Files in this Repo

```
ids568-milestone6-<netid>/
├── README.md                       # this file
├── requirements.txt                # pinned Python dependencies
├── rag_pipeline.py                 # Part 1: RAG implementation
├── evaluate_rag.py                 # Part 1: runs the 10 eval queries
├── rag_pipeline_diagram.md         # Part 1: architecture diagram
├── rag_evaluation_report.md        # Part 1: evaluation writeup
├── rag_eval_results.json           # Part 1: raw metrics from my run
├── agent_controller.py             # Part 2: ReAct-style agent
├── agent_report.md                 # Part 2: agent writeup
├── agent_traces/
│   ├── task_01.json                # Part 2: 10 task traces
│   ├── task_02.json
│   ├── ...
│   └── task_10.json
└── data/
    ├── doc1_feature_stores.md      # the 7 corpus docs
    ├── doc2_model_monitoring.md
    ├── doc3_ab_testing.md
    ├── doc4_cicd_ml.md
    ├── doc5_model_deployment.md
    ├── doc6_data_versioning.md
    └── doc7_model_registry.md
```




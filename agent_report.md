# Agent Controller Report

## Setup

- **Model:** `mistral:7b-instruct` through Ollama (same one I used in Part 1)
- **Tools:** `retrieve` (wraps the FAISS index from Part 1) and `summarize` (calls the LLM with a summarization prompt)
- **Control:** ReAct-style loop. The model replies with a JSON object each turn — either a tool call or a final answer. Max 6 steps before I kill it.
- **JSON mode:** I use Ollama's `format="json"` and `num_predict=1024` to guarantee parseable output. Without this the agent was truncating mid-JSON.

All 10 tasks are in `agent_traces/task_01.json` through `task_10.json`.

## 1. How the agent decides which tool to use

The policy lives in the system prompt (you can see it in `agent_controller.py::SYSTEM_PROMPT`). Three main rules:

1. **Always retrieve before answering.** The corpus is the ground truth. I don't trust a 7B model's memory on specific MLOps terminology, so the prompt forces it to search first.
2. **Summarize only after retrieving.** The summarize tool is for condensing text the agent already has. The agent shouldn't use it to answer from its own memory.
3. **Emit `final_answer` when confident.** The JSON output has two branches — either a tool call or a final answer. This gives the model a clean way to say "I'm done."

For out-of-scope questions (like "moons of Jupiter"), the prompt tells the agent to admit it doesn't know rather than make something up.

## 2. How retrieval fits in

Retrieval is a tool the agent can call whenever — not something that runs automatically at the start. The agent decides on every turn whether to retrieve, summarize, or finish. Every decision gets logged in the trace so you can see exactly what the agent was thinking.

The retriever uses the same FAISS index from Part 1. I just import `build_or_load_index` from `rag_pipeline.py`. Zero code duplication.

For multi-hop tasks, the agent can issue two retrievals with different queries. Task 4 (DVC vs model registry) is a good example — the agent retrieved "DVC data versioning" first, then "model registry purpose" separately, then synthesized an answer. This is actually how it dodges the same weakness that hurt my RAG eval queries 8 and 9.

## 3. How the agent did on 10 tasks

All numbers below are from my actual run:

| # | Task | Steps | Time (ms) | Outcome | Notes |
|---|------|-------|-----------|---------|-------|
| 1 | Feature store + 1-sentence summary | 3 | 11,205 | ✅ | retrieve → summarize → final |
| 2 | Data drift vs concept drift | 3 | 12,077 | ✅ | retrieve → summarize → final |
| 3 | Canary summary (2 sentences) | 3 | 10,088 | ✅ | retrieve → summarize → final |
| 4 | DVC vs model registry | 4 | 14,577 | ✅ | Two retrievals, then synthesis — this is the good multi-hop pattern |
| 5 | A/B test stages | 3 | 13,791 | ✅ | retrieve → summarize → final |
| 6 | CI/CD briefing | 3 | 13,697 | ✅ | Got a clean plain-text final answer here |
| 7 | Shadow vs canary | 6 | 30,928 | ❌ | **Hit the 6-step limit without finishing.** See failure analysis. |
| 8 | Moons of Jupiter (out-of-scope) | 2 | 4,124 | ✅ | Correctly said: "The corpus does not contain any information about the moons of Jupiter." |
| 9 | Canary + drift monitoring | 2 | 6,937 | ⚠️ | Finished, but only retrieved from one of two relevant docs |
| 10 | Model registry summary | 2 | 16,665 | ✅ | Quick retrieve → final |

**9 out of 10 tasks worked.** Task 7 failed by running out of steps. Task 9 technically finished but missed a relevant document (similar problem to RAG query 9).

Average completion time on successful tasks: about 11 seconds. Average step count: 2.8, so the agent usually uses 1-2 tool calls before answering.

## 4. Failures (where things went wrong)

### Task 7: hit the step limit
This was my real failure. The task was: "Look up shadow deployment, then look up canary deployment, then explain when to use one vs the other." The agent correctly did the two retrievals, but then kept retrieving instead of synthesizing an answer. It used all 6 steps and got cut off.

I think what happened is the 7B model wasn't confident enough to emit `final_answer` after the two retrievals. The system prompt says "emit final_answer when you have enough information" but apparently that's not forceful enough.

**How I'd fix it next time:**
- Add a rule like "after two retrievals on different topics, you MUST synthesize and answer"
- Track the step count in the controller and force a final answer at step 4+
- Use a bigger model (14B) for synthesis-heavy tasks

### Task 9: missed a relevant document
The task was: "If I deploy a new recommender via canary, what should I monitor to detect drift?" The agent did one retrieval with the query "canary deployment monitoring" and answered using only the deployment doc. It never retrieved from the monitoring doc.

The answer it gave (accuracy, precision, CTR, etc.) is reasonable but not grounded in the monitoring doc's actual content about PSI, KS tests, etc. This is the same failure mode as my RAG query 9 — the word "canary" pulls everything to the deployment doc.

The agent *could* have done a second retrieval but decided not to. I could strengthen the policy to force two retrievals when the task mentions two topics, but that would slow down simpler tasks.

### JSON format weirdness
A few tasks (4, 5, 10) returned `final_answer` as a nested JSON object instead of a plain string. Something like `{"thought": "...", "summary": "..."}` instead of just text. This happens because `format="json"` mode sometimes makes the model over-structure its output.

My code handles this — if the final answer isn't a string, I flatten it with `json.dumps()` before saving. So the traces are still readable, they just have some JSON inside the final answer field.

## 5. Model size tradeoffs

**Why I used a 7B model.** Running locally on CPU, each LLM call takes 2–5 seconds. With an average of 3 calls per task, I'm already at ~11 seconds per task. A 14B model would roughly double that. For a classroom assignment that's fine, but for anything interactive it would be rough.

**What the 7B model did well:**
- Picked the right tool 9 out of 10 times
- Always produced parseable JSON (after I added `format="json"` + `num_predict=1024`)
- Correctly refused the out-of-scope question
- Handled the two-retrieval synthesis on task 4

**What the 7B model struggled with:**
- Knowing when to stop (task 7 step-cap failure)
- Deciding it needs to retrieve twice when the task has two topics (task 9)
- Producing a consistent output format — sometimes prose, sometimes a JSON object (tasks 4, 5, 10)

**If I had more time:** I'd try a 14B model just for the final-answer step. The tool-selection decisions are simple enough that 7B handles them fine. The synthesis/final-answer step is where model size matters, so spending the extra latency only on that step would be a good tradeoff.

## 6. Why the traces are readable

Every step in every trace file logs:
- `thought`: what the model said it was doing
- `action`: which tool it picked
- `action_input`: what it passed to that tool
- `observation`: what the tool returned
- `latency_ms`: how long this step took
- `raw_llm_output`: the exact model response for debugging

If you want to grade this, you can open any `agent_traces/task_NN.json` and read it top to bottom. Task 4's trace shows the ideal multi-hop pattern (two separate retrievals then a synthesis). Task 7's trace shows the failure — you can see it retrieving over and over without emitting a final answer. The failure is fully visible, not hidden.

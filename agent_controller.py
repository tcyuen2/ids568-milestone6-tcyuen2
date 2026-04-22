"""
Agent Controller with Retrieval Integration
-------------------------------------------
A lightweight ReAct-style agent that picks between two tools:

  1. retrieve(query)      -> top-k chunks from the Part-1 RAG index
  2. summarize(text)      -> short summary produced by the same LLM

The agent runs a loop where the LLM outputs a JSON action, the controller
executes the corresponding tool, appends the observation, and loops until
the model emits a final answer.

Every step (thought, tool, tool_input, observation) is logged to
agent_traces/task_NN.json.

Run:
  ollama pull mistral:7b-instruct
  pip install -r requirements.txt
  python agent_controller.py
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import ollama

from rag_pipeline import (
    build_or_load_index,
    RAGIndex,
    LLM_MODEL_NAME,
)


TRACE_DIR = Path(__file__).parent / "agent_traces"
TRACE_DIR.mkdir(exist_ok=True)

MAX_STEPS = 6


# -----------------------------------------------------------------------------
# Tool interface
# -----------------------------------------------------------------------------

@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[[Dict[str, Any]], Any]


def make_retrieve_tool(index: RAGIndex) -> Tool:
    def _retrieve(args: Dict[str, Any]) -> Dict[str, Any]:
        query = args.get("query", "")
        k = int(args.get("k", 4))
        results = index.search(query, k=k)
        return {
            "query": query,
            "hits": [
                {"source": r.chunk.source, "score": round(r.score, 4), "text": r.chunk.text}
                for r in results
            ],
        }

    return Tool(
        name="retrieve",
        description=(
            "Search the MLOps document corpus for information. "
            "Input: {\"query\": <search text>, \"k\": <int, default 4>}. "
            "Returns the top-k matching chunks with their source filenames."
        ),
        fn=_retrieve,
    )


def make_summarize_tool(model: str = LLM_MODEL_NAME) -> Tool:
    def _summarize(args: Dict[str, Any]) -> Dict[str, Any]:
        text = args.get("text", "")
        max_sentences = int(args.get("max_sentences", 3))
        prompt = (
            f"Summarize the following text in at most {max_sentences} sentences. "
            f"Focus on the most important facts. Do not add information that is "
            f"not in the text.\n\n---\n{text}\n---\n\nSummary:"
        )
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        return {"summary": resp["message"]["content"].strip()}

    return Tool(
        name="summarize",
        description=(
            "Summarize a block of text into a concise summary. "
            "Input: {\"text\": <text to summarize>, \"max_sentences\": <int, default 3>}. "
            "Returns a summary string."
        ),
        fn=_summarize,
    )


# -----------------------------------------------------------------------------
# Agent loop
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a careful research agent. You solve the user's task by
calling tools and reasoning over their outputs.

You have these tools:
{tool_descriptions}

On every turn you MUST reply with ONE JSON object and NOTHING ELSE.
Use either of these two shapes:

  Take an action:
    {{"thought": "<your reasoning>", "action": "<tool_name>", "action_input": {{...}}}}

  Emit the final answer:
    {{"thought": "<your reasoning>", "final_answer": "<the answer to the user's task>"}}

Rules:
- Prefer using the retrieve tool before answering from memory. The corpus is the
  ground truth for this task.
- If retrieval returns nothing relevant, say so in final_answer rather than
  fabricating facts.
- Keep thoughts short. Do not wrap the JSON in code fences.
"""


@dataclass
class TraceStep:
    step: int
    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[Any] = None
    final_answer: Optional[str] = None
    latency_ms: float = 0.0
    raw_llm_output: str = ""


@dataclass
class Trace:
    task_id: int
    task: str
    steps: List[TraceStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    total_ms: float = 0.0
    success: Optional[bool] = None   # set externally after grading
    notes: str = ""


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> Dict[str, Any]:
    """Pull the first {...} block out of the model response and parse it."""
    m = _JSON_OBJ_RE.search(text)
    if not m:
        raise ValueError(f"No JSON object found in model output:\n{text}")
    raw = m.group(0)
    # Small models sometimes emit trailing commas; attempt a forgiving parse.
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
        return json.loads(cleaned)


def run_agent(
    task_id: int,
    task: str,
    tools: List[Tool],
    model: str = LLM_MODEL_NAME,
    max_steps: int = MAX_STEPS,
) -> Trace:
    tool_map = {t.name: t for t in tools}
    tool_descriptions = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    system = SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Task: {task}"},
    ]

    trace = Trace(task_id=task_id, task=task)
    t_start = time.perf_counter()

    for step_i in range(1, max_steps + 1):
        t0 = time.perf_counter()
        resp = ollama.chat(
            model=model,
            messages=messages,
            format="json",
            options={
                "temperature": 0.0,
                "num_predict": 1024,   # cap on generated tokens; default is 128, which truncates JSON
            },
        )
        raw = resp["message"]["content"]
        latency_ms = (time.perf_counter() - t0) * 1000

        try:
            decision = _extract_json(raw)
        except (ValueError, json.JSONDecodeError) as e:
            step = TraceStep(
                step=step_i,
                thought=f"[parse error] {e}",
                latency_ms=latency_ms,
                raw_llm_output=raw,
            )
            trace.steps.append(step)
            trace.final_answer = f"Agent halted: could not parse output. Raw: {raw[:200]}"
            break

        thought = decision.get("thought", "")

        # Final answer branch
        if "final_answer" in decision:
            fa = decision["final_answer"]
            # Some models in JSON mode return a dict/list for final_answer; flatten it.
            if not isinstance(fa, str):
                fa = json.dumps(fa, ensure_ascii=False)
            step = TraceStep(
                step=step_i,
                thought=thought,
                final_answer=fa,
                latency_ms=latency_ms,
                raw_llm_output=raw,
            )
            trace.steps.append(step)
            trace.final_answer = fa
            break

        # Tool-call branch
        action = decision.get("action")
        action_input = decision.get("action_input", {})
        if action not in tool_map:
            observation = {"error": f"Unknown tool: {action}. Available: {list(tool_map)}"}
        else:
            try:
                observation = tool_map[action].fn(action_input)
            except Exception as e:
                observation = {"error": f"Tool raised: {e}"}

        step = TraceStep(
            step=step_i,
            thought=thought,
            action=action,
            action_input=action_input,
            observation=observation,
            latency_ms=latency_ms,
            raw_llm_output=raw,
        )
        trace.steps.append(step)

        # Append the assistant message and an observation message to the history
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": f"Observation from {action}:\n{json.dumps(observation)[:2000]}",
        })
    else:
        trace.final_answer = "Agent halted: maximum steps reached without final_answer."

    trace.total_ms = (time.perf_counter() - t_start) * 1000
    return trace


def save_trace(trace: Trace, trace_dir: Path = TRACE_DIR) -> Path:
    path = trace_dir / f"task_{trace.task_id:02d}.json"
    payload = {
        "task_id": trace.task_id,
        "task": trace.task,
        "final_answer": trace.final_answer,
        "total_ms": round(trace.total_ms, 2),
        "n_steps": len(trace.steps),
        "steps": [asdict(s) for s in trace.steps],
        "notes": trace.notes,
        "success": trace.success,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


# -----------------------------------------------------------------------------
# 10 evaluation tasks
# -----------------------------------------------------------------------------

EVAL_TASKS: List[Dict[str, Any]] = [
    {
        "id": 1,
        "task": "Explain what a feature store is. Then summarize your explanation in one sentence.",
        "expected_tools": ["retrieve", "summarize"],
    },
    {
        "id": 2,
        "task": "What is the difference between data drift and concept drift?",
        "expected_tools": ["retrieve"],
    },
    {
        "id": 3,
        "task": "Find what the corpus says about canary deployments, then give me a 2-sentence summary.",
        "expected_tools": ["retrieve", "summarize"],
    },
    {
        "id": 4,
        "task": "Compare how the corpus describes DVC versus model registries. What problem does each solve?",
        "expected_tools": ["retrieve", "retrieve"],
    },
    {
        "id": 5,
        "task": "Is A/B testing for ML mentioned in the corpus? If so, list the main stages.",
        "expected_tools": ["retrieve"],
    },
    {
        "id": 6,
        "task": "Write a short briefing (3 sentences) for a new engineer on why CI/CD for ML is harder than regular CI/CD.",
        "expected_tools": ["retrieve", "summarize"],
    },
    {
        "id": 7,
        "task": "Look up shadow deployment in the corpus, then look up canary deployment, then explain when you would use one instead of the other.",
        "expected_tools": ["retrieve", "retrieve"],
    },
    {
        "id": 8,
        "task": "Does the corpus mention anything about the moons of Jupiter? Answer honestly.",
        "expected_tools": ["retrieve"],
        "notes": "Designed-to-fail retrieval - tests honest 'no relevant info' response.",
    },
    {
        "id": 9,
        "task": "If I deploy a new recommender model via canary, what should I monitor to detect drift? Cite your sources.",
        "expected_tools": ["retrieve", "retrieve"],
    },
    {
        "id": 10,
        "task": "Summarize the role of a model registry in MLOps governance in 2 sentences.",
        "expected_tools": ["retrieve", "summarize"],
    },
]


def main() -> None:
    print("=" * 70)
    print("Agent Controller - 10 evaluation tasks")
    print("=" * 70)
    index = build_or_load_index()
    tools = [make_retrieve_tool(index), make_summarize_tool()]

    for task_def in EVAL_TASKS:
        print(f"\n--- Task {task_def['id']}: {task_def['task'][:80]}...")
        trace = run_agent(task_def["id"], task_def["task"], tools)
        if "notes" in task_def:
            trace.notes = task_def["notes"]
        path = save_trace(trace)
        print(f"    steps={len(trace.steps)} time={trace.total_ms:.0f}ms -> {path.name}")
        answer_preview = str(trace.final_answer) if trace.final_answer else "(none)"
        print(f"    answer: {answer_preview[:200]}")


if __name__ == "__main__":
    main()
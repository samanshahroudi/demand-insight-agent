import json
import os
from pathlib import Path
from typing import NotRequired, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from .models import Product, ProductAssessment, RunRequest
from .scoring import SCORING_VERSION, assess


class RunState(TypedDict):
    request: dict
    run_id: str
    candidates: list[dict]
    assessments: list[dict]
    summary: NotRequired[str]


def _catalog() -> list[dict]:
    path = Path(__file__).with_name("catalog.json")
    return json.loads(path.read_text(encoding="utf-8"))


def select_candidates(state: RunState) -> dict:
    request = RunRequest.model_validate(state["request"])
    catalog = _catalog()
    if request.mode == "shortlist":
        requested = set(request.product_ids)
        catalog = [item for item in catalog if item["product_id"] in requested]
        missing = requested - {item["product_id"] for item in catalog}
        if missing:
            raise ValueError(f"Unknown product IDs: {', '.join(sorted(missing))}")
    return {"candidates": catalog}


def score_candidates(state: RunState) -> dict:
    request = RunRequest.model_validate(state["request"])
    results = [assess(Product.model_validate(item)).model_dump() for item in state["candidates"]]
    results.sort(key=lambda item: item["score"], reverse=True)
    return {"assessments": results[: request.limit]}


def deterministic_summary(state: RunState) -> dict:
    results = [ProductAssessment.model_validate(item) for item in state["assessments"]]
    if not results:
        return {"summary": "No products matched this run."}
    leaders = ", ".join(f"{item.name} ({item.score:.1f})" for item in results[:3])
    return {"summary": f"Top opportunities by the current scoring rules: {leaders}."}


def model_summary(state: RunState) -> dict:
    from langchain_openai import ChatOpenAI

    model_name = os.getenv("OPENAI_MODEL")
    if not os.getenv("OPENAI_API_KEY") or not model_name:
        raise ValueError("Set OPENAI_API_KEY and OPENAI_MODEL to enable model-written summaries.")
    model = ChatOpenAI(model=model_name, temperature=0)
    prompt = (
        "Summarize these product opportunity scores for a business operator in 2-3 sentences. "
        "Use only the supplied results. Do not change scores or claim that they predict actual sales.\n\n"
        "Results: {results}"
    )
    response = model.invoke(prompt.format(results=json.dumps(state["assessments"], ensure_ascii=False)))
    return {"summary": str(response.content)}


def _summary_route(state: RunState) -> str:
    request = RunRequest.model_validate(state["request"])
    return "model_summary" if request.use_llm_summary else "deterministic_summary"


builder = StateGraph(RunState)
builder.add_node("select_candidates", select_candidates)
builder.add_node("score_candidates", score_candidates)
builder.add_node("deterministic_summary", deterministic_summary)
builder.add_node("model_summary", model_summary)
builder.add_edge(START, "select_candidates")
builder.add_edge("select_candidates", "score_candidates")
builder.add_conditional_edges(
    "score_candidates",
    _summary_route,
    {"deterministic_summary": "deterministic_summary", "model_summary": "model_summary"},
)
builder.add_edge("deterministic_summary", END)
builder.add_edge("model_summary", END)
run_graph = builder.compile()


def run(request: RunRequest) -> dict:
    result = run_graph.invoke(
        {"request": request.model_dump(), "run_id": str(uuid4()), "candidates": [], "assessments": []}
    )
    return {
        "run_id": result["run_id"],
        "mode": request.mode,
        "assessments": result["assessments"],
        "summary": result["summary"],
        "scoring_version": SCORING_VERSION,
    }

from fastapi import FastAPI, HTTPException

from .graph import run
from .models import RunRequest, RunResponse

app = FastAPI(title="Demand Insight Agent", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/runs", response_model=RunResponse)
def create_run(request: RunRequest) -> RunResponse:
    try:
        return RunResponse.model_validate(run(request))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ._deps import state

router = APIRouter(prefix="/agent", tags=["agent"])


class RunRequest(BaseModel):
    prompt: str
    model: str | None = None
    allowed_tools: list[str] | None = None
    source: str = "web"


@router.post("/run")
def run(body: RunRequest, s=Depends(state)):
    if not body.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    try:
        result = s.agent.run(
            body.prompt,
            source=body.source,
            model=body.model,
            allowed_tools=body.allowed_tools,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error))

    return {
        "id": result.id,
        "final": result.final,
        "events": [
            {"kind": event.kind, "payload": event.payload}
            for event in result.events
        ],
    }


@router.get("/runs")
def list_runs(s=Depends(state), limit: int = 50):
    return s.db.list_runs(limit=limit)
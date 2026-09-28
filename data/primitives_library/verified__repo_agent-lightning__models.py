from __future__ import annotations

from fastapi import APIRouter

from agentlightning.schemas import Model
from agentlightning.server.store import _models

router = APIRouter(tags=["models"])


@router.post("/models", status_code=201, response_model=list[Model])
async def register_models(body: list[Model]) -> list[Model]:
    """Register model server(s). Upsert by (model, endpoint)."""
    registered: list[Model] = []
    for item in body:
        _models.setdefault(item.model, {})[item.endpoint] = item
        registered.append(item)
    return registered


@router.delete("/models")
async def delete_all_models() -> dict[str, str]:
    """Remove all model servers."""
    _models.clear()
    return {"status": "ok"}
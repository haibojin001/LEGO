from __future__ import annotations

from fastapi import APIRouter, Depends

from ._deps import state

router = APIRouter(tags=["tools"], prefix="/tools")


@router.get("")
def list_tools(s=Depends(state)):
    result: dict[str, list[dict]] = {}
    for item in s.registry.all():
        result.setdefault(item.category, []).append(
            {
                "name": item.name,
                "description": item.description,
                "parameters": item.parameters,
                "requires": item.requires,
                "danger": item.danger,
            }
        )
    return result


@router.get("/{tool_name}")
def describe(tool_name: str, s=Depends(state)):
    item = s.registry.get(tool_name)
    if not item:
        return {"error": "not found"}
    return {
        "name": item.name,
        "description": item.description,
        "parameters": item.parameters,
        "category": item.category,
        "requires": item.requires,
        "danger": item.danger,
    }
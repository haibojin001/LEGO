"""LLM provider configuration."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ...providers import CATALOG, get_spec
from ._deps import state

router = APIRouter(tags=["models"], prefix="/models")


class ProviderUpdate(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    default_model: str | None = None
    enabled: bool | None = None


class ActiveSelection(BaseModel):
    provider_id: str
    model: str | None = None


@router.get("/catalog")
def catalog():
    """Static catalog — what providers we know about."""
    entries = []
    for spec in CATALOG:
        entries.append(
            {
                "id": spec.id,
                "display_name": spec.display_name,
                "region": spec.region,
                "base_url": spec.base_url,
                "api_key_url": spec.api_key_url,
                "docs_url": spec.docs_url,
                "models": list(spec.models),
                "requires_key": spec.requires_key,
                "notes": spec.notes,
            }
        )
    return entries


@router.get("")
def list_providers(s=Depends(state)):
    return s.db.list_providers()


@router.get("/active")
def active(s=Depends(state)):
    return {
        "provider_id": s.providers.active_provider_id(),
        "model": s.providers.active_model(),
    }


@router.post("/active")
def set_active(sel: ActiveSelection, s=Depends(state)):
    if not s.db.get_provider(sel.provider_id):
        raise HTTPException(404, "unknown provider")
    s.providers.set_active_provider(sel.provider_id, sel.model)
    return {"ok": True}


@router.patch("/{provider_id}")
def update_provider(provider_id: str, body: ProviderUpdate, s=Depends(state)):
    spec = get_spec(provider_id)
    if not spec:
        raise HTTPException(404, "unknown provider")

    existing = s.db.get_provider(provider_id) or {}
    base_url = body.base_url
    if base_url is None:
        base_url = existing.get("base_url") or spec.base_url

    default_model = body.default_model
    if default_model is None:
        default_model = existing.get("default_model")

    enabled = body.enabled
    if enabled is None:
        enabled = bool(existing.get("enabled", True))

    s.db.upsert_provider(
        id=provider_id,
        display_name=spec.display_name,
        base_url=base_url,
        api_key=body.api_key,
        default_model=default_model,
        enabled=enabled,
    )
    return s.db.get_provider(provider_id)


@router.post("/{provider_id}/test")
def test_provider(provider_id: str, s=Depends(state)):
    """Cheap smoke test — call the provider with a 1-token reply."""
    from ...providers import ChatMessage

    try:
        client = s.providers.client(provider_id)
        response = client.chat(
            [ChatMessage(role="user", content="Reply with the single word: OK")],
            model=s.db.get_provider(provider_id)["default_model"],
            max_tokens=8,
        )
        return {"ok": True, "reply": response.content[:200]}
    except Exception as exc:
        raise HTTPException(400, f"{type(exc).__name__}: {exc}")


@router.delete("/{provider_id}")
def delete_provider(provider_id: str, s=Depends(state)):
    s.db.delete_provider(provider_id)
    return {"ok": True}
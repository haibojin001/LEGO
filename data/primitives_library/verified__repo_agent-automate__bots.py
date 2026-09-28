from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel

from ._deps import state

router = APIRouter(tags=["bots"], prefix="/bots")


class BotConfigIn(BaseModel):
    config: dict


@router.get("/catalog")
def catalog(s=Depends(state)):
    return s.bots.list_kinds()


@router.get("")
def list_(s=Depends(state)):
    return s.bots.list_instances()


@router.get("/{bot_id}")
def get_one(bot_id: str, s=Depends(state)):
    instances = {bot["id"]: bot for bot in s.bots.list_instances()}
    return instances.get(bot_id) or {
        "id": bot_id,
        "enabled": False,
        "status": "stopped",
        "config_set": False,
    }


@router.put("/{bot_id}/config")
def save_config(bot_id: str, body: BotConfigIn, s=Depends(state)):
    try:
        s.bots.save_config(bot_id, body.config)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"ok": True}


@router.post("/{bot_id}/start")
def start(bot_id: str, s=Depends(state)):
    try:
        return s.bots.start(bot_id)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"{type(exc).__name__}: {exc}",
        )


@router.post("/{bot_id}/stop")
def stop(bot_id: str, s=Depends(state)):
    s.bots.stop(bot_id)
    return {"ok": True}


@router.get("/wechat_oa/webhook", response_class=PlainTextResponse)
def wechat_oa_handshake(
    signature: str = Query(""),
    timestamp: str = Query(""),
    nonce: str = Query(""),
    echostr: str = Query(""),
    s=Depends(state),
):
    instance = s.bots.get_instance("wechat_oa")
    if not instance or not instance.verify_handshake(
        signature=signature,
        timestamp=timestamp,
        nonce=nonce,
    ):
        raise HTTPException(status_code=403, detail="invalid signature")
    return echostr


@router.post("/wechat_oa/webhook")
async def wechat_oa_message(request: Request, s=Depends(state)):
    instance = s.bots.get_instance("wechat_oa")
    if not instance:
        raise HTTPException(status_code=503, detail="bot not running")
    raw = await request.body()
    reply_xml = instance.handle_inbound(raw)
    return Response(reply_xml, media_type="application/xml")


@router.get("/wecom/webhook", response_class=PlainTextResponse)
def wecom_handshake(
    msg_signature: str = Query(""),
    timestamp: str = Query(""),
    nonce: str = Query(""),
    echostr: str = Query(""),
    s=Depends(state),
):
    instance = s.bots.get_instance("wecom")
    if not instance or not instance.verify_handshake(
        signature=msg_signature,
        timestamp=timestamp,
        nonce=nonce,
        echostr=echostr,
    ):
        raise HTTPException(status_code=403, detail="invalid signature")
    return echostr


@router.post("/wecom/webhook")
async def wecom_message(request: Request, s=Depends(state)):
    instance = s.bots.get_instance("wecom")
    if not instance:
        raise HTTPException(status_code=503, detail="bot not running")
    raw = await request.body()
    reply_xml = instance.handle_inbound(raw)
    return Response(reply_xml, media_type="application/xml")
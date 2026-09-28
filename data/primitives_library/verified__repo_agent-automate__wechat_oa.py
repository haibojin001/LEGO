from __future__ import annotations

import hashlib
import logging
import time
import xml.etree.ElementTree as ET
from typing import Any

from .base import Bot, BotMeta

log = logging.getLogger("automate.bots.wechat_oa")

META = BotMeta(
    id="wechat_oa",
    label="微信公众号",
    description=(
        "Subscription / Service account on mp.weixin.qq.com. Massive C-end "
        "reach, but passive-reply only. Needs a public callback URL."
    ),
    docs_url=(
        "https://developers.weixin.qq.com/doc/offiaccount/"
        "Basic_Information/Access_Overview.html"
    ),
    config_fields=[
        {
            "name": "token",
            "label": "Token (you choose)",
            "kind": "string",
            "required": True,
            "hint": (
                "Pick any string. Paste the same value into the 公众号 backend "
                "→ 开发 → 基本配置."
            ),
        },
        {
            "name": "app_id",
            "label": "AppID",
            "kind": "string",
            "required": True,
        },
        {
            "name": "app_secret",
            "label": "AppSecret",
            "kind": "password",
            "required": True,
        },
        {
            "name": "encoding_aes_key",
            "label": "EncodingAESKey (optional, for safe mode)",
            "kind": "password",
            "required": False,
        },
    ],
)


class WeChatOABot(Bot):
    kind = "wechat_oa"

    def __init__(self, *, config: dict, agent):
        self.config = config
        self.agent = agent
        self._status = "ready"

    @property
    def status(self) -> str:
        return self._status

    def start(self) -> None:
        self._status = "ready"

    def stop(self) -> None:
        self._status = "stopped"

    def verify_handshake(
        self,
        *,
        signature: str,
        timestamp: str,
        nonce: str,
    ) -> bool:
        token = (self.config.get("token") or "").strip()
        if not token:
            return False

        source = "".join(sorted((token, timestamp, nonce)))
        digest = hashlib.sha1(source.encode()).hexdigest()
        return digest == signature

    def handle_inbound(self, raw_xml: bytes) -> str:
        try:
            document = ET.fromstring(raw_xml)
            sender = (document.findtext("FromUserName") or "").strip()
            recipient = (document.findtext("ToUserName") or "").strip()
            message_type = (document.findtext("MsgType") or "").strip()
            text = (document.findtext("Content") or "").strip()
        except Exception as exc:
            log.warning("malformed wechat_oa payload: %s", exc)
            return ""

        if message_type != "text" or not text:
            return self._make_reply(recipient, sender, "目前只支持文本消息哦。")

        log.info("[wechat_oa] from %s: %s", sender, text[:80])
        try:
            response = self.agent(text) if self.agent else "(agent not wired)"
        except Exception as exc:
            response = f"⚠ {type(exc).__name__}: {exc}"

        if len(response) > 1800:
            response = response[:1800] + "…\n(后续内容稍后通过客服消息发送)"

        return self._make_reply(recipient, sender, response)

    def _make_reply(self, from_user: str, to_user: str, content: str) -> str:
        return (
            "<xml>"
            f"<ToUserName><![CDATA[{to_user}]]></ToUserName>"
            f"<FromUserName><![CDATA[{from_user}]]></FromUserName>"
            f"<CreateTime>{int(time.time())}</CreateTime>"
            "<MsgType><![CDATA[text]]></MsgType>"
            f"<Content><![CDATA[{content}]]></Content>"
            "</xml>"
        )
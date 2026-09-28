from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from .base import Bot, BotMeta

log = logging.getLogger("automate.bots.wecom")

META = BotMeta(
    id="wecom",
    label="企业微信 (WeCom)",
    description=(
        "Self-built app on work.weixin.qq.com. Active push, no 5s reply limit. "
        "Best for internal/B-end use."
    ),
    docs_url="https://developer.work.weixin.qq.com/document/path/90664",
    config_fields=[
        {
            "name": "corp_id",
            "label": "企业 ID (CorpID)",
            "kind": "string",
            "required": True,
        },
        {
            "name": "agent_id",
            "label": "应用 AgentID",
            "kind": "string",
            "required": True,
        },
        {
            "name": "secret",
            "label": "应用 Secret",
            "kind": "password",
            "required": True,
        },
        {
            "name": "token",
            "label": "Token",
            "kind": "string",
            "required": True,
        },
        {
            "name": "encoding_aes_key",
            "label": "EncodingAESKey",
            "kind": "password",
            "required": False,
        },
    ],
)


class WeComBot(Bot):
    kind = "wecom"

    def __init__(self, *, config: dict, agent):
        self.config = config
        self.agent = agent
        self._status = "ready"
        self._access_token = ""
        self._access_token_expires = 0.0

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
        echostr: str | None = None,
    ) -> bool:
        token = (self.config.get("token") or "").strip()
        if not token:
            return False

        values = [token, timestamp, nonce]
        if echostr:
            values.append(echostr)

        digest = hashlib.sha1("".join(sorted(values)).encode()).hexdigest()
        return digest == signature

    def handle_inbound(self, raw_xml: bytes) -> str:
        try:
            root = ET.fromstring(raw_xml)
            from_user = (root.findtext("FromUserName") or "").strip()
            content = (root.findtext("Content") or "").strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("malformed wecom payload: %s", exc)
            return ""

        if not content:
            return ""

        log.info("[wecom] from %s: %s", from_user, content[:80])

        try:
            answer = self.agent(content) if self.agent else "(agent not wired)"
        except Exception as exc:  # noqa: BLE001
            answer = f"⚠ {type(exc).__name__}: {exc}"

        try:
            self.send_text(from_user, answer)
            return ""
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "active push failed, falling back to passive reply: %s",
                exc,
            )
            return (
                "<xml><MsgType>text</MsgType><Content><![CDATA["
                f"{answer[:1500]}"
                "]]></Content></xml>"
            )

    def _ensure_access_token(self) -> str:
        if (
            self._access_token
            and time.time() < self._access_token_expires - 60
        ):
            return self._access_token

        params = urllib.parse.urlencode(
            {
                "corpid": self.config["corp_id"],
                "corpsecret": self.config["secret"],
            }
        )
        url = f"https://qyapi.weixin.qq.com/cgi-bin/gettoken?{params}"

        with urllib.request.urlopen(url, timeout=15) as response:
            data = json.loads(response.read().decode())

        if data.get("errcode"):
            raise RuntimeError(f"wecom gettoken failed: {data}")

        self._access_token = data["access_token"]
        self._access_token_expires = time.time() + int(
            data.get("expires_in", 7200)
        )
        return self._access_token

    def send_text(self, to_user: str, text: str) -> dict:
        token = self._ensure_access_token()
        body = {
            "touser": to_user,
            "msgtype": "text",
            "agentid": int(self.config["agent_id"]),
            "text": {"content": text[:2048]},
        }
        url = (
            "https://qyapi.weixin.qq.com/cgi-bin/message/send"
            f"?access_token={token}"
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(request, timeout=15) as response:
            data = json.loads(response.read().decode())

        if data.get("errcode"):
            raise RuntimeError(f"wecom send failed: {data}")

        return data
"""Telegram long-polling bot integration."""

from __future__ import annotations

import json
import logging
import threading
import urllib.request

from .base import Bot, BotMeta

log = logging.getLogger("automate.bots.telegram")

META = BotMeta(
    id="telegram",
    label="Telegram",
    description=(
        "Personal Telegram bot via long polling. No public URL needed — "
        "works from your laptop / NAS / Docker."
    ),
    docs_url="https://core.telegram.org/bots/api",
    config_fields=[
        {
            "name": "bot_token",
            "label": "Bot token",
            "kind": "password",
            "required": True,
            "hint": "From @BotFather on Telegram. Looks like 1234:ABC...",
        },
    ],
)


class TelegramBot(Bot):
    kind = "telegram"

    def __init__(self, *, config: dict, agent):
        self.config = config
        self.agent = agent
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._offset = 0
        self._status = "stopped"
        self._last_error = ""

    @property
    def status(self) -> str:
        return self._status

    @property
    def last_error(self) -> str:
        return self._last_error

    def _api(self, method: str, params: dict | None = None) -> dict:
        token = self.config.get("bot_token", "").strip()
        if not token:
            raise RuntimeError("bot_token is empty")

        endpoint = f"https://api.telegram.org/bot{token}/{method}"
        payload = json.dumps(params or {}).encode()
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(request, timeout=35) as response:
            return json.loads(response.read().decode())

    def start(self) -> None:
        if self._thread:
            return

        self._stop.clear()

        try:
            response = self._api("getMe")
            if not response.get("ok"):
                raise RuntimeError(response.get("description", "getMe failed"))

            log.info("Telegram bot started: @%s", response["result"].get("username"))
            self._status = "running"
            self._last_error = ""
        except Exception as exc:
            self._status = "error"
            self._last_error = str(exc)
            raise

        self._thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="automate-tg-bot",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._status = "stopped"

    def _poll_loop(self) -> None:
        while not self._stop.is_set():
            try:
                response = self._api(
                    "getUpdates",
                    {"offset": self._offset, "timeout": 25},
                )

                for update in response.get("result", []):
                    self._offset = update["update_id"] + 1
                    message = update.get("message")

                    if not message or "text" not in message:
                        continue

                    self._handle(message)
            except Exception as exc:
                log.warning("Telegram poll error: %s", exc)
                self._last_error = str(exc)

                if self._stop.wait(5):
                    break

    def _handle(self, msg: dict) -> None:
        chat_id = msg["chat"]["id"]
        text = msg["text"].strip()

        if not text:
            return

        log.info("[telegram] from %s: %s", chat_id, text[:80])

        try:
            self._api(
                "sendChatAction",
                {"chat_id": chat_id, "action": "typing"},
            )
        except Exception:
            pass

        try:
            reply = self.agent(text) if self.agent else "(agent not wired)"
        except Exception as exc:
            reply = f"⚠ {type(exc).__name__}: {exc}"

        for chunk in _chunks(reply, 3500):
            try:
                self._api(
                    "sendMessage",
                    {"chat_id": chat_id, "text": chunk},
                )
            except Exception as exc:
                log.warning("telegram sendMessage failed: %s", exc)
                break


def _chunks(text: str, n: int):
    for start in range(0, len(text), n):
        yield text[start:start + n]
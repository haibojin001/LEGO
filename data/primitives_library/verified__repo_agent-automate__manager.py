from __future__ import annotations

import json
import logging
import time
from typing import Any

from ..agent import AgentLoop
from ..store import Database
from .base import Bot, BotMeta
from .telegram import META as TELEGRAM_META, TelegramBot
from .wechat_oa import META as OA_META, WeChatOABot
from .wechat_personal import META as WX_PERSONAL_META, WeChatPersonalBot
from .wecom import META as WECOM_META, WeComBot

log = logging.getLogger("automate.bots.manager")

CATALOG: dict[str, tuple[BotMeta, type[Bot]]] = {
    TELEGRAM_META.id: (TELEGRAM_META, TelegramBot),
    OA_META.id: (OA_META, WeChatOABot),
    WECOM_META.id: (WECOM_META, WeComBot),
    WX_PERSONAL_META.id: (WX_PERSONAL_META, WeChatPersonalBot),
}


class BotManager:
    def __init__(self, *, db: Database, agent: AgentLoop):
        self.db = db
        self.agent = agent
        self._instances: dict[str, Bot] = {}

    def _agent_run(self, prompt: str) -> str:
        try:
            response = self.agent.run(prompt, source="bot")
            return response.final or "(no reply)"
        except RuntimeError as exc:
            return f"⚠ {exc}"

    def list_kinds(self) -> list[dict]:
        result = []
        for metadata, _bot_class in CATALOG.values():
            result.append(
                {
                    "id": metadata.id,
                    "label": metadata.label,
                    "description": metadata.description,
                    "config_fields": metadata.config_fields,
                    "risks": metadata.risks,
                    "docs_url": metadata.docs_url,
                }
            )
        return result

    def list_instances(self) -> list[dict]:
        records = self.db.fetchall("SELECT * FROM bots ORDER BY id") or []
        result = []

        for record in records:
            active = self._instances.get(record["id"])
            result.append(
                {
                    "id": record["id"],
                    "enabled": bool(record["enabled"]),
                    "status": active.status if active else record["status"],
                    "last_error": record.get("last_error")
                    or (active and active.last_error)
                    or "",
                    "config_set": bool(record["config_enc"]),
                    "updated_at": record["updated_at"],
                }
            )

        return result

    def get_config(self, bot_id: str) -> dict:
        record = self.db.fetchone(
            "SELECT config_enc FROM bots WHERE id = ?",
            (bot_id,),
        )
        if not record or not record["config_enc"]:
            return {}

        try:
            decrypted = self.db.vault.decrypt(record["config_enc"])
            return json.loads(decrypted)
        except Exception as exc:
            log.warning("config decrypt failed for %s: %s", bot_id, exc)
            return {}

    def save_config(self, bot_id: str, config: dict) -> None:
        if bot_id not in CATALOG:
            raise ValueError(f"unknown bot kind: {bot_id}")

        current = self.get_config(bot_id)
        for key, value in config.items():
            if value == "" or value is None:
                continue
            current[key] = value

        encrypted = self.db.vault.encrypt(json.dumps(current))
        self.db.execute(
            "INSERT INTO bots (id, enabled, config_enc, status, updated_at) "
            "VALUES (?, 0, ?, 'stopped', ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "config_enc = excluded.config_enc, updated_at = excluded.updated_at",
            (bot_id, encrypted, time.time()),
        )

    def start(self, bot_id: str) -> dict:
        entry = CATALOG.get(bot_id)
        if entry is None:
            raise ValueError(f"unknown bot: {bot_id}")

        _metadata, bot_class = entry
        configuration = self.get_config(bot_id)
        if not configuration:
            raise RuntimeError("no config saved yet — fill in the form first")

        previous = self._instances.pop(bot_id, None)
        if previous is not None:
            try:
                previous.stop()
            except Exception:
                pass

        bot = bot_class(config=configuration, agent=self._agent_run)

        try:
            bot.start()
        except Exception as exc:
            self.db.execute(
                "UPDATE bots SET enabled=0, status='error', last_error=?, "
                "updated_at=? WHERE id=?",
                (str(exc), time.time(), bot_id),
            )
            raise

        self._instances[bot_id] = bot
        self.db.execute(
            "UPDATE bots SET enabled=1, status=?, last_error=NULL, "
            "updated_at=? WHERE id=?",
            (bot.status, time.time(), bot_id),
        )
        return {"status": bot.status}

    def stop(self, bot_id: str) -> None:
        bot = self._instances.pop(bot_id, None)
        if bot is not None:
            try:
                bot.stop()
            except Exception:
                pass

        self.db.execute(
            "UPDATE bots SET enabled=0, status='stopped', updated_at=? WHERE id=?",
            (time.time(), bot_id),
        )

    def get_instance(self, bot_id: str) -> Bot | None:
        return self._instances.get(bot_id)

    def restart_enabled(self) -> None:
        records = self.db.fetchall("SELECT id FROM bots WHERE enabled = 1") or []
        for record in records:
            try:
                self.start(record["id"])
            except Exception as exc:
                log.warning("could not auto-start bot %s: %s", record["id"], exc)
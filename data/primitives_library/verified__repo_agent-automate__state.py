from __future__ import annotations

from dataclasses import dataclass

from ..agent import AgentLoop
from ..bots import BotManager
from ..providers import ProviderManager
from ..reminders import Scheduler
from ..settings import PATHS, SERVER
from ..store import Database, get_db
from ..tools import ToolRegistry, build_default_registry


@dataclass
class AppState:
    db: Database
    providers: ProviderManager
    registry: ToolRegistry
    agent: AgentLoop
    scheduler: Scheduler
    bots: BotManager


def build_state() -> AppState:
    PATHS.ensure()

    db = get_db()

    providers = ProviderManager(db)
    providers.seed_catalog()

    registry = build_default_registry()

    agent = AgentLoop(
        db=db,
        providers=providers,
        registry=registry,
    )

    scheduler = Scheduler(db)
    scheduler.start()

    bots = BotManager(db=db, agent=agent)
    bots.restart_enabled()

    return AppState(
        db=db,
        providers=providers,
        registry=registry,
        agent=agent,
        scheduler=scheduler,
        bots=bots,
    )
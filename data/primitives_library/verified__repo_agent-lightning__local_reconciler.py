from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import json
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass

import httpx
import structlog
from omegaconf import DictConfig

from agentlightning.client import AgentLightningAsyncClient
from agentlightning.schemas import (
    DEFAULT_ATTEMPT_ID,
    Rollout,
    RolloutPatch,
    RolloutState,
    RolloutStatusPatch,
)

log = structlog.get_logger()

_SHUTDOWN_WAIT_TIMEOUT = 5.0


def _is_native_windows() -> bool:
    return os.name == "nt"


def _run_local_reconciler_worker(agent_class_path: str) -> int:
    try:
        if ":" in agent_class_path:
            module_name, class_name = agent_class_path.split(":", 1)
        else:
            module_name, class_name = agent_class_path.rsplit(".", 1)

        agent_class = getattr(importlib.import_module(module_name), class_name)
        if not isinstance(agent_class, type):
            raise TypeError(f"{agent_class_path} is not a class")

        outcome = agent_class().run()
        if inspect.isawaitable(outcome):
            asyncio.run(outcome)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


@dataclass
class Proc:
    """In-flight local subprocess."""

    attempt_id: str
    proc: asyncio.subprocess.Process
    spawned_at: float
    killed: bool = False


def _resolve_input_path(task_input: object, path: str) -> object:
    if path == "input":
        return task_input

    if not path.startswith("input."):
        return path

    current = task_input
    for component in path.split(".")[1:]:
        if isinstance(current, dict) and component in current:
            current = current[component]
        elif isinstance(current, list) and component.isdigit() and int(component) < len(current):
            current = current[int(component)]
        else:
            raise ValueError(f"local.env_map path not found: {path}")
    return current


def _build_env_from_map(task_input: object, env_map: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for variable, source_path in env_map.items():
        value = _resolve_input_path(task_input, source_path)
        if isinstance(value, str):
            result[variable] = value
            continue
        try:
            result[variable] = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"local.env_map.{variable} value is not JSON serializable"
            ) from exc
    return result


class LocalReconciler:
    """Local-mode rollout reconciler."""

    def __init__(
        self,
        api: AgentLightningAsyncClient,
        config: DictConfig,
    ) -> None:
        assert config.runner_type == "local"
        self._api = api
        self._config = config
        self._runner_config = config.local_runner
        self._pool_size = int(self._runner_config.maximum_size)
        self._tick_interval = float(self._runner_config.poll_interval)
        self._rid_to_proc: dict[str, Proc] = {}
        self._stop = asyncio.Event()

    async def run(self) -> None:
        if _is_native_windows():
            raise RuntimeError(
                "runner_type=local is not supported on native Windows; use Linux "
                "(for example, WSL) instead."
            )

        log.info(
            "LocalReconciler starting",
            pool_size=self._pool_size,
            tick=self._tick_interval,
        )
        try:
            await self._reconcile_loop()
        finally:
            await self._shutdown()

    def stop(self) -> None:
        self._stop.set()

    async def _reconcile_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._reconcile_once()
            except Exception:
                log.exception("Local reconcile error")

            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self._tick_interval,
                )
                break
            except TimeoutError:
                continue

    async def _reconcile_once(self, *, spawn_queued: bool = True) -> None:
        query = httpx.QueryParams()
        query = query.add("state_in", RolloutState.QUEUING.value)
        query = query.add("state_in", RolloutState.RUNNING.value)
        query = query.add("limit", 50)

        response = await self._api.get("/api/rollouts", params=query)
        response.raise_for_status()

        rollouts = [Rollout.model_validate(item) for item in response.json()]
        by_id = {rollout.rollout_id: rollout for rollout in rollouts}
        live_count = sum(
            process.proc.returncode is None
            for process in self._rid_to_proc.values()
        )

        for rollout in rollouts:
            existing = self._rid_to_proc.get(rollout.rollout_id)

            if existing is None:
                if (
                    spawn_queued
                    and not self._stop.is_set()
                    and rollout.status.state == RolloutState.QUEUING
                    and live_count < self._pool_size
                ):
                    if await self._spawn_for(rollout):
                        live_count += 1
                elif rollout.status.state == RolloutState.RUNNING:
                    await self._patch(
                        rollout.rollout_id,
                        RolloutState.FAILED,
                        "local subprocess is not running",
                    )
                continue

            if existing.proc.returncode is None:
                if rollout.status.state == RolloutState.QUEUING:
                    await self._patch(
                        rollout.rollout_id,
                        RolloutState.RUNNING,
                        last_attempt_id=existing.attempt_id,
                    )
                continue

            await self._finish_proc(rollout, existing)

        now = time.monotonic()
        for rollout_id, process in list(self._rid_to_proc.items()):
            if process.proc.returncode is not None:
                continue

            rollout = by_id.get(rollout_id)
            timeout = (
                float(rollout.config.timeout_seconds)
                if rollout is not None and rollout.config.timeout_seconds
                else None
            )
            if (
                timeout is not None
                and now - process.spawned_at > timeout
                and await self._kill_process_group(rollout_id, process)
            ):
                await self._patch(
                    rollout_id,
                    RolloutState.FAILED,
                    "local subprocess timed out",
                )

    async def _finish_proc(self, rollout: Rollout, item: Proc) -> bool:
        if rollout.status.state == RolloutState.QUEUING:
            moved_to_running = await self._patch(
                rollout.rollout_id,
                RolloutState.RUNNING,
                last_attempt_id=item.attempt_id,
            )
            if not moved_to_running:
                return False

        if item.proc.returncode == 0:
            return await self._patch(
                rollout.rollout_id,
                RolloutState.SUCCEEDED,
                last_attempt_id=item.attempt_id,
            )

        return await self._patch(
            rollout.rollout_id,
            RolloutState.FAILED,
            f"subprocess exited with code {item.proc.returncode}",
        )

    async def _kill_process_group(self, rollout_id: str, item: Proc) -> bool:
        """SIGKILL the worker process group and wait for exit."""
        if item.proc.returncode is not None:
            return True

        if not item.killed:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(item.proc.pid, signal.SIGKILL)
            item.killed = True
            log.info(
                "SIGKILL sent to subprocess group",
                rollout_id=rollout_id,
                pid=item.proc.pid,
            )

        try:
            await asyncio.wait_for(
                item.proc.wait(),
                timeout=_SHUTDOWN_WAIT_TIMEOUT,
            )
            return True
        except TimeoutError:
            log.warning(
                "Subprocess did not exit after SIGKILL within 5s",
                rollout_id=rollout_id,
                pid=item.proc.pid,
            )
            return False

    async def _spawn_for(self, rollout: Rollout) -> bool:
        """Spawn one local subprocess for a rollout."""
        try:
            attempt_id = DEFAULT_ATTEMPT_ID
            if rollout.config.local is None or not rollout.config.local.agent_class:
                raise ValueError(
                    "invalid rollout config: missing config.local.agent_class"
                )

            agent_class_path = rollout.config.local.agent_class
            mode = "train" if rollout.is_train else "val"
            configured_agent_url = self._config.agl_server.get("agent_url", None)
            base_url = str(configured_agent_url or self._config.agl_server.url).rstrip("/")

            environment = {
                **os.environ,
                "AGL_KEY": str(self._config.agl_server.key or ""),
                "AGL_OPENAI_BASE_URL": (
                    f"{base_url}/proxy/rollout/{rollout.rollout_id}/attempt/"
                    f"{attempt_id}/mode/{mode}/openai/v1"
                ),
                "AGL_EVENT_URL": (
                    f"{base_url}/api/rollouts/{rollout.rollout_id}/attempt/"
                    f"{attempt_id}/events"
                ),
            }
            environment.update(
                _build_env_from_map(
                    rollout.input,
                    rollout.config.local.env_map,
                )
            )

            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "from agentlightning.controller.local_reconciler "
                    "import _run_local_reconciler_worker; "
                    "sys.exit(_run_local_reconciler_worker(sys.argv[1]))"
                ),
                agent_class_path,
                env=environment,
                start_new_session=True,
            )
            self._rid_to_proc[rollout.rollout_id] = Proc(
                attempt_id=attempt_id,
                proc=proc,
                spawned_at=time.monotonic(),
            )
            log.info(
                "Local subprocess started",
                rollout_id=rollout.rollout_id,
                pid=proc.pid,
                attempt_id=attempt_id,
            )
            return True
        except Exception as exc:
            log.exception(
                "Failed to spawn local subprocess",
                rollout_id=rollout.rollout_id,
            )
            await self._patch(
                rollout.rollout_id,
                RolloutState.FAILED,
                f"failed to spawn local subprocess: {exc}",
            )
            return False

    async def _patch(
        self,
        rollout_id: str,
        state: RolloutState,
        message: str | None = None,
        *,
        last_attempt_id: str | None = None,
    ) -> bool:
        try:
            status = RolloutStatusPatch(
                state=state,
                message=message,
                last_attempt_id=last_attempt_id,
            )
            patch = RolloutPatch(status=status)
            response = await self._api.patch(
                f"/api/rollouts/{rollout_id}",
                json=patch.model_dump(mode="json", exclude_none=True),
            )
            response.raise_for_status()
            return True
        except Exception:
            log.exception(
                "Failed to patch rollout status",
                rollout_id=rollout_id,
                state=state.value,
            )
            return False

    async def _shutdown(self) -> None:
        live = [
            (rollout_id, item)
            for rollout_id, item in self._rid_to_proc.items()
            if item.proc.returncode is None
        ]

        if live:
            await asyncio.gather(
                *(
                    self._kill_process_group(rollout_id, item)
                    for rollout_id, item in live
                ),
                return_exceptions=True,
            )

        with contextlib.suppress(Exception):
            await self._reconcile_once(spawn_queued=False)
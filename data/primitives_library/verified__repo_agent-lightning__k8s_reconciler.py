from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, AsyncIterator, cast

import httpx
import kr8s
import kr8s.asyncio
import structlog
import yaml
from jinja2 import Environment
from kr8s.asyncio import objects as k8s_objects
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

MANAGED_BY_SELECTOR = "app.kubernetes.io/managed-by=agentlightning"
JOB_CREATION_WINDOW_SECONDS = 60


def build_job_name(rollout_id: str) -> str:
    """Return the deterministic Kubernetes Job name for a rollout."""
    return f"agl-rollout-{rollout_id}"


def build_job_spec(rollout: Rollout, controller_config: DictConfig) -> dict[str, Any]:
    """Render and augment a Kubernetes Job manifest for a rollout."""
    k8s_config = getattr(rollout.config, "k8s", None)
    template = getattr(k8s_config, "job_template", None) if k8s_config else None
    if not template:
        raise ValueError("invalid rollout config: missing config.k8s.job_template")

    environment = Environment()
    environment.filters["yaml_escape"] = lambda value: json.dumps(str(value), ensure_ascii=True)
    rendered = environment.from_string(template).render(
        job_name=build_job_name(rollout.rollout_id),
        input=rollout.input,
    )

    documents = [document for document in yaml.safe_load_all(rendered) if document is not None]
    if len(documents) != 1:
        raise ValueError(
            "invalid rollout config: config.k8s.job_template must render exactly one YAML document"
        )

    job = documents[0]
    if not isinstance(job, dict) or job.get("kind") != "Job":
        raise ValueError(
            "invalid rollout config: config.k8s.job_template must render a Kubernetes Job"
        )

    metadata = job.setdefault("metadata", {})
    metadata["name"] = build_job_name(rollout.rollout_id)
    metadata["namespace"] = controller_config.k8s_runner.namespace

    labels = metadata.setdefault("labels", {})
    labels["app.kubernetes.io/managed-by"] = "agentlightning"
    labels["agentlightning/rollout-id"] = rollout.rollout_id
    labels["agentlightning/attempt-id"] = DEFAULT_ATTEMPT_ID

    spec = job.setdefault("spec", {})
    spec["backoffLimit"] = 0
    spec["ttlSecondsAfterFinished"] = controller_config.k8s_runner.ttl_after_finished

    timeout_seconds = getattr(rollout.config, "timeout_seconds", None)
    if timeout_seconds:
        spec["activeDeadlineSeconds"] = timeout_seconds

    pod_spec = spec.setdefault("template", {}).setdefault("spec", {})
    pod_spec["restartPolicy"] = "Never"

    mode = "train" if rollout.is_train else "val"
    configured_agent_url = controller_config.agl_server.get("agent_url", None)
    agent_base_url = str(configured_agent_url or controller_config.agl_server.url).rstrip("/")

    openai_base_url = (
        f"{agent_base_url}/proxy/rollout/{rollout.rollout_id}/attempt/"
        f"{DEFAULT_ATTEMPT_ID}/mode/{mode}/openai/v1"
    )
    event_url = (
        f"{agent_base_url}/api/rollouts/{rollout.rollout_id}/attempt/"
        f"{DEFAULT_ATTEMPT_ID}/events"
    )
    key = str(controller_config.agl_server.key or "")

    for container in pod_spec.get("containers", []):
        container_env = container.setdefault("env", [])
        for name, value in {
            "AGL_OPENAI_BASE_URL": openai_base_url,
            "AGL_EVENT_URL": event_url,
            "AGL_KEY": key,
        }.items():
            existing = next(
                (item for item in container_env if isinstance(item, dict) and item.get("name") == name),
                None,
            )
            if existing is None:
                container_env.append({"name": name, "value": value})
            else:
                existing.clear()
                existing.update({"name": name, "value": value})

    return job


class K8sReconciler:
    """Controller which reconciles rollout records into Kubernetes Jobs."""

    def __init__(self, api: AgentLightningAsyncClient, config: DictConfig) -> None:
        self._api = api
        self._config = config
        self._runner_config = config.k8s_runner
        self._namespace = str(self._runner_config.namespace)
        self._k8s_api: Any | None = None
        self._stop = asyncio.Event()
        self._job_creation_timestamps: deque[float] = deque()

    async def _get_k8s_api(self) -> Any:
        if self._k8s_api is None:
            self._k8s_api = await kr8s.asyncio.api()
        return self._k8s_api

    async def run(self) -> None:
        """Run periodic reconciliation and Job watch loops until stopped."""
        log.info(
            "Controller starting",
            namespace=self._namespace,
            poll_interval=self._runner_config.poll_interval,
        )
        try:
            await asyncio.gather(
                self._periodic_reconcile_loop(),
                self._watch_jobs_loop(),
            )
        except asyncio.CancelledError:
            log.info("Controller stopped")

    def stop(self) -> None:
        """Signal this controller to stop."""
        self._stop.set()

    async def _periodic_reconcile_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self._reconcile_once()
            except Exception:
                log.exception("Periodic reconcile error")

            try:
                await asyncio.wait_for(
                    self._stop.wait(),
                    timeout=self._runner_config.poll_interval,
                )
                return
            except TimeoutError:
                continue

    async def _reconcile_once(self) -> None:
        rollouts = await self._query_rollouts(
            state_in=[RolloutState.QUEUING, RolloutState.RUNNING],
            limit=500,
        )
        api = await self._get_k8s_api()

        jobs = [
            cast(k8s_objects.Job, job).raw
            async for job in k8s_objects.Job.async_list(
                namespace=self._namespace,
                label_selector=MANAGED_BY_SELECTOR,
                api=api,
            )
        ]
        jobs_by_name = {
            str(job.get("metadata", {}).get("name", "")): job
            for job in jobs
            if isinstance(job, dict)
        }

        active_jobs = 0
        for job in jobs:
            state, _ = self._job_terminal_state(job)
            if state is None:
                active_jobs += 1

        max_concurrent = self._config_value("max_concurrent_jobs", None)

        for rollout in rollouts:
            job_name = rollout.status.k8s_job_name or build_job_name(rollout.rollout_id)
            job = jobs_by_name.get(job_name)

            if job is None:
                if rollout.status.state == RolloutState.QUEUING:
                    if max_concurrent is not None and active_jobs >= int(max_concurrent):
                        continue
                    created = await self._create_job(rollout)
                    if created:
                        active_jobs += 1
                    continue

                log.warning(
                    "Orphaned running rollout — Job gone",
                    rollout_id=rollout.rollout_id,
                    job_name=job_name,
                )
                await self._patch_status(
                    rollout.rollout_id,
                    state=RolloutState.FAILED,
                    error_message="Job disappeared",
                )
                continue

            labels = job.get("metadata", {}).get("labels", {})
            attempt_id = labels.get("agentlightning/attempt-id") or DEFAULT_ATTEMPT_ID
            state, error_message = self._job_terminal_state(job)

            if state is None:
                if rollout.status.state == RolloutState.QUEUING:
                    await self._patch_status(
                        rollout.rollout_id,
                        state=RolloutState.RUNNING,
                        k8s_job_name=job_name,
                        last_attempt_id=attempt_id,
                    )
                continue

            if rollout.status.state == RolloutState.QUEUING and state == RolloutState.SUCCEEDED:
                await self._patch_status(
                    rollout.rollout_id,
                    state=RolloutState.RUNNING,
                    k8s_job_name=job_name,
                    last_attempt_id=attempt_id,
                )

            await self._patch_status(
                rollout.rollout_id,
                state=state,
                k8s_job_name=job_name,
                last_attempt_id=attempt_id,
                error_message=error_message,
            )

    async def _create_job(self, rollout: Rollout) -> bool:
        if self._creation_rate_limited():
            log.warning(
                "Job creation rate limit reached",
                rollout_id=rollout.rollout_id,
            )
            return False

        manifest = build_job_spec(rollout, self._config)
        api = await self._get_k8s_api()
        job = k8s_objects.Job(manifest, api=api)

        try:
            await job.create()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 409:
                log.info(
                    "Job already exists",
                    rollout_id=rollout.rollout_id,
                    job_name=build_job_name(rollout.rollout_id),
                )
            else:
                raise
        except Exception as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code != 409:
                raise

        self._job_creation_timestamps.append(time.monotonic())
        await self._patch_status(
            rollout.rollout_id,
            state=RolloutState.RUNNING,
            k8s_job_name=build_job_name(rollout.rollout_id),
            last_attempt_id=DEFAULT_ATTEMPT_ID,
        )
        log.info(
            "Created rollout Job",
            rollout_id=rollout.rollout_id,
            job_name=build_job_name(rollout.rollout_id),
        )
        return True

    def _creation_rate_limited(self) -> bool:
        now = time.monotonic()
        while (
            self._job_creation_timestamps
            and now - self._job_creation_timestamps[0] >= JOB_CREATION_WINDOW_SECONDS
        ):
            self._job_creation_timestamps.popleft()

        maximum = self._config_value("max_job_creations_per_minute", None)
        if maximum is None:
            maximum = self._config_value("max_job_creations", None)
        return maximum is not None and len(self._job_creation_timestamps) >= int(maximum)

    async def _watch_jobs_loop(self) -> None:
        while not self._stop.is_set():
            try:
                api = await self._get_k8s_api()
                watch = k8s_objects.Job.async_watch(
                    namespace=self._namespace,
                    label_selector=MANAGED_BY_SELECTOR,
                    api=api,
                )
                async for event in cast(AsyncIterator[Any], watch):
                    if self._stop.is_set():
                        return
                    try:
                        await self._handle_job_event(event)
                    except Exception:
                        log.exception("Error handling Job watch event")
            except asyncio.CancelledError:
                raise
            except Exception:
                if not self._stop.is_set():
                    log.exception("Job watch error")
                    try:
                        await asyncio.wait_for(self._stop.wait(), timeout=1)
                    except TimeoutError:
                        pass

    async def _handle_job_event(self, event: Any) -> None:
        event_type: str | None = None
        job: Any = event

        if isinstance(event, tuple) and len(event) == 2:
            event_type, job = event
        elif isinstance(event, dict) and "object" in event:
            event_type = event.get("type")
            job = event.get("object")

        if event_type == "DELETED":
            return

        raw = getattr(job, "raw", job)
        if not isinstance(raw, dict):
            return

        state, error_message = self._job_terminal_state(raw)
        if state is None:
            return

        metadata = raw.get("metadata", {})
        labels = metadata.get("labels", {})
        rollout_id = labels.get("agentlightning/rollout-id")
        if not rollout_id:
            return

        attempt_id = labels.get("agentlightning/attempt-id") or DEFAULT_ATTEMPT_ID
        job_name = metadata.get("name") or build_job_name(rollout_id)

        await self._patch_status(
            rollout_id,
            state=state,
            k8s_job_name=job_name,
            last_attempt_id=attempt_id,
            error_message=error_message,
        )

    @staticmethod
    def _job_terminal_state(job: dict[str, Any]) -> tuple[RolloutState | None, str | None]:
        status = job.get("status") or {}
        for condition in status.get("conditions") or []:
            if condition.get("status") != "True":
                continue
            if condition.get("type") == "Complete":
                return RolloutState.SUCCEEDED, None
            if condition.get("type") == "Failed":
                reason = condition.get("reason", "Unknown")
                message = condition.get("message", "")
                error_message = f"Job failed: {reason}"
                if message:
                    error_message += f" — {message}"
                return RolloutState.FAILED, error_message

        if status.get("succeeded", 0) > 0:
            return RolloutState.SUCCEEDED, None
        if status.get("failed", 0) > 0:
            return RolloutState.FAILED, "Job failed"
        return None, None

    async def _query_rollouts(
        self,
        *,
        state_in: list[RolloutState],
        limit: int,
    ) -> list[Rollout]:
        result = await self._call_api(
            ("query_rollouts", "list_rollouts"),
            state_in=state_in,
            limit=limit,
        )
        if isinstance(result, list):
            return cast(list[Rollout], result)
        for attribute in ("rollouts", "items", "data"):
            value = getattr(result, attribute, None)
            if value is not None:
                return list(value)
            if isinstance(result, dict) and attribute in result:
                return list(result[attribute])
        return list(result)

    async def _patch_status(
        self,
        rollout_id: str,
        *,
        state: RolloutState,
        k8s_job_name: str | None = None,
        last_attempt_id: str | None = None,
        error_message: str | None = None,
    ) -> Any:
        values: dict[str, Any] = {"state": state}
        if k8s_job_name is not None:
            values["k8s_job_name"] = k8s_job_name
        if last_attempt_id is not None:
            values["last_attempt_id"] = last_attempt_id
        if error_message is not None:
            values["error_message"] = error_message

        patch = RolloutPatch(status=RolloutStatusPatch(**values))
        return await self._call_api(
            ("patch_rollout", "update_rollout"),
            rollout_id,
            patch,
        )

    async def _call_api(self, names: tuple[str, ...], *args: Any, **kwargs: Any) -> Any:
        targets = [self._api]
        rollouts = getattr(self._api, "rollouts", None)
        if rollouts is not None:
            targets.insert(0, rollouts)

        for target in targets:
            for name in names:
                method = getattr(target, name, None)
                if method is None:
                    continue
                result = method(*args, **kwargs)
                if inspect.isawaitable(result):
                    return await result
                return result

        raise AttributeError(f"AgentLightningAsyncClient does not provide any of {names!r}")

    def _config_value(self, name: str, default: Any) -> Any:
        value = getattr(self._runner_config, name, None)
        if value is None and hasattr(self._runner_config, "get"):
            value = self._runner_config.get(name, default)
        return default if value is None else value
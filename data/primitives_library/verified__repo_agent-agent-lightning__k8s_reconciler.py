from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from typing import Any, cast

import httpx
import kr8s
import kr8s.asyncio
import structlog
import yaml
from jinja2 import Environment
from kr8s.asyncio import objects as k8s_objects
from omegaconf import DictConfig

from agentlightning.client import AgentLightningAsyncClient
from agentlightning.schemas import DEFAULT_ATTEMPT_ID, Rollout, RolloutPatch, RolloutState, RolloutStatusPatch

log = structlog.get_logger()

MANAGED_BY_SELECTOR = "app.kubernetes.io/managed-by=agentlightning"
JOB_CREATION_WINDOW_SECONDS = 60


def build_job_name(rollout_id: str) -> str:
    """Deterministically derive the Kubernetes Job name for a rollout."""
    return f"agl-rollout-{rollout_id}"


def build_job_spec(rollout: Rollout, controller_config: DictConfig) -> dict[str, Any]:
    """Render the rollout Job template and add controller-owned fields."""
    template = rollout.config.k8s.job_template if rollout.config.k8s else None
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

    if rollout.config.timeout_seconds:
        spec["activeDeadlineSeconds"] = rollout.config.timeout_seconds

    pod_spec = spec.setdefault("template", {}).setdefault("spec", {})
    pod_spec["restartPolicy"] = "Never"

    mode = "train" if rollout.is_train else "val"
    agent_base_url = str(
        controller_config.agl_server.get("agent_url", None) or controller_config.agl_server.url
    ).rstrip("/")
    openai_base_url = (
        f"{agent_base_url}/proxy/rollout/{rollout.rollout_id}/attempt/"
        f"{DEFAULT_ATTEMPT_ID}/mode/{mode}/openai/v1"
    )
    event_url = (
        f"{agent_base_url}/api/rollouts/{rollout.rollout_id}/attempt/"
        f"{DEFAULT_ATTEMPT_ID}/events"
    )

    for container in pod_spec.get("containers", []):
        container_env = container.setdefault("env", [])
        values = {
            "AGL_OPENAI_BASE_URL": openai_base_url,
            "AGL_EVENT_URL": event_url,
            "AGL_KEY": str(controller_config.agl_server.key or ""),
        }
        for name, value in values.items():
            existing = next((item for item in container_env if item.get("name") == name), None)
            if existing is None:
                container_env.append({"name": name, "value": value})
            else:
                existing.clear()
                existing.update({"name": name, "value": value})

    return job


class K8sReconciler:
    """Reconcile rollout records with Kubernetes Jobs."""

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
        """Run the polling reconciler and Kubernetes Job watcher."""
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
        """Request that all controller loops stop."""
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
                pass

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
            job.get("metadata", {}).get("name", ""): job
            for job in jobs
        }

        for rollout in rollouts:
            job_name = rollout.status.k8s_job_name or build_job_name(rollout.rollout_id)
            job = jobs_by_name.get(job_name)

            if job is None:
                if rollout.status.state == RolloutState.QUEUING:
                    await self._create_job(rollout)
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

            await self._reconcile_job(rollout, job)

    async def _reconcile_job(self, rollout: Rollout, job: dict[str, Any]) -> None:
        metadata = job.get("metadata", {})
        labels = metadata.get("labels", {})
        job_name = metadata.get("name") or build_job_name(rollout.rollout_id)
        attempt_id = labels.get("agentlightning/attempt-id") or DEFAULT_ATTEMPT_ID

        state, error_message = self._job_result(job)
        if state is None:
            if rollout.status.state == RolloutState.QUEUING:
                await self._patch_status(
                    rollout.rollout_id,
                    state=RolloutState.RUNNING,
                    k8s_job_name=job_name,
                    last_attempt_id=attempt_id,
                )
            return

        if rollout.status.state == RolloutState.QUEUING:
            await self._patch_status(
                rollout.rollout_id,
                state=RolloutState.RUNNING,
                k8s_job_name=job_name,
                last_attempt_id=attempt_id,
            )

        await self._patch_status(
            rollout.rollout_id,
            state=state,
            error_message=error_message,
            k8s_job_name=job_name,
            last_attempt_id=attempt_id,
        )

    @staticmethod
    def _job_result(job: dict[str, Any]) -> tuple[RolloutState | None, str | None]:
        status = job.get("status", {})

        for condition in status.get("conditions", []):
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

    async def _create_job(self, rollout: Rollout) -> None:
        now = time.monotonic()
        while (
            self._job_creation_timestamps
            and now - self._job_creation_timestamps[0] >= JOB_CREATION_WINDOW_SECONDS
        ):
            self._job_creation_timestamps.popleft()

        maximum = getattr(self._runner_config, "max_job_creations_per_minute", None)
        if maximum is not None and maximum > 0 and len(self._job_creation_timestamps) >= maximum:
            log.warning(
                "Job creation rate limit reached",
                rollout_id=rollout.rollout_id,
                limit=maximum,
            )
            return

        manifest = build_job_spec(rollout, self._config)
        api = await self._get_k8s_api()

        try:
            job = k8s_objects.Job(manifest, api=api)
            await job.create()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 409:
                raise
            log.info(
                "Job already exists",
                rollout_id=rollout.rollout_id,
                job_name=build_job_name(rollout.rollout_id),
            )
        else:
            self._job_creation_timestamps.append(now)
            log.info(
                "Created Job for rollout",
                rollout_id=rollout.rollout_id,
                job_name=build_job_name(rollout.rollout_id),
            )

        await self._patch_status(
            rollout.rollout_id,
            state=RolloutState.RUNNING,
            k8s_job_name=build_job_name(rollout.rollout_id),
            last_attempt_id=DEFAULT_ATTEMPT_ID,
        )

    async def _watch_jobs_loop(self) -> None:
        while not self._stop.is_set():
            try:
                api = await self._get_k8s_api()
                async for event, watched_job in k8s_objects.Job.async_watch(
                    namespace=self._namespace,
                    label_selector=MANAGED_BY_SELECTOR,
                    api=api,
                ):
                    if self._stop.is_set():
                        return
                    if event not in {"ADDED", "MODIFIED"}:
                        continue
                    await self._handle_job_event(cast(k8s_objects.Job, watched_job).raw)
            except asyncio.CancelledError:
                raise
            except Exception:
                if self._stop.is_set():
                    return
                log.exception("Kubernetes Job watcher error")
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=1)
                    return
                except TimeoutError:
                    pass

    async def _handle_job_event(self, job: dict[str, Any]) -> None:
        metadata = job.get("metadata", {})
        labels = metadata.get("labels", {})
        rollout_id = labels.get("agentlightning/rollout-id")
        if not rollout_id:
            return

        state, _ = self._job_result(job)
        if state is None:
            return

        rollout = await self._get_rollout(rollout_id)
        if rollout is None:
            log.warning(
                "Received Job event for missing rollout",
                rollout_id=rollout_id,
                job_name=metadata.get("name"),
            )
            return

        if rollout.status.state in {RolloutState.SUCCEEDED, RolloutState.FAILED}:
            return

        await self._reconcile_job(rollout, job)

    async def _query_rollouts(
        self,
        state_in: list[RolloutState],
        limit: int,
    ) -> list[Rollout]:
        result = await self._api.list_rollouts(state_in=state_in, limit=limit)
        if isinstance(result, list):
            return result
        if hasattr(result, "items"):
            return list(result.items)
        if hasattr(result, "rollouts"):
            return list(result.rollouts)
        return list(result)

    async def _get_rollout(self, rollout_id: str) -> Rollout | None:
        try:
            return await self._api.get_rollout(rollout_id)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise

    async def _patch_status(
        self,
        rollout_id: str,
        *,
        state: RolloutState,
        error_message: str | None = None,
        k8s_job_name: str | None = None,
        last_attempt_id: str | None = None,
    ) -> Rollout:
        status_values: dict[str, Any] = {
            "state": state,
            "error_message": error_message,
        }
        if k8s_job_name is not None:
            status_values["k8s_job_name"] = k8s_job_name
        if last_attempt_id is not None:
            status_values["last_attempt_id"] = last_attempt_id

        patch = RolloutPatch(status=RolloutStatusPatch(**status_values))
        return await self._api.update_rollout(rollout_id, patch)
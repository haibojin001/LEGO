from __future__ import annotations

import asyncio
import contextlib
import signal

import hydra
from omegaconf import DictConfig

from agentlightning.client import AgentLightningAsyncClient


async def _run_controller(config: DictConfig) -> None:
    server_config = config.agl_server

    async with AgentLightningAsyncClient(
        base_url=str(server_config.url),
        key=str(server_config.key or "") or None,
    ) as api:
        runner_type = config.runner_type

        if runner_type == "k8s":
            try:
                from agentlightning.controller.k8s_reconciler import K8sReconciler
            except ImportError:
                raise RuntimeError(
                    "kr8s unavailable - install agentlightning[controller]"
                ) from None
            reconciler = K8sReconciler(api=api, config=config)
        elif runner_type == "local":
            from agentlightning.controller.local_reconciler import LocalReconciler

            reconciler = LocalReconciler(api=api, config=config)
        else:
            raise ValueError(f"unknown runner_type: {runner_type}")

        loop = asyncio.get_running_loop()
        for signum in (signal.SIGTERM, signal.SIGINT):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(signum, reconciler.stop)

        await reconciler.run()


@hydra.main(version_base=None, config_path="../config", config_name="controller")
def main(config: DictConfig) -> None:
    asyncio.run(_run_controller(config))


if __name__ == "__main__":
    main()
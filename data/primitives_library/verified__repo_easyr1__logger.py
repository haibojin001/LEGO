import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch

from ..py_functional import convert_dict_to_str, flatten_dict, is_package_available, unflatten_dict
from .gen_logger import AggregateGenerationsLogger

if is_package_available("mlflow"):
    import mlflow  # type: ignore

if is_package_available("tensorboard"):
    from torch.utils.tensorboard import SummaryWriter

if is_package_available("wandb"):
    import wandb  # type: ignore

if is_package_available("swanlab"):
    import swanlab  # type: ignore


class Logger(ABC):
    @abstractmethod
    def __init__(self, config: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def log(self, data: dict[str, Any], step: int) -> None:
        ...

    def finish(self) -> None:
        pass


class ConsoleLogger(Logger):
    def __init__(self, config: dict[str, Any]) -> None:
        print("Config\n" + convert_dict_to_str(config))

    def log(self, data: dict[str, Any], step: int) -> None:
        formatted = convert_dict_to_str(unflatten_dict(data))
        print(f"Step {step}\n" + formatted)


class FileLogger(Logger):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        directory = config["trainer"]["save_checkpoint_path"]
        print(f"Initializing logging file to {directory}.")
        os.makedirs(directory, exist_ok=True)

        with open(os.path.join(directory, "experiment_config.json"), "w") as handle:
            json.dump(config, handle, indent=2)

        with open(os.path.join(directory, "experiment_log.jsonl"), "w"):
            pass

        with open(os.path.join(directory, "generations.log"), "w"):
            pass

    def log(self, data: dict[str, Any], step: int) -> None:
        directory = self.config["trainer"]["save_checkpoint_path"]
        payload = {"step": step, **unflatten_dict(data)}
        with open(os.path.join(directory, "experiment_log.jsonl"), "a") as handle:
            handle.write(json.dumps(payload) + "\n")


class MlflowLogger(Logger):
    def __init__(self, config: dict[str, Any]) -> None:
        mlflow.start_run(run_name=config["trainer"]["experiment_name"])
        mlflow.log_params(flatten_dict(config))

    def log(self, data: dict[str, Any], step: int) -> None:
        mlflow.log_metrics(metrics=data, step=step)


class SwanlabLogger(Logger):
    def __init__(self, config: dict[str, Any]) -> None:
        key = os.getenv("SWANLAB_API_KEY")
        directory = os.getenv("SWANLAB_DIR", "swanlab_log")
        run_mode = os.getenv("SWANLAB_MODE", "cloud")

        if key:
            swanlab.login(key)

        swanlab.init(
            project=config["trainer"]["project_name"],
            experiment_name=config["trainer"]["experiment_name"],
            config={"UPPERFRAMEWORK": "EasyR1", "FRAMEWORK": "veRL", **config},
            logdir=directory,
            mode=run_mode,
        )

    def log(self, data: dict[str, Any], step: int) -> None:
        swanlab.log(data=data, step=step)

    def finish(self) -> None:
        swanlab.finish()


class TensorBoardLogger(Logger):
    def __init__(self, config: dict[str, Any]) -> None:
        base_directory = os.getenv("TENSORBOARD_DIR", "tensorboard_log")
        directory = os.path.join(
            base_directory,
            config["trainer"]["project_name"],
            config["trainer"]["experiment_name"],
        )
        os.makedirs(directory, exist_ok=True)
        print(f"Saving tensorboard log to {directory}.")
        self.writer = SummaryWriter(directory)

        hparams = {}
        for name, value in flatten_dict(config).items():
            if isinstance(value, (int, float, str, bool, torch.Tensor)):
                hparams[name] = value
            else:
                hparams[name] = str(value)

        self.writer.add_hparams(hparam_dict=hparams, metric_dict={"placeholder": 0})

    def log(self, data: dict[str, Any], step: int) -> None:
        for name, value in data.items():
            self.writer.add_scalar(name, value, step)

    def finish(self):
        self.writer.close()


class WandbLogger(Logger):
    def __init__(self, config: dict[str, Any]) -> None:
        wandb.init(
            project=config["trainer"]["project_name"],
            name=config["trainer"]["experiment_name"],
            config=config,
        )

    def log(self, data: dict[str, Any], step: int) -> None:
        wandb.log(data=data, step=step)

    def finish(self) -> None:
        wandb.finish()


LOGGERS = {
    "console": ConsoleLogger,
    "file": FileLogger,
    "mlflow": MlflowLogger,
    "swanlab": SwanlabLogger,
    "tensorboard": TensorBoardLogger,
    "wandb": WandbLogger,
}


class Tracker:
    def __init__(self, loggers: Union[str, list[str]] = "console", config: Optional[dict[str, Any]] = None):
        if isinstance(loggers, str):
            loggers = [loggers]

        self.loggers: list[Logger] = []
        for name in loggers:
            if name not in LOGGERS:
                raise ValueError(f"{name} is not supported.")
            self.loggers.append(LOGGERS[name](config))

        self.gen_logger = AggregateGenerationsLogger(loggers, config)

    def log(self, data: dict[str, Any], step: int) -> None:
        for logger in self.loggers:
            logger.log(data=data, step=step)

    def log_generation(self, samples: list[tuple[str, str, str, float]], step: int) -> None:
        self.gen_logger.log(samples, step)

    def __del__(self):
        for logger in self.loggers:
            logger.finish()
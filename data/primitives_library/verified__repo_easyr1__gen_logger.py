import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from ..py_functional import is_package_available

if is_package_available("wandb"):
    import wandb  # type: ignore

if is_package_available("swanlab"):
    import swanlab  # type: ignore


@dataclass
class GenerationLogger(ABC):
    config: dict[str, Any]

    @abstractmethod
    def log(self, samples: List[Tuple[str, str, str, float]], step: int) -> None:
        raise NotImplementedError


@dataclass
class ConsoleGenerationLogger(GenerationLogger):
    def log(self, samples: List[Tuple[str, str, str, float]], step: int) -> None:
        for prompt, output, label, score in samples:
            message = (
                f"[prompt] {prompt}\n"
                f"[output] {output}\n"
                f"[ground_truth] {label}\n"
                f"[score] {score}\n"
            )
            print(message)


@dataclass
class FileGenerationLogger(GenerationLogger):
    def log(self, samples: List[Tuple[str, str, str, float]], step: int) -> None:
        path = os.path.join(
            self.config["trainer"]["save_checkpoint_path"],
            "generations.log",
        )
        with open(path, "a") as file:
            for prompt, output, label, score in samples:
                file.write(
                    f"[prompt] {prompt}\n"
                    f"[output] {output}\n"
                    f"[ground_truth] {label}\n"
                    f"[score] {score}\n\n"
                )


@dataclass
class WandbGenerationLogger(GenerationLogger):
    def log(self, samples: List[Tuple[str, str, str, float]], step: int) -> None:
        columns = ["step"]
        for index in range(len(samples)):
            sample_number = index + 1
            columns.extend(
                (
                    f"input_{sample_number}",
                    f"output_{sample_number}",
                    f"label_{sample_number}",
                    f"score_{sample_number}",
                )
            )

        if not hasattr(self, "validation_table"):
            self.validation_table = wandb.Table(columns=columns)

        updated_table = wandb.Table(
            columns=columns,
            data=self.validation_table.data,
        )

        row = [step]
        for sample in samples:
            row.extend(sample)

        updated_table.add_data(*row)
        wandb.log({"val/generations": updated_table}, step=step)
        self.validation_table = updated_table


@dataclass
class SwanlabGenerationLogger(GenerationLogger):
    def log(self, samples: List[Tuple[str, str, str, float]], step: int) -> None:
        entries = []

        for index, sample in enumerate(samples):
            text = "\n\n---\n\n".join(
                (
                    f"input: {sample[0]}",
                    f"output: {sample[1]}",
                    f"label: {sample[2]}",
                    f"score: {sample[3]}",
                )
            )
            entries.append(swanlab.Text(text, caption=f"sample {index + 1}"))

        swanlab.log({"val/generations": entries}, step=step)


GEN_LOGGERS = {
    "console": ConsoleGenerationLogger,
    "file": FileGenerationLogger,
    "wandb": WandbGenerationLogger,
    "swanlab": SwanlabGenerationLogger,
}


class AggregateGenerationsLogger:
    def __init__(self, loggers: List[str], config: Optional[dict[str, Any]] = None):
        self.loggers: List[GenerationLogger] = []

        for logger_name in loggers:
            logger_class = GEN_LOGGERS.get(logger_name)
            if logger_class is not None:
                self.loggers.append(logger_class(config))

    def log(self, samples: List[Tuple[str, str, str, float]], step: int) -> None:
        for logger in self.loggers:
            logger.log(samples, step)
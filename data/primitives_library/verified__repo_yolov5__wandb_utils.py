import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from utils.general import LOGGER, colorstr

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

RANK = int(os.getenv("RANK", "-1"))
DEPRECATION_WARNING = (
    f"{colorstr('wandb')}: wandb is deprecated and will be removed in a future release. "
    "See supported integrations at https://github.com/ultralytics/yolov5#integrations."
)

try:
    import wandb

    assert hasattr(wandb, "__version__")
    LOGGER.warning(DEPRECATION_WARNING)
except (ImportError, AssertionError):
    wandb = None


class WandbLogger:
    """Manage YOLOv5 logging through Weights & Biases."""

    def __init__(self, opt, run_id=None, job_type="Training"):
        self.job_type = job_type
        self.wandb = wandb
        self.wandb_run = wandb.run if wandb else None

        self.val_artifact = None
        self.train_artifact = None
        self.train_artifact_path = None
        self.val_artifact_path = None
        self.result_artifact = None
        self.val_table = None
        self.result_table = None
        self.max_imgs_to_log = 16
        self.data_dict = None

        if self.wandb:
            project = "YOLOv5" if opt.project == "runs/train" else Path(opt.project).stem
            run_name = opt.name if opt.name != "exp" else None
            self.wandb_run = wandb.run or wandb.init(
                config=opt,
                resume="allow",
                project=project,
                entity=opt.entity,
                name=run_name,
                job_type=job_type,
                id=run_id,
                allow_val_change=True,
            )

        if self.wandb_run and self.job_type == "Training":
            if isinstance(opt.data, dict):
                self.data_dict = opt.data
            self.setup_training(opt)

    def setup_training(self, opt):
        self.log_dict = {}
        self.current_epoch = 0
        self.bbox_interval = opt.bbox_interval

        if opt.bbox_interval == -1:
            interval = opt.epochs // 10 if opt.epochs > 10 else 1
            self.bbox_interval = interval
            opt.bbox_interval = interval

            if opt.evolve or opt.noplots:
                self.bbox_interval = opt.epochs + 1
                opt.bbox_interval = opt.epochs + 1

    def log_model(self, path, opt, epoch, fitness_score, best_model=False):
        artifact = wandb.Artifact(
            f"run_{wandb.run.id}_model",
            type="model",
            metadata={
                "original_url": str(path),
                "epochs_trained": epoch + 1,
                "save period": opt.save_period,
                "project": opt.project,
                "total_epochs": opt.epochs,
                "fitness_score": fitness_score,
            },
        )
        artifact.add_file(str(path / "last.pt"), name="last.pt")
        wandb.log_artifact(
            artifact,
            aliases=[
                "latest",
                "last",
                f"epoch {self.current_epoch!s}",
                "best" if best_model else "",
            ],
        )
        LOGGER.info(f"Saving model artifact on epoch {epoch + 1}")

    def log(self, log_dict):
        if self.wandb_run:
            for key, value in log_dict.items():
                self.log_dict[key] = value

    def end_epoch(self):
        if self.wandb_run:
            with all_logging_disabled():
                try:
                    wandb.log(self.log_dict)
                except BaseException as e:
                    LOGGER.info(
                        f"An error occurred in wandb logger. The training will proceed without interruption. More info\n{e}"
                    )
                    self.wandb_run.finish()
                    self.wandb_run = None
                self.log_dict = {}

    def finish_run(self):
        if self.wandb_run:
            if self.log_dict:
                with all_logging_disabled():
                    wandb.log(self.log_dict)
            wandb.run.finish()
            LOGGER.warning(DEPRECATION_WARNING)


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)
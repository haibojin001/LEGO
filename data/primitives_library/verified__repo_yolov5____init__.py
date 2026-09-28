import glob
import json
import logging
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    import comet_ml

    _comet_config = comet_ml.config.get_config()
    COMET_PROJECT_NAME = _comet_config.get_string(
        os.getenv("COMET_PROJECT_NAME"), "comet.project_name", default="yolov5"
    )
except ImportError:
    comet_ml = None
    COMET_PROJECT_NAME = None

import PIL
import torch
import torchvision.transforms as T
import yaml

from utils.dataloaders import img2label_paths
from utils.general import check_dataset, scale_boxes, xywh2xyxy
from utils.metrics import box_iou

COMET_PREFIX = "comet://"
COMET_MODE = os.getenv("COMET_MODE", "online")
COMET_MODEL_NAME = os.getenv("COMET_MODEL_NAME", "yolov5")
COMET_DEFAULT_CHECKPOINT_FILENAME = os.getenv("COMET_DEFAULT_CHECKPOINT_FILENAME", "last.pt")
COMET_UPLOAD_DATASET = os.getenv("COMET_UPLOAD_DATASET", "false").lower() == "true"
COMET_LOG_CONFUSION_MATRIX = os.getenv("COMET_LOG_CONFUSION_MATRIX", "true").lower() == "true"
COMET_LOG_PREDICTIONS = os.getenv("COMET_LOG_PREDICTIONS", "true").lower() == "true"
COMET_MAX_IMAGE_UPLOADS = int(os.getenv("COMET_MAX_IMAGE_UPLOADS", "100"))
CONF_THRES = float(os.getenv("CONF_THRES", "0.001"))
IOU_THRES = float(os.getenv("IOU_THRES", "0.6"))
COMET_LOG_BATCH_METRICS = os.getenv("COMET_LOG_BATCH_METRICS", "false").lower() == "true"
COMET_BATCH_LOGGING_INTERVAL = int(os.getenv("COMET_BATCH_LOGGING_INTERVAL", "1"))
COMET_LOG_PER_CLASS_METRICS = os.getenv("COMET_LOG_PER_CLASS_METRICS", "false").lower() == "true"
RANK = int(os.getenv("RANK", "-1"))

to_pil = T.ToPILImage()


def download_model_checkpoint(opt, experiment):
    """Download a selected checkpoint asset and update the weights option."""
    destination_dir = f"{opt.project}/{experiment.name}"
    os.makedirs(destination_dir, exist_ok=True)

    assets = experiment.get_model_asset_list(COMET_MODEL_NAME)
    if not assets:
        logger.error(f"COMET ERROR: No checkpoints found for model name : {COMET_MODEL_NAME}")
        return

    newest_first = sorted(assets, key=lambda item: item["step"], reverse=True)
    assets_by_name = {item["fileName"]: item["assetId"] for item in newest_first}

    requested_name = urlparse(opt.weights).query
    if not requested_name:
        requested_name = COMET_DEFAULT_CHECKPOINT_FILENAME

    asset_id = assets_by_name.get(requested_name)
    if asset_id is None:
        logger.error(f"COMET ERROR: Checkpoint {requested_name} not found in the given Experiment")
        return

    try:
        logger.info(f"COMET INFO: Downloading checkpoint {requested_name}")
        contents = experiment.get_asset(asset_id, return_type="binary", stream=False)
        target = f"{destination_dir}/{requested_name}"
        with open(target, "wb") as f:
            f.write(contents)
        opt.weights = target
    except Exception:
        logger.exception("COMET WARNING: Unable to download checkpoint from Comet")


def set_opt_parameters(opt, experiment):
    """Restore saved experiment options and create the local hyperparameter file."""
    resume_value = opt.resume

    for asset in experiment.get_asset_list():
        if asset["fileName"] != "opt.yaml":
            continue
        contents = experiment.get_asset(asset["assetId"], return_type="binary", stream=False)
        options = yaml.safe_load(contents) or {}
        for key, value in options.items():
            setattr(opt, key, value)
        opt.resume = resume_value
        break

    run_directory = f"{opt.project}/{experiment.name}"
    os.makedirs(run_directory, exist_ok=True)
    hyp_path = f"{run_directory}/hyp.yaml"
    with open(hyp_path, "w") as f:
        yaml.dump(opt.hyp, f)
    opt.hyp = hyp_path


def check_comet_resume(opt):
    """Restore a training invocation when its resume value identifies a Comet run."""
    if comet_ml is None:
        return

    if not isinstance(opt.resume, str) or not opt.resume.startswith(COMET_PREFIX):
        return

    try:
        resource = urlparse(opt.resume)
        parts = [part for part in resource.path.split("/") if part]
        if len(parts) < 2:
            logger.error("COMET ERROR: Invalid Comet resume URI")
            return

        workspace = resource.netloc
        project_name, experiment_key = parts[:2]
        api = comet_ml.API()
        experiment = api.get_experiment(workspace, project_name, experiment_key)
        set_opt_parameters(opt, experiment)
        download_model_checkpoint(opt, experiment)
    except Exception:
        logger.exception("COMET WARNING: Unable to restore the requested Comet experiment")


class CometLogger:
    """Comet.ml integration used by YOLOv5 training and validation callbacks."""

    def __init__(self, opt, hyp, run_id=None, job_type="Training", **experiment_kwargs) -> None:
        self.job_type = job_type
        self.opt = opt
        self.hyp = hyp
        self.comet_mode = COMET_MODE
        self.save_model = opt.save_period > -1
        self.model_name = COMET_MODEL_NAME
        self.log_batch_metrics = COMET_LOG_BATCH_METRICS
        self.comet_log_batch_interval = COMET_BATCH_LOGGING_INTERVAL
        self.upload_dataset = opt.upload_dataset or COMET_UPLOAD_DATASET
        self.resume = opt.resume

        self.default_experiment_kwargs = {
            "log_code": False,
            "log_env_gpu": True,
            "log_env_cpu": True,
            "project_name": COMET_PROJECT_NAME,
            **experiment_kwargs,
        }
        self.experiment = self._get_experiment(self.comet_mode, run_id)
        self.experiment.set_name(opt.name)

        self.data_dict = self.check_dataset(opt.data)
        self.class_names = self.data_dict["names"]
        self.num_classes = self.data_dict["nc"]
        self.logged_images_count = 0
        self.max_images = COMET_MAX_IMAGE_UPLOADS

        if run_id is None:
            self.experiment.log_other("Created from", "YOLOv5")
            if comet_ml is not None and not isinstance(self.experiment, comet_ml.OfflineExperiment):
                try:
                    workspace, project, experiment_id = self.experiment.url.rstrip("/").split("/")[-3:]
                    self.experiment.log_other("Run Path", f"{workspace}/{project}/{experiment_id}")
                except Exception:
                    pass

            self.log_parameters(vars(opt))
            self.log_parameters(opt.hyp)
            self.log_asset_data(opt.hyp, name="hyperparameters.json", metadata={"type": "hyp-config-file"})
            self.log_asset(f"{opt.save_dir}/opt.yaml", metadata={"type": "opt-config-file"})

        self.comet_log_confusion_matrix = COMET_LOG_CONFUSION_MATRIX
        self.conf_thres = getattr(opt, "conf_thres", CONF_THRES)
        self.iou_thres = getattr(opt, "iou_thres", IOU_THRES)
        self.log_parameters({"val_iou_threshold": self.iou_thres, "val_conf_threshold": self.conf_thres})

        self.comet_log_predictions = COMET_LOG_PREDICTIONS
        if opt.bbox_interval == -1:
            self.comet_log_prediction_interval = 1 if opt.epochs < 10 else opt.epochs // 10
        else:
            self.comet_log_prediction_interval = opt.bbox_interval

        if self.comet_log_predictions:
            self.metadata_dict = {}
            self.logged_image_names = []

        self.comet_log_per_class_metrics = COMET_LOG_PER_CLASS_METRICS
        self.experiment.log_others(
            {
                "comet_mode": COMET_MODE,
                "comet_max_image_uploads": COMET_MAX_IMAGE_UPLOADS,
                "comet_log_per_class_metrics": COMET_LOG_PER_CLASS_METRICS,
                "comet_log_batch_metrics": COMET_LOG_BATCH_METRICS,
                "comet_log_confusion_matrix": COMET_LOG_CONFUSION_MATRIX,
                "comet_model_name": COMET_MODEL_NAME,
            }
        )

        if hasattr(opt, "comet_optimizer_id"):
            self.experiment.log_other("optimizer_id", opt.comet_optimizer_id)
            self.experiment.log_other("optimizer_objective", opt.comet_optimizer_objective)
            self.experiment.log_other("optimizer_metric", opt.comet_optimizer_metric)
            self.experiment.log_other("optimizer_parameters", json.dumps(hyp))

    def _get_experiment(self, mode, experiment_id=None):
        """Create or reopen an online or offline Comet experiment."""
        if mode == "offline":
            if experiment_id is not None:
                return comet_ml.ExistingOfflineExperiment(
                    previous_experiment=experiment_id, **self.default_experiment_kwargs
                )
            return comet_ml.OfflineExperiment(**self.default_experiment_kwargs)

        try:
            if experiment_id is not None:
                return comet_ml.ExistingExperiment(previous_experiment=experiment_id, **self.default_experiment_kwargs)
            return comet_ml.Experiment(**self.default_experiment_kwargs)
        except ValueError:
            logger.warning(
                "COMET WARNING: Comet credentials have not been set. "
                "Comet will default to offline logging. "
                "Please set your credentials to enable online logging."
            )
            return self._get_experiment("offline", experiment_id)

    def log_metrics(self, log_dict, **kwargs):
        self.experiment.log_metrics(log_dict, **kwargs)

    def log_parameters(self, log_dict, **kwargs):
        self.experiment.log_parameters(log_dict, **kwargs)

    def log_asset(self, asset_path, **kwargs):
        self.experiment.log_asset(asset_path, **kwargs)

    def log_asset_data(self, asset, **kwargs):
        self.experiment.log_asset_data(asset, **kwargs)

    def log_image(self, img, **kwargs):
        self.experiment.log_image(img, **kwargs)

    def log_model(self, path, opt, epoch, fitness_score, best_model=False):
        """Upload checkpoint files generated for an epoch."""
        if not self.save_model:
            return

        metadata = {
            "fitness_score": fitness_score[-1],
            "epochs_trained": epoch + 1,
            "save_period": opt.save_period,
            "total_epochs": opt.epochs,
        }
        for model_path in glob.glob(f"{path}/*.pt"):
            filename = Path(model_path).name
            model_metadata = dict(metadata)
            model_metadata["best_model"] = best_model
            self.experiment.log_model(
                self.model_name,
                file_or_folder=model_path,
                file_name=filename,
                metadata=model_metadata,
                overwrite=True,
            )

    def check_dataset(self, data_file):
        """Resolve a normal dataset definition or a Comet artifact dataset."""
        if isinstance(data_file, str) and data_file.startswith(COMET_PREFIX):
            return self.download_dataset_artifact(data_file)
        return check_dataset(data_file)

    def download_dataset_artifact(self, artifact_path):
        """Download a dataset artifact and return its dataset configuration."""
        if comet_ml is None:
            raise ImportError("comet_ml is required to download a Comet dataset artifact")

        parsed = urlparse(artifact_path)
        parts = [item for item in parsed.path.split("/") if item]
        workspace = parsed.netloc
        project_name = parts[0] if parts else None
        artifact_name = parts[1] if len(parts) > 1 else None
        version = parsed.query or "latest"

        if not workspace or not project_name or not artifact_name:
            raise ValueError(f"Invalid Comet dataset artifact URI: {artifact_path}")

        artifact = comet_ml.API().get_artifact(workspace, project_name, artifact_name, version)
        local_directory = artifact.download()
        yaml_files = list(Path(local_directory).rglob("*.yaml")) + list(Path(local_directory).rglob("*.yml"))
        if not yaml_files:
            raise FileNotFoundError("COMET ERROR: Dataset artifact does not contain a YAML dataset definition")
        return check_dataset(str(yaml_files[0]))

    def log_dataset_artifact(self, paths):
        """Create and upload a Comet artifact containing dataset metadata and labels."""
        if comet_ml is None or not hasattr(comet_ml, "Artifact"):
            return

        try:
            artifact = comet_ml.Artifact(name="yolov5-dataset", artifact_type="dataset")
            data_path = Path(self.opt.data)
            if data_path.exists():
                artifact.add(data_path, name=data_path.name)

            for split_path in paths or []:
                split_path = Path(split_path)
                if split_path.exists():
                    artifact.add(split_path, name=split_path.name)

                for label_path in img2label_paths([str(split_path)]):
                    if Path(label_path).exists():
                        artifact.add(label_path, name=Path(label_path).name)

            self.experiment.log_artifact(artifact)
        except Exception:
            logger.exception("COMET WARNING: Failed to upload dataset artifact")

    def preprocess_predictions(self, detections, image_shape, image_name, label_type):
        """Turn YOLO boxes into Comet image annotation records."""
        if detections is None:
            detections = []

        height, width = int(image_shape[0]), int(image_shape[1])
        annotations = []
        names = self.class_names

        if isinstance(detections, torch.Tensor):
            detections = detections.detach().cpu().tolist()

        for detection in detections:
            if len(detection) < 5:
                continue

            x1, y1, x2, y2 = map(float, detection[:4])
            confidence = float(detection[4]) if len(detection) > 5 else None
            class_id = int(detection[5]) if len(detection) > 5 else int(detection[4])
            label = names[class_id] if isinstance(names, (list, tuple)) else names.get(class_id, str(class_id))

            box = {
                "x": max(0.0, x1),
                "y": max(0.0, y1),
                "width": max(0.0, min(float(width), x2) - max(0.0, x1)),
                "height": max(0.0, min(float(height), y2) - max(0.0, y1)),
                "label": label,
            }
            if confidence is not None:
                box["score"] = confidence
            annotations.append(box)

        return {"name": label_type, "image": image_name, "data": annotations}

    def log_predictions(self, image, labels, path, shape=None, predn=None):
        """Log ground-truth labels and detections as Comet image annotations."""
        if self.logged_images_count >= self.max_images:
            return

        image_name = Path(path).name
        if isinstance(image, torch.Tensor):
            image_to_log = to_pil(image.detach().cpu())
            image_shape = image.shape[-2:]
        elif isinstance(image, PIL.Image.Image):
            image_to_log = image
            image_shape = (image.height, image.width)
        else:
            image_to_log = image
            image_shape = image.shape[:2]

        annotations = []
        if image_name not in self.logged_image_names:
            annotations.append(self.preprocess_predictions(labels, image_shape, image_name, "ground_truth"))
            self.logged_image_names.append(image_name)

        annotations.append(self.preprocess_predictions(predn, image_shape, image_name, "prediction"))
        self.log_image(image_to_log, name=image_name, annotations=annotations)
        self.logged_images_count += 1

    def log_confusion_matrix(self, matrix, epoch=None):
        """Upload a confusion matrix when the Comet SDK supports it."""
        if not self.comet_log_confusion_matrix:
            return

        try:
            labels = self.class_names
            if isinstance(labels, dict):
                labels = [labels[i] for i in sorted(labels)]
            self.experiment.log_confusion_matrix(
                matrix=matrix,
                labels=labels,
                title="Confusion Matrix",
                step=epoch,
            )
        except Exception:
            logger.exception("COMET WARNING: Failed to log confusion matrix")

    def on_pretrain_routine_end(self, paths):
        if self.upload_dataset:
            self.log_dataset_artifact(paths)

    def on_train_batch_end(self, model, ni, imgs, targets, paths, vals):
        if not self.log_batch_metrics or ni % self.comet_log_batch_interval:
            return

        metrics = {}
        if vals is not None:
            for key, value in zip(("train/box_loss", "train/obj_loss", "train/cls_loss"), vals):
                metrics[key] = value
        metrics["train/epoch"] = getattr(model, "epoch", 0)
        self.log_metrics(metrics, step=ni)

    def on_val_image_end(self, pred, predn, path, names, im):
        if not self.comet_log_predictions or self.logged_images_count >= self.max_images:
            return

        if predn is None:
            return

        if isinstance(predn, torch.Tensor):
            prediction_boxes = predn.detach().cpu()
        else:
            prediction_boxes = predn

        self.log_predictions(im, [], path, predn=prediction_boxes)

    def on_fit_epoch_end(self, x, epoch, best_fitness=None, fi=None):
        self.log_metrics(x, epoch=epoch)

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        if self.save_model and (final_epoch or epoch % max(getattr(self.opt, "save_period", 1), 1) == 0):
            self.log_model(Path(last).parent, self.opt, epoch, fi, best_model=best_fitness == fi)

    def on_train_end(self, last, best, epoch, results):
        try:
            if Path(last).exists():
                self.log_model(Path(last).parent, self.opt, epoch, results, best_model=Path(best).exists())
            self.experiment.end()
        except Exception:
            logger.exception("COMET WARNING: Failed to finalize Comet experiment")
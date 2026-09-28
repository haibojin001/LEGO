import glob
import re
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics.utils.plotting import Annotator, colors

try:
    import clearml
    from clearml import Dataset, Task

    try:
        from clearml.backend_api.session.defs import MissingConfigError
    except ImportError:
        MissingConfigError = ValueError

    assert hasattr(clearml, "__version__")
except (ImportError, AssertionError):
    clearml = None


CLEARML_CONFIG_ERRORS = (
    "ClearML configuration could not be found",
    "Missing access_key.",
    "Missing secret_key.",
    "host is required in init or config",
    "Could not get access credentials",
)


class ClearmlNotConfiguredError(ValueError):
    """Raised when ClearML is available but has not been configured."""


def construct_dataset(clearml_info_string):
    """Retrieve a ClearML dataset and convert its YAML definition to local absolute paths."""
    dataset_identifier = clearml_info_string.replace("clearml://", "")
    dataset = Dataset.get(dataset_id=dataset_identifier)
    dataset_root = Path(dataset.get_local_copy())

    yaml_files = list(glob.glob(str(dataset_root / "*.yaml")) + glob.glob(str(dataset_root / "*.yml")))
    if len(yaml_files) > 1:
        raise ValueError(
            "More than one yaml file was found in the dataset root, cannot determine which one contains "
            "the dataset definition this way."
        )
    if not yaml_files:
        raise ValueError(
            "No yaml definition found in dataset root path, check that there is a correct yaml file "
            "inside the dataset root path."
        )

    with open(yaml_files[0]) as yaml_file:
        definition = yaml.safe_load(yaml_file)

    assert set(definition.keys()).issuperset({"train", "test", "val", "nc", "names"}), (
        "The right keys were not found in the yaml file, make sure it at least has the following keys: "
        "('train', 'test', 'val', 'nc', 'names')"
    )

    def local_path(value):
        return str((dataset_root / value).resolve()) if value else None

    return {
        "train": local_path(definition["train"]),
        "test": local_path(definition["test"]),
        "val": local_path(definition["val"]),
        "nc": definition["nc"],
        "names": definition["names"],
    }


class ClearmlLogger:
    """ClearML integration for training metrics, artifacts, datasets, and prediction images."""

    def __init__(self, opt, hyp):
        """Create the ClearML task and optionally resolve a ClearML-hosted dataset."""
        self.current_epoch = 0
        self.current_epoch_logged_images = set()
        self.max_imgs_to_log_per_epoch = 16

        if "bbox_interval" in opt:
            self.bbox_interval = opt.bbox_interval

        self.clearml = clearml
        self.task = None
        self.data_dict = None

        if self.clearml:
            try:
                self.task = Task.init(
                    project_name="YOLOv5" if str(opt.project).startswith("runs/") else opt.project,
                    task_name=opt.name if opt.name != "exp" else "Training",
                    tags=["YOLOv5"],
                    output_uri=True,
                    reuse_last_task_id=opt.exist_ok,
                    auto_connect_frameworks={"pytorch": False, "matplotlib": False},
                )
            except MissingConfigError as error:
                if MissingConfigError is ValueError and not str(error).startswith(CLEARML_CONFIG_ERRORS):
                    raise
                raise ClearmlNotConfiguredError from error

            self.task.connect(hyp, name="Hyperparameters")
            self.task.connect(opt, name="Args")
            self.task.set_base_docker(
                "ultralytics/yolov5:latest",
                docker_arguments='--ipc=host -e="CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1"',
                docker_setup_bash_script="pip install clearml",
            )

            if opt.data.startswith("clearml://"):
                self.data_dict = construct_dataset(opt.data)
                opt.data = self.data_dict

    def log_scalars(self, metrics, epoch):
        """Send scalar metric values to the current ClearML task."""
        logger = self.task.get_logger()
        for key, value in metrics.items():
            title, series = key.split("/")
            logger.report_scalar(title, series, value, epoch)

    def log_model(self, model_path, model_name, epoch=0):
        """Upload a model checkpoint to ClearML."""
        self.task.update_output_model(
            model_path=str(model_path),
            name=model_name,
            iteration=epoch,
            auto_delete_file=False,
        )

    def log_summary(self, metrics):
        """Store final metric values in ClearML's summary view."""
        logger = self.task.get_logger()
        for key, value in metrics.items():
            logger.report_single_value(key, value)

    def log_plot(self, title, plot_path):
        """Report an image file as a non-interactive matplotlib plot."""
        image = mpimg.imread(plot_path)
        figure = plt.figure()
        axes = figure.add_axes([0, 0, 1, 1], frameon=False, aspect="auto", xticks=[], yticks=[])
        axes.imshow(image)
        self.task.get_logger().report_matplotlib_figure(title, "", figure=figure, report_interactive=False)

    def log_debug_samples(self, files, title="Debug Samples"):
        """Upload existing image files as ClearML debug samples."""
        logger = self.task.get_logger()
        for file in files:
            if file.exists():
                match = re.search(r"_batch(\d+)", file.name)
                iteration = int(match.groups()[0]) if match else 0
                logger.report_image(
                    title=title,
                    series=file.name.replace(f"_batch{iteration}", ""),
                    local_path=str(file),
                    iteration=iteration,
                )

    def log_image_with_boxes(self, image_path, boxes, class_names, image, conf_threshold=0.25):
        """Annotate detections on an image and upload the resulting debug sample."""
        if (
            len(self.current_epoch_logged_images) < self.max_imgs_to_log_per_epoch
            and image_path not in self.current_epoch_logged_images
        ):
            annotated = image.copy()
            annotator = Annotator(annotated, line_width=1, example=str(class_names))

            for *xyxy, confidence, class_id in boxes:
                if confidence > conf_threshold:
                    class_index = int(class_id)
                    annotator.box_label(
                        xyxy,
                        f"{class_names[class_index]} {confidence:.2f}",
                        color=colors(class_index),
                    )

            self.task.get_logger().report_image(
                title="Bounding Boxes",
                series=image_path.name,
                iteration=self.current_epoch,
                image=annotator.result(),
            )
            self.current_epoch_logged_images.add(image_path)

    def log_dataset(self, dataset_path, dataset_name=None):
        """Create, upload, finalize, and associate a ClearML dataset with this task."""
        name = dataset_name or f"YOLOv5-{self.task.name}"
        dataset = Dataset.create(dataset_project=self.task.get_project_name(), dataset_name=name)
        dataset.add_files(dataset_path)
        dataset.upload()
        dataset.finalize()
        self.task.connect({"dataset_id": dataset.id}, name="General")
        return dataset
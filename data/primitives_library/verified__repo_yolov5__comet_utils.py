import logging
import os
from urllib.parse import urlparse

try:
    import comet_ml
except ImportError:
    comet_ml = None

import yaml

logger = logging.getLogger(__name__)

COMET_PREFIX = "comet://"
COMET_MODEL_NAME = os.getenv("COMET_MODEL_NAME", "yolov5")
COMET_DEFAULT_CHECKPOINT_FILENAME = os.getenv("COMET_DEFAULT_CHECKPOINT_FILENAME", "last.pt")


def download_model_checkpoint(opt, experiment):
    """Fetch a checkpoint asset from a Comet experiment and assign its local path to opt.weights."""
    destination = f"{opt.project}/{experiment.name}"
    os.makedirs(destination, exist_ok=True)

    assets = experiment.get_model_asset_list(COMET_MODEL_NAME)
    if len(assets) == 0:
        logger.error(f"COMET ERROR: No checkpoints found for model name : {COMET_MODEL_NAME}")
        return

    checkpoint_assets = sorted(assets, key=lambda item: item["step"], reverse=True)
    asset_ids = {item["fileName"]: item["assetId"] for item in checkpoint_assets}

    parsed_weights = urlparse(opt.weights)
    filename = parsed_weights.query

    if filename:
        asset_id = asset_ids.get(filename)
    else:
        filename = COMET_DEFAULT_CHECKPOINT_FILENAME
        asset_id = asset_ids.get(filename)

    if asset_id is None:
        logger.error(f"COMET ERROR: Checkpoint {filename} not found in the given Experiment")
        return

    try:
        logger.info(f"COMET INFO: Downloading checkpoint {filename}")
        contents = experiment.get_asset(asset_id, return_type="binary", stream=False)
        output_path = f"{destination}/{filename}"

        with open(output_path, "wb") as file:
            file.write(contents)

        opt.weights = output_path
    except Exception:
        logger.exception("COMET WARNING: Unable to download checkpoint from Comet")


def set_opt_parameters(opt, experiment):
    """Restore saved run options and write the associated hyperparameters locally."""
    original_resume = opt.resume

    for asset in experiment.get_asset_list():
        if asset["fileName"] == "opt.yaml":
            content = experiment.get_asset(asset["assetId"], return_type="binary", stream=False)
            saved_options = yaml.safe_load(content)

            for key, value in saved_options.items():
                setattr(opt, key, value)

            opt.resume = original_resume

    directory = f"{opt.project}/{experiment.name}"
    os.makedirs(directory, exist_ok=True)

    hyp_path = f"{directory}/hyp.yaml"
    with open(hyp_path, "w") as file:
        yaml.dump(opt.hyp, file)

    opt.hyp = hyp_path


def check_comet_resume(opt):
    """Restore a Comet-backed training run when opt.resume contains a Comet URL."""
    if comet_ml is None:
        return

    if isinstance(opt.resume, str) and opt.resume.startswith(COMET_PREFIX):
        api = comet_ml.API()
        resource = urlparse(opt.resume)
        experiment = api.get(f"{resource.netloc}{resource.path}")

        set_opt_parameters(opt, experiment)
        download_model_checkpoint(opt, experiment)
        return True

    return None
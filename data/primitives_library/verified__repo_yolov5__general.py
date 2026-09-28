from __future__ import annotations

import contextlib
import glob
import inspect
import os
import platform
import random
import re
import subprocess
import sys
import time
import urllib
from copy import deepcopy
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from tarfile import is_tarfile
from zipfile import ZipFile, is_zipfile

import cv2
import numpy as np
import packaging
import pandas as pd
import torch
import torchvision
import yaml

try:
    import ultralytics

    assert hasattr(ultralytics, "__version__")
except (ImportError, AssertionError):
    os.system("pip install -U ultralytics")
    import ultralytics

from ultralytics.data.converter import coco80_to_coco91_class
from ultralytics.utils import LOGGER, TQDM, colorstr, get_default_args
from ultralytics.utils.checks import check_requirements as check_requirements_ultralytics
from ultralytics.utils.checks import is_ascii
from ultralytics.utils.files import WorkingDirectory, file_date, file_size, get_latest_run
from ultralytics.utils.git import GitRepo
from ultralytics.utils.ops import (
    Profile,
    clip_boxes,
    make_divisible,
    segments2boxes,
    xywh2xyxy,
    xywhn2xyxy,
    xyxy2xywhn,
)
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import intersect_dicts, one_cycle

from utils import TryExcept, emojis
from utils.downloads import curl_download, gsutil_getsize
from utils.metrics import box_iou, fitness

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
RANK = int(os.getenv("RANK", "-1"))

NUM_THREADS = min(8, max(1, (os.cpu_count() or 1) - 1))
DATASETS_DIR = Path(os.getenv("YOLOv5_DATASETS_DIR", ROOT.parent / "datasets"))
AUTOINSTALL = str(os.getenv("YOLOv5_AUTOINSTALL", "true")).lower() == "true"
FONT = "Arial.ttf"

torch.set_printoptions(linewidth=320, precision=5, profile="long")
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})
pd.options.display.max_columns = 10
cv2.setNumThreads(0)
os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_THREADS)
os.environ["OMP_NUM_THREADS"] = "1" if platform.system() == "Darwin" else str(NUM_THREADS)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["KINETO_LOG_LEVEL"] = "5"


def check_requirements(requirements=ROOT / "requirements.txt", exclude=(), install=True, cmds="", **kwargs):
    if isinstance(requirements, Path) and sys.version_info < (3, 9):
        exclude = (*exclude, "urllib3")
    return check_requirements_ultralytics(requirements, exclude=exclude, install=install, cmds=cmds, **kwargs)


def is_colab():
    return "google.colab" in sys.modules


def is_jupyter():
    with contextlib.suppress(Exception):
        from IPython import get_ipython

        return get_ipython() is not None
    return False


def is_kaggle():
    return os.environ.get("PWD") == "/kaggle/working" and os.environ.get("KAGGLE_URL_BASE") == "https://www.kaggle.com"


def is_docker() -> bool:
    if Path("/.dockerenv").exists():
        return True
    try:
        with open("/proc/self/cgroup") as f:
            return any("docker" in line for line in f)
    except OSError:
        return False


def is_writeable(dir, test=False):
    if not test:
        return os.access(dir, os.W_OK)
    p = Path(dir) / "tmp.txt"
    try:
        with open(p, "w"):
            pass
        p.unlink()
        return True
    except OSError:
        return False


def user_config_dir(dir="Ultralytics", env_var="YOLOV5_CONFIG_DIR"):
    env = os.getenv(env_var)
    if env:
        path = Path(env)
    else:
        locations = {"Windows": "AppData/Roaming", "Linux": ".config", "Darwin": "Library/Application Support"}
        path = Path.home() / locations.get(platform.system(), "")
        path = (path if is_writeable(path) else Path("/tmp")) / dir
    path.mkdir(exist_ok=True)
    return path


CONFIG_DIR = user_config_dir()


def methods(instance):
    return [name for name in dir(instance) if callable(getattr(instance, name)) and not name.startswith("__")]


def init_seeds(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic and check_version(torch.__version__, "1.12.0"):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)


def check_online():
    import socket

    def attempt():
        try:
            socket.create_connection(("1.1.1.1", 443), 5)
            return True
        except OSError:
            return False

    return attempt() or attempt()


def git_describe(path=ROOT):
    try:
        assert (Path(path) / ".git").exists()
        return check_output(f"git -C {path} describe --tags --long --always", shell=True).decode().strip()
    except Exception:
        return ""


@TryExcept()
@WorkingDirectory(ROOT)
def check_git_status(repo="ultralytics/yolov5", branch="master"):
    url = f"https://github.com/{repo}"
    suffix = f", for updates see {url}"
    prefix = colorstr("github: ")
    assert Path(".git").exists(), prefix + "skipping check (not a git repository)" + suffix
    assert check_online(), prefix + "skipping check (offline)" + suffix

    words = re.split(r"\s", check_output("git remote -v", shell=True).decode())
    found = [repo in x for x in words]
    if any(found):
        remote = words[found.index(True) - 1]
    else:
        remote = "ultralytics"
        check_output(f"git remote add {remote} {url}", shell=True)
    check_output(f"git fetch {remote}", shell=True, timeout=5)
    local = check_output("git rev-parse --abbrev-ref HEAD", shell=True).decode().strip()
    branch = branch or local
    n = int(check_output(f"git rev-list {local}..{remote}/{branch} --count", shell=True).decode())
    if n:
        LOGGER.info(f"{prefix}YOLOv5 is out of date by {n} commit{'s' if n > 1 else ''}. Use 'git pull' or 'git clone {url}' to update.")
    else:
        LOGGER.info(f"{prefix}up to date with {url} ✅")


def check_git_info():
    try:
        return GitRepo().get_git_info()
    except Exception:
        return None


def check_python(minimum="3.8.0"):
    return check_version(platform.python_version(), minimum, name="Python ", hard=True)


def check_font(font=FONT):
    font = Path(font)
    if font.exists():
        return font
    font = CONFIG_DIR / font.name
    if not font.exists():
        try:
            url = f"https://github.com/ultralytics/assets/releases/download/v0.0.0/{font.name}"
            torch.hub.download_url_to_file(url, str(font), progress=False)
        except Exception as e:
            LOGGER.warning(f"WARNING ⚠️ Unable to download font '{font}': {e}")
    return font


def check_suffix(file="yolov5s.pt", suffix=(".pt",), msg=""):
    if file and suffix:
        suffix = (suffix,) if isinstance(suffix, str) else suffix
        files = [file] if isinstance(file, (str, Path)) else file
        for f in files:
            s = Path(f).suffix.lower()
            if s:
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


def check_yaml(file, suffix=(".yaml", ".yml")):
    return check_file(file, suffix)


def check_file(file, suffix="", download=True, hard=True):
    check_suffix(file, suffix)
    file = str(file)
    if not file:
        return file
    if Path(file).is_file():
        return file

    if file.startswith(("http:/", "https:/")):
        url = file.replace(":/", "://", 1)
        name = Path(urllib.parse.unquote(url)).name.split("?")[0]
        if Path(name).is_file():
            LOGGER.info(f"Found {url} locally at {name}")
        elif download:
            LOGGER.info(f"Downloading {url} to {name}...")
            torch.hub.download_url_to_file(url, name, progress=True)
        return name

    candidates = []
    for root in (ROOT / "data", ROOT / "models", ROOT / "utils"):
        candidates.extend(glob.glob(str(root / "**" / file), recursive=True))
    if not candidates:
        if hard:
            raise FileNotFoundError(f"'{file}' does not exist")
        return []
    if len(candidates) > 1:
        raise FileNotFoundError(f"Multiple files match '{file}', specify an exact path: {candidates}")
    return candidates[0]


def check_img_size(imgsz, s=32, floor=0):
    stride = int(s.max()) if isinstance(s, torch.Tensor) else int(s)
    sizes = [imgsz] if isinstance(imgsz, int) else list(imgsz)
    rounded = [max(make_divisible(x, stride), floor) for x in sizes]
    if rounded != sizes:
        LOGGER.warning(f"WARNING ⚠️ --img-size {sizes} must be multiple of max stride {stride}, updating to {rounded}")
    return rounded[0] if isinstance(imgsz, int) else rounded


def check_imshow(warn=False):
    try:
        assert not is_docker(), "imshow() is disabled in Docker environments"
        assert not is_colab(), "imshow() is disabled in Google Colab environments"
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        cv2.imshow("test", image)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f"WARNING ⚠️ Environment does not support cv2.imshow() or PIL Image.show()\n{e}")
        return False


def check_version(current="0.0.0", minimum="0.0.0", name="version ", pinned=False, hard=False, verbose=False):
    current, minimum = (str(x) for x in (current, minimum))
    current = current.split("+")[0]
    try:
        result = packaging.version.parse(current) == packaging.version.parse(minimum) if pinned else packaging.version.parse(current) >= packaging.version.parse(minimum)
    except Exception:
        result = False
    warning = f"WARNING ⚠️ {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed"
    if hard:
        assert result, emojis(warning)
    elif verbose and not result:
        LOGGER.warning(warning)
    return result


def check_dataset(data, autodownload=True):
    from utils.downloads import download

    extract_dir = None
    if isinstance(data, (str, Path)):
        data = check_file(data)
        if is_zipfile(data) or is_tarfile(data):
            extract_dir = Path(data).with_suffix("")
            if not extract_dir.exists():
                if is_zipfile(data):
                    with ZipFile(data) as z:
                        z.extractall(extract_dir)
                else:
                    import tarfile

                    with tarfile.open(data) as t:
                        t.extractall(extract_dir)
            yamls = list(extract_dir.rglob("*.yaml")) + list(extract_dir.rglob("*.yml"))
            assert yamls, f"No YAML file found in {data}"
            data = yamls[0]

        with open(data, errors="ignore") as f:
            data = yaml.safe_load(f)

    data = deepcopy(data)
    if extract_dir:
        data["path"] = extract_dir
    path = Path(data.get("path") or "")
    if not path.is_absolute():
        path = (ROOT / path).resolve()

    for key in ("train", "val", "test"):
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, (str, Path)):
            p = Path(value)
            data[key] = str((path / p).resolve()) if not p.is_absolute() else str(p)
        else:
            data[key] = [str((path / x).resolve()) if not Path(x).is_absolute() else str(x) for x in value]

    assert "nc" in data, "Dataset configuration missing 'nc'"
    assert "train" in data, "Dataset configuration missing 'train'"
    assert "val" in data, "Dataset configuration missing 'val'"
    if "names" not in data:
        data["names"] = [str(i) for i in range(data["nc"])]
    elif isinstance(data["names"], dict):
        data["names"] = [data["names"][i] for i in range(data["nc"])]
    assert len(data["names"]) == data["nc"], "Dataset names count must match nc"

    val = data["val"]
    vals = [val] if isinstance(val, str) else val
    missing = [x for x in vals if not Path(x).exists()]
    if missing:
        message = f"Dataset not found ⚠️, missing paths {missing}"
        download_cmd = data.get("download")
        if autodownload and download_cmd:
            LOGGER.info(f"{message}\nAttempting autodownload...")
            start = time.time()
            if isinstance(download_cmd, str) and download_cmd.startswith("http") and download_cmd.endswith(".zip"):
                download(download_cmd, dir=DATASETS_DIR, delete=True)
            elif isinstance(download_cmd, str) and download_cmd.startswith("bash "):
                os.system(download_cmd)
            else:
                exec(download_cmd, {"yaml": data})
            LOGGER.info(f"Dataset autodownload {'success ✅' if not missing else 'failure ⚠️'}, {time.time() - start:.1f}s")
        else:
            raise Exception(emojis(message))
    return data


def check_amp(model):
    device = next(model.parameters()).device
    if device.type == "cpu":
        return False
    try:
        from models.common import AutoShape

        im = np.zeros((640, 640, 3), dtype=np.uint8)
        m = AutoShape(deepcopy(model).eval(), verbose=False)
        a = m(im, size=640)
        with torch.cuda.amp.autocast(True):
            b = m(im, size=640)
        return a[0].shape == b[0].shape
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ AMP checks failed: {e}")
        return False


def check_yolo(verbose=True, device=""):
    if RANK in (-1, 0):
        try:
            import psutil

            total = psutil.virtual_memory().total / 1e9
            disk = psutil.disk_usage("/").free / 1e9
            if verbose:
                LOGGER.info(f"Setup complete ✅ ({os.cpu_count()} CPUs, {total:.1f} GB RAM, {disk:.1f} GB disk)")
        except Exception:
            pass
        check_font()
    return ROOT


def file_search(filename, path="."):
    files = glob.glob(str(Path(path) / "**" / filename), recursive=True)
    assert files, f"File not found: {filename}"
    assert len(files) == 1, f"Multiple files found: {files}"
    return files[0]


def increment_path(path, exist_ok=False, sep="", mkdir=False):
    path = Path(path)
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix("")
        for n in range(2, 10000):
            candidate = Path(f"{path}{sep}{n}{suffix}")
            if not candidate.exists():
                path = candidate
                break
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,
):
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are 0.0 to 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU threshold {iou_thres}, valid values are 0.0 to 1.0"

    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    device = prediction.device
    mps = "mps" in device.type
    if mps:
        prediction = prediction.cpu()

    bs = prediction.shape[0]
    nc = prediction.shape[2] - nm - 5
    xc = prediction[..., 4] > conf_thres
    max_wh = 7680
    max_nms = 30000
    time_limit = 0.5 + 0.05 * bs
    multi_label &= nc > 1
    merge = False
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    start = time.time()

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]
            v[:, 4] = 1.0
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0
            x = torch.cat((x, v), 0)

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]
        box = xywh2xyxy(x[:, :4])
        mask = x[:, 5 + nc:]

        if multi_label:
            i, j = (x[:, 5:5 + nc] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float(), mask[i]), 1)
        else:
            conf, j = x[:, 5:5 + nc].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        offsets = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + offsets, x[:, 4]
        keep = torchvision.ops.nms(boxes, scores, iou_thres)
        keep = keep[:max_det]

        if merge and 1 < n < 3000:
            iou = box_iou(boxes[keep], boxes) > iou_thres
            weights = iou * scores[None]
            x[keep, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if True:
                keep = keep[iou.sum(1) > 1]

        output[xi] = x[keep]
        if time.time() - start > time_limit:
            LOGGER.warning(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break

    return [x.to(device) for x in output] if mps else output


def strip_optimizer(f="best.pt", s=""):
    x = torch_load(f, map_location=torch.device("cpu"))
    if x.get("ema"):
        x["model"] = x["ema"]
    for key in ("optimizer", "best_fitness", "ema", "updates"):
        x[key] = None
    x["epoch"] = -1
    x["model"].half()
    for p in x["model"].parameters():
        p.requires_grad = False
    target = s or f
    torch.save(x, target)
    mb = os.path.getsize(target) / 1e6
    LOGGER.info(f"Optimizer stripped from {f}, saved as {target}, {mb:.1f}MB")


def print_args(args=None, show_file=True, show_func=False):
    frame = inspect.currentframe().f_back
    file = Path(frame.f_code.co_filename).resolve()
    func = frame.f_code.co_name
    if args is None:
        args = frame.f_locals
    elif hasattr(args, "__dict__"):
        args = vars(args)
    text = ", ".join(f"{k}={v}" for k, v in args.items())
    prefix = (f"{file.stem}: " if show_file else "") + (f"{func}: " if show_func else "")
    LOGGER.info(colorstr(prefix) + text)


def yaml_load(file="data.yaml"):
    with open(file, errors="ignore") as f:
        return yaml.safe_load(f)


def yaml_save(file="data.yaml", data=None):
    data = {} if data is None else data
    clean = {}
    for k, v in data.items():
        clean[k] = str(v) if isinstance(v, Path) else v
    with open(file, "w") as f:
        yaml.safe_dump(clean, f, sort_keys=False, allow_unicode=True)


def print_mutation(keys, results, hyp, save_dir, bucket, prefix=colorstr("evolve: ")):
    values = [f"{x:.5g}" for x in results]
    LOGGER.info(prefix + ", ".join(values))
    row = dict(zip(keys, results))
    row.update(hyp)
    csv_file = Path(save_dir) / "evolve.csv"
    header = not csv_file.exists()
    pd.DataFrame([row]).to_csv(csv_file, mode="a", header=header, index=False)
    yaml_save(Path(save_dir) / "hyp_evolve.yaml", hyp)
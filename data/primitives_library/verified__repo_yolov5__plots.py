import contextlib
import math
import os
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d
from ultralytics.utils.plotting import Annotator, colors

from utils import TryExcept, threaded
from utils.general import LOGGER, xywh2xyxy, xyxy2xywh
from utils.metrics import fitness

RANK = int(os.getenv("RANK", "-1"))
matplotlib.rc("font", size=11)
matplotlib.use("Agg")


def hist2d(x, y, n=100):
    x = np.asarray(x)
    y = np.asarray(y)
    xe = np.linspace(x.min(), x.max(), n)
    ye = np.linspace(y.min(), y.max(), n)
    counts, xe, ye = np.histogram2d(x, y, bins=(xe, ye))
    xi = np.clip(np.digitize(x, xe) - 1, 0, counts.shape[0] - 1)
    yi = np.clip(np.digitize(y, ye) - 1, 0, counts.shape[1] - 1)
    return np.log(counts[xi, yi])


def output_to_target(output, max_det=300):
    result = []
    for image_index, prediction in enumerate(output):
        boxes, scores, categories = prediction[:max_det, :6].cpu().split((4, 1, 1), dim=1)
        image_ids = torch.full((scores.shape[0], 1), image_index)
        result.append(torch.cat((image_ids, categories, xyxy2xywh(boxes), scores), dim=1))
    return torch.cat(result, dim=0).numpy()


@threaded
def plot_images(images, targets, paths=None, fname="images.jpg", names=None):
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    maximum = 16
    max_dimension = 1920
    batch, _, height, width = images.shape
    batch = min(batch, maximum)
    side = np.ceil(batch**0.5)

    if np.max(images[0]) <= 1:
        images *= 255

    canvas = np.full((int(side * height), int(side * width), 3), 255, dtype=np.uint8)
    for index, image in enumerate(images[:maximum]):
        left = int(width * (index // side))
        top = int(height * (index % side))
        canvas[top : top + height, left : left + width] = image.transpose(1, 2, 0)

    ratio = max_dimension / side / max(height, width)
    if ratio < 1:
        height = math.ceil(height * ratio)
        width = math.ceil(width * ratio)
        canvas = cv2.resize(canvas, tuple(int(v * side) for v in (width, height)))

    fontsize = int((height + width) * side * 0.01)
    painter = Annotator(
        canvas,
        line_width=round(fontsize / 10),
        font_size=fontsize,
        pil=True,
        example=names,
    )

    for index in range(batch):
        left = int(width * (index // side))
        top = int(height * (index % side))
        painter.rectangle([left, top, left + width, top + height], None, (255, 255, 255), width=2)

        if paths:
            painter.text([left + 5, top + 5], Path(paths[index]).name[:40], txt_color=(220, 220, 220))

        if len(targets):
            rows = targets[targets[:, 0] == index]
            boxes = xywh2xyxy(rows[:, 2:6]).T
            classes = rows[:, 1].astype(int)
            is_label = rows.shape[1] == 6
            confidence = None if is_label else rows[:, 6]

            if boxes.shape[1]:
                if boxes.max() <= 1.01:
                    boxes[[0, 2]] *= width
                    boxes[[1, 3]] *= height
                elif ratio < 1:
                    boxes *= ratio

                boxes[[0, 2]] += left
                boxes[[1, 3]] += top

                for row, box in enumerate(boxes.T.tolist()):
                    category = classes[row]
                    color = colors(category)
                    display_name = names[category] if names else category
                    if is_label or confidence[row] > 0.25:
                        text = f"{display_name}" if is_label else f"{display_name} {confidence[row]:.1f}"
                        painter.box_label(box, text, color=color)

    painter.im.save(fname)


def plot_val_study(file="", dir="", x=None):
    destination = Path(file).parent if file else Path(dir)
    additional = False

    if additional:
        axes = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)[1].ravel()

    _, speed_axis = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    for study in sorted(destination.glob("study*.txt")):
        values = np.loadtxt(study, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        epochs = np.arange(values.shape[1]) if x is None else np.array(x)

        if additional:
            titles = (
                "P",
                "R",
                "mAP@.5",
                "mAP@.5:.95",
                "t_preprocess (ms/img)",
                "t_inference (ms/img)",
                "t_NMS (ms/img)",
            )
            for i, title in enumerate(titles):
                axes[i].plot(epochs, values[i], ".-", linewidth=2, markersize=8)
                axes[i].set_title(title)

        cutoff = values[3].argmax() + 1
        speed_axis.plot(
            values[5, 1:cutoff],
            values[3, 1:cutoff] * 100,
            ".-",
            linewidth=2,
            markersize=8,
            label=study.stem.replace("study_coco_", "").replace("yolo", "YOLO"),
        )

    speed_axis.plot(
        1000 / np.array([209, 140, 97, 58, 35, 18]),
        [34.6, 40.5, 43.0, 47.5, 49.7, 51.5],
        "k.-",
        linewidth=2,
        markersize=8,
        alpha=0.25,
        label="EfficientDet",
    )
    speed_axis.grid(alpha=0.2)
    speed_axis.set_yticks(np.arange(20, 60, 5))
    speed_axis.set_xlim(0, 57)
    speed_axis.set_ylim(25, 55)
    speed_axis.set_xlabel("GPU Speed (ms/img)")
    speed_axis.set_ylabel("COCO AP val")
    speed_axis.legend(loc="lower right")

    output = destination / "study.png"
    LOGGER.info(f"Saving {output}...")
    plt.savefig(output, dpi=300)


@TryExcept()
def plot_labels(labels, names=(), save_dir=Path("")):
    save_dir = Path(save_dir)
    LOGGER.info(f"Plotting labels to {save_dir / 'labels.jpg'}... ")

    category, box_values = labels[:, 0], labels[:, 1:].transpose()
    class_count = int(category.max() + 1)
    data = pd.DataFrame(box_values.transpose(), columns=["x", "y", "width", "height"])

    sn.pairplot(
        data,
        corner=True,
        diag_kind="auto",
        kind="hist",
        diag_kws={"bins": 50},
        plot_kws={"pmax": 0.9},
    )
    plt.savefig(save_dir / "labels_correlogram.jpg", dpi=200)
    plt.close()

    matplotlib.use("svg")
    axes = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)[1].ravel()

    histogram = axes[0].hist(category, bins=np.linspace(0, class_count, class_count + 1) - 0.5, rwidth=0.8)
    with contextlib.suppress(Exception):
        for i in range(class_count):
            histogram[2].patches[i].set_color(np.array(colors(i)) / 255)

    axes[0].set_ylabel("instances")
    if 0 < len(names) < 30:
        axes[0].set_xticks(range(len(names)))
        axes[0].set_xticklabels(list(names.values()), rotation=90, fontsize=10)
    else:
        axes[0].set_xlabel("classes")

    sn.histplot(data, x="x", y="y", ax=axes[2], bins=50, pmax=0.9)
    sn.histplot(data, x="width", y="height", ax=axes[3], bins=50, pmax=0.9)

    labels[:, 1:3] = 0.5
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
    image = Image.fromarray(np.full((2000, 2000, 3), 255, dtype=np.uint8))
    drawer = ImageDraw.Draw(image)
    for cls, *coordinates in labels[:1000]:
        drawer.rectangle(coordinates, width=1, outline=colors(cls))

    axes[1].imshow(image)
    axes[1].axis("off")

    for axis in axes:
        for spine in ("top", "right", "left", "bottom"):
            axis.spines[spine].set_visible(False)

    plt.savefig(save_dir / "labels.jpg", dpi=200)
    matplotlib.use("Agg")
    plt.close()


def imshow_cls(im, labels=None, pred=None, names=None, nmax=25, verbose=False, f=Path("images.jpg")):
    from utils.augmentations import denormalize

    names = names or [f"class{i}" for i in range(1000)]
    blocks = torch.chunk(denormalize(im.clone()).cpu().float(), len(im), dim=0)
    count = min(len(blocks), nmax)
    grid = math.ceil(count**0.5)
    figure, axes = plt.subplots(grid, grid, figsize=(12, 12), tight_layout=True)
    axes = np.asarray(axes).reshape(-1)

    for i in range(count):
        axes[i].imshow(blocks[i].squeeze().permute(1, 2, 0))
        title = ""
        if labels is not None:
            value = labels[i]
            value = value.item() if hasattr(value, "item") else value
            title = str(names[int(value)])
        if pred is not None:
            value = pred[i]
            value = value.item() if hasattr(value, "item") else value
            prediction = str(names[int(value)])
            title = f"{title}{' ' * 30}{prediction}" if title else prediction
        axes[i].set_title(title)
        axes[i].axis("off")

    for axis in axes[count:]:
        axis.axis("off")

    f = Path(f)
    plt.savefig(f, dpi=300, bbox_inches="tight")
    plt.close()
    if verbose:
        LOGGER.info(f"Saving {f}...")


def feature_visualization(x, module_type, stage, n=32, save_dir=Path("runs/detect/exp")):
    if "Detect" in module_type:
        return

    save_dir = Path(save_dir)
    name = module_type.split(".")[-1]
    png = save_dir / f"stage{stage}_{name}_features.png"
    LOGGER.info(f"Saving {png}... ({n}/{x.shape[1]})")

    features = x[0].detach().cpu()
    n = min(n, features.shape[0])
    columns = 8
    rows = math.ceil(n / columns)
    figure, axes = plt.subplots(rows, columns, tight_layout=True)
    axes = np.asarray(axes).reshape(-1)

    for i in range(n):
        axes[i].imshow(features[i], cmap="gray")
        axes[i].axis("off")
    for axis in axes[n:]:
        axis.axis("off")

    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.close()
    np.save(str(png.with_suffix(".npy")), features.numpy())


def plot_evolve(evolve_csv="path/to/evolve.csv"):
    evolve_csv = Path(evolve_csv)
    if not evolve_csv.exists():
        LOGGER.info(f"File not found: {evolve_csv}")
        return

    dataframe = pd.read_csv(evolve_csv)
    values = dataframe.values
    if values.ndim != 2 or values.shape[0] == 0:
        return

    score = fitness(values[:, :7])
    keys = list(dataframe.columns)
    parameter_columns = keys[7:]

    figure = plt.figure(figsize=(10, 12), tight_layout=True)
    sn.set_theme(context="notebook", style="whitegrid")

    for i, key in enumerate(parameter_columns):
        y = values[:, i + 7].astype(float)
        x = values[:, i].astype(float) if i < values.shape[1] else np.arange(len(y))
        best = score.argmax()

        axis = figure.add_subplot(5, 5, i + 1)
        axis.scatter(x, y, c=hist2d(x, y), cmap="viridis", alpha=0.8, edgecolors="none")
        axis.plot(x[best], y[best], "k.", markersize=15)
        axis.set_title(f"{key} = {y[best]:.3g}", fontdict={"size": 9})
        axis.set_xlabel(key)
        axis.set_ylabel("fitness")

    output = evolve_csv.with_suffix(".png")
    LOGGER.info(f"Saving {output}...")
    plt.savefig(output, dpi=200)
    plt.close()


def plot_val_txt():
    values = np.loadtxt("val.txt", dtype=np.float32, ndmin=2)
    boxes = xywh2xyxy(values[:, 1:5])
    centers = values[:, 1:3]

    figure, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
    axes[0].scatter(centers[:, 0], centers[:, 1], c="k", s=2)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Image centers")

    axes[1].scatter(boxes[:, 0], boxes[:, 1], c="k", s=2)
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("y1")
    axes[1].set_title("Image corners")
    plt.savefig("val.jpg", dpi=200)
    plt.close()


def plot_targets_txt():
    values = np.loadtxt("targets.txt", dtype=np.float32, ndmin=2)
    labels = ("x targets", "y targets", "width targets", "height targets")
    figure, axes = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)

    for i, axis in enumerate(axes.ravel()):
        if i < values.shape[1]:
            axis.hist(values[:, i], bins=100)
        axis.set_title(labels[i])

    plt.savefig("targets.jpg", dpi=200)
    plt.close()


def plot_results(file="path/to/results.csv", dir=""):
    files = [Path(file)] if file else sorted(Path(dir).glob("results*.csv"))
    metric_names = [
        "train/box_loss",
        "train/obj_loss",
        "train/cls_loss",
        "metrics/precision",
        "metrics/recall",
        "val/box_loss",
        "val/obj_loss",
        "val/cls_loss",
        "metrics/mAP_0.5",
        "metrics/mAP_0.5:0.95",
    ]

    figure, axes = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    axes = axes.ravel()

    for result_file in files:
        try:
            frame = pd.read_csv(result_file)
            columns = [column.strip() for column in frame.columns]
            frame.columns = columns
            epoch = frame.iloc[:, 0].astype(float).to_numpy()

            for i, axis in enumerate(axes):
                if i + 1 >= frame.shape[1]:
                    continue
                series = frame.iloc[:, i + 1].astype(float).to_numpy()
                axis.plot(epoch, series, marker=".", label=result_file.stem, linewidth=2, markersize=8)
                axis.set_title(columns[i + 1] if i + 1 < len(columns) else metric_names[i])
        except Exception as error:
            LOGGER.info(f"WARNING: Plotting error for {result_file}: {error}")

    for axis in axes:
        axis.legend()

    output = (Path(dir) if dir else files[0].parent if files else Path(".")) / "results.png"
    LOGGER.info(f"Saving {output}...")
    plt.savefig(output, dpi=200)
    plt.close()


def profile_idetection(start=0, stop=0, labels=(), save_dir=""):
    save_dir = Path(save_dir)
    files = [save_dir / f"frames_{x}.txt" for x in ("iDetection", "iOS")]
    titles = [
        "Images",
        "Free Storage (GB)",
        "Battery (%)",
        "Mem Usage (MB)",
        "Frame Rate (FPS)",
        "Inference (ms)",
        "WiFi (Mbps)",
        "LTE (Mbps)",
    ]

    figure, axes = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)
    axes = axes.ravel()

    for file in files:
        if not file.exists():
            continue
        data = np.loadtxt(file, ndmin=2).T
        if data.shape[0] < 2:
            continue

        x = data[0]
        indices = (x >= start) & (x <= stop) if stop else x >= start
        x = x[indices]

        for i, axis in enumerate(axes):
            if i + 1 >= data.shape[0]:
                continue
            y = data[i + 1][indices]
            axis.plot(x, y, ".-", linewidth=1, markersize=3, label=file.stem)
            axis.set_title(titles[i])
            axis.grid(alpha=0.2)

    for axis in axes:
        if axis.lines:
            axis.legend()

    output = save_dir / "idetection_profile.png"
    plt.savefig(output, dpi=200)
    plt.close()


def save_one_box(xyxy, im, file=Path("im.jpg"), gain=1.02, pad=10, square=False, BGR=False, save=True):
    if isinstance(xyxy, torch.Tensor):
        box = xyxy.detach().clone().view(-1, 4).float()
    else:
        box = torch.as_tensor(np.asarray(xyxy), dtype=torch.float32).view(-1, 4)

    xywh = xyxy2xywh(box)
    if square:
        xywh[:, 2:] = xywh[:, 2:].max(1, keepdim=True)[0]
    xywh[:, 2:] = xywh[:, 2:] * gain + pad
    box = xywh2xyxy(xywh).round().long().cpu().numpy()[0]

    height, width = im.shape[:2]
    box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
    x1, y1, x2, y2 = box.tolist()
    crop = im[y1:y2, x1:x2]

    if save:
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(file), crop if BGR else crop[..., ::-1])

    return crop
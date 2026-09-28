import math
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from ultralytics.utils.metrics import bbox_ioa

from utils.general import LOGGER, check_version, colorstr, resample_segments, segment2box

IMAGENET_MEAN = 0.485, 0.456, 0.406
IMAGENET_STD = 0.229, 0.224, 0.225


class Albumentations:
    """Optional Albumentations image and bounding-box augmentation pipeline."""

    def __init__(self, size=640):
        self.transform = None
        prefix = colorstr("albumentations: ")
        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)
            v2 = check_version(A.__version__, "2.0.0")
            transforms = [
                A.RandomResizedCrop(size=(size, size), scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0)
                if v2
                else A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.0), ratio=(0.9, 1.11), p=0.0),
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_range=(75, 100), p=0.0)
                if v2
                else A.ImageCompression(quality_lower=75, p=0.0),
            ]
            self.transform = A.Compose(
                transforms,
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
            )
            LOGGER.info(prefix + ", ".join(str(x).replace("always_apply=False, ", "") for x in transforms if x.p))
        except ImportError:
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            result = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])
            im = result["image"]
            labels = np.array([[cls, *box] for cls, box in zip(result["class_labels"], result["bboxes"])])
        return im, labels


def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Undo per-channel ImageNet normalization in place for BCHW tensors."""
    for i in range(3):
        x[:, i] = x[:, i] * std[i] + mean[i]
    return x


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    """Apply random HSV-space color augmentation in place."""
    if hgain or sgain or vgain:
        gains = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
        hue, saturation, value = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype
        values = np.arange(256, dtype=gains.dtype)

        hue_lut = ((values * gains[0]) % 180).astype(dtype)
        saturation_lut = np.clip(values * gains[1], 0, 255).astype(dtype)
        value_lut = np.clip(values * gains[2], 0, 255).astype(dtype)

        hsv = cv2.merge(
            (
                cv2.LUT(hue, hue_lut),
                cv2.LUT(saturation, saturation_lut),
                cv2.LUT(value, value_lut),
            )
        )
        cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR, dst=im)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resize an image with padding while preserving its aspect ratio."""
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    ratio_value = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        ratio_value = min(ratio_value, 1.0)

    ratio = ratio_value, ratio_value
    new_unpad = round(shape[1] * ratio_value), round(shape[0] * ratio_value)
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = new_shape[1], new_shape[0]
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top = round(dh - 0.1)
    bottom = round(dh + 0.1)
    left = round(dw - 0.1)
    right = round(dw + 0.1)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


def random_perspective(
    im, targets=(), segments=(), degrees=10, translate=0.1, scale=0.1, shear=10, perspective=0.0, border=(0, 0)
):
    """Randomly transform an image and its targets by affine or perspective projection."""
    height = im.shape[0] + border[0] * 2
    width = im.shape[1] + border[1] * 2

    center = np.eye(3)
    center[0, 2] = -im.shape[1] / 2
    center[1, 2] = -im.shape[0] / 2

    perspective_matrix = np.eye(3)
    perspective_matrix[2, 0] = random.uniform(-perspective, perspective)
    perspective_matrix[2, 1] = random.uniform(-perspective, perspective)

    rotation = np.eye(3)
    angle = random.uniform(-degrees, degrees)
    scale_factor = random.uniform(1 - scale, 1 + scale)
    rotation[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale_factor)

    shear_matrix = np.eye(3)
    shear_matrix[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    shear_matrix[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    translation = np.eye(3)
    translation[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    translation[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height

    matrix = translation @ shear_matrix @ rotation @ perspective_matrix @ center

    if border[0] != 0 or border[1] != 0 or (matrix != np.eye(3)).any():
        if perspective:
            im = cv2.warpPerspective(im, matrix, dsize=(width, height), borderValue=(114, 114, 114))
        else:
            im = cv2.warpAffine(im, matrix[:2], dsize=(width, height), borderValue=(114, 114, 114))

    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments) and len(segments) == n
        new_boxes = np.zeros((n, 4))

        if use_segments:
            segments = resample_segments(segments)
            for i, segment in enumerate(segments):
                points = np.ones((len(segment), 3))
                points[:, :2] = segment
                points = points @ matrix.T
                points = points[:, :2] / points[:, 2:3] if perspective else points[:, :2]
                new_boxes[i] = segment2box(points, width, height)
        else:
            points = np.ones((n * 4, 3))
            points[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
            points = points @ matrix.T
            points = (points[:, :2] / points[:, 2:3] if perspective else points[:, :2]).reshape(n, 8)

            x = points[:, [0, 2, 4, 6]]
            y = points[:, [1, 3, 5, 7]]
            new_boxes = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clip(0, width)
            new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clip(0, height)

        keep = box_candidates(
            box1=targets[:, 1:5].T * scale_factor,
            box2=new_boxes.T,
            area_thr=0.01 if use_segments else 0.10,
        )
        targets = targets[keep]
        targets[:, 1:5] = new_boxes[keep]

    return im, targets


def copy_paste(im, labels, segments, p=0.5):
    """Apply Copy-Paste augmentation by horizontally mirroring selected object masks."""
    n = len(segments)
    if p and n:
        h, w, c = im.shape
        im_new = np.zeros(im.shape, np.uint8)

        for j in random.sample(range(n), k=round(p * n)):
            label = labels[j]
            segment = segments[j]
            box = w - label[3], label[2], w - label[1], label[4]
            ioa = bbox_ioa(box, labels[:, 1:5])

            if (ioa < 0.30).all():
                labels = np.concatenate((labels, [[label[0], *box]]), 0)
                segments.append(np.concatenate((w - segment[:, 0:1], segment[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[-1].astype(np.int32)], -1, (1, 1, 1), cv2.FILLED)

        result = cv2.bitwise_and(src1=im, src2=im_new)
        result = cv2.flip(result, 1)
        mask = result > 0
        im[mask] = result[mask]

    return im, labels, segments


def cutout(im, labels, p=0.5):
    """Randomly cover rectangular regions of an image and remove heavily obscured labels."""
    if random.random() < p:
        height, width = im.shape[:2]
        scales = [0.5] + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16

        for scale_value in scales:
            mask_h = random.randint(1, int(height * scale_value))
            mask_w = random.randint(1, int(width * scale_value))

            xmin = max(0, random.randint(0, width) - mask_w // 2)
            ymin = max(0, random.randint(0, height) - mask_h // 2)
            xmax = min(width, xmin + mask_w)
            ymax = min(height, ymin + mask_h)

            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            if len(labels) and scale_value > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                ioa = bbox_ioa(box, labels[:, 1:5])
                labels = labels[ioa < 0.60]

    return labels


def mixup(im, labels, im2, labels2):
    """Blend two images and concatenate their labels using a beta-distributed ratio."""
    ratio = np.random.beta(32.0, 32.0)
    im = (im * ratio + im2 * (1 - ratio)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
    """Return validity flags for transformed bounding boxes."""
    w1 = box1[2] - box1[0]
    h1 = box1[3] - box1[1]
    w2 = box2[2] - box2[0]
    h2 = box2[3] - box2[1]
    aspect_ratio = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (aspect_ratio < ar_thr)


def classify_albumentations(
    augment=True,
    size=224,
    scale=(0.08, 1.0),
    ratio=(0.75, 1.0 / 0.75),
    hflip=0.5,
    vflip=0.0,
    jitter=0.4,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    auto_aug=False,
):
    """Create an Albumentations preprocessing pipeline for image classification."""
    prefix = colorstr("albumentations: ")
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        check_version(A.__version__, "1.0.3", hard=True)
        v2 = check_version(A.__version__, "2.0.0")

        if augment:
            transforms = [
                A.RandomResizedCrop(size=(size, size), scale=scale, ratio=ratio)
                if v2
                else A.RandomResizedCrop(height=size, width=size, scale=scale, ratio=ratio)
            ]

            if auto_aug:
                LOGGER.info(f"{prefix}auto augmentations are currently not supported")
            else:
                if hflip > 0:
                    transforms.append(A.HorizontalFlip(p=hflip))
                if vflip > 0:
                    transforms.append(A.VerticalFlip(p=vflip))
                if jitter > 0:
                    transforms.append(A.ColorJitter(jitter, jitter, jitter, 0, p=0.5))
        else:
            transforms = [
                A.SmallestMaxSize(max_size=size),
                A.CenterCrop(height=size, width=size),
            ]

        transforms += [A.Normalize(mean=mean, std=std), ToTensorV2()]
        LOGGER.info(prefix + ", ".join(str(x).replace("always_apply=False, ", "") for x in transforms))
        return A.Compose(transforms)
    except ImportError:
        LOGGER.warning(f"{prefix}⚠️ not found, install with 'pip install albumentations'")
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")


def classify_transforms(size=224):
    """Create torchvision transforms for classification image preprocessing."""
    assert isinstance(size, int), f"ERROR: classify_transforms size {size} must be integer, not (list, tuple)"
    return T.Compose([T.CenterCrop(size), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
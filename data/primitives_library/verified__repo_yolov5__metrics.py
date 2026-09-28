import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import box_iou, mask_iou, plot_mc_curve, plot_pr_curve, smooth

from utils import TryExcept

trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def fitness(x):
    """Calculate the weighted validation score for an array of metric results."""
    weights = np.asarray((0.0, 0.0, 0.1, 0.9))
    return np.sum(x[:, :4] * weights, axis=1)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir=".", names=(), eps=1e-16, prefix=""):
    """Calculate per-class precision, recall, F1 score, and average precision."""
    order = np.argsort(-conf)
    tp = tp[order]
    conf = conf[order]
    pred_cls = pred_cls[order]

    unique_classes, counts = np.unique(target_cls, return_counts=True)
    class_count = len(unique_classes)
    x = np.linspace(0.0, 1.0, 1000)
    curves = []

    ap = np.zeros((class_count, tp.shape[1]))
    precision_samples = np.zeros((class_count, x.size))
    recall_samples = np.zeros((class_count, x.size))

    for class_index, class_id in enumerate(unique_classes):
        selected = pred_cls == class_id
        labels_count = counts[class_index]
        predictions_count = selected.sum()

        if predictions_count == 0 or labels_count == 0:
            continue

        false_positives = (1 - tp[selected]).cumsum(axis=0)
        true_positives = tp[selected].cumsum(axis=0)

        recall_curve = true_positives / (labels_count + eps)
        recall_samples[class_index] = np.interp(
            -x, -conf[selected], recall_curve[:, 0], left=0
        )

        precision_curve = true_positives / (true_positives + false_positives)
        precision_samples[class_index] = np.interp(
            -x, -conf[selected], precision_curve[:, 0], left=1
        )

        for threshold_index in range(tp.shape[1]):
            ap_value, envelope_precision, envelope_recall = compute_ap(
                recall_curve[:, threshold_index], precision_curve[:, threshold_index]
            )
            ap[class_index, threshold_index] = ap_value
            if plot and threshold_index == 0:
                curves.append(np.interp(x, envelope_recall, envelope_precision))

    f1 = 2 * precision_samples * recall_samples / (precision_samples + recall_samples + eps)

    selected_names = [value for key, value in names.items() if key in unique_classes]
    selected_names = dict(enumerate(selected_names))

    if plot:
        output_dir = Path(save_dir)
        plot_pr_curve(x, curves, ap, output_dir / f"{prefix}PR_curve.png", selected_names)
        plot_mc_curve(x, f1, output_dir / f"{prefix}F1_curve.png", selected_names, ylabel="F1")
        plot_mc_curve(
            x,
            precision_samples,
            output_dir / f"{prefix}P_curve.png",
            selected_names,
            ylabel="Precision",
        )
        plot_mc_curve(
            x,
            recall_samples,
            output_dir / f"{prefix}R_curve.png",
            selected_names,
            ylabel="Recall",
        )

    best = smooth(f1.mean(axis=0), 0.1).argmax()
    p = precision_samples[:, best]
    r = recall_samples[:, best]
    f1 = f1[:, best]

    true_positives = (r * counts).round()
    false_positives = (true_positives / (p + eps) - true_positives).round()

    return true_positives, false_positives, p, r, f1, ap, unique_classes.astype(int)


def compute_ap(recall, precision):
    """Integrate a precision-recall curve and return AP with its padded envelopes."""
    recall_envelope = np.concatenate((np.array([0.0]), recall, np.array([1.0])))
    precision_envelope = np.concatenate((np.array([1.0]), precision, np.array([0.0])))

    precision_envelope = np.flip(np.maximum.accumulate(np.flip(precision_envelope)))

    points = np.linspace(0.0, 1.0, 101)
    average_precision = trapezoid(
        np.interp(points, recall_envelope, precision_envelope),
        points,
    )

    return average_precision, precision_envelope, recall_envelope


class ConfusionMatrix:
    """Maintain a detection confusion matrix with an additional background class."""

    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        """Create a matrix for nc foreground classes."""
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """Add one set of detections and labels to the confusion matrix."""
        if detections is None:
            ground_truth = labels.int()
            for class_id in ground_truth:
                self.matrix[self.nc, class_id] += 1
            return

        detections = detections[detections[:, 4] > self.conf]
        ground_truth = labels[:, 0].int()
        predicted = detections[:, 5].int()

        overlaps = box_iou(labels[:, 1:], detections[:, :4])
        label_indices, detection_indices = torch.where(overlaps > self.iou_thres)

        if label_indices.shape[0]:
            matches = torch.cat(
                (
                    torch.stack((label_indices, detection_indices), dim=1),
                    overlaps[label_indices, detection_indices].unsqueeze(1),
                ),
                dim=1,
            ).cpu().numpy()

            if label_indices.shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        has_matches = matches.shape[0] > 0
        matched_labels, matched_detections, _ = matches.transpose().astype(int)

        for label_index, class_id in enumerate(ground_truth):
            current = matched_labels == label_index
            if has_matches and current.sum() == 1:
                self.matrix[predicted[matched_detections[current]], class_id] += 1
            else:
                self.matrix[self.nc, class_id] += 1

        if has_matches:
            for detection_index, class_id in enumerate(predicted):
                if not any(matched_detections == detection_index):
                    self.matrix[class_id, self.nc] += 1

    @TryExcept("ConfusionMatrix plot failure")
    def plot(self, normalize=True, save_dir="", names=()):
        """Render and save the confusion matrix image."""
        import seaborn as sn

        denominator = self.matrix.sum(axis=0).reshape(1, -1) + 1e-9 if normalize else 1
        values = self.matrix / denominator
        values[values < 0.005] = np.nan

        figure, axis = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        name_count = len(names)

        sn.set(font_scale=1.0 if self.nc < 50 else 0.8)
        use_names = 0 < name_count < 99 and name_count == self.nc
        tick_labels = [*names, "background"] if use_names else "auto"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sn.heatmap(
                values,
                ax=axis,
                annot=self.nc < 30,
                annot_kws={"size": 8},
                cmap="Blues",
                fmt=".2f",
                square=True,
                vmin=0.0,
                xticklabels=tick_labels,
                yticklabels=tick_labels,
            ).set_facecolor((1, 1, 1))

        axis.set_xlabel("True")
        axis.set_ylabel("Predicted")
        axis.set_title("Confusion Matrix")
        figure.savefig(Path(save_dir) / "confusion_matrix.png", dpi=250)
        plt.close(figure)

    def print(self):
        """Log each row of the confusion matrix."""
        for row in range(self.nc + 1):
            LOGGER.info(" ".join(map(str, self.matrix[row])))
import random

import numpy as np
import torch
import yaml

from utils import TryExcept
from utils.general import LOGGER, TQDM, colorstr

PREFIX = colorstr("AutoAnchor: ")


def check_anchor_order(m):
    """Ensure anchor scale ordering follows the ordering of detection strides."""
    anchor_areas = m.anchors.prod(dim=-1).mean(dim=-1).reshape(-1)
    area_delta = anchor_areas[-1] - anchor_areas[0]
    stride_delta = m.stride[-1] - m.stride[0]

    if area_delta and area_delta.sign() != stride_delta.sign():
        LOGGER.info(f"{PREFIX}Reversing anchor order")
        m.anchors[:] = m.anchors.flip(0)


@TryExcept(colorstr("AutoAnchor"))
def check_anchors(dataset, model, thr=4.0, imgsz=640):
    """Measure current anchor suitability and generate improved anchors when needed."""
    detect_layer = model.module.model[-1] if hasattr(model, "module") else model.model[-1]

    resized_shapes = imgsz * dataset.shapes / dataset.shapes.max(axis=1, keepdims=True)
    random_scale = np.random.uniform(0.9, 1.1, size=(len(resized_shapes), 1))
    widths_heights = torch.tensor(
        np.concatenate(
            [
                labels[:, 3:5] * shape
                for shape, labels in zip(resized_shapes * random_scale, dataset.labels)
            ]
        )
    ).float()

    def evaluate(candidate):
        ratios = widths_heights[:, None] / candidate[None]
        quality = torch.minimum(ratios, 1.0 / ratios).min(dim=2)[0]
        best_quality = quality.max(dim=1)[0]
        anchors_per_target = (quality > 1.0 / thr).float().sum(dim=1).mean()
        recall = (best_quality > 1.0 / thr).float().mean()
        return recall, anchors_per_target

    strides = detect_layer.stride.to(detect_layer.anchors.device).reshape(-1, 1, 1)
    pixel_anchors = detect_layer.anchors.detach().clone() * strides
    bpr, aat = evaluate(pixel_anchors.cpu().reshape(-1, 2))

    report = f"\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). "
    if bpr > 0.98:
        LOGGER.info(f"{report}Current anchors are a good fit to dataset ✅")
        return

    LOGGER.info(f"{report}Anchors are a poor fit to dataset ⚠️, attempting to improve...")
    count = detect_layer.anchors.numel() // 2
    proposed = kmean_anchors(dataset, n=count, img_size=imgsz, thr=thr, gen=1000, verbose=False)
    proposed_bpr = evaluate(proposed)[0]

    if proposed_bpr > bpr:
        proposed = torch.tensor(proposed, device=detect_layer.anchors.device).type_as(detect_layer.anchors)
        detect_layer.anchors[:] = proposed.clone().view_as(detect_layer.anchors)
        check_anchor_order(detect_layer)
        detect_layer.anchors /= strides
        message = f"{PREFIX}Done ✅ (optional: update model *.yaml to use these anchors in the future)"
    else:
        message = f"{PREFIX}Done ⚠️ (original anchors better than new anchors, proceeding with original anchors)"

    LOGGER.info(message)


def kmean_anchors(dataset="./data/coco128.yaml", n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """Generate k-means initialized and genetically evolved anchors for a dataset."""
    from scipy.cluster.vq import kmeans

    rng = np.random
    threshold = 1.0 / thr

    def ratios_and_best(anchors, sizes):
        ratios = sizes[:, None] / anchors[None]
        scores = torch.minimum(ratios, 1.0 / ratios).min(dim=2)[0]
        return scores, scores.max(dim=1)[0]

    def fitness(anchors):
        _, best = ratios_and_best(torch.tensor(anchors, dtype=torch.float32), filtered_wh)
        return (best * (best > threshold).float()).mean()

    def describe(anchors, verbose=True):
        ordered = anchors[np.argsort(anchors.prod(axis=1))]
        scores, best = ratios_and_best(ordered, original_wh)
        bpr = (best > threshold).float().mean()
        aat = (scores > threshold).float().mean() * n

        text = (
            f"{PREFIX}thr={threshold:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n"
            f"{PREFIX}n={n}, img_size={img_size}, metric_all={scores.mean():.3f}/{best.mean():.3f}-mean/best, "
            f"past_thr={scores[scores > threshold].mean():.3f}-mean: "
        )
        text += "".join(f"{round(anchor[0])},{round(anchor[1])}, " for anchor in ordered)

        if verbose:
            LOGGER.info(text[:-2])
        return ordered

    if isinstance(dataset, str):
        with open(dataset, errors="ignore") as file:
            configuration = yaml.safe_load(file)

        from utils.dataloaders import LoadImagesAndLabels

        dataset = LoadImagesAndLabels(configuration["train"], augment=True, rect=True)

    scaled_shapes = img_size * dataset.shapes / dataset.shapes.max(axis=1, keepdims=True)
    wh_array = np.concatenate(
        [labels[:, 3:5] * shape for shape, labels in zip(scaled_shapes, dataset.labels)]
    )

    tiny_count = (wh_array < 3.0).any(axis=1).sum()
    if tiny_count:
        LOGGER.warning(
            f"{PREFIX}Extremely small objects found: {tiny_count} of {len(wh_array)} labels are <3 pixels in size"
        )

    retained = wh_array[(wh_array >= 2.0).any(axis=1)].astype(np.float32)

    try:
        LOGGER.info(f"{PREFIX}Running kmeans for {n} anchors on {len(retained)} points...")
        assert n <= len(retained)
        deviation = retained.std(axis=0)
        anchors = kmeans(retained / deviation, n, iter=30)[0] * deviation
        assert len(anchors) == n
    except Exception:
        LOGGER.warning(f"{PREFIX}switching strategies from kmeans to random init")
        anchors = np.sort(rng.rand(n * 2)).reshape(n, 2) * img_size

    filtered_wh, original_wh = (
        torch.tensor(values, dtype=torch.float32) for values in (retained, wh_array)
    )
    anchors = describe(anchors, verbose=False)

    current_fitness = fitness(anchors)
    anchor_shape = anchors.shape
    mutation_probability = 0.9
    mutation_sigma = 0.1

    progress = TQDM(range(gen))
    for _ in progress:
        mutation = np.ones(anchor_shape)

        while (mutation == 1).all():
            mutation = (
                (rng.random(anchor_shape) < mutation_probability)
                * random.random()
                * rng.randn(*anchor_shape)
                * mutation_sigma
                + 1
            ).clip(0.3, 3.0)

        candidate = (anchors.copy() * mutation).clip(min=2.0)
        candidate_fitness = fitness(candidate)

        if candidate_fitness > current_fitness:
            current_fitness = candidate_fitness
            anchors = candidate.copy()
            progress.desc = f"{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {current_fitness:.4f}"
            if verbose:
                describe(anchors, verbose=True)

    return describe(anchors).astype(np.float32)
# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg511::torch.max+torch.zeros
# name: torch_primitive
# summary: Uses torch.max, torch.zeros across 2 repos
# anchor_symbols: ['torch.max', 'torch.zeros']
# observed in 2 repos: ['Lightning-AI__torchmetrics', 'sb-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::lightautoml/ml_algo/torch_based/node_nn_model.py::to_one_hot ---
def to_one_hot(y, depth=None):
    """Takes integer with n dims and converts it to 1-hot representation with n + 1 dims.

    The n+1'st dimension will have zeros everywhere but at y'th index, where it will be equal to 1.

    Args:
        y : input integer (IntTensor, LongTensor or Variable) of any shape
        depth : the size of the one hot dimension

    Returns:
        one hot Tensor
    """
    y_flat = y.to(torch.int64).view(-1, 1)
    depth = depth if depth is not None else int(torch.max(y_flat)) + 1
    y_one_hot = torch.zeros(y_flat.size()[0], depth, device=y.device).scatter_(1, y_flat, 1)
    y_one_hot = y_one_hot.view(*(tuple(y.shape) + (-1,)))
    return y_one_hot

# --- from Lightning-AI__torchmetrics::src/torchmetrics/functional/segmentation/hausdorff_distance.py::hausdorff_distance ---
def hausdorff_distance(
    preds: Tensor,
    target: Tensor,
    num_classes: int,
    include_background: bool = False,
    distance_metric: Literal["euclidean", "chessboard", "taxicab"] = "euclidean",
    spacing: Optional[Union[Tensor, list[float]]] = None,
    directed: bool = False,
    input_format: Literal["one-hot", "index", "mixed"] = "one-hot",
) -> Tensor:
    """Calculate `Hausdorff Distance`_ for semantic segmentation.

    Args:
        preds: predicted binarized segmentation map
        target: target binarized segmentation map
        num_classes: number of classes
        include_background: whether to include background class in calculation
        distance_metric: distance metric to calculate surface distance. Choose one of `"euclidean"`,
          `"chessboard"` or `"taxicab"`
        spacing: spacing between pixels along each spatial dimension. If not provided the spacing is assumed to be 1
        directed: whether to calculate directed or undirected Hausdorff distance
        input_format: What kind of input the function receives.
            Choose between ``"one-hot"`` for one-hot encoded tensors, ``"index"`` for index tensors
            or ``"mixed"`` for one one-hot encoded and one index tensor

    Returns:
        Hausdorff Distance for each class and batch element

    Example:
        >>> from torch import randint
        >>> from torchmetrics.functional.segmentation import hausdorff_distance
        >>> preds = randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 prediction
        >>> target = randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 target
        >>> hausdorff_distance(preds, target, num_classes=5)
        tensor([[2.0000, 1.4142, 2.0000, 2.0000],
                [1.4142, 2.0000, 2.0000, 2.0000],
                [2.0000, 2.0000, 1.4142, 2.0000],
                [2.0000, 2.8284, 2.0000, 2.2361]])

    """
    _hausdorff_distance_validate_args(num_classes, include_background, distance_metric, spacing, directed, input_format)

    preds, target = _segmentation_inputs_format(preds, target, include_background, num_classes, input_format)

    distances = torch.zeros(preds.shape[0], preds.shape[1], device=preds.device)

    # TODO: add support for batched inputs
    for b in range(preds.shape[0]):
        for c in range(preds.shape[1]):
            dist = edge_surface_distance(
                preds=preds[b, c],
                target=target[b, c],
                distance_metric=distance_metric,
                spacing=spacing,
                symmetric=not directed,
            )
            distances[b, c] = torch.max(dist) if directed else torch.max(dist[0].max(), dist[1].max())  # type: ignore
    return distances

# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg617::torch.abs+torch.where
# name: torch_primitive
# summary: Uses torch.abs, torch.where across 2 repos
# anchor_symbols: ['torch.abs', 'torch.where']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::lightautoml/tasks/losses/torch.py::torch_quantile ---
def torch_quantile(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
    q: float = 0.9,
):
    """Computes Mean Quantile Error.

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.
        q: metric coefficient.

    Returns:
        metric value.

    """
    err = y_pred - y_true
    s = err < 0
    err = torch.abs(err)
    err = torch.where(s, err * (1 - q), err * q)

    if len(err.shape) == 2:
        err = err.sum(dim=1)

    if sample_weight is not None:
        err = err * sample_weight
        return err.mean() / sample_weight.mean()

    return err.mean()

# --- from sberbank-ai-lab__LightAutoML::lightautoml/tasks/losses/torch.py::torch_quantile ---
def torch_quantile(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
    q: float = 0.9,
):
    """Computes Mean Quantile Error.

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.
        q: metric coefficient.

    Returns:
        metric value.

    """
    err = y_pred - y_true
    s = err < 0
    err = torch.abs(err)
    err = torch.where(s, err * (1 - q), err * q)

    if len(err.shape) == 2:
        err = err.sum(dim=1)

    if sample_weight is not None:
        err = err * sample_weight
        return err.mean() / sample_weight.mean()

    return err.mean()

# --- from sb-ai-lab__LightAutoML::lightautoml/tasks/losses/torch.py::torch_huber ---
def torch_huber(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
    a: float = 0.9,
):
    """Computes Mean Huber Error.

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.
        a: metric coefficient.

    Returns:
        metric value.

    """
    err = y_pred - y_true
    s = torch.abs(err) < a
    err = torch.where(s, 0.5 * (err ** 2), a * torch.abs(err) - 0.5 * (a ** 2))

    if len(err.shape) == 2:
        err = err.sum(dim=1)

    if sample_weight is not None:
        err = err * sample_weight
        return err.mean() / sample_weight.mean()

    return err.mean()

# --- from sberbank-ai-lab__LightAutoML::lightautoml/tasks/losses/torch.py::torch_huber ---
def torch_huber(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
    a: float = 0.9,
):
    """Computes Mean Huber Error.

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.
        a: metric coefficient.

    Returns:
        metric value.

    """
    err = y_pred - y_true
    s = torch.abs(err) < a
    err = torch.where(s, 0.5 * (err ** 2), a * torch.abs(err) - 0.5 * (a ** 2))

    if len(err.shape) == 2:
        err = err.sum(dim=1)

    if sample_weight is not None:
        err = err * sample_weight
        return err.mean() / sample_weight.mean()

    return err.mean()

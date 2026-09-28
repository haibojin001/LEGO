# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg411::torch.mean+torch.sigmoid
# name: torch_primitive
# summary: Uses torch.mean, torch.sigmoid across 2 repos
# anchor_symbols: ['torch.mean', 'torch.sigmoid']
# observed in 2 repos: ['ruc-datalab__DeepAnalyze', 'sb-ai-lab__LightAutoML']...

# --- from ruc-datalab__DeepAnalyze::deepanalyze/ms-swift/swift/tuners/scetuning/scetuning_components.py::get_weight_value ---
def get_weight_value(weight_type, scaling, x):
    if weight_type in ["gate"]:
        scaling = torch.mean(torch.sigmoid(scaling(x)), dim=1).view(-1, 1, 1)
    elif weight_type in ["scale", "scale_channel"] or weight_type.startswith("scalar"):
        scaling = scaling
    else:
        scaling = None
    return scaling

# --- from ruc-datalab__DeepAnalyze::deepanalyze/ms-swift/swift/tuners/restuning_components.py::apply_data_weight ---
def apply_data_weight(data, scaling, weight_type):
    if weight_type in ["gate"]:
        scaling = torch.mean(torch.sigmoid(scaling(data)), dim=1).view(-1, 1, 1)
    elif weight_type in ["scale", "scale_channel"] or weight_type.startswith("scalar"):
        scaling = scaling
    else:
        scaling = None
    if scaling is not None:
        data = data * scaling
    return data

# --- from sb-ai-lab__LightAutoML::lightautoml/tasks/losses/torch.py::torch_f1 ---
def torch_f1(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    sample_weight: Optional[torch.Tensor] = None,
):
    """Computes F1 macro.

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.

    Returns:
        metric value.

    """
    y_pred = torch.sigmoid(y_pred)
    y_true = y_true[:, 0].type(torch.int64)
    y_true_ohe = torch.zeros_like(y_pred)

    y_true_ohe[range(y_true.shape[0]), y_true] = 1
    tp = y_true_ohe * y_pred
    if sample_weight is not None:
        sample_weight = sample_weight.unsqueeze(-1)
        sm = sample_weight.mean()
        tp = (tp * sample_weight).mean(dim=0) / sm
        f1 = (2 * tp) / (
            (y_pred * sample_weight).mean(dim=0) / sm + (y_true_ohe * sample_weight).mean(dim=0) / sm + 1e-7
        )

        return -f1.mean()

    tp = torch.mean(tp, dim=0)

    f1 = (2 * tp) / (y_pred.mean(dim=0) + y_true_ohe.mean(dim=0) + 1e-7)

    f1[f1 != f1] = 0

    return -f1.mean()

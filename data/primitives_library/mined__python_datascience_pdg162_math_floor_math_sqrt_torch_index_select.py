# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg162::math.floor+math.sqrt+torch.index_select
# name: math_torch_primitive
# summary: Uses math.floor, math.sqrt, torch.index_select across 3 repos
# anchor_symbols: ['math.floor', 'math.sqrt', 'torch.index_select']
# observed in 3 repos: ['AgnostiqHQ__covalent', 'Lightning-AI__torchmetrics', 'microsoft__nni']...

# --- from AgnostiqHQ__covalent::tests/stress_tests/scripts/tasks.py::test_prime ---
def test_prime(n):
    m = int(math.floor(math.sqrt(n)))
    for d in range(2, m + 1):
        if n % d == 0:
            return False
    return True

# --- from AgnostiqHQ__covalent::tests/stress_tests/scripts/parallel_mixed.py::test_prime ---
def test_prime(n):
    m = int(math.floor(math.sqrt(n)))
    for d in range(2, m + 1):
        if n % d == 0:
            return False
    return True

# --- from Lightning-AI__torchmetrics::src/torchmetrics/utilities/plot.py::_get_col_row_split ---
def _get_col_row_split(n: int) -> tuple[int, int]:
    """Split `n` figures into `rows` x `cols` figures."""
    nsq = sqrt(n)
    if int(nsq) == nsq:  # square number
        return int(nsq), int(nsq)
    if floor(nsq) * ceil(nsq) >= n:
        return floor(nsq), ceil(nsq)
    return ceil(nsq), ceil(nsq)

# --- from microsoft__nni::nni/compression/speedup/replacement.py::convert_to_coarse_mask ---
def convert_to_coarse_mask(t_mask, dim):
    """
    Convert the mask tensor to the coarse-grained mask tensor.

    Parameters
    ---------
    t_mask: torch.Tensor
        The tensor only have 1s and 0s, 0 indicates this value is masked
        and 1 indicates the corresponding value is not masked.
    dim: int
        Try to reduce the mask tensor on this dimension.

    Returns
    -------
    indexes: torch.Tensor
        The indexes of the sparsity that can be structurally removed.
    remained_indexes: torch.Tensor
        The indexes of values that need to be remained.
    """
    assert isinstance(t_mask, torch.Tensor)
    shape = list(t_mask.size())
    n_dims = len(shape)
    dim_list = list(range(n_dims))
    # try to reduce the mask from the dim-th dimension
    dim = dim if dim >= 0 else n_dims + dim
    dim_list.remove(dim)

    t_merged = torch.sum(t_mask, dim_list)
    assert t_merged.size(0) == shape[dim]
    all_pruned = t_merged == 0
    need_remain = t_merged != 0
    # return the indexes of the sparsity that can be removed
    indexes = torch.nonzero(all_pruned, as_tuple=True)[0]
    remained_indexes = torch.nonzero(need_remain, as_tuple=True)[0]
    return indexes, remained_indexes

# --- from microsoft__nni::nni/compression/speedup/replacement.py::replace_layernorm ---
def replace_layernorm(layernorm, masks):
    in_masks, _, _ = masks
    assert isinstance(layernorm, nn.LayerNorm)
    if len(in_masks) != 1:
        raise InputsNumberError()
    in_mask = in_masks[0]

    old_normalized_shape = layernorm.normalized_shape
    new_normalized_shape = []
    remained_list = []
    for i in range(-len(old_normalized_shape), 0):
        pruned, remained = convert_to_coarse_mask(in_mask, i)
        new_normalized_shape.append(old_normalized_shape[i] - pruned.size()[i])
        remained_list.append(remained)

    new_layernorm = nn.LayerNorm(tuple(new_normalized_shape), layernorm.eps, layernorm.elementwise_affine)
    _logger.info(f"replace LayerNorm with new normalized_shape: {tuple(new_normalized_shape)}")

    if new_layernorm.elementwise_affine:
        new_layernorm.to(layernorm.weight.device)
        # NOTE: should we keep the weight & bias?
        with torch.no_grad():
            tmp_weight_data = layernorm.weight.data
            tmp_bias_data = layernorm.bias.data
            for i, remained in enumerate(remained_list):
                tmp_weight_data = torch.index_select(tmp_weight_data, i, remained)
                tmp_bias_data = torch.index_select(tmp_bias_data, i, remained)
            new_layernorm.weight.data = tmp_weight_data
            new_layernorm.bias.data = tmp_bias_data
    return new_layernorm

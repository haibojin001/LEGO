# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg152::torch.no_grad+typing.cast
# name: torch_typing_primitive
# summary: Uses torch.no_grad, typing.cast across 2 repos
# anchor_symbols: ['torch.no_grad', 'typing.cast']
# observed in 2 repos: ['allenai__allennlp', 'microsoft__nni']...

# --- from microsoft__nni::nni/nas/oneshot/pytorch/supermodule/_operation_utils.py::zeros_like ---
def zeros_like(arr: T) -> T:
    if isinstance(arr, np.ndarray):
        return np.zeros_like(arr)
    elif isinstance(arr, torch.Tensor):
        return torch.zeros_like(arr)
    else:
        raise TypeError(f'Unsupported type for {arr}: {type(arr)}')

# --- from allenai__allennlp::allennlp/commands/diff.py::_finalize ---
def _finalize(
    history: List[Union[Keep, Insert, Remove]],
    state_dict_a: Dict[str, torch.Tensor],
    state_dict_b: Dict[str, torch.Tensor],
    scale: float,
    threshold: float,
) -> List[Union[Keep, Insert, Remove, Modify]]:
    out = cast(List[Union[Keep, Insert, Remove, Modify]], history)
    for i, step in enumerate(out):
        if isinstance(step, Keep):
            a_tensor = state_dict_a[step.key]
            b_tensor = state_dict_b[step.key]
            with torch.no_grad():
                dist = (scale * torch.nn.functional.mse_loss(a_tensor, b_tensor).sqrt()).item()
            if dist > threshold:
                out[i] = Modify(step.key, step.shape, dist)
    return out

# --- from microsoft__nni::nni/nas/oneshot/pytorch/supermodule/_operation_utils.py::_slice_weight ---
def _slice_weight(weight: T, slice_: multidim_slice | list[tuple[multidim_slice, float]]) -> T:
    # slice_ can be a tuple of slice, e.g., ([3:6], [2:4])
    # or tuple of slice -> float, e.g. {([3:6],): 0.6, ([2:4],): 0.3}

    if isinstance(slice_, list):
        # for weighted case, we get the corresponding masks. e.g.,
        # {([3:6],): 0.6, ([2:4],): 0.3} => [0, 0, 0.3, 0.9, 0.6, 0.6] (if the whole length is 6)
        # this mask is broadcasted and multiplied onto the weight

        masks = []

        # the accepted argument is list of tuple here
        # because slice can't be key of dict
        for sl, wt in slice_:
            # create a mask with weight w
            with torch.no_grad():
                mask = zeros_like(weight)
                mask[_eliminate_list_slice(weight.shape, sl)] = 1  # type: ignore

            # track gradients here
            masks.append(mask * wt)  # type: ignore

        masks = sum(masks)

        return masks * weight  # type: ignore

    else:
        # for unweighted case, we slice it directly.

        def _do_slice(arr, slice_):
            return arr[_eliminate_list_slice(arr.shape, slice_)]  # type: ignore

        # sometimes, we don't need slice.
        # this saves an op on computational graph, which will hopefully make training faster

        # Use a dummy array to check this. Otherwise it would be too complex.
        dummy_arr = np.zeros(weight.shape, dtype=bool)  # type: ignore
        no_effect = cast(Any, _do_slice(dummy_arr, slice_)).shape == dummy_arr.shape

        if no_effect:
            return weight

        return _do_slice(weight, slice_)

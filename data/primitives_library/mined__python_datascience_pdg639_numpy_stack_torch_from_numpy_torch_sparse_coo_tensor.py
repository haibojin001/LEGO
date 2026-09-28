# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg639::numpy.stack+torch.from_numpy+torch.sparse_coo_tensor
# name: numpy_torch_primitive
# summary: Uses numpy.stack, torch.from_numpy, torch.sparse_coo_tensor across 2 repos
# anchor_symbols: ['numpy.stack', 'torch.from_numpy', 'torch.sparse_coo_tensor']
# observed in 2 repos: ['sb-ai-lab__LightAutoML', 'sberbank-ai-lab__LightAutoML']...

# --- from sb-ai-lab__LightAutoML::lightautoml/ml_algo/torch_based/linear_model.py::convert_scipy_sparse_to_torch_float ---
def convert_scipy_sparse_to_torch_float(matrix: sparse.spmatrix) -> torch.Tensor:
    """Convert scipy sparse matrix to torch sparse tensor.

    Args:
        matrix: Matrix to convert.

    Returns:
        Matrix in torch.Tensor format.

    """
    matrix = sparse.coo_matrix(matrix, dtype=np.float32)
    np_idx = np.stack([matrix.row, matrix.col], axis=0).astype(np.int64)
    idx = torch.from_numpy(np_idx)
    values = torch.from_numpy(matrix.data)
    sparse_tensor = torch.sparse_coo_tensor(idx, values, size=matrix.shape)

    return sparse_tensor

# --- from sberbank-ai-lab__LightAutoML::lightautoml/ml_algo/torch_based/linear_model.py::convert_scipy_sparse_to_torch_float ---
def convert_scipy_sparse_to_torch_float(matrix: sparse.spmatrix) -> torch.Tensor:
    """Convert scipy sparse matrix to torch sparse tensor.

    Args:
        matrix: Matrix to convert.

    Returns:
        Matrix in torch.Tensor format.

    """
    matrix = sparse.coo_matrix(matrix, dtype=np.float32)
    np_idx = np.stack([matrix.row, matrix.col], axis=0).astype(np.int64)
    idx = torch.from_numpy(np_idx)
    values = torch.from_numpy(matrix.data)
    sparse_tensor = torch.sparse_coo_tensor(idx, values, size=matrix.shape)

    return sparse_tensor

# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg160::torch.Tensor+torch.prod
# name: torch_primitive
# summary: Uses torch.Tensor, torch.prod across 2 repos
# anchor_symbols: ['torch.Tensor', 'torch.prod']
# observed in 2 repos: ['DeepWisdom__AutoDL', 'microsoft__nni']...

# --- from DeepWisdom__AutoDL::AutoDL_sample_code_submission/Auto_Video/skeleton/nn/modules/profile.py::count_maxpool ---
def count_maxpool(m, x, y):
    kernel_ops = torch.prod(torch.Tensor([m.kernel_size])) - 1
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    return int(total_ops)

# --- from DeepWisdom__AutoDL::AutoDL_sample_code_submission/Auto_Video/skeleton/nn/modules/profile.py::count_avgpool ---
def count_avgpool(m, x, y):
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    return int(total_ops)

# --- from microsoft__nni::nni/compression/utils/counter.py::ModelProfiler._count_adap_avgpool ---
def _count_adap_avgpool(self, m, x, y):
        kernel = torch.Tensor([*(x[0].shape[2:])]) // torch.Tensor(list((m.output_size,))).squeeze()
        total_add = int(torch.prod(kernel))
        total_div = 1
        kernel_ops = total_add + total_div
        num_elements = y[0].numel()
        total_ops = kernel_ops * num_elements

        return self._get_result(m, total_ops)

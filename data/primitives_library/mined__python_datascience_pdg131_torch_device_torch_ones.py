# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg131::torch.device+torch.ones
# name: torch_primitive
# summary: Uses torch.device, torch.ones across 2 repos
# anchor_symbols: ['torch.device', 'torch.ones']
# observed in 2 repos: ['OML-Team__open-metric-learning', 'microsoft__nni']...

# --- from microsoft__nni::test/algo/nas/graph_converter/test_convert_basic.py::TestConvert.test_basic_new_empty ---
def test_basic_new_empty(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                out = x.new_empty((2, 3), dtype=torch.int8, device=torch.device('cpu'))
                return out
        self.checkExportImport(SimpleOp(), (torch.ones(()), ), check_value=False)

# --- from microsoft__nni::test/algo/nas/graph_converter/test_convert_basic.py::TestConvert.test_basic_new_full ---
def test_basic_new_full(self):
        class SimpleOp(nn.Module):
            def forward(self, x):
                # requires_grad is not supported by jit
                # aten::new_full(Tensor self, int[] size, Scalar fill_value, *, int? dtype=None, int? layout=None, Device? device=None, bool? pin_memory=None) -> (Tensor):
                # Keyword argument requires_grad unknown.
                out = x.new_full((3, 4), 3.141592, dtype=torch.float32, device=torch.device('cpu'))
                return out
        self.checkExportImport(SimpleOp(), (torch.ones((2,), dtype=torch.float64), ))

# --- from OML-Team__open-metric-learning::oml/models/utils.py::patch_device_and_float ---
def patch_device_and_float(module: nn.Module, device: Union[str, torch.device] = "cuda") -> None:
    """
    This function is for patching jitted weights with hardcoded ``.to(device)`` and ``.to(dtype)`` operations.
    You may need this if you want to correctly load some jitted model which uses half-precision and(or) which
    device was hardcoded.
    """
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]
    patch_device(module, device_node)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_node = list(float_holder.graph.findNode("aten::to").inputs())[1].node()
        patch_float(module, float_node)

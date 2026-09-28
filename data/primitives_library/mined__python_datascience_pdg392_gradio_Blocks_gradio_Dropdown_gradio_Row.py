# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg392::gradio.Blocks+gradio.Dropdown+gradio.Row
# name: gradio_primitive
# summary: Uses gradio.Blocks, gradio.Dropdown, gradio.Row, gradio.Textbox across 3 repos
# anchor_symbols: ['gradio.Blocks', 'gradio.Dropdown', 'gradio.Row', 'gradio.Textbox']
# observed in 3 repos: ['Lightning-AI__torchmetrics', 'ruc-datalab__DeepAnalyze', 'sematic-ai__sematic']...

# --- from ruc-datalab__DeepAnalyze::deepanalyze/ms-swift/swift/llm/infer/infer_engine/infer_client.py::InferClient.get_model_list_async ---
async def get_model_list_async(self) -> ModelList:
        url = f"{self.base_url.rstrip('/')}/models"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, **self._get_request_kwargs()) as resp:
                resp_obj = await resp.json()
        return from_dict(ModelList, resp_obj)

# --- from ruc-datalab__DeepAnalyze::deepanalyze/ms-swift/swift/llm/infer/protocol.py::ChatCompletionRequest.parse ---
def parse(self) -> Tuple["InferRequest", "RequestConfig"]:
        data = asdict(self)
        res = []
        for cls_type in [InferRequest, RequestConfig]:
            parameters = set(f.name for f in fields(cls_type))
            _data = {k: v for k, v in data.items() if k in parameters}
            res.append(cls_type(**_data))
        return tuple(res)

# --- from Lightning-AI__torchmetrics::tests/unittests/video/test_vmaf.py::_reference_vmaf_no_features ---
def _reference_vmaf_no_features(preds, target) -> Union[dict[str, torch.Tensor], torch.Tensor]:
    """Reference implementation of VMAF metric.

    This should preferably be replaced with the python version of the netflix library
    https://github.com/Netflix/vmaf
    but that requires it to be compiled on the system.

    """
    b = preds.shape[0]
    orig_dtype, device = preds.dtype, preds.device
    preds_luma = calculate_luma(preds)
    target_luma = calculate_luma(target)
    vmaf = VMAF().to(device)

    # we need to compute the model for each video separately
    scores = [
        vmaf.compute_vmaf_score(
            rearrange(target_luma[video], "c f h w -> f c h w"), rearrange(preds_luma[video], "c f h w -> f c h w")
        )
        for video in range(b)
    ]
    return torch.cat(scores, dim=1).t().to(orig_dtype)

# --- from sematic-ai__sematic::sematic/examples/summarization_finetune/interactive.py::launch_summary_app ---
def launch_summary_app(
    model,
    tokenizer,
    model_type,
    max_input_tokens,
    max_new_tokens,
    prompt_format,
):
    """Launch an interactive Gradio app to generate summaries using the given model.

    Parameters
    ----------
    model:
        The model to run inference with.
    tokenizer:
        The tokenizer to use with the model.
    model_type:
        The kind of model (sequence to sequence or causal).
    max_input_tokens:
        The maximum number of tokens that can be used for the input.
    max_new_tokens:
        The maximum number of new tokens generated fot the output.
    prompt_format:
        The format of the prompt.

    Returns
    -------
    A transcript of all the context/summary pairs generated during the app session.
    """
    context = gr.Textbox()
    should_close = False
    history = []

    def close():
        nonlocal should_close
        should_close = True

    with gr.Blocks() as blocks:
        context = gr.Textbox(label="Context")
        summary = gr.Textbox(label="Summary")
        run_button = gr.Button("Run", variant="primary")
        stop_button = gr.Button("Stop", variant="secondary")
        run_button.click(
            lambda ctx: summarize(
                ctx,
                history,
                model,
                tokenizer,
                model_type,
                max_input_tokens,
                max_new_tokens,
                prompt_format,
            ),
            inputs=[context],
            outputs=[summary],
        )
        stop_button.click(lambda: close())

    blocks.launch(server_name="0.0.0.0", inbrowser=True, prevent_thread_lock=True)
    while not should_close:
        time.sleep(1)
    blocks.close()

    return history

# --- from Lightning-AI__torchmetrics::src/torchmetrics/functional/video/vmaf.py::video_multi_method_assessment_fusion ---
def video_multi_method_assessment_fusion(
    preds: Tensor,
    target: Tensor,
    features: bool = False,
) -> Union[Tensor, Dict[str, Tensor]]:
    """Calculates Video Multi-Method Assessment Fusion (VMAF) metric.

    VMAF is a full-reference video quality assessment algorithm that combines multiple quality assessment features
    such as detail loss, motion, and contrast using a machine learning model to predict human perception of video
    quality more accurately than traditional metrics like PSNR or SSIM.

    The metric works by:

       1. Converting input videos to luma component (grayscale)
       2. Computing multiple elementary features:
          - Additive Detail Measure (ADM): Evaluates detail preservation at different scales
          - Visual Information Fidelity (VIF): Measures preservation of visual information across frequency bands
          - Motion: Quantifies the amount of motion in the video
       3. Combining these features using a trained SVM model to predict quality

    .. note::
        This implementation requires you to have vmaf-torch installed: https://github.com/alvitrioliks/VMAF-torch.
        Install either by cloning the repository and running `pip install .` or with `pip install torchmetrics[video]`.

    Args:
        preds: Video tensor of shape (batch, channels, frames, height, width). Expected to be in RGB format
            with values in range [0, 1].
        target: Video tensor of shape (batch, channels, frames, height, width). Expected to be in RGB format
            with values in range [0, 1].
        features: If True, all the elementary features (ADM, VIF, motion) are returned along with the VMAF score in
            a dictionary. This corresponds to the output you would get from the VMAF command line tool with the `--csv`
            option enabled. If False, only the VMAF score is returned as a tensor.

    Returns:
        - If `features` is False, returns a tensor with shape (batch, frame) of VMAF score for each frame in
          each video. Higher scores indicate better quality, with typical values ranging from 0 to 100.

        - If `features` is True, returns a dictionary where each value is a (batch, frame) tensor of the
          corresponding feature. The keys are:

            - 'integer_motion2': Integer motion feature
            - 'integer_motion': Integer motion feature
            - 'integer_adm2': Integer ADM feature
            - 'integer_adm_scale0': Integer ADM feature at scale 0
            - 'integer_adm_scale1': Integer ADM feature at scale 1
            - 'integer_adm_scale2': Integer ADM feature at scale 2
            - 'integer_adm_scale3': Integer ADM feature at scale 3
            - 'integer_vif_scale0': Integer VIF feature at scale 0
            - 'integer_vif_scale1': Integer VIF feature at scale 1
            - 'integer_vif_scale2': Integer VIF feature at scale 2
            - 'integer_vif_scale3': Integer VIF feature at scale 3
            - 'vmaf': VMAF score for each frame in each video

    Example:
        >>> import torch
        >>> from torchmetrics.functional.video import video_multi_method_assessment_fusion
        >>> # 2 videos, 3 channels, 10 frames, 32x32 resolution
        >>> preds = torch.rand(2, 3, 10, 32, 32, generator=torch.manual_seed(42))
        >>> target = torch.rand(2, 3, 10, 32, 32, generator=torch.manual_seed(43))
        >>> vmaf_score = video_multi_method_assessment_fusion(preds, target)
        >>> torch.round(vmaf_score, decimals=2)
        tensor([[ 9.9900, 15.9000, 14.2600, 16.6100, 15.9100, 14.3000, 13.5800, 13.4900, 15.4700, 20.2800],
                [ 6.2500, 11.3000, 17.3000, 11.4600, 19.0600, 14.9300, 14.0500, 14.4100, 12.4700, 14.8200]])
        >>> vmaf_dict = video_multi_method_assessment_fusion(preds, target, features=True)
        >>> # show a couple of features, more features are available
        >>> vmaf_dict['vmaf'].round(decimals=2)
        tensor([[ 9.9900, 15.9000, 14.2600, 16.6100, 15.9100, 14.3000, 13.5800, 13.4900, 15.4700, 20.2800],
                [ 6.2500, 11.3000, 17.3000, 11.4600, 19.0600, 14.9300, 14.0500, 14.4100, 12.4700, 14.8200]])
        >>> vmaf_dict['integer_adm2'].round(decimals=2)
        tensor([[0.4500, 0.4500, 0.3600, 0.4700, 0.4300, 0.3600, 0.3900, 0.4100, 0.3700, 0.4700],
                [0.4200, 0.3900, 0.4400, 0.3700, 0.4500, 0.3900, 0.3800, 0.4800, 0.3900, 0.3900]])

    """
    if not _TORCH_VMAF_AVAILABLE:
        raise RuntimeError("vmaf-torch is not installed. Please install with `pip install torchmetrics[video]`.")
    b = preds.shape[0]
    orig_dtype, device = preds.dtype, preds.device
    preds_luma = calculate_luma(preds)
    target_luma = calculate_luma(target)

    vmaf = VMAF().to(device)

    # we need to compute the model for each video separately
    if not features:
        scores = [
            vmaf.compute_vmaf_score(
                rearrange(target_luma[video], "c f h w -> f c h w"), rearrange(preds_luma[video], "c f h w -> f c h w")
            )
            for video in range(b)
        ]
        return torch.cat(scores, dim=1).t().to(orig_dtype)

    scores_and_features = [
        vmaf.table(
            rearrange(target_luma[video], "c f h w -> f c h w"), rearrange(preds_luma[video], "c f h w -> f c h w")
        )
        for video in range(b)
    ]
    dfs = [scores_and_features[video].apply(pd.to_numeric, errors="coerce") for video in range(b)]
    result = [
        {col: torch.tensor(dfs[video][col].values, dtype=orig_dtype) for col in dfs[video].columns if col != "Frame"}
        for video in range(b)
    ]
    return {col: torch.stack([result[video][col] for video in range(b)]) for col in result[0]}

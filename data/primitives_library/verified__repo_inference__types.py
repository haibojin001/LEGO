from typing import Any, Callable, Dict, ForwardRef, Iterable, List, Optional, Union

from typing_extensions import Literal, NotRequired, TypedDict

from ._compat import (
    BaseModel,
    create_model,
    create_model_from_typeddict,
    validate_arguments,
)
from .fields import (
    echo_field,
    max_tokens_field,
    none_field,
    repeat_penalty_field,
    stop_field,
    stream_field,
    stream_interval_field,
    stream_option_field,
    temperature_field,
    top_k_field,
    top_p_field,
)


class Image(TypedDict):
    url: Optional[str]
    b64_json: Optional[str]


class ImageList(TypedDict):
    created: int
    data: List[Image]


class ImageEditRequest(TypedDict, total=False):
    image: Union[Union[str, bytes], List[Union[str, bytes]]]
    mask: Optional[Union[str, bytes]]
    prompt: str
    n: int
    size: Optional[str]
    response_format: str


class SDAPIResult(TypedDict):
    images: List[str]
    parameters: dict
    info: dict


class Video(TypedDict):
    url: Optional[str]
    b64_json: Optional[str]


class VideoList(TypedDict):
    created: int
    data: List[Video]


class EmbeddingUsage(TypedDict):
    prompt_tokens: int
    total_tokens: int


class EmbeddingData(TypedDict):
    index: int
    object: str
    embedding: Union[List[float], Dict[str, float]]


class Embedding(TypedDict):
    object: Literal["list"]
    model: str
    model_replica: str
    data: List[EmbeddingData]
    usage: EmbeddingUsage


class AudioEmbedding(TypedDict):
    object: Literal["embedding"]
    model: str
    dimensions: int
    embedding: List[float]


class Document(TypedDict):
    text: str


class DocumentObj(TypedDict):
    index: int
    relevance_score: float
    document: Optional[Document]


class ApiVersion(TypedDict):
    version: str
    is_deprecated: bool
    is_experimental: bool


class BilledUnit(TypedDict):
    input_tokens: int
    output_tokens: int
    search_units: int
    classifications: int


class RerankTokens(TypedDict):
    input_tokens: int
    output_tokens: int


class Meta(TypedDict):
    api_version: Optional[ApiVersion]
    billed_units: Optional[BilledUnit]
    tokens: RerankTokens
    warnings: Optional[List[str]]


class Rerank(TypedDict):
    id: str
    results: List[DocumentObj]
    meta: Meta


class CompletionLogprobs(TypedDict):
    text_offset: List[int]
    token_logprobs: List[Optional[float]]
    tokens: List[str]
    top_logprobs: List[Optional[Dict[str, float]]]


class ChatCompletionTopLogprob(TypedDict):
    token: str
    bytes: Optional[List[int]]
    logprob: float


class ChatCompletionLogprob(TypedDict):
    token: str
    bytes: Optional[List[int]]
    logprob: float
    top_logprobs: List[ChatCompletionTopLogprob]


class ChatCompletionLogprobs(TypedDict):
    content: Optional[List[ChatCompletionLogprob]]


class ToolCallFunction(TypedDict):
    name: str
    arguments: str


class ToolCalls(TypedDict):
    id: str
    type: Literal["function"]
    function: ToolCallFunction


class ToolCallDeltaFunction(TypedDict, total=False):
    name: str
    arguments: str


class ToolCallDelta(TypedDict):
    index: int
    id: NotRequired[str]
    type: NotRequired[Literal["function"]]
    function: NotRequired[ToolCallDeltaFunction]


class CompletionChoice(TypedDict):
    text: NotRequired[str]
    index: int
    logprobs: Optional[CompletionLogprobs]
    finish_reason: Optional[str]
    tool_calls: NotRequired[List[ToolCalls]]


class PromptTokensDetails(TypedDict):
    cached_tokens: int


class CompletionUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: NotRequired[PromptTokensDetails]


class CompletionChunk(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: NotRequired[CompletionUsage]


class Completion(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


class ChatCompletionAudio(TypedDict):
    id: str
    data: str
    expires_at: int
    transcript: str


class ChatCompletionMessage(TypedDict):
    role: str
    reasoning_content: NotRequired[str]
    content: Optional[str]
    audio: NotRequired[ChatCompletionAudio]
    user: NotRequired[str]
    tool_calls: NotRequired[List]


class ChatCompletionChoice(TypedDict):
    index: int
    message: ChatCompletionMessage
    logprobs: NotRequired[Optional[ChatCompletionLogprobs]]
    finish_reason: Optional[str]


class ChatCompletion(TypedDict):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: CompletionUsage


class ChatCompletionChunkDelta(TypedDict):
    role: NotRequired[str]
    reasoning_content: NotRequired[Union[str, None]]
    content: NotRequired[Union[str, None]]
    tool_calls: NotRequired[List[ToolCallDelta]]


class ChatCompletionChunkChoice(TypedDict):
    index: int
    delta: ChatCompletionChunkDelta
    logprobs: NotRequired[Optional[ChatCompletionLogprobs]]
    finish_reason: Optional[str]


class ChatCompletionChunk(TypedDict):
    id: str
    model: str
    object: Literal["chat.completion.chunk"]
    created: int
    choices: List[ChatCompletionChunkChoice]
    usage: NotRequired[CompletionUsage]


StoppingCriteria = Callable[[List[int], List[float]], bool]


class StoppingCriteriaList(List[StoppingCriteria]):
    def __call__(self, input_ids: List[int], logits: List[float]) -> bool:
        return any(criteria(input_ids, logits) for criteria in self)


LogitsProcessor = Callable[[List[int], List[float]], List[float]]


class LogitsProcessorList(List[LogitsProcessor]):
    def __call__(self, input_ids: List[int], scores: List[float]) -> List[float]:
        for processor in self:
            scores = processor(input_ids, scores)
        return scores


class PytorchGenerateConfig(TypedDict, total=False):
    temperature: float
    repetition_penalty: float
    top_p: float
    top_k: int
    stream: bool
    max_tokens: int
    echo: bool
    stop: Optional[Union[str, List[str]]]
    stop_token_ids: Optional[Union[int, List[int]]]
    stream_interval: int
    model: Optional[str]
    tools: Optional[List[Dict]]
    lora_name: Optional[str]
    stream_options: Optional[Union[dict, None]]
    request_id: Optional[str]


class CogagentGenerateConfig(PytorchGenerateConfig, total=False):
    platform: Optional[Literal["Mac", "WIN", "Mobile"]]
    format: Optional[
        Literal[
            "(Answer in Action-Operation-Sensitive format.)",
            "(Answer in Status-Plan-Action-Operation format.)",
            "(Answer in Status-Action-Operation-Sensitive format.)",
            "(Answer in Status-Action-Operation format.)",
            "(Answer in Action-Operation format.)",
        ]
    ]


class PytorchModelConfig(TypedDict, total=False):
    revision: Optional[str]
    device: str
    gpus: Optional[str]
    num_gpus: int
    max_gpu_memory: str
    gptq_ckpt: Optional[str]
    gptq_wbits: int
    gptq_groupsize: int
    gptq_act_order: bool
    trust_remote_code: bool
    revision_type: Optional[str]
    lora_model: Optional[str]
    lora_model_path: Optional[str]
    lora_model_name: Optional[str]
    kv_cache_dtype: Optional[str]
    max_num_seqs: Optional[int]
    max_model_len: Optional[int]
    max_batch_size: Optional[int]
    max_seq_len: Optional[int]
    dtype: Optional[str]
    quantization: Optional[str]
    gpu_memory_utilization: Optional[float]
    swap_space: Optional[int]
    enforce_eager: Optional[bool]
    tensor_parallel_size: Optional[int]
    pipeline_parallel_size: Optional[int]
    block_size: Optional[int]
    seed: Optional[int]
    download_dir: Optional[str]
    tokenizer: Optional[str]
    tokenizer_mode: Optional[str]
    tokenizer_revision: Optional[str]
    limit_mm_per_prompt: Optional[Dict[str, int]]
    enable_prefix_caching: Optional[bool]
    disable_log_stats: Optional[bool]
    disable_log_requests: Optional[bool]
    enable_chunked_prefill: Optional[bool]
    served_model_name: Optional[Union[str, List[str]]]
    chat_template: Optional[str]
    response_role: Optional[str]
    guided_decoding_backend: Optional[str]
    enable_lora: Optional[bool]
    max_loras: Optional[int]
    max_lora_rank: Optional[int]
    enable_prompt_adapter: Optional[bool]
    max_prompt_adapters: Optional[int]
    max_prompt_adapter_token: Optional[int]
    speculative_model: Optional[str]
    num_speculative_tokens: Optional[int]
    use_v2_block_manager: Optional[bool]
    rope_scaling: Optional[Dict[str, Any]]
    rope_theta: Optional[float]
    worker_use_ray: Optional[bool]
    engine_use_ray: Optional[bool]
    disable_custom_all_reduce: Optional[bool]
    max_logprobs: Optional[int]
    max_num_batched_tokens: Optional[int]
    enable_auto_tool_choice: Optional[bool]
    tool_call_parser: Optional[str]
    reasoning_parser: Optional[str]
    extra: Optional[Dict[str, Any]]
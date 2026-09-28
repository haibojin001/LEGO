from typing import Optional

from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizer, ProcessorMixin


def get_tokenizer(
    model_path: str,
    override_chat_template: Optional[str] = None,
    **kwargs,
) -> PreTrainedTokenizer:
    """Load and configure a Hugging Face tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)

    if override_chat_template is not None:
        with open(override_chat_template) as template_file:
            tokenizer.chat_template = template_file.read()
        print(f"New chat template: {tokenizer.chat_template}")

    if tokenizer.bos_token == "<bos>" and tokenizer.eos_token == "<eos>":
        print("Found gemma model. Set eos_token and eos_token_id to <end_of_turn> and 107.")
        tokenizer.eos_token = "<end_of_turn>"

    if tokenizer.pad_token_id is None:
        print("Pad token is None. Set it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_processor(
    model_path: str,
    override_chat_template: Optional[str] = None,
    **kwargs,
) -> Optional[ProcessorMixin]:
    """Load a Hugging Face processor when the resolved object is a processor."""
    processor = AutoProcessor.from_pretrained(model_path, **kwargs)

    if override_chat_template is not None:
        with open(override_chat_template) as template_file:
            processor.chat_template = template_file.read()
        print(f"New chat template: {processor.chat_template}")

    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return processor
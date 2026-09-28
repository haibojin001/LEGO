from threading import Lock

from tiktoken import get_encoding


class TokenEncoder:
    _encoder_instance = None
    _model = None
    _lock = Lock()

    @classmethod
    def get_token_encoder(cls):
        if cls._encoder_instance is None:
            with cls._lock:
                if cls._encoder_instance is None:
                    cls._encoder_instance = get_encoding("o200k_base")
        return cls._encoder_instance


class TokenHandler:
    def __init__(self):
        self.encoder = TokenEncoder.get_token_encoder()

    def count_tokens(self, patch: str) -> int:
        return len(self.encoder.encode(patch))


def clip_tokens(
    text: str,
    max_tokens: int,
    add_three_dots=True,
    num_input_tokens=None,
    delete_last_line=False,
) -> str:
    if not text:
        return text

    try:
        if num_input_tokens is None:
            encoding = TokenEncoder.get_token_encoder()
            num_input_tokens = len(encoding.encode(text))

        if num_input_tokens <= max_tokens:
            return text

        if max_tokens <= 0:
            return ""

        character_ratio = len(text) / num_input_tokens
        character_limit = int(0.9 * character_ratio * max_tokens)

        if character_limit > 0:
            result = text[:character_limit]
            if delete_last_line:
                result = result.rsplit("\n", 1)[0]
            if add_three_dots:
                result += "\n...(truncated)"
            return result

        return ""
    except Exception as error:
        print(f"Failed to clip tokens: {error}")
        return text
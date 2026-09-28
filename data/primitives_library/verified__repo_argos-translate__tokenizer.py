from pathlib import Path
from typing import List

import sentencepiece as spm


class Tokenizer:
    def encode(self, sentence: str) -> List[str]:
        raise NotImplementedError()

    def decode(self, tokens: List[str]) -> str:
        raise NotImplementedError()


class SentencePieceTokenizer(Tokenizer):
    def __init__(self, model_file: Path):
        self.model_file = model_file
        self.processor = None

    def lazy_processor(self) -> spm.SentencePieceProcessor:
        if self.processor is None:
            self.processor = spm.SentencePieceProcessor(
                model_file=str(self.model_file)
            )
        return self.processor

    def encode(self, sentence: str) -> List[str]:
        return self.lazy_processor().encode(sentence, out_type=str)

    def decode(self, tokens: List[str]) -> str:
        text = self.lazy_processor().decode_pieces(tokens)
        return text.replace("▁", " ").replace("_", " ")


class BPETokenizer(Tokenizer):
    def __init__(self, model_file: Path, from_code: str, to_code: str):
        self.model_file = model_file
        self.from_code = from_code
        self.to_code = to_code
        self.tokenizer = None
        self.detokenizer = None
        self.bpe_source = None
        self.normalizer = None

    def lazy_load(self):
        if self.tokenizer is None:
            from sacremoses.normalize import MosesPunctNormalizer
            from sacremoses.tokenize import MosesDetokenizer, MosesTokenizer
            from argostranslate.apply_bpe import BPE

            self.tokenizer = MosesTokenizer(self.from_code)
            self.detokenizer = MosesDetokenizer(self.to_code)
            self.normalizer = MosesPunctNormalizer(self.from_code)

            with open(str(self.model_file), "r", encoding="utf-8") as model:
                self.bpe_source = BPE(model)

    def encode(self, sentence: str) -> List[str]:
        self.lazy_load()
        normalized = self.normalizer.normalize(sentence)
        tokenized = " ".join(self.tokenizer.tokenize(normalized))
        words = tokenized.strip("\r\n ").split(" ")
        return self.bpe_source.segment_tokens(words)

    def decode(self, tokens: List[str]) -> str:
        self.lazy_load()
        text = " ".join(tokens).replace("@@ ", "")
        return self.detokenizer.detokenize(text.split(" "))
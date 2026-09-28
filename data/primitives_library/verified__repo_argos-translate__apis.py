from __future__ import annotations

import json
import sys
from urllib import parse, request

from argostranslate.models import ILanguageModel


class LibreTranslateAPI:
    """Client for a LibreTranslate-compatible server."""

    DEFAULT_URL = "https://translate.argosopentech.com/"

    def __init__(self, url: str = None, api_key: str = None):
        self.url = self.DEFAULT_URL if url is None else url
        self.api_key = api_key

        assert len(self.url) > 0
        if not self.url.endswith("/"):
            self.url += "/"

    def translate(self, q: str, source: str = "en", target: str = "es") -> str:
        data = {"q": q, "source": source, "target": target}
        if self.api_key is not None:
            data["api_key"] = self.api_key

        payload = parse.urlencode(data).encode()
        req = request.Request(self.url + "translate", data=payload)
        result = request.urlopen(req).read().decode()
        return json.loads(result)["translatedText"]

    def languages(self):
        data = {}
        if self.api_key is not None:
            data["api_key"] = self.api_key

        payload = parse.urlencode(data).encode()
        req = request.Request(self.url + "languages", data=payload)
        result = request.urlopen(req).read().decode()
        return json.loads(result)

    def detect(self, q: str):
        data = {"q": q}
        if self.api_key is not None:
            data["api_key"] = self.api_key

        payload = parse.urlencode(data).encode()
        req = request.Request(self.url + "detect", data=payload)
        result = request.urlopen(req).read().decode()
        return json.loads(result)


class OpenAIAPI(ILanguageModel):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def infer(self, prompt: str) -> str | None:
        payload = json.dumps({"prompt": prompt, "max_tokens": 100}).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
        }
        req = request.Request(
            "https://api.openai.com/v1/engines/davinci/completions",
            data=payload,
            headers=headers,
            method="POST",
        )

        try:
            response = request.urlopen(req)
        except Exception as e:
            print(e, sys.stderr)
            return None

        try:
            response_text = response.read().decode()
        except Exception as e:
            print(e, sys.stderr)
            return None

        return json.loads(response_text)["choices"][0]["text"]
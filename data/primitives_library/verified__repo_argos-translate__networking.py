from __future__ import annotations

import random
import urllib.request
from os import makedirs
from pathlib import Path

from argostranslate import settings
from argostranslate.utils import error, info

USER_AGENT = "ArgosTranslate"


def get_protocol(url: str) -> str | None:
    separator_position = url.find(":")
    if separator_position > 0:
        return url[:separator_position]
    return None


supported_protocols = {"http", "https"}


def get(url: str, retry_count: int = 3) -> bytes | None:
    if get_protocol(url) not in supported_protocols:
        return None

    info(f"Get {url}")
    attempts = 0

    while attempts <= retry_count:
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT},
            )
            response = urllib.request.urlopen(request)
            result = response.read()
            info(f"Got {url}")
            return result
        except Exception as exception:
            attempts += 1
            error(exception)

    return None


def get_from(urls: list[str], retry_count: int = 3) -> bytes | None:
    for url in random.sample(urls, len(urls)):
        result = get(url, retry_count)
        if result is not None:
            return result
    return None


def cache_spacy() -> Path | None:
    from spacy import load as spacy_load
    from spacy.cli import download as spacy_download

    cache_path = Path(settings.cache_dir / "spacy")
    makedirs(cache_path, exist_ok=True)

    info("Looking for cached Spacy xx_sent_ud_sm.")
    model_path = Path(cache_path / "senter" / "model")

    if model_path.exists():
        return cache_path

    try:
        info("Downloading Spacy xx_sent_ud_sm.")
        spacy_download("xx_sent_ud_sm")
        pipeline = spacy_load("xx_sent_ud_sm", exclude=["parser"])
        pipeline.to_disk(cache_path)
        info("Spacy xx_sent_ud_sm successfully cached.")
        return cache_path
    except Exception as exception:
        error(str(exception))
        return None
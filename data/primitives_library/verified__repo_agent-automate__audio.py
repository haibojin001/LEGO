from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .store import Database, get_db

log = logging.getLogger("automate.audio")

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]{3,}|[一-龥]{2,4}")

_STOP = {
    "the",
    "and",
    "for",
    "you",
    "this",
    "that",
    "with",
    "are",
    "was",
    "but",
    "have",
    "has",
    "from",
    "they",
    "will",
    "your",
    "what",
    "when",
    "how",
    "我们",
    "他们",
    "你们",
    "什么",
    "为什么",
    "怎么",
    "因为",
    "所以",
    "可以",
    "现在",
    "时候",
    "知道",
    "觉得",
    "可能",
    "需要",
    "已经",
}


def mine_vocabulary(db: Database, *, top_n: int = 50) -> list[str]:
    documents: list[str] = []

    try:
        notes = db.fetchall(
            "SELECT title, body FROM notes ORDER BY updated_at DESC LIMIT 200"
        )
        documents.extend(f"{row['title']} {row['body']}" for row in notes)
    except Exception:
        pass

    try:
        runs = db.fetchall(
            "SELECT result FROM runs WHERE status='done' ORDER BY started_at DESC LIMIT 200"
        )
        documents.extend(row["result"] or "" for row in runs)
    except Exception:
        pass

    if not documents:
        return []

    document_frequency: Counter[str] = Counter()
    total_frequency: Counter[str] = Counter()

    for document in documents:
        tokens = _TOKEN_RE.findall(document)
        useful = [token for token in tokens if token.lower() not in _STOP]

        for token in set(useful):
            document_frequency[token] += 1

        total_frequency.update(useful)

    total_documents = len(documents)
    ranking: list[tuple[float, str]] = []

    for token, frequency in total_frequency.items():
        if frequency < 2:
            continue

        inverse_frequency = max(
            0.5,
            _log(total_documents / (1 + document_frequency[token])),
        )
        capitalization_bonus = (
            1.5
            if token[0].isupper() and token[0].isalpha()
            else 1.0
        )
        ranking.append(
            (frequency * inverse_frequency * capitalization_bonus, token)
        )

    ranking.sort(reverse=True)
    return [token for _, token in ranking[:top_n]]


def _log(x: float) -> float:
    import math

    return math.log(max(x, 1e-9))


@dataclass
class Provider:
    id: str
    label: str


def pick_provider(db: Database, prefer: str | None = None) -> Provider | None:
    if prefer:
        if prefer == "tencent_asr" and _tencent_connected(db):
            return Provider(id="tencent_asr", label="Tencent ASR")
        if prefer == "openai_whisper" and _openai_key(db):
            return Provider(id="openai_whisper", label="OpenAI Whisper")

    if _tencent_connected(db):
        return Provider(id="tencent_asr", label="Tencent ASR")

    if _openai_key(db):
        return Provider(id="openai_whisper", label="OpenAI Whisper")

    return None


def _tencent_connected(db: Database) -> bool:
    connection = db.get_connection("tencent_asr", decrypt=True)
    return bool(
        connection
        and connection.get("status") == "connected"
        and connection.get("secret_id")
    )


def _openai_key(db: Database) -> str | None:
    for provider in db.list_providers():
        provider_id = provider.get("id", "")
        if not provider_id.startswith("openai"):
            continue
        if not provider.get("api_key_set"):
            continue

        configured = db.get_provider(provider_id, decrypt=True)
        if configured and configured.get("api_key"):
            return configured["api_key"]

    return None


def transcribe(
    *,
    audio_path: Path,
    db: Database | None = None,
    provider: str = "",
    language: str = "",
    with_vocab: bool = True,
) -> dict:
    database = db or get_db()
    chosen = pick_provider(database, prefer=provider or None)

    if not chosen:
        raise RuntimeError(
            "no transcription provider configured — connect Tencent ASR "
            "(Settings → Integrations) or set an OpenAI api_key"
        )

    vocabulary = mine_vocabulary(database) if with_vocab else []
    started_at = time.time()

    if chosen.id == "tencent_asr":
        transcript = _transcribe_tencent(
            audio_path,
            database,
            vocab=vocabulary,
            language=language,
        )
    elif chosen.id == "openai_whisper":
        transcript = _transcribe_openai(
            audio_path,
            database,
            vocab=vocabulary,
            language=language,
        )
    else:
        raise RuntimeError(f"unknown provider: {chosen.id}")

    return {
        "provider": chosen.id,
        "text": transcript,
        "ms": int((time.time() - started_at) * 1000),
        "vocabulary_used": vocabulary[:20],
    }


def _transcribe_openai(
    path: Path,
    db: Database,
    *,
    vocab: list[str],
    language: str,
) -> str:
    api_key = _openai_key(db)
    if not api_key:
        raise RuntimeError("OpenAI api_key not configured")

    import urllib.request

    boundary = "----automateAudioBoundary"
    chunks: list[bytes] = []

    def field(name: str, value: str) -> None:
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        )
        chunks.append(value.encode() + b"\r\n")

    field("model", "whisper-1")

    if language:
        field("language", language)

    if vocab:
        field("prompt", ", ".join(vocab[:50]))

    chunks.append(f"--{boundary}\r\n".encode())
    chunks.append(
        (
            f'Content-Disposition: form-data; name="file"; '
            f'filename="{path.name}"\r\n'
            "Content-Type: audio/mpeg\r\n\r\n"
        ).encode()
    )
    chunks.append(path.read_bytes())
    chunks.append(f"\r\n--{boundary}--\r\n".encode())

    request = urllib.request.Request(
        "https://api.openai.com/v1/audio/transcriptions",
        data=b"".join(chunks),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read().decode())

    return payload.get("text", "").strip()


def _transcribe_tencent(
    path: Path,
    db: Database,
    *,
    vocab: list[str],
    language: str,
) -> str:
    connection = db.get_connection("tencent_asr", decrypt=True)
    if (
        not connection
        or not connection.get("secret_id")
        or not connection.get("secret_key")
    ):
        raise RuntimeError("Tencent ASR connection not configured")

    try:
        from tencentcloud.common import credential
        from tencentcloud.asr.v20190614 import asr_client, models
    except ImportError as exc:
        raise RuntimeError(
            "tencentcloud-sdk-python not installed — pip install "
            "tencentcloud-sdk-python-asr"
        ) from exc

    region = connection.get("region") or os.environ.get(
        "TENCENT_ASR_REGION",
        "ap-guangzhou",
    )
    credentials = credential.Credential(
        connection["secret_id"],
        connection["secret_key"],
    )
    client = asr_client.AsrClient(credentials, region)

    audio = path.read_bytes()
    suffix = path.suffix.lower().lstrip(".")
    voice_format = suffix if suffix in {
        "mp3",
        "wav",
        "pcm",
        "m4a",
        "aac",
        "amr",
        "flac",
        "opus",
        "silk",
    } else "mp3"

    normalized_language = language.lower().replace("_", "-")
    engine = "16k_zh"
    if normalized_language.startswith("en"):
        engine = "16k_en"
    elif normalized_language.startswith("ja"):
        engine = "16k_ja"
    elif normalized_language.startswith("ko"):
        engine = "16k_ko"
    elif normalized_language.startswith(("yue", "zh-yue")):
        engine = "16k_yue"

    params: dict[str, object] = {
        "EngSerViceType": engine,
        "SourceType": 1,
        "VoiceFormat": voice_format,
        "Data": base64.b64encode(audio).decode(),
        "DataLen": len(audio),
    }

    if vocab:
        params["HotwordList"] = ",".join(
            f"{word}|5" for word in vocab[:50]
        )

    request = models.SentenceRecognitionRequest()
    request.from_json_string(json.dumps(params))
    response = client.SentenceRecognition(request)
    return (getattr(response, "Result", "") or "").strip()
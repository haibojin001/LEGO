from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from promptify.core.config import CacheConfig


class PromptCache:
    """An in-memory LRU cache keyed by prompt request details."""

    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        self.config = config if config is not None else CacheConfig()
        self._cache: OrderedDict[str, Any] = OrderedDict()

    @staticmethod
    def _make_key(messages: List[Dict[str, str]], model: str, **kwargs: Any) -> str:
        payload = {"messages": messages, "model": model, **kwargs}
        serialized = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def get(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs: Any,
    ) -> Optional[Any]:
        if not self.config.enabled:
            return None

        key = self._make_key(messages, model, **kwargs)
        if key not in self._cache:
            return None

        self._cache.move_to_end(key)
        return self._cache[key]

    def put(
        self,
        messages: List[Dict[str, str]],
        model: str,
        value: Any,
        **kwargs: Any,
    ) -> None:
        if not self.config.enabled:
            return

        key = self._make_key(messages, model, **kwargs)
        self._cache[key] = value
        self._cache.move_to_end(key)

        while len(self._cache) > self.config.maxsize:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()
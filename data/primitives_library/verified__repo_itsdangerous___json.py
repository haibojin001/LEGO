from __future__ import annotations

import json as _json
import typing as t


class _CompactJSON:
    """Small adapter for JSON with compact output defaults."""

    @staticmethod
    def loads(payload: str | bytes) -> t.Any:
        return _json.loads(payload)

    @staticmethod
    def dumps(obj: t.Any, **kwargs: t.Any) -> str:
        if "ensure_ascii" not in kwargs:
            kwargs["ensure_ascii"] = False

        if "separators" not in kwargs:
            kwargs["separators"] = (",", ":")

        return _json.dumps(obj, **kwargs)
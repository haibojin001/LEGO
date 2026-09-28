import json
import re
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from finance.data.providers import ProviderError


@dataclass(frozen=True)
class TradingView:
    timeout: float = 20

    def recommendations(self, symbols: list[str], interval: str = "1d") -> pd.DataFrame:
        """Return current TradingView scanner recommendations for qualified symbols."""
        intervals = {
            "1m": "|1",
            "5m": "|5",
            "15m": "|15",
            "1h": "|60",
            "4h": "|240",
            "1d": "",
            "1w": "|1W",
            "1mo": "|1M",
        }

        if (
            interval not in intervals
            or not symbols
            or len(symbols) > 100
            or len(set(symbols)) != len(symbols)
        ):
            raise ValueError("supported interval and 1–100 distinct symbols required")

        if self.timeout <= 0 or any(
            re.fullmatch(r"[A-Z0-9_]+:[A-Z0-9_.!-]+", symbol) is None
            for symbol in symbols
        ):
            raise ValueError("positive timeout and EXCHANGE:SYMBOL identifiers required")

        ending = intervals[interval]
        fields = [
            f"{field}{ending}"
            for field in (
                "close",
                "Recommend.All",
                "Recommend.MA",
                "Recommend.Other",
            )
        ]
        body = {
            "symbols": {"tickers": symbols, "query": {"types": []}},
            "columns": fields,
            "range": [0, len(symbols)],
        }
        request = Request(
            "https://scanner.tradingview.com/america/scan",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0",
            },
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_data = json.load(response)

            names = ("price", "overall", "moving_averages", "oscillators")
            records = []
            for entry in response_data["data"]:
                values = dict(zip(names, entry["d"], strict=True))
                records.append({"symbol": entry["s"], **values})

            frame = pd.DataFrame(records).set_index("symbol").reindex(symbols)

            invalid = (
                not np.isfinite(frame).all().all()
                or (frame.price <= 0).any()
                or (frame.drop(columns="price").abs() > 1).any().any()
            )
            if invalid:
                raise ValueError("missing symbols or invalid recommendations")

            frame.attrs.update(
                provider="TradingView public scanner",
                interval=interval,
                retrieved_at=pd.Timestamp.now(tz="UTC").isoformat(),
                historical=False,
            )
            return frame
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError, TypeError) as error:
            raise ProviderError(f"TradingView scanner failed: {error}") from error
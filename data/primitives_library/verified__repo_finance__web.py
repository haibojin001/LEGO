import json
import math
import time
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from finance.data.providers import ProviderError


@dataclass
class PublicWeb:
    timeout: float = 20
    interval: float = 1
    cache_seconds: float = 300
    _cache: dict = field(default_factory=dict, init=False, repr=False)
    _last_request: float = field(default=0, init=False, repr=False)

    def text(self, url: str) -> str:
        values = (self.timeout, self.interval, self.cache_seconds)
        if (
            not all(math.isfinite(value) for value in values)
            or self.timeout <= 0
            or self.interval < 0
            or self.cache_seconds < 0
        ):
            raise ValueError("positive timeout and nonnegative interval/cache required")

        parsed_url = urlparse(url)
        if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
            raise ValueError("public data requires an HTTP(S) URL")

        entry = self._cache.get(url)
        if entry and time.monotonic() - entry[0] < self.cache_seconds:
            return entry[1]

        for attempt in range(3):
            elapsed = time.monotonic() - self._last_request
            time.sleep(max(0, self.interval - elapsed))
            self._last_request = time.monotonic()

            request = Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; FinanceToolkit research)",
                    "Accept": "text/html,application/json,application/xml",
                },
            )

            try:
                with urlopen(request, timeout=self.timeout) as response:
                    result = response.read().decode("utf-8")
                self._cache[url] = (time.monotonic(), result)
                return result
            except HTTPError as exc:
                retryable = exc.code in (429, 500, 502, 503, 504)
                if not retryable or attempt == 2:
                    raise ProviderError(f"HTTP {exc.code} from {url}") from exc
                time.sleep(min(10, 2 ** (attempt + 1)))
            except (URLError, TimeoutError, UnicodeError) as exc:
                raise ProviderError(f"Unable to retrieve {url}: {exc}") from exc

        raise ProviderError(f"Unable to retrieve {url}")

    def json(self, url: str):
        try:
            return json.loads(self.text(url))
        except ValueError as exc:
            raise ProviderError(f"Expected JSON from {url}") from exc
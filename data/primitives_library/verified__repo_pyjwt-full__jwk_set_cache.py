import time
from typing import Optional

from .api_jwk import PyJWKSet, PyJWTSetWithTimestamp


class JWKSetCache:
    def __init__(self, lifespan: float) -> None:
        self.jwk_set_with_timestamp: Optional[PyJWTSetWithTimestamp] = None
        self.lifespan = lifespan

    def put(self, jwk_set: PyJWKSet) -> None:
        if jwk_set is None:
            self.jwk_set_with_timestamp = None
            return

        self.jwk_set_with_timestamp = PyJWTSetWithTimestamp(jwk_set)

    def get(self) -> Optional[PyJWKSet]:
        cached_set = self.jwk_set_with_timestamp
        if cached_set is None:
            return None
        if self.is_expired():
            return None
        return cached_set.get_jwk_set()

    def is_expired(self) -> bool:
        cached_set = self.jwk_set_with_timestamp
        if cached_set is None or self.lifespan <= -1:
            return False
        return time.monotonic() > cached_set.get_timestamp() + self.lifespan
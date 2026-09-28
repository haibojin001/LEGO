import http.cookies
from typing import Optional


class SimpleCookieJar:
    def __init__(self) -> None:
        self.jar: dict = {}

    def add(self, set_cookie: Optional[str]) -> None:
        if not set_cookie:
            return

        parsed = http.cookies.SimpleCookie(set_cookie)
        for morsel in parsed.values():
            domain = morsel.get("domain")
            if not domain:
                continue

            if not domain.startswith("."):
                domain = "." + domain

            stored = self.jar.get(domain)
            if stored is None:
                stored = http.cookies.SimpleCookie()

            stored.update(parsed)
            self.jar[domain.lower()] = stored

    def set(self, set_cookie: str) -> None:
        if not set_cookie:
            return

        parsed = http.cookies.SimpleCookie(set_cookie)
        for morsel in parsed.values():
            domain = morsel.get("domain")
            if not domain:
                continue

            if not domain.startswith("."):
                domain = "." + domain

            self.jar[domain.lower()] = parsed

    def get(self, host: str) -> str:
        if not host:
            return ""

        matching = []
        for domain in self.jar:
            host = host.lower()
            if host.endswith(domain) or host == domain[1:]:
                matching.append(self.jar.get(domain))

        values = [
            f"{name}={morsel.value}"
            for cookie in filter(None, matching)
            for name, morsel in cookie.items()
        ]
        return "; ".join(filter(None, sorted(values)))
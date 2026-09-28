from .exceptions import *

import random

import requests


def extract_video_id_from_url(url, headers={}, proxy=None):
    resolved_url = requests.head(
        url=url,
        allow_redirects=True,
        headers=headers,
        proxies=proxy,
    ).url

    if "@" in resolved_url and "/video/" in resolved_url:
        return resolved_url.split("/video/")[1].split("?")[0]

    raise TypeError(
        "URL format not supported. Below is an example of a supported url.\n"
        "https://www.tiktok.com/@therock/video/6829267836783971589"
    )


def random_choice(choices: list):
    """Return a random choice from a list, or None if the list is empty"""
    if choices is None or len(choices) == 0:
        return None
    return random.choice(choices)


def requests_cookie_to_playwright_cookie(req_c):
    cookie = {
        "name": req_c.name,
        "value": req_c.value,
        "domain": req_c.domain,
        "path": req_c.path,
        "secure": req_c.secure,
    }
    if req_c.expires:
        cookie["expires"] = req_c.expires
    return cookie
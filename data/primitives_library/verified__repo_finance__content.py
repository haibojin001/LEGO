import re
from urllib.parse import quote
from xml.etree import ElementTree

import pandas as pd

from finance.data.providers import ProviderError
from finance.data.web import PublicWeb


def rss_news(url: str, *, web: PublicWeb | None = None) -> pd.DataFrame:
    source = web if web is not None else PublicWeb()
    xml_text = source.text(url)
    try:
        document = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as error:
        raise ProviderError("Feed is not valid XML") from error

    entries = []
    for node in document.findall(".//item"):
        headline = node.findtext("title")
        target = node.findtext("link")
        if headline and target:
            entries.append(
                {
                    "title": headline,
                    "url": target,
                    "published": node.findtext("pubDate"),
                }
            )

    if not entries:
        raise ProviderError("No RSS news items found")

    frame = pd.DataFrame(entries).drop_duplicates("url")
    frame["published"] = pd.to_datetime(
        frame.published,
        utc=True,
        errors="raise",
    )
    return frame.reset_index(drop=True)


def article_text(url: str, *, web: PublicWeb | None = None) -> str:
    """Extract a publicly accessible article/transcript by URL; no login or paywall fallback."""
    from lxml import html

    source = web if web is not None else PublicWeb()
    page = html.fromstring(source.text(url))
    candidates = page.xpath('//*[@id="article-body-transcript"]')
    if not candidates:
        candidates = page.xpath("//article")
    if not candidates:
        raise ProviderError("No article element; publisher needs a dedicated parser")

    selected = max(candidates, key=lambda element: len(element.text_content()))
    blocks = [
        " ".join(paragraph.text_content().split())
        for paragraph in selected.xpath(".//p")
    ]
    content = "\n\n".join(block for block in blocks if block)

    if len(content) < 200:
        raise ProviderError("Article is missing or truncated")
    return content


def reddit_posts(
    subreddit: str,
    limit: int = 25,
    *,
    web: PublicWeb | None = None,
) -> pd.DataFrame:
    """Public recent posts; fail explicitly if Reddit requires authenticated access."""
    if not re.fullmatch(r"\w{2,30}", subreddit) or not 1 <= limit <= 100:
        raise ValueError("invalid subreddit or limit")

    source = web if web is not None else PublicWeb()
    response = source.json(
        f"https://www.reddit.com/r/{quote(subreddit)}/new.json?limit={limit}&raw_json=1"
    )

    try:
        posts = [child["data"] for child in response["data"]["children"]]
        frame = pd.DataFrame(
            [
                {
                    "id": post["id"],
                    "text": post["title"] + "\n" + post.get("selftext", ""),
                    "published": pd.to_datetime(
                        post["created_utc"],
                        unit="s",
                        utc=True,
                    ),
                    "score": post["score"],
                    "url": "https://www.reddit.com" + post["permalink"],
                }
                for post in posts
            ]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ProviderError("Reddit response schema changed") from error

    return frame


def transcript_index(*, web: PublicWeb | None = None) -> pd.DataFrame:
    """Discover recent publicly listed Motley Fool transcripts; not a complete historical archive."""
    from urllib.parse import urljoin

    from lxml import html

    index_url = "https://www.fool.com/earnings-call-transcripts/"
    source = web if web is not None else PublicWeb()
    page = html.fromstring(source.text(index_url))

    entries = []
    for anchor in page.xpath("//a[@href]"):
        href = anchor.get("href")
        if "/earnings/call-transcripts/" in href:
            headline = " ".join(anchor.text_content().split())
            if headline:
                entries.append(
                    {
                        "title": headline,
                        "url": urljoin(index_url, href),
                    }
                )

    if not entries:
        raise ProviderError("Transcript index has no article links")

    return pd.DataFrame(entries).drop_duplicates("url").reset_index(drop=True)
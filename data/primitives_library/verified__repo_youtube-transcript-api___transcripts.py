import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from html import unescape
from itertools import chain
from typing import Dict, Iterable, Iterator, List, Optional, Pattern

from defusedxml import ElementTree
from requests import HTTPError, Response, Session

from ._errors import (
    AgeRestricted,
    InvalidVideoId,
    IpBlocked,
    NoTranscriptFound,
    NotTranslatable,
    RequestBlocked,
    TranscriptsDisabled,
    TranslationLanguageNotAvailable,
    VideoUnavailable,
    VideoUnplayable,
    YouTubeDataUnparsable,
    YouTubeRequestFailed,
    YouTubeTranscriptApiException,
)
from ._settings import INNERTUBE_API_URL, INNERTUBE_CONTEXT, WATCH_URL
from .proxies import ProxyConfig

try:
    from ._errors import FailedToCreateConsentCookie
except ImportError:
    class FailedToCreateConsentCookie(YouTubeTranscriptApiException):
        pass


try:
    from ._errors import PoTokenRequired
except ImportError:
    class PoTokenRequired(YouTubeTranscriptApiException):
        pass


@dataclass
class FetchedTranscriptSnippet:
    text: str
    start: float
    """
    The timestamp at which this transcript snippet appears on screen in seconds.
    """
    duration: float
    """
    The duration of how long the snippet in seconds. Be aware that this is not the
    duration of the transcribed speech, but how long the snippet stays on screen.
    Therefore, there can be overlaps between snippets!
    """


@dataclass
class FetchedTranscript:
    """
    Represents a fetched transcript. This object is iterable, which allows you to
    iterate over the transcript snippets.
    """

    snippets: List[FetchedTranscriptSnippet]
    video_id: str
    language: str
    language_code: str
    is_generated: bool

    def __iter__(self) -> Iterator[FetchedTranscriptSnippet]:
        return iter(self.snippets)

    def __getitem__(self, index) -> FetchedTranscriptSnippet:
        return self.snippets[index]

    def __len__(self) -> int:
        return len(self.snippets)

    def to_raw_data(self) -> List[Dict]:
        return [asdict(snippet) for snippet in self]


@dataclass
class _TranslationLanguage:
    language: str
    language_code: str


class _PlayabilityStatus(str, Enum):
    OK = "OK"
    ERROR = "ERROR"
    LOGIN_REQUIRED = "LOGIN_REQUIRED"


class _PlayabilityFailedReason(str, Enum):
    BOT_DETECTED = "Sign in to confirm you’re not a bot"
    AGE_RESTRICTED = "This video may be inappropriate for some users."
    VIDEO_UNAVAILABLE = "This video is unavailable"


def _raise_http_errors(response: Response, video_id: str) -> Response:
    try:
        if response.status_code == 429:
            raise IpBlocked(video_id)
        response.raise_for_status()
        return response
    except HTTPError as error:
        raise YouTubeRequestFailed(video_id, error)


class Transcript:
    def __init__(
        self,
        http_client: Session,
        video_id: str,
        url: str,
        language: str,
        language_code: str,
        is_generated: bool,
        translation_languages: List[_TranslationLanguage],
    ):
        self._http_client = http_client
        self.video_id = video_id
        self._url = url
        self.language = language
        self.language_code = language_code
        self.is_generated = is_generated
        self.translation_languages = translation_languages
        self._translation_languages_dict = {
            translation_language.language_code: translation_language.language
            for translation_language in translation_languages
        }

    def fetch(self, preserve_formatting: bool = False) -> FetchedTranscript:
        if "&exp=xpe" in self._url:
            raise PoTokenRequired(self.video_id)

        response = self._http_client.get(self._url)
        snippets = _TranscriptParser(preserve_formatting=preserve_formatting).parse(
            _raise_http_errors(response, self.video_id).text
        )
        return FetchedTranscript(
            snippets=snippets,
            video_id=self.video_id,
            language=self.language,
            language_code=self.language_code,
            is_generated=self.is_generated,
        )

    def __str__(self) -> str:
        return '{language_code} ("{language}"){translation_description}'.format(
            language=self.language,
            language_code=self.language_code,
            translation_description="[TRANSLATABLE]" if self.is_translatable else "",
        )

    @property
    def is_translatable(self) -> bool:
        return len(self.translation_languages) > 0

    def translate(self, language_code: str) -> "Transcript":
        if not self.is_translatable:
            raise NotTranslatable(self.video_id)

        if language_code not in self._translation_languages_dict:
            raise TranslationLanguageNotAvailable(self.video_id)

        return Transcript(
            self._http_client,
            self.video_id,
            "{url}&tlang={language_code}".format(
                url=self._url, language_code=language_code
            ),
            self._translation_languages_dict[language_code],
            language_code,
            True,
            [],
        )


class TranscriptList:
    """
    Represents all transcripts available for a video.
    """

    def __init__(
        self,
        video_id: str,
        manually_created_transcripts: Dict[str, Transcript],
        generated_transcripts: Dict[str, Transcript],
        translation_languages: List[_TranslationLanguage],
    ):
        self.video_id = video_id
        self._manually_created_transcripts = manually_created_transcripts
        self._generated_transcripts = generated_transcripts
        self._translation_languages = translation_languages

    @staticmethod
    def build(
        http_client: Session, video_id: str, captions_json: Dict
    ) -> "TranscriptList":
        translation_languages = [
            _TranslationLanguage(
                language=item["languageName"]["runs"][0]["text"],
                language_code=item["languageCode"],
            )
            for item in captions_json.get("translationLanguages", [])
        ]

        manually_created_transcripts: Dict[str, Transcript] = {}
        generated_transcripts: Dict[str, Transcript] = {}

        for caption in captions_json["captionTracks"]:
            is_generated = caption.get("kind", "") == "asr"
            target = generated_transcripts if is_generated else manually_created_transcripts
            target[caption["languageCode"]] = Transcript(
                http_client=http_client,
                video_id=video_id,
                url=caption["baseUrl"].replace("&fmt=srv3", ""),
                language=caption["name"]["runs"][0]["text"],
                language_code=caption["languageCode"],
                is_generated=is_generated,
                translation_languages=(
                    translation_languages if caption.get("isTranslatable", False) else []
                ),
            )

        return TranscriptList(
            video_id,
            manually_created_transcripts,
            generated_transcripts,
            translation_languages,
        )

    def __iter__(self) -> Iterator[Transcript]:
        return chain(
            self._manually_created_transcripts.values(),
            self._generated_transcripts.values(),
        )

    def __str__(self) -> str:
        manual = "\n".join(map(str, self._manually_created_transcripts.values())) or "None"
        generated = "\n".join(map(str, self._generated_transcripts.values())) or "None"
        translations = (
            "\n".join(
                '{0.language_code} ("{0.language}")'.format(item)
                for item in self._translation_languages
            )
            or "None"
        )
        return (
            "For this video ({}) transcripts are available in the following languages:\n\n"
            "(MANUALLY CREATED)\n"
            "{}\n\n"
            "(GENERATED)\n"
            "{}\n\n"
            "(TRANSLATION LANGUAGES)\n"
            "{}"
        ).format(self.video_id, manual, generated, translations)

    def find_transcript(self, language_codes: Iterable[str]) -> Transcript:
        return self._find_transcript(
            language_codes,
            [
                self._manually_created_transcripts,
                self._generated_transcripts,
            ],
        )

    def find_generated_transcript(self, language_codes: Iterable[str]) -> Transcript:
        return self._find_transcript(language_codes, [self._generated_transcripts])

    def find_manually_created_transcript(
        self, language_codes: Iterable[str]
    ) -> Transcript:
        return self._find_transcript(language_codes, [self._manually_created_transcripts])

    def _find_transcript(
        self,
        language_codes: Iterable[str],
        transcript_dicts: List[Dict[str, Transcript]],
    ) -> Transcript:
        for language_code in language_codes:
            for transcript_dict in transcript_dicts:
                if language_code in transcript_dict:
                    return transcript_dict[language_code]

        raise NoTranscriptFound(
            self.video_id,
            list(language_codes),
            self._manually_created_transcripts,
            self._generated_transcripts,
        )


class TranscriptListFetcher:
    _INNERTUBE_API_KEY_REGEX = re.compile(r'"INNERTUBE_API_KEY":"([^"]+)"')

    def __init__(
        self,
        http_client: Session,
        proxy_config: Optional[ProxyConfig] = None,
    ):
        self._http_client = http_client
        self._proxy_config = proxy_config

    def fetch(self, video_id: str) -> TranscriptList:
        if video_id.startswith(("http://", "https://")):
            raise InvalidVideoId(video_id)

        try:
            return self._fetch_transcript_list(video_id)
        except RequestBlocked:
            if self._proxy_config is None:
                raise

            retries = getattr(self._proxy_config, "retries_when_blocked", 0)
            if retries <= 0:
                raise

            last_error = None
            for _ in range(retries):
                rotate_proxy = getattr(self._proxy_config, "rotate_proxy", None)
                if callable(rotate_proxy):
                    rotate_proxy()
                try:
                    return self._fetch_transcript_list(video_id)
                except RequestBlocked as error:
                    last_error = error

            if last_error is not None:
                raise last_error
            raise

    def _fetch_transcript_list(self, video_id: str) -> TranscriptList:
        html = self._fetch_video_html(video_id)
        api_key = self._extract_innertube_api_key(html, video_id)
        innertube_data = self._fetch_innertube_data(video_id, api_key)
        captions_json = self._extract_captions_json(innertube_data, video_id)
        return TranscriptList.build(self._http_client, video_id, captions_json)

    def _fetch_video_html(self, video_id: str) -> str:
        response = self._http_client.get(WATCH_URL.format(video_id=video_id))

        response_url = getattr(response, "url", "")
        if response_url and "consent.youtube.com" in response_url:
            self._create_consent_cookie(response, video_id)
            response = self._http_client.get(WATCH_URL.format(video_id=video_id))

        return _raise_http_errors(response, video_id).text

    def _create_consent_cookie(self, response: Response, video_id: str) -> None:
        match = re.search(r'name="v"\s+value="([^"]+)"', response.text)
        if match is None:
            raise FailedToCreateConsentCookie(video_id)

        self._http_client.cookies.set(
            "CONSENT",
            "YES+{}".format(match.group(1)),
            domain=".youtube.com",
        )

    def _extract_innertube_api_key(self, html: str, video_id: str) -> str:
        match = self._INNERTUBE_API_KEY_REGEX.search(html)
        if match is not None:
            return match.group(1)

        try:
            config_match = re.search(r"ytcfg\.set\((\{.*?\})\);", html)
            if config_match is not None:
                return json.loads(config_match.group(1))["INNERTUBE_API_KEY"]
        except (KeyError, TypeError, ValueError):
            pass

        raise YouTubeDataUnparsable(video_id)

    def _fetch_innertube_data(self, video_id: str, api_key: str) -> Dict:
        response = self._http_client.post(
            INNERTUBE_API_URL.format(api_key=api_key),
            json={
                "context": INNERTUBE_CONTEXT,
                "videoId": video_id,
            },
        )
        response = _raise_http_errors(response, video_id)

        try:
            return response.json()
        except (ValueError, AttributeError):
            try:
                return json.loads(response.text)
            except (TypeError, ValueError):
                raise YouTubeDataUnparsable(video_id)

    def _extract_captions_json(self, innertube_data: Dict, video_id: str) -> Dict:
        self._assert_playability(innertube_data.get("playabilityStatus", {}), video_id)

        try:
            captions_json = innertube_data["captions"][
                "playerCaptionsTracklistRenderer"
            ]
        except (KeyError, TypeError):
            raise TranscriptsDisabled(video_id)

        if "captionTracks" not in captions_json:
            raise TranscriptsDisabled(video_id)

        return captions_json

    def _assert_playability(self, playability_status: Dict, video_id: str) -> None:
        status = playability_status.get("status")
        reason = playability_status.get("reason", "")

        if status == _PlayabilityStatus.OK:
            return

        if status == _PlayabilityStatus.LOGIN_REQUIRED:
            if reason == _PlayabilityFailedReason.BOT_DETECTED:
                raise RequestBlocked(video_id)
            if reason == _PlayabilityFailedReason.AGE_RESTRICTED:
                raise AgeRestricted(video_id)

        if (
            status == _PlayabilityStatus.ERROR
            and reason == _PlayabilityFailedReason.VIDEO_UNAVAILABLE
        ):
            raise VideoUnavailable(video_id)

        raise VideoUnplayable(video_id, reason)


class _TranscriptParser:
    _FORMATTING_TAGS = (
        "strong",
        "em",
        "b",
        "i",
        "mark",
        "small",
        "del",
        "ins",
        "sub",
        "sup",
    )

    def __init__(self, preserve_formatting: bool = False):
        self._html_regex = self._get_html_regex(preserve_formatting)

    def parse(self, plain_data: str) -> List[FetchedTranscriptSnippet]:
        return [
            FetchedTranscriptSnippet(
                text=unescape(
                    self._html_regex.sub("", "".join(xml_element.itertext()))
                ),
                start=float(xml_element.attrib["start"]),
                duration=float(xml_element.attrib.get("dur", "0.0")),
            )
            for xml_element in ElementTree.fromstring(plain_data)
            if xml_element.text is not None
        ]

    def _get_html_regex(self, preserve_formatting: bool) -> Pattern:
        if not preserve_formatting:
            return re.compile(r"<[^>]*>", re.IGNORECASE)

        formatting_tags = "|".join(self._FORMATTING_TAGS)
        return re.compile(
            r"<\/?(?!(?:{})(?:\s|>|\/))[^>]*>".format(formatting_tags),
            re.IGNORECASE,
        )
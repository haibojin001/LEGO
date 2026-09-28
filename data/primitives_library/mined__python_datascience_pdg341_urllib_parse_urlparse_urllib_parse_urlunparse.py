# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg341::urllib.parse.urlparse+urllib.parse.urlunparse
# name: urllib_primitive
# summary: Uses urllib.parse.urlparse, urllib.parse.urlunparse across 5 repos
# anchor_symbols: ['urllib.parse.urlparse', 'urllib.parse.urlunparse']
# observed in 5 repos: ['airbnb__knowledge-repo', 'insitro__redun', 'okfn-brasil__querido-diario', 'turicas__rows', 'xorbitsai__xorbits']...

# --- from okfn-brasil__querido-diario::data_collection/gazette/spiders/ba/ba_barreiras.py::BaBarreirasSpider._url_fix ---
def _url_fix(self, link):
        link = urlparse(link)
        link = link._replace(scheme="https")
        return urlunparse(link)

# --- from insitro__redun::examples/scraping/workflow.py::get_base_url ---
def get_base_url(url: str) -> str:
    """
    Strip the query and fragment from a URL.
    """
    parts = urlparse(url)
    return urlunparse((parts.scheme, parts.netloc, parts.path, "", "", ""))

# --- from airbnb__knowledge-repo::knowledge_repo/app/utils/auth.py::is_safe_url ---
def is_safe_url(target):
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') \
        and ref_url.netloc == test_url.netloc

# --- from insitro__redun::examples/scraping/workflow.py::clean_url ---
def clean_url(url: str, base_url: str) -> str:
    """
    Clean a URL for use in crawling.
    """
    # Make url absolute.
    if not urlparse(url).netloc:
        url = urljoin(base_url, url)

    # Discard query params and fragment.
    return get_base_url(url)

# --- from turicas__rows::tests/tests_plugin_postgresql.py::PluginPostgreSQLTestCase.tearDownClass ---
def tearDownClass(cls):
        """Delete the test database for this Python version"""

        if DATABASE_URL is None:
            return
        parsed = urlparse(DATABASE_URL)
        database_url_no_db = urlunparse(
            (parsed.scheme, parsed.netloc, "/", parsed.params, parsed.query, parsed.fragment)
        )
        connection = pgconnect(database_url_no_db)
        connection.autocommit = True
        cursor = connection.cursor()
        cursor.execute("DROP DATABASE IF EXISTS {}".format(TEST_DATABASE_NAME))
        cursor.close()

# --- from xorbitsai__xorbits::python/xorbits/_mars/lib/filesystem/fsspec_adapter.py::FsSpecAdapter._normalize_path ---
def _normalize_path(path: path_type) -> str:
        """
        Stringify path and remove its scheme.
        """
        path_str = stringify_path(path)
        parsed = urlparse(path_str)
        if parsed.scheme:
            return urlunparse(
                ParseResult(
                    scheme="",
                    netloc=parsed.netloc,
                    path=parsed.path,
                    params="",
                    query="",
                    fragment="",
                )
            )
        else:
            return path_str

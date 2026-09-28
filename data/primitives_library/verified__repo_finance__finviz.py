import re
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import numpy as np
import pandas as pd

from finance.data.providers import ProviderError
from finance.data.web import PublicWeb


def number(value: str) -> float:
    cleaned = value.strip().replace(",", "").replace("$", "").replace("−", "-")
    if cleaned in {"", "-", "--", "N/A", "None"}:
        return np.nan

    multipliers = {
        "K": 1e3,
        "M": 1e6,
        "B": 1e9,
        "T": 1e12,
        "%": 0.01,
    }
    suffix = cleaned[-1]
    multiplier = multipliers.get(suffix, 1)
    if suffix in multipliers:
        cleaned = cleaned[:-1]

    try:
        parsed = float(cleaned) * multiplier
        if not np.isfinite(parsed):
            raise ValueError("nonfinite number")
        return parsed
    except ValueError as exc:
        raise ProviderError(f"Unexpected numeric field: {cleaned!r}") from exc


def _root(body):
    from lxml import html

    return html.fromstring(body)


def _text(node):
    return " ".join(node.text_content().split())


def _symbol(cell):
    for anchor in cell.xpath(".//a[@href]"):
        params = parse_qs(urlparse(anchor.get("href")).query)
        if "t" in params:
            return params["t"][0]
    raise ProviderError("Ticker link missing from Finviz row")


def _table(root, required):
    for candidate in root.xpath("//table"):
        rows = candidate.xpath("./tr|./thead/tr|./tbody/tr")
        if not rows:
            continue
        labels = [_text(item) for item in rows[0].xpath("./td|./th")]
        if set(required).issubset(labels):
            return labels, rows[1:]
    raise ProviderError(f"Finviz table missing columns {required}")


def _stamp(result, url):
    result.attrs.update(
        source=url,
        retrieved_at=pd.Timestamp.now(tz="UTC").isoformat(),
        historical=False,
    )
    return result


@dataclass
class Finviz:
    """Public snapshots requiring the data extra; attrs record source and retrieval time."""

    web: PublicWeb = field(default_factory=PublicWeb)

    def _get(self, path, **params):
        address = "https://finviz.com/" + path
        if params:
            address += "?" + urlencode(params)
        return _root(self.web.text(address)), address

    def screen(
        self, filters: list[str] | None = None, *, order: str = "ticker", limit: int = 100
    ) -> pd.DataFrame:
        """Fetch up to limit rows using Finviz filter codes, e.g. ['cap_largeover'].

        Index: source ticker. Columns: name, sector, industry, country, market_cap,
        pe, price, change, volume. Percent change is fractional; missing numbers are NaN.
        Inspect attrs['complete'] and attrs['total_matches'] before treating this as
        the full result. These are current snapshots, not historical constituents.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10000:
            raise ValueError("limit must be an integer in [1,10000]")

        requested_filters = filters or []
        if any(not re.fullmatch(r"[\w.-]+", item) for item in [order, *requested_filters]):
            raise ValueError("invalid Finviz filter/order code")

        entries = []
        total_matches = None
        source_url = ""

        for first in range(1, limit + 1, 20):
            document, source_url = self._get(
                "screener.ashx",
                v=111,
                f=",".join(requested_filters),
                o=order,
                r=first,
            )

            report = _text(document)
            found = re.search(r"/\s*([\d,]+)\s+Total", report)
            if found:
                observed_total = int(found.group(1).replace(",", ""))
                if total_matches is not None and observed_total != total_matches:
                    raise ProviderError("Screener changed during pagination; retry the snapshot")
                total_matches = observed_total

            if total_matches == 0:
                break

            headings, data_rows = _table(
                document,
                ["Ticker", "Company", "Market Cap", "Price"],
            )
            current_page = []

            for data_row in data_rows:
                data_cells = data_row.xpath("./td")
                if len(data_cells) != len(headings):
                    continue

                cells_by_name = dict(zip(headings, data_cells, strict=True))
                current_page.append(
                    {
                        "ticker": _symbol(cells_by_name["Ticker"]),
                        "name": _text(cells_by_name["Company"]),
                        "sector": _text(cells_by_name["Sector"]),
                        "industry": _text(cells_by_name["Industry"]),
                        "country": _text(cells_by_name["Country"]),
                        "market_cap": number(_text(cells_by_name["Market Cap"])),
                        "pe": number(_text(cells_by_name["P/E"])),
                        "price": number(_text(cells_by_name["Price"])),
                        "change": number(_text(cells_by_name["Change %"])),
                        "volume": number(_text(cells_by_name["Volume"])),
                    }
                )

            if not current_page:
                raise ProviderError("Finviz returned no parseable screener rows")

            entries.extend(current_page)
            if total_matches is not None and len(entries) >= total_matches:
                break

        frame = pd.DataFrame(
            entries,
            columns=[
                "ticker",
                "name",
                "sector",
                "industry",
                "country",
                "market_cap",
                "pe",
                "price",
                "change",
                "volume",
            ],
        ).set_index("ticker")

        if frame.index.has_duplicates:
            raise ProviderError("Duplicate tickers across screener pages; retry the snapshot")

        if total_matches is None:
            raise ProviderError("Finviz total count missing; completeness cannot be established")

        frame = frame.iloc[:limit]
        frame.attrs.update(
            total_matches=total_matches,
            complete=len(frame) == total_matches,
            filters=requested_filters,
        )
        return _stamp(frame, source_url)

    def universe(self, index: str, *, limit: int = 3000) -> pd.DataFrame:
        """Return screen rows for sp500, dow, nasdaq100 or russell2000.

        Membership follows Finviz's current classification. The index contains tickers;
        attrs['complete'] reports whether all provider matches fit within limit.
        """
        index_codes = {
            "sp500": "sp500",
            "dow": "dji",
            "nasdaq100": "ndx",
            "russell2000": "rut",
        }
        if index not in index_codes:
            raise ValueError(f"index must be one of {list(index_codes)}")
        return self.screen([f"idx_{index_codes[index]}"], limit=limit)

    def company(self, ticker: str) -> pd.Series:
        """Numeric snapshot named by ticker; growth, yield and ownership are fractions.

        RSI uses 0–100. Unavailable numbers are NaN; source-absent optional fields may
        be omitted. attrs['raw_fields'] retains source labels and duplicate values.
        """
        document, source_url = self._get("quote.ashx", t=ticker)

        names = {
            "Market Cap": "market_cap",
            "Enterprise Value": "enterprise_value",
            "P/E": "pe",
            "Forward P/E": "forward_pe",
            "PEG": "peg",
            "P/S": "price_sales",
            "P/B": "price_book",
            "P/FCF": "price_fcf",
            "ROE": "roe",
            "ROA": "roa",
            "ROIC": "roic",
            "Gross Margin": "gross_margin",
            "Oper. Margin": "operating_margin",
            "Profit Margin": "profit_margin",
            "EPS Q/Q": "earnings_growth",
            "Sales Q/Q": "revenue_growth",
            "EPS Y/Y TTM": "earnings_growth_ttm",
            "Sales Y/Y TTM": "revenue_growth_ttm",
            "EPS next 5Y": "expected_earnings_growth_5y",
            "Insider Own": "insider_ownership",
            "Insider Trans": "insider_transactions",
            "Inst Own": "institutional_ownership",
            "Inst Trans": "institutional_transactions",
            "SMA20": "distance_sma20",
            "SMA50": "distance_sma50",
            "SMA200": "distance_sma200",
            "RSI (14)": "rsi",
            "Rel Volume": "relative_volume",
            "Avg Volume": "average_volume",
            "Volume": "volume",
            "Price": "price",
            "Target Price": "target_price",
            "Recom": "analyst_rating",
            "Shs Outstand": "shares",
            "Short Float": "short_float",
            "Beta": "beta",
            "Debt/Eq": "debt_equity",
            "Perf Year": "return_1y",
            "Perf Quarter": "return_quarter",
            "Perf Month": "return_month",
        }

        values = {}
        raw_fields = {}

        for row in document.xpath('//table[contains(@class,"snapshot-table2")]//tr'):
            cells = row.xpath("./td")
            for position in range(0, len(cells) - 1, 2):
                label = _text(cells[position])
                field_value = _text(cells[position + 1])

                raw_fields.setdefault(label, []).append(field_value)

                mapped_name = names.get(label)
                if mapped_name is not None:
                    values[mapped_name] = number(field_value)

                if label == "EPS next Y" and field_value.endswith("%"):
                    values["expected_earnings_growth_1y"] = number(field_value)

        snapshot = pd.Series(values, dtype=float, name=ticker)
        snapshot.attrs["raw_fields"] = raw_fields
        return _stamp(snapshot, source_url)
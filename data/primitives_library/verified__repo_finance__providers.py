from dataclasses import dataclass
from io import StringIO
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from finance.data.normalize import normalize_ohlcv, normalize_ticker


class ProviderError(RuntimeError):
    """Provider unavailable, missing data, or unexpected schema."""


class PriceProvider(Protocol):
    def history(
        self, ticker: str, start: str, end: str, *, interval: str = "1d"
    ) -> pd.DataFrame: ...


def fetch_text(url: str, *, timeout: float = 15) -> str:
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    request = Request(url, headers={"User-Agent": "FinanceToolkit research"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError, UnicodeError) as exc:
        raise ProviderError(f"Unable to retrieve {url}: {exc}") from exc


@dataclass(frozen=True)
class YahooFinance:
    """Optional Yahoo adapter. End is exclusive; adjusted OHLC includes splits/dividends."""

    timeout: float = 15
    adjusted: bool = True

    def _ticker(self, ticker: str):
        import yfinance as yf
        from curl_cffi.requests import Session

        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

        return yf.Ticker(
            normalize_ticker(ticker),
            session=Session(impersonate="chrome", timeout=self.timeout),
        )

    def history(
        self, ticker: str, start: str, end: str, *, interval: str = "1d"
    ) -> pd.DataFrame:
        """Return float open/high/low/close/volume columns for [start, end).

        The sorted, unique DatetimeIndex retains the provider timezone. Missing or
        invalid OHLCV raises ProviderError. attrs['adjusted'] records the price basis;
        the default includes split and dividend adjustments in all OHLC prices.
        """
        if pd.Timestamp(start) >= pd.Timestamp(end):
            raise ValueError("start must precede end")

        handle = self._ticker(ticker)
        try:
            history = handle.history(
                start=start,
                end=end,
                interval=interval,
                auto_adjust=self.adjusted,
                actions=False,
                timeout=self.timeout,
            )
            frame = normalize_ohlcv(history)
        except Exception as exc:
            raise ProviderError(
                f"Yahoo history failed for {ticker} ({interval}): {exc}"
            ) from exc

        frame.attrs.update(
            {
                "provider": "Yahoo Finance",
                "ticker": normalize_ticker(ticker),
                "adjusted": self.adjusted,
            }
        )
        return frame

    def dividends(self, ticker: str, start: str, end: str) -> pd.Series:
        handle = self._ticker(ticker)
        try:
            history = handle.history(
                start=start,
                end=end,
                auto_adjust=False,
                actions=True,
                timeout=self.timeout,
            )
            if history.empty or "Dividends" not in history:
                raise ValueError("missing dividend history")

            dividends = pd.to_numeric(history["Dividends"], errors="raise")
            return dividends[dividends != 0].rename("dividend")
        except Exception as exc:
            raise ProviderError(f"Yahoo dividends failed for {ticker}: {exc}") from exc

    def company(self, ticker: str) -> pd.Series:
        """Current snapshot, unsuitable for historical fundamental backtests; rates are fractions."""
        handle = self._ticker(ticker)
        mapping = {
            "longName": "name",
            "currentPrice": "price",
            "marketCap": "market_cap",
            "trailingPE": "pe",
            "returnOnEquity": "roe",
            "returnOnAssets": "roa",
            "revenueGrowth": "revenue_growth",
            "earningsGrowth": "earnings_growth",
            "grossMargins": "gross_margin",
            "dividendRate": "annual_dividend",
            "totalDebt": "debt",
            "totalCash": "cash",
            "freeCashflow": "free_cash_flow",
            "sharesOutstanding": "shares",
            "recommendationMean": "analyst_rating",
        }

        try:
            info = handle.get_info()
            if not info or "marketCap" not in info:
                raise ValueError("company snapshot lacks marketCap")

            company = pd.Series(
                {destination: info.get(source) for source, destination in mapping.items()},
                name=normalize_ticker(ticker),
            )
            price = company["price"]
            dividend = company["annual_dividend"]
            company["dividend_yield"] = (
                dividend / price if dividend is not None and price else float("nan")
            )
            return company
        except Exception as exc:
            raise ProviderError(f"Yahoo company failed for {ticker}: {exc}") from exc

    def statements(
        self, ticker: str, kind: str = "income", frequency: str = "annual"
    ) -> pd.DataFrame:
        """Period-end rows, provider statement labels, reported currency units; restatements possible."""
        statement_names = {
            "income": "income_stmt",
            "balance": "balance_sheet",
            "cashflow": "cashflow",
        }
        if kind not in statement_names or frequency not in ("annual", "quarterly"):
            raise ValueError("kind: income/balance/cashflow; frequency: annual/quarterly")

        handle = self._ticker(ticker)
        prefix = "quarterly_" if frequency == "quarterly" else ""
        attribute = prefix + statement_names[kind]

        try:
            statement = getattr(handle, attribute)
            if (
                not isinstance(statement, pd.DataFrame)
                or statement.empty
                or statement.index.has_duplicates
            ):
                raise ValueError("missing or duplicate statement fields")

            frame = statement.T.apply(pd.to_numeric, errors="raise").sort_index()
            frame = frame.dropna(how="all")
            frame.index = pd.DatetimeIndex(frame.index, name="period_end")
            frame.attrs.update(
                {
                    "provider": "Yahoo Finance",
                    "ticker": ticker,
                    "frequency": frequency,
                    "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(),
                    "currency": handle.get_info().get("financialCurrency"),
                    "restated": True,
                }
            )
            return frame
        except Exception as exc:
            raise ProviderError(
                f"Yahoo {kind} statement failed for {ticker}: {exc}"
            ) from exc

    def earnings(self, ticker: str) -> pd.DataFrame:
        return self._table(ticker, "get_earnings_dates", ["EPS Estimate", "Reported EPS"])

    def insider_transactions(self, ticker: str) -> pd.DataFrame:
        return self._table(ticker, "get_insider_transactions", ["Shares", "Start Date"])

    def _table(self, ticker: str, method: str, required: list[str]) -> pd.DataFrame:
        handle = self._ticker(ticker)
        try:
            table = getattr(handle, method)()
            if (
                not isinstance(table, pd.DataFrame)
                or table.empty
                or not set(required).issubset(table.columns)
            ):
                raise ValueError(f"missing table/columns: {required}")
            return table.copy()
        except Exception as exc:
            raise ProviderError(f"Yahoo {method} failed for {ticker}: {exc}") from exc

    def news(self, ticker: str) -> pd.DataFrame:
        handle = self._ticker(ticker)
        try:
            stories = handle.get_news()
            rows = []
            for story in stories:
                content = story.get("content", story)
                rows.append(
                    {
                        "title": content["title"],
                        "published": content.get("pubDate"),
                        "url": content.get(
                            "canonicalUrl", {}
                        ).get("url", content.get("link")),
                    }
                )

            if not rows:
                raise ValueError("empty news response")
            return pd.DataFrame(rows)
        except Exception as exc:
            raise ProviderError(f"Yahoo news failed for {ticker}: {exc}") from exc


def sp500_constituents(*, timeout: float = 15) -> pd.DataFrame:
    """Current constituents, not a point-in-time universe (survivorship bias). Requires data extra."""
    html = fetch_text(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        timeout=timeout,
    )
    try:
        tables = pd.read_html(StringIO(html))
        source = next(
            table
            for table in tables
            if {"Symbol", "Security", "GICS Sector"}.issubset(table.columns)
        )
        result = source[["Symbol", "Security", "GICS Sector"]].copy()
        result.columns = ["ticker", "name", "sector"]
        result["ticker"] = result.ticker.map(normalize_ticker)

        if not 400 <= len(result) <= 600 or result.ticker.duplicated().any():
            raise ValueError("unexpected constituent count or duplicates")

        return result.sort_values("ticker").reset_index(drop=True)
    except Exception as exc:
        raise ProviderError(f"S&P 500 constituents failed: {exc}") from exc
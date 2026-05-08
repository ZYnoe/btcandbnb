"""Data layer: download, align, return statistics. Built on yfinance."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger("portfolio_optimizer")


@dataclass
class MarketData:
    prices: pd.DataFrame  # aligned, dropna
    returns: pd.DataFrame  # daily simple returns
    mu: np.ndarray  # annualized mean return per asset
    Sigma: np.ndarray  # annualized covariance matrix
    vol: np.ndarray  # annualized vol per asset
    corr: np.ndarray  # correlation matrix
    tickers: tuple[str, ...]
    annualization_factor: int

    def basic_stats(self) -> dict:
        return {
            "tickers": list(self.tickers),
            "rows": int(len(self.returns)),
            "start": str(self.returns.index.min().date()) if len(self.returns) else None,
            "end": str(self.returns.index.max().date()) if len(self.returns) else None,
            "annualization_factor": self.annualization_factor,
            "annualized_return": {t: float(r) for t, r in zip(self.tickers, self.mu)},
            "annualized_volatility": {t: float(v) for t, v in zip(self.tickers, self.vol)},
            "covariance": [[float(x) for x in row] for row in self.Sigma],
            "correlation": [[float(x) for x in row] for row in self.corr],
        }


def _extract_price_series(raw: pd.DataFrame, ticker: str) -> pd.Series:
    """Pull a price series from yfinance output, preferring Adj Close.

    Handles both single-ticker (flat columns) and multi-ticker (MultiIndex) shapes,
    plus yfinance's evolving auto_adjust behavior where ``Adj Close`` may be missing.
    """
    candidates = ("Adj Close", "Close")

    if isinstance(raw.columns, pd.MultiIndex):
        for field in candidates:
            try:
                col = raw[field][ticker]
                if isinstance(col, pd.Series) and col.dropna().size > 0:
                    return col.rename(ticker)
            except KeyError:
                continue
            except Exception:
                continue
        # Some yfinance versions reverse the level order
        for field in candidates:
            try:
                col = raw[(ticker, field)]
                if isinstance(col, pd.Series) and col.dropna().size > 0:
                    return col.rename(ticker)
            except Exception:
                continue
    else:
        for field in candidates:
            if field in raw.columns:
                col = raw[field]
                if isinstance(col, pd.Series) and col.dropna().size > 0:
                    return col.rename(ticker)
    raise ValueError(
        f"could not find Adj Close or Close for {ticker}; columns={list(raw.columns)}"
    )


def download_prices(
    tickers: tuple[str, ...],
    start: str,
    end: str,
    frequency: str = "1d",
) -> pd.DataFrame:
    """Download adjusted prices via yfinance and return one column per ticker."""
    import yfinance as yf

    series_map: dict[str, pd.Series] = {}

    raw = yf.download(
        tickers=list(tickers),
        start=start,
        end=end,
        interval=frequency,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError(
            f"yfinance returned empty data for {tickers} between {start} and {end}. "
            "Check your network connection or the ticker symbols."
        )

    for t in tickers:
        try:
            series_map[t] = _extract_price_series(raw, t)
        except Exception as e:
            logger.warning(
                "Bulk download missing %s (%s); falling back to single-ticker fetch", t, e
            )
            single = yf.download(
                tickers=t,
                start=start,
                end=end,
                interval=frequency,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if single is None or single.empty:
                raise RuntimeError(f"yfinance returned no data for {t}.") from e
            series_map[t] = _extract_price_series(single, t)

    prices = pd.concat([series_map[t] for t in tickers], axis=1)
    prices.columns = list(tickers)
    prices = prices.sort_index().dropna(how="any")
    if len(prices) < 30:
        raise RuntimeError(
            f"only {len(prices)} aligned price rows after dropna; need at least 30. "
            "Try a wider date range."
        )
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily percentage returns, with the first NaN row dropped."""
    return prices.pct_change().dropna(how="any")


def compute_market_data(
    tickers: tuple[str, ...],
    start: str,
    end: str,
    frequency: str,
    annualization_factor: int,
) -> MarketData:
    """End-to-end: download + align + compute mu/Sigma/vol/corr."""
    prices = download_prices(tickers, start=start, end=end, frequency=frequency)
    returns = compute_returns(prices)
    if returns.empty:
        raise RuntimeError("returns dataframe is empty after pct_change/dropna")

    mu = returns.mean().to_numpy() * annualization_factor
    cov_daily = returns.cov().to_numpy()
    Sigma = cov_daily * annualization_factor
    vol = np.sqrt(np.diag(Sigma))
    corr = returns.corr().to_numpy()

    logger.info(
        "Loaded %d daily rows for %s from %s to %s.",
        len(returns),
        ", ".join(tickers),
        returns.index.min().date(),
        returns.index.max().date(),
    )
    return MarketData(
        prices=prices,
        returns=returns,
        mu=mu,
        Sigma=Sigma,
        vol=vol,
        corr=corr,
        tickers=tuple(tickers),
        annualization_factor=annualization_factor,
    )

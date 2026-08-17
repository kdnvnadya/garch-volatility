from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ["BTC-USD", "^GSPC"]
DEFAULT_START = "2018-01-01"
DEFAULT_END = "2025-12-31"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def download_prices(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    use_cache: bool = True,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    cache_path = cache_path or RAW_DIR / "prices.csv"

    if use_cache and cache_path.exists():
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return prices

    data = yf.download(tickers, start=start, end=end)
    prices = data["Close"].dropna()

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(cache_path)

    return prices

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna()

def train_test_split(
    returns: pd.DataFrame, train_frac: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_point = int(len(returns) * train_frac)
    train = returns.iloc[:split_point]
    test = returns.iloc[split_point:]
    return train, test

def load_dataset(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    train_frac: float = 0.8,
    use_cache: bool = True,
):
    prices = download_prices(tickers, start, end, use_cache=use_cache)
    returns = compute_log_returns(prices)
    train, test = train_test_split(returns, train_frac=train_frac)

    return {
        "prices": prices,
        "returns": returns,
        "train": train,
        "test": test,
    }


if __name__ == "__main__":
    ds = load_dataset()

    print(f"Период: {ds['returns'].index[0].date()} — {ds['returns'].index[-1].date()}")
    print(f"Всего наблюдений: {len(ds['returns'])}")
    print(f"Train: {len(ds['train'])}, Test: {len(ds['test'])}")

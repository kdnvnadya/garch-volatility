from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import adfuller


def _clean(series: pd.Series) -> pd.Series:
    return series.dropna().replace([np.inf, -np.inf], np.nan).dropna()


def descriptive_stats(series: pd.Series) -> dict:
    s = _clean(series)

    adf_stat, adf_p, *_ = adfuller(s)
    jb_stat, jb_p = stats.jarque_bera(s)[:2]

    return {
        "Ст. откл. (%)": round(s.std() * 100, 2),
        "Годовая волатильность (%)": round(s.std() * np.sqrt(252) * 100, 0),
        "Минимум (%)": round(s.min() * 100, 2),
        "Максимум (%)": round(s.max() * 100, 2),
        "Асимметрия": round(s.skew(), 4),
        "Эксцесс": round(s.kurtosis(), 4),
        "ADF p-value": round(adf_p, 4),
        "Стационарность": "ДА" if adf_p < 0.05 else "НЕТ",
        "JB p-value": round(jb_p, 4),
        "Нормальность": "НЕТ" if jb_p < 0.05 else "ДА",
    }


def descriptive_stats_table(series_dict: dict[str, pd.Series]) -> pd.DataFrame:
    rows = {label: descriptive_stats(series) for label, series in series_dict.items()}
    return pd.DataFrame(rows).T


def arch_lm_test(series: pd.Series, nlags: int = 10) -> dict:
    s = _clean(series)
    resid = s - s.mean()

    lm_stat, lm_pval, f_stat, f_pval = het_arch(resid, nlags=nlags)

    return {
        "LM-статистика (χ²)": round(lm_stat, 2),
        "p-значение": "< 0.001" if lm_pval < 0.001 else round(lm_pval, 4),
        "Число лагов": nlags,
        "Вывод": "ARCH-эффекты есть" if lm_pval < 0.05 else "ARCH-эффекты не обнаружены",
    }


def arch_lm_table(series_dict: dict[str, pd.Series], nlags: int = 10) -> pd.DataFrame:
    rows = {label: arch_lm_test(series, nlags=nlags) for label, series in series_dict.items()}
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    from src.data import load_dataset

    ds = load_dataset()
    train = ds["train"]
    series_dict = {"BTC-USD": train["BTC-USD"], "S&P 500": train["^GSPC"]}

    print("ПРЕДВАРИТЕЛЬНЫЙ АНАЛИЗ (обучающая выборка)")
    print(descriptive_stats_table(series_dict).to_string())
    print()
    print("ARCH-LM ТЕСТ")
    print(arch_lm_table(series_dict).to_string())
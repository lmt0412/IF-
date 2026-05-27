import pandas as pd
import numpy as np


def rolling_zscore(series, window=20):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def build_factors(df):
    df = df.copy()
    df = df.sort_values("日期")

    price_col = "收盘价"
    turnover_col = "成交占总市值占比"
    pe_col = "pe"
    macd_col = "macd"
    basis_col = "基差"

    # =====================
    # 量价因子
    # =====================

    df["ret_1d"] = df[price_col].pct_change()

    df["mom_5"] = df[price_col] / df[price_col].shift(5) - 1
    df["mom_20"] = df[price_col] / df[price_col].shift(20) - 1

    df["vol_20"] = df["ret_1d"].rolling(20).std()

    df["ma_20"] = df[price_col].rolling(20).mean()
    df["bias_20"] = df[price_col] / df["ma_20"] - 1

    df["macd_factor"] = df[macd_col]
    df["macd_slope"] = df[macd_col] - df[macd_col].shift(1)

    df["turnover_factor"] = df[turnover_col]
    df["turnover_z20"] = rolling_zscore(df[turnover_col], 20)

    df["basis_factor"] = df[basis_col]
    df["basis_z20"] = rolling_zscore(df[basis_col], 20)
    df["basis_change"] = df[basis_col] - df[basis_col].shift(1)

    # =====================
    # 基本面因子
    # =====================

    df["pe_factor"] = df[pe_col]
    df["pe_z20"] = rolling_zscore(df[pe_col], 20)
    df["pe_change"] = df[pe_col] - df[pe_col].shift(1)

    df["pe_rank_60"] = (
        df[pe_col]
        .rolling(60)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    )

    # =====================
    # target 
    # 预测五日后涨跌百分比, 避开短期噪音
    # =====================

    df["target_5d"] = df[price_col].shift(-5) / df[price_col] - 1

    return df


def get_factor_columns():
    price_volume_factors = [
        "mom_5",
        "mom_20",
        "vol_20",
        "bias_20",
        "macd_factor",
        "macd_slope",
        "turnover_factor",
        "turnover_z20",
        "basis_factor",
        "basis_z20",
        "basis_change",
    ]

    fundamental_factors = [
        "pe_factor",
        "pe_z20",
        "pe_change",
        "pe_rank_60",
    ]

    return price_volume_factors, fundamental_factors
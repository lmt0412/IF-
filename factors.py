import pandas as pd
import numpy as np


def rolling_zscore(series, window=20):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


def build_factors(df):
    df = df.copy()
    df = df.rename(columns={"Unnamed: 0": "日期"})
    df["日期"] = pd.to_datetime(df["日期"], format="%m/%d/%y")
    df = df.sort_values("日期").reset_index(drop=True)

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

    if turnover_col in df.columns:
        df["turnover_factor"] = df[turnover_col]
        df["turnover_z20"] = rolling_zscore(df[turnover_col], 20)

    df["basis_factor"] = df[basis_col]
    df["basis_z20"] = rolling_zscore(df[basis_col], 20)
    df["basis_change"] = df[basis_col] - df[basis_col].shift(1)
    df["basis_pct"] = df[basis_col] / df[price_col]

    # RSI(14)
    delta = df[price_col].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

    # Bollinger Band %B：价格在布林带内的相对位置
    bb_mid = df[price_col].rolling(20).mean()
    bb_std = df[price_col].rolling(20).std()
    df["bb_pct"] = (df[price_col] - (bb_mid - 2 * bb_std)) / (4 * bb_std)

    # 短期反转：1日收益
    df["ret_1d_rev"] = -df["ret_1d"]

    # 波动率比值：近期波动 / 历史波动，判断波动率扩张/收缩
    df["vol_ratio"] = df["ret_1d"].rolling(5).std() / df["ret_1d"].rolling(60).std()

    # 2日、3日收益（补充短期动量粒度）
    df["ret_2d"] = df[price_col] / df[price_col].shift(2) - 1
    df["ret_3d"] = df[price_col] / df[price_col].shift(3) - 1

    # MACD 二阶导：斜率的变化，捕捉动量加速/减速
    df["macd_accel"] = df["macd_slope"] - df["macd_slope"].shift(1)

    # 基差加速度：基差变化的变化
    df["basis_accel"] = df["basis_change"] - df["basis_change"].shift(1)

    # 价量背离：价格涨跌幅 / 换手率，换手低但涨幅大说明上涨缺乏成交量支撑
    if turnover_col in df.columns:
        df["price_vol_diverge"] = df["ret_1d"] / df[turnover_col].replace(0, np.nan)

    # 布林带位置变化：今日 bb_pct 相对昨日的移动方向
    df["bb_pct_change"] = df["bb_pct"] - df["bb_pct"].shift(1)

    # PE 变化率：相对变化比绝对变化更稳定
    df["pe_change_rate"] = df[pe_col].pct_change()

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
    # 因子交叉项
    # =====================

    df["pe_x_mom20"] = rolling_zscore(df[pe_col], 20) * df["mom_20"]
    df["vol_x_bias"] = df["vol_20"] * df["bias_20"]
    df["rsi_x_mom5"] = rolling_zscore(df["rsi_14"], 20) * df["mom_5"]

    # =====================
    # target
    # =====================

    df["target_1d"] = df[price_col].shift(-1) / df[price_col] - 1
    df["target_3d"] = df[price_col].shift(-3) / df[price_col] - 1
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
        "macd_accel",
        "turnover_factor",
        "turnover_z20",
        "basis_factor",
        "basis_z20",
        "basis_change",
        "basis_accel",
        "basis_pct",
        "rsi_14",
        "bb_pct",
        "bb_pct_change",
        "ret_1d_rev",
        "ret_2d",
        "ret_3d",
        "vol_ratio",
        "price_vol_diverge",
    ]

    fundamental_factors = [
        "pe_factor",
        "pe_z20",
        "pe_change",
        "pe_change_rate",
        "pe_rank_60",
    ]

    interaction_factors = [
        "pe_x_mom20",
        "vol_x_bias",
        "rsi_x_mom5",
    ]

    return price_volume_factors, fundamental_factors, interaction_factors
import pandas as pd

from factors import build_factors, get_factor_columns
from factor_analysis import run_factor_analysis
from model import run_all_models
from backtest import run_all_backtests
from report import generate_report


# =====================
# 配置：预测目标（可选 target_1d / target_3d / target_5d）
# SIGNAL_THRESHOLD：过滤最弱信号，0=不过滤，0.33=最弱33%置空仓
# =====================

TARGET = "target_1d"
SIGNAL_FILTER = True      # True = 预测强度 < 0.5×std 时空仓，False = 每天都持仓
SHORT_MULTIPLIER = 3.0    # 做空门槛 = 做多门槛 × 此倍数，越大越难做空
LONG_LEVERAGE = 3.0       # 做多时的杠杆倍数，1.0 = 无杠杆
SHORT_LEVERAGE = 2.0      # 做空时的杠杆倍数，1.0 = 无杠杆


# =====================
# load data
# =====================

df = pd.read_csv("data/cleaned_data.csv")


# =====================
# build factors
# =====================

df = build_factors(df)


# =====================
# factor list & basic info
# =====================

price_volume_factors, fundamental_factors, interaction_factors = get_factor_columns()
all_factors = [f for f in price_volume_factors + fundamental_factors + interaction_factors if f in df.columns]

print(df[all_factors + [TARGET]].head())

df_clean = df.dropna(subset=all_factors + [TARGET]).reset_index(drop=True)
selected_features = run_factor_analysis(df_clean, all_factors, target_col=TARGET, output_dir="Results/factor")


# =====================
# model
# =====================

run_all_models(df, selected_features, target_col=TARGET, output_dir="Results/model")


# =====================
# backtest
# =====================

run_all_backtests(df, selected_features, target_col=TARGET, signal_filter=SIGNAL_FILTER, short_multiplier=SHORT_MULTIPLIER, long_leverage=LONG_LEVERAGE, short_leverage=SHORT_LEVERAGE, output_dir="Results/backtest")


# =====================
# report
# =====================

generate_report(
    target_col=TARGET,
    signal_filter=SIGNAL_FILTER,
    short_multiplier=SHORT_MULTIPLIER,
    output_path="Results/report.pdf",
)
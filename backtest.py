import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "Arial Unicode MS"

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import torch
import torch.nn as nn
from model import LSTMModel, make_sequences


# =====================
# 生成交易信号
# signal_filter=True 时：|y_pred| < 0.5*std 视为噪声 → 空仓(0)
# short_multiplier：做空门槛 = 做多门槛 × short_multiplier，越大越难做空
# =====================

def predictions_to_signals(y_pred, signal_filter=False, short_multiplier=1.0, long_leverage=1.0, short_leverage=1.0):
    if signal_filter:
        noise_band = 0.5 * np.std(y_pred)
        long_side  = y_pred >  noise_band
        short_side = y_pred < -(noise_band * short_multiplier)
        return np.where(long_side, long_leverage, np.where(short_side, -short_leverage, 0))
    return np.where(y_pred > 0, long_leverage, -short_leverage)


# =====================
# 回测引擎
# 用信号乘以实际次日收益，得到策略每期损益
# =====================

def run_backtest(signals, actual_returns, dates):
    strategy_returns = signals * actual_returns
    cumulative = (1 + strategy_returns).cumprod()
    benchmark = (1 + actual_returns).cumprod()
    return pd.DataFrame({
        "日期": dates,
        "策略收益": strategy_returns,
        "累计策略": cumulative,
        "累计基准": benchmark,
    }).set_index("日期")


# =====================
# 绩效指标
# =====================

def calc_performance(result_df, model_name):
    ret = result_df["策略收益"]
    annual_return = ret.mean() * 252
    annual_vol = ret.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else np.nan

    cumulative = result_df["累计策略"]
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    trading_days = ret[ret != 0]
    win_rate = (trading_days > 0).mean() if len(trading_days) > 0 else 0.0
    total_return = cumulative.iloc[-1] - 1

    return {
        "模型": model_name,
        "累计收益": round(total_return, 4),
        "年化收益": round(annual_return, 4),
        "年化波动": round(annual_vol, 4),
        "Sharpe": round(sharpe, 4),
        "最大回撤": round(max_drawdown, 4),
        "胜率": round(win_rate, 4),
    }


# =====================
# 各模型回测
# =====================

def split_train_test(df, selected_features, target_col="target_1d", test_ratio=0.3):
    df_model = df.dropna(subset=selected_features + [target_col]).reset_index(drop=True)
    split_idx = int(len(df_model) * (1 - test_ratio))
    return df_model.iloc[:split_idx], df_model.iloc[split_idx:]


def backtest_ridge(df, selected_features, target_col="target_1d", signal_filter=False, short_multiplier=1.0, long_leverage=1.0, short_leverage=1.0):
    train, test = split_train_test(df, selected_features, target_col)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[selected_features].values)
    X_test = scaler.transform(test[selected_features].values)

    model = Ridge(alpha=10.0)
    model.fit(X_train, train[target_col].values)
    y_pred = model.predict(X_test)

    signals = predictions_to_signals(y_pred, signal_filter, short_multiplier, long_leverage, short_leverage)
    return run_backtest(signals, test[target_col].values, test["日期"].values), y_pred


def backtest_logistic(df, selected_features, target_col="target_1d", signal_filter=False, short_multiplier=1.0, long_leverage=1.0, short_leverage=1.0):
    train, test = split_train_test(df, selected_features, target_col)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[selected_features].values)
    X_test = scaler.transform(test[selected_features].values)

    y_train_cls = (train[target_col].values > 0).astype(int)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train_cls)

    # 用概率衡量置信度：score = P(涨) - 0.5，范围 (-0.5, 0.5)
    proba = model.predict_proba(X_test)[:, 1]
    scores = proba - 0.5
    signals = predictions_to_signals(scores, signal_filter, short_multiplier, long_leverage, short_leverage)
    return run_backtest(signals, test[target_col].values, test["日期"].values), scores


def backtest_xgboost(df, selected_features, target_col="target_1d", signal_filter=False, short_multiplier=1.0, long_leverage=1.0, short_leverage=1.0):
    train, test = split_train_test(df, selected_features, target_col)

    X_tr_all = train[selected_features].values
    y_tr_all = train[target_col].values
    val_size = max(1, int(len(X_tr_all) * 0.1))
    X_tr, y_tr = X_tr_all[:-val_size], y_tr_all[:-val_size]
    X_val, y_val = X_tr_all[-val_size:], y_tr_all[-val_size:]

    model = XGBRegressor(
        n_estimators=500, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
        early_stopping_rounds=20,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict(test[selected_features].values)

    signals = predictions_to_signals(y_pred, signal_filter, short_multiplier, long_leverage, short_leverage)
    return run_backtest(signals, test[target_col].values, test["日期"].values), y_pred


def backtest_lstm(df, selected_features, target_col="target_1d", signal_filter=False, short_multiplier=1.0, long_leverage=1.0, short_leverage=1.0, lookback=10, epochs=100):
    train, test = split_train_test(df, selected_features, target_col)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[selected_features].values)
    X_test = scaler.transform(test[selected_features].values)

    X_train_seq, y_train_seq = make_sequences(X_train, train[target_col].values, lookback)
    X_test_seq, y_test_seq = make_sequences(X_test, test[target_col].values, lookback)

    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)

    model = LSTMModel(input_size=len(selected_features))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X_train_t), y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).numpy()

    signals = predictions_to_signals(y_pred, signal_filter, short_multiplier, long_leverage, short_leverage)
    return run_backtest(signals, y_test_seq, test["日期"].values[lookback:]), y_pred


# =====================
# 汇总运行所有回测
# =====================

def run_all_backtests(df, selected_features, target_col="target_1d", signal_filter=False, short_multiplier=1.0, long_leverage=1.0, short_leverage=1.0, output_dir="Results/backtest"):
    print("\n" + "=" * 50)
    print(f"策略回测（target: {target_col}，信号过滤: {'开' if signal_filter else '关'}，做空倍数: {short_multiplier}x，后30%为测试集）")
    print("=" * 50)

    ridge_result, _ = backtest_ridge(df, selected_features, target_col, signal_filter, short_multiplier, long_leverage, short_leverage)
    logistic_result, _ = backtest_logistic(df, selected_features, target_col, signal_filter, short_multiplier, long_leverage, short_leverage)
    xgb_result, _ = backtest_xgboost(df, selected_features, target_col, signal_filter, short_multiplier, long_leverage, short_leverage)
    lstm_result, _ = backtest_lstm(df, selected_features, target_col, signal_filter, short_multiplier, long_leverage, short_leverage)

    perf_ridge = calc_performance(ridge_result, "Ridge")
    perf_logistic = calc_performance(logistic_result, "Logistic")
    perf_xgb = calc_performance(xgb_result, "XGBoost")
    perf_lstm = calc_performance(lstm_result, "LSTM")

    perf_df = pd.DataFrame([perf_ridge, perf_logistic, perf_xgb, perf_lstm])
    print(perf_df.to_string(index=False))

    # =====================
    # 净值曲线图
    # =====================

    # =====================
    # Buy & Hold Benchmark
    # 测试集起始买入 IF 并持有到结束，不做任何择时
    # 用于衡量策略是否跑赢被动持有
    # =====================

    benchmark = xgb_result["累计基准"]

    _, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ridge_result["累计策略"].values, label="Ridge")
    ax.plot(logistic_result["累计策略"].values, label="Logistic")
    ax.plot(xgb_result["累计策略"].values, label="XGBoost")
    ax.plot(lstm_result["累计策略"].values, label="LSTM")
    ax.plot(benchmark.values, label="Buy & Hold", linestyle="--", color="gray")
    ax.axhline(1, color="black", linewidth=0.8)
    ax.set_title("策略净值曲线（测试集）")
    ax.set_ylabel("累计净值")
    ax.set_xlabel("交易日")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/backtest_curve.png", dpi=150)
    print(f"\n净值曲线已保存至 {output_dir}/backtest_curve.png")
    plt.show()

    # =====================
    # 保存结果
    # =====================

    perf_df.to_csv(f"{output_dir}/backtest_performance.csv", index=False)
    ridge_result.to_csv(f"{output_dir}/ridge_daily_returns.csv")
    logistic_result.to_csv(f"{output_dir}/logistic_daily_returns.csv")
    xgb_result.to_csv(f"{output_dir}/xgb_daily_returns.csv")
    lstm_result.to_csv(f"{output_dir}/lstm_daily_returns.csv")
    print(f"绩效报告已保存至 {output_dir}/backtest_performance.csv")

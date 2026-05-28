import numpy as np
import pandas as pd

from factors import build_factors, get_factor_columns
from factor_analysis import run_factor_analysis
from backtest import (split_train_test, predictions_to_signals,
                      run_backtest, calc_performance)

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# =====================
# 参数扫描范围
# =====================

PARAM_RANGE     = np.arange(1.0, 3.5, 0.5)   # 1.0, 1.5, 2.0, 2.5, 3.0
SIGNAL_FILTER   = True
TARGET          = "target_1d"
OUTPUT_PATH     = "Results/optimize_results.csv"


# =====================
# 加载数据 & 因子
# =====================

df = pd.read_csv("data/cleaned_data.csv")
df = build_factors(df)

price_volume_factors, fundamental_factors, interaction_factors = get_factor_columns()
all_factors = [f for f in price_volume_factors + fundamental_factors + interaction_factors if f in df.columns]
df_clean = df.dropna(subset=all_factors + [TARGET]).reset_index(drop=True)

selected_features = run_factor_analysis(df_clean, all_factors, target_col=TARGET, output_dir="Results/factor")
print(f"\n入选因子: {selected_features}\n")


# =====================
# 训练各模型，保存预测值（只训练一次）
# =====================

def get_predictions():
    predictions = {}

    train, test = split_train_test(df, selected_features, TARGET)
    actual = test[TARGET].values
    dates  = test["日期"].values

    print("  训练 Ridge...")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[selected_features].values)
    X_te = scaler.transform(test[selected_features].values)
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_tr, train[TARGET].values)
    predictions["Ridge"] = (ridge.predict(X_te), actual, dates)
    print("  Ridge 完成")

    print("  训练 Logistic...")
    y_cls = (train[TARGET].values > 0).astype(int)
    scaler2 = StandardScaler()
    X_tr2 = scaler2.fit_transform(train[selected_features].values)
    X_te2 = scaler2.transform(test[selected_features].values)
    logit = LogisticRegression(max_iter=1000, class_weight="balanced")
    logit.fit(X_tr2, y_cls)
    scores = logit.predict_proba(X_te2)[:, 1] - 0.5
    predictions["Logistic"] = (scores, actual, dates)
    print("  Logistic 完成")

    print("  训练 XGBoost...")
    val_size = max(1, int(len(train) * 0.1))
    X_all = train[selected_features].values
    y_all = train[TARGET].values
    xgb = XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, random_state=42,
                       verbosity=0, early_stopping_rounds=20)
    xgb.fit(X_all[:-val_size], y_all[:-val_size],
            eval_set=[(X_all[-val_size:], y_all[-val_size:])], verbose=False)
    predictions["XGBoost"] = (xgb.predict(test[selected_features].values), actual, dates)
    print("  XGBoost 完成")

    return predictions


print("训练模型中（只训练一次）...")
predictions = get_predictions()
print("训练完成，开始参数扫描...\n")


# =====================
# 参数网格扫描
# =====================

rows = []
total = len(PARAM_RANGE) ** 3
done  = 0

for sm in PARAM_RANGE:
    for ll in PARAM_RANGE:
        for sl in PARAM_RANGE:
            for model_name, (y_pred, actual, dates) in predictions.items():
                signals = predictions_to_signals(y_pred, SIGNAL_FILTER, sm, ll, sl)
                result  = run_backtest(signals, actual, dates)
                perf    = calc_performance(result, model_name)
                rows.append({
                    "模型":             model_name,
                    "short_multiplier": sm,
                    "long_leverage":    ll,
                    "short_leverage":   sl,
                    "累计收益":          perf["累计收益"],
                    "年化收益":          perf["年化收益"],
                    "Sharpe":           perf["Sharpe"],
                    "最大回撤":          perf["最大回撤"],
                    "胜率":             perf["胜率"],
                })
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{total} 组合完成")

results_df = pd.DataFrame(rows)
results_df = results_df.sort_values("Sharpe", ascending=False)
results_df.to_csv(OUTPUT_PATH, index=False)

print(f"\n扫描完成，共 {len(results_df)} 条结果")
print(f"结果已保存至 {OUTPUT_PATH}")
print("\n=== 各模型最优参数（按 Sharpe）===")
for model_name in ["Ridge", "Logistic", "XGBoost"]:
    best = results_df[results_df["模型"] == model_name].iloc[0]
    print(f"\n{model_name}:")
    print(f"  short_multiplier={best['short_multiplier']}  "
          f"long_leverage={best['long_leverage']}  "
          f"short_leverage={best['short_leverage']}")
    print(f"  Sharpe={best['Sharpe']:.4f}  "
          f"累计收益={best['累计收益']:.2%}  "
          f"最大回撤={best['最大回撤']:.2%}")

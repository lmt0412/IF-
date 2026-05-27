import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor


# =====================
# 工具函数
# =====================

def train_test_split(df, selected_features, target_col="target_1d", test_ratio=0.3):
    df_model = df.dropna(subset=selected_features + [target_col]).reset_index(drop=True)
    split_idx = int(len(df_model) * (1 - test_ratio))
    train = df_model.iloc[:split_idx]
    test = df_model.iloc[split_idx:]
    X_train = train[selected_features].values
    y_train = train[target_col].values
    X_test = test[selected_features].values
    y_test = test[target_col].values
    return X_train, y_train, X_test, y_test


def evaluate(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    direction_acc = np.mean(np.sign(y_pred) == np.sign(y_true))
    ic = pd.Series(y_pred).corr(pd.Series(y_true), method="spearman")
    return {
        "模型": model_name,
        "RMSE": round(rmse, 6),
        "MAE": round(mae, 6),
        "方向准确率": round(direction_acc, 4),
        "IC": round(ic, 4),
    }


# =====================
# Ridge 回归
# =====================

def run_ridge(df, selected_features, target_col="target_1d"):
    X_train, y_train, X_test, y_test = train_test_split(df, selected_features, target_col)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = Ridge(alpha=10.0)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    result = evaluate(y_test, y_pred, "Ridge")
    print(f"[Ridge] RMSE={result['RMSE']}  方向准确率={result['方向准确率']}  IC={result['IC']}")
    return result, y_pred, y_test


# =====================
# Logistic Regression
# 直接预测涨跌方向，target 转为二分类
# =====================

def run_logistic(df, selected_features, target_col="target_1d"):
    X_train, y_train, X_test, y_test = train_test_split(df, selected_features, target_col)

    # 转为二分类标签：> 0 涨=1，≤ 0 跌=0
    y_train_cls = (y_train > 0).astype(int)
    y_test_cls = (y_test > 0).astype(int)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_s, y_train_cls)
    y_pred_cls = model.predict(X_test_s)

    direction_acc = np.mean(y_pred_cls == y_test_cls)
    ic = pd.Series(y_pred_cls.astype(float)).corr(pd.Series(y_test_cls.astype(float)), method="spearman")

    result = {
        "模型": "Logistic",
        "RMSE": "-",
        "MAE": "-",
        "方向准确率": round(direction_acc, 4),
        "IC": round(ic, 4),
    }
    print(f"[Logistic] 方向准确率={result['方向准确率']}  IC={result['IC']}")
    return result, y_pred_cls, y_test_cls


# =====================
# XGBoost
# =====================

def run_xgboost(df, selected_features, target_col="target_1d"):
    X_train, y_train, X_test, y_test = train_test_split(df, selected_features, target_col)

    # 从训练集末尾切10%做验证集，用于 early stopping
    val_size = max(1, int(len(X_train) * 0.1))
    X_tr, y_tr = X_train[:-val_size], y_train[:-val_size]
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]

    model = XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        early_stopping_rounds=20,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict(X_test)

    result = evaluate(y_test, y_pred, "XGBoost")
    print(f"[XGBoost] RMSE={result['RMSE']}  方向准确率={result['方向准确率']}  IC={result['IC']}")
    return result, y_pred, y_test


# =====================
# LSTM
# =====================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)


def make_sequences(X, y, lookback=20):
    xs, ys = [], []
    for i in range(lookback, len(X)):
        xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(xs), np.array(ys)


def run_lstm(df, selected_features, target_col="target_1d", lookback=10, epochs=100):
    X_train, y_train, X_test, y_test = train_test_split(df, selected_features, target_col)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_train_seq, y_train_seq = make_sequences(X_train_s, y_train, lookback)
    X_test_seq, y_test_seq = make_sequences(X_test_s, y_test, lookback)

    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)

    model = LSTMModel(input_size=len(selected_features))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t).numpy()

    result = evaluate(y_test_seq, y_pred, "LSTM")
    print(f"[LSTM]  RMSE={result['RMSE']}  方向准确率={result['方向准确率']}  IC={result['IC']}")
    return result, y_pred, y_test_seq


# =====================
# 汇总运行所有模型
# =====================

def run_all_models(df, selected_features, target_col="target_1d", output_dir="Results"):
    print("\n" + "=" * 50)
    print(f"模型训练与评估（target: {target_col}）")
    print(f"使用因子：{selected_features}")
    print("=" * 50)

    results = []
    ridge_result, _, _ = run_ridge(df, selected_features, target_col)
    logistic_result, _, _ = run_logistic(df, selected_features, target_col)
    xgb_result, _, _ = run_xgboost(df, selected_features, target_col)
    lstm_result, _, _ = run_lstm(df, selected_features, target_col)

    results = [ridge_result, logistic_result, xgb_result, lstm_result]
    comparison_df = pd.DataFrame(results)

    print("\n" + "=" * 50)
    print("模型对比")
    print("=" * 50)
    print(comparison_df.to_string(index=False))

    comparison_df.to_csv(f"{output_dir}/model_comparison.csv", index=False)
    print(f"\n对比结果已保存至 {output_dir}/model_comparison.csv")

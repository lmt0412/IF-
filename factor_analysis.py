import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "Arial Unicode MS"

from factors import build_factors, get_factor_columns


# =====================
# load & build factors
# =====================

df = pd.read_csv("data/cleaned_data.csv")
df = build_factors(df)

price_volume_factors, fundamental_factors, interaction_factors = get_factor_columns()
all_factors = price_volume_factors + fundamental_factors + interaction_factors

# 因子计算造成了新的缺失值, 直接去掉带有NaN的行
df_clean = df.dropna(subset=all_factors + ["target_5d"]).reset_index(drop=True)


# =====================
# IC 分析
# IC = 每期因子值与未来收益的 Spearman 相关系数
# =====================

def calc_ic_series(df, factor_col, target_col="target_1d"):
    """逐行滚动计算不适合时序数据，这里直接用截面IC（单资产每日点的rank相关）"""
    # 对于单标的时序数据，IC = 因子值与未来收益的 rank 相关（rolling window）
    window = 60
    ic_list = []
    dates = df["日期"].values

    for i in range(window, len(df)):
        window_df = df.iloc[i - window:i]
        corr, _ = stats.spearmanr(window_df[factor_col], window_df[target_col])
        ic_list.append({"日期": dates[i], "IC": corr})

    return pd.DataFrame(ic_list).set_index("日期")


def calc_all_ic(df, factors, target_col="target_1d"):
    summary = []
    ic_series_dict = {}

    for f in factors:
        ic_df = calc_ic_series(df, f, target_col)
        ic_series_dict[f] = ic_df["IC"]

        ic_mean = ic_df["IC"].mean()
        ic_std = ic_df["IC"].std()
        icir = ic_mean / ic_std if ic_std > 0 else np.nan
        ic_positive_rate = (ic_df["IC"] > 0).mean()

        n = len(ic_df)
        t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else np.nan
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)) if not np.isnan(t_stat) else np.nan

        summary.append({
            "因子": f,
            "IC均值": round(ic_mean, 4),
            "IC标准差": round(ic_std, 4),
            "ICIR": round(icir, 4),
            "IC>0占比": round(ic_positive_rate, 4),
            "t统计量": round(t_stat, 4),
            "p值": round(p_value, 4),
        })

    summary_df = pd.DataFrame(summary).sort_values("ICIR", key=abs, ascending=False)
    return summary_df, ic_series_dict


# =====================
# 因子相关性矩阵（检验多重共线性）
# =====================

def calc_factor_correlation(df, factors):
    return df[factors].corr(method="spearman").round(3)


# =====================
# 单因子分组收益（五分组）
# =====================

def calc_quintile_returns(df, factor_col, target_col="target_1d", n_groups=5):
    df = df.copy().dropna(subset=[factor_col, target_col])
    df["group"] = pd.qcut(df[factor_col], q=n_groups, labels=False, duplicates="drop")
    group_ret = df.groupby("group")[target_col].mean()
    return group_ret


# =====================
# 因子自动筛选
# =====================

def select_factors(summary_df, corr_matrix, icir_threshold=0.3, p_threshold=0.05, corr_threshold=0.7):
    # 第一步：按 ICIR 和 p 值过滤
    candidates = summary_df[
        (summary_df["ICIR"].abs() >= icir_threshold) &
        (summary_df["p值"] < p_threshold)
    ].copy()
    candidates = candidates.sort_values("ICIR", key=abs, ascending=False)

    # 第二步：贪心去共线性，按 |ICIR| 从强到弱遍历，若与已选因子相关性 > threshold 则跳过
    selected = []
    for factor in candidates["因子"]:
        too_correlated = any(
            abs(corr_matrix.loc[factor, kept]) > corr_threshold
            for kept in selected
        )
        if not too_correlated:
            selected.append(factor)

    return selected


# =====================
# 主流程
# =====================

def run_factor_analysis(df_clean, all_factors, target_col="target_1d", output_dir="Results"):
    print("=" * 50)
    print(f"因子 IC / ICIR 分析（target: {target_col}）")
    print("=" * 50)

    summary_df, ic_series_dict = calc_all_ic(df_clean, all_factors, target_col)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 50)
    print("因子 Spearman 相关性矩阵")
    print("=" * 50)

    corr_matrix = calc_factor_correlation(df_clean, all_factors)
    print(corr_matrix)

    print("\n" + "=" * 50)
    print("各因子五分组平均收益（多头 - 空头 = 多空收益）")
    print("=" * 50)

    quintile_rows = []
    for f in all_factors:
        qret = calc_quintile_returns(df_clean, f, target_col)
        spread = qret.iloc[-1] - qret.iloc[0]
        print(f"{f:20s}  组1={qret.iloc[0]:.4f}  组5={qret.iloc[-1]:.4f}  多空={spread:.4f}")
        quintile_rows.append({
            "因子": f,
            "组1均收益": round(qret.iloc[0], 4),
            "组5均收益": round(qret.iloc[-1], 4),
            "多空收益": round(spread, 4),
        })
    quintile_df = pd.DataFrame(quintile_rows)

    # =====================
    # 保存结果到 CSV
    # =====================

    selected = select_factors(summary_df, corr_matrix)
    print("\n" + "=" * 50)
    print("自动筛选结果")
    print("=" * 50)
    print("入选因子：", selected)

    summary_df.to_csv(f"{output_dir}/factor_ic_summary.csv", index=False)
    corr_matrix.to_csv(f"{output_dir}/factor_correlation.csv")
    quintile_df.to_csv(f"{output_dir}/factor_quintile_returns.csv", index=False)
    pd.DataFrame({"selected_features": selected}).to_csv(f"{output_dir}/selected_features.csv", index=False)
    print(f"\n结果已保存至 {output_dir}/")

    # =====================
    # 可视化
    # =====================

    _, axes = plt.subplots(2, 1, figsize=(12, 10))

    ax1 = axes[0]
    x = range(len(summary_df))
    ax1.bar(x, summary_df["IC均值"], color=["steelblue" if v > 0 else "tomato" for v in summary_df["IC均值"]])
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(summary_df["因子"], rotation=45, ha="right", fontsize=9)
    ax1.set_title("因子 IC 均值（按 |ICIR| 排序）")
    ax1.set_ylabel("IC 均值")

    ax2 = axes[1]
    top_factors = summary_df.head(3)["因子"].tolist()
    for f in top_factors:
        cumIC = ic_series_dict[f].cumsum()
        ax2.plot(cumIC.values, label=f)
    ax2.set_title("Top3 因子 IC 累积和")
    ax2.set_ylabel("累积 IC")
    ax2.legend()
    ax2.axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/factor_analysis.png", dpi=150)
    print(f"图表已保存至 {output_dir}/factor_analysis.png")
    plt.close()

    return selected


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_data.csv")
    df = build_factors(df)

    price_volume_factors, fundamental_factors, interaction_factors = get_factor_columns()
    all_factors = price_volume_factors + fundamental_factors + interaction_factors

    TARGET = "target_1d"
    df_clean = df.dropna(subset=all_factors + [TARGET]).reset_index(drop=True)

    run_factor_analysis(df_clean, all_factors, target_col=TARGET)

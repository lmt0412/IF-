import pandas as pd

from factors import build_factors


# =====================
# load data
# =====================

df = pd.read_csv("data/cleaned_data.csv")


# =====================
# build factors
# =====================

df = build_factors(df)


# =====================
# feature list
# =====================

features = [
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
    "pe_factor",
    "pe_z20",
    "pe_change",
    "pe_rank_60"
]


# =====================
# print basic info
# =====================

print(df_model.head())

print("\n")
print(df_model.columns)

print("\n")
print(df_model[features + ["target_5d"]].head())


# =====================
# train test split
# =====================

split_idx = int(len(df_model) * 0.7)

train_df = df_model.iloc[:split_idx]
test_df = df_model.iloc[split_idx:]


print("\n")
print("train size:", len(train_df))
print("test size:", len(test_df))
import pandas as pd
from xgboost import XGBRegressor
import joblib

# === Load dataset ===
url_data5 = "https://docs.google.com/spreadsheets/d/16cyvXwvucVb7EM1qiikZpbK8J8isbktuiw-MR1EJDEY/export?format=csv&gid=2075790964"
df5 = pd.read_csv(url_data5)
df5.columns = df5.columns.str.strip()

# Pastikan kolom tanggal valid
df5 = df5.dropna(subset=["TGL BAYAR"]).copy()
df5["TGL BAYAR"] = pd.to_datetime(df5["TGL BAYAR"], errors="coerce")
df5 = df5.dropna(subset=["TGL BAYAR"])
df5["Tanggal"] = df5["TGL BAYAR"].dt.normalize()

# ==================================================
# ===== Model Harian (model_daily.pkl) =============
# ==================================================
daily = df5.groupby("Tanggal")["No"].nunique().rename("y").reset_index()
s = daily.set_index("Tanggal")["y"].asfreq("D").fillna(0)

df_feat = s.to_frame().reset_index().rename(columns={"Tanggal": "ds", "y": "y"})
df_feat["dayofweek"] = df_feat["ds"].dt.dayofweek
df_feat["month"] = df_feat["ds"].dt.month
for L in [1, 2, 3, 7]:
    df_feat[f"lag{L}"] = df_feat["y"].shift(L)
df_feat = df_feat.dropna().reset_index(drop=True)

FEATURES = ["dayofweek", "month", "lag1", "lag2", "lag3", "lag7"]

model_daily = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
model_daily.fit(df_feat[FEATURES], df_feat["y"])

# 🔹 Simpan model harian
joblib.dump(model_daily, "model_daily.pkl")
print("✅ Model harian disimpan ke model_daily.pkl")

# ==================================================
# ===== Model Bulanan (model_monthly.pkl) ==========
# ==================================================
monthly = df5.groupby(df5["Tanggal"].dt.to_period("M"))["No"].nunique().rename("y").reset_index()
monthly["Tanggal"] = monthly["Tanggal"].dt.to_timestamp()

df_month = monthly.copy()
df_month["month"] = df_month["Tanggal"].dt.month
df_month["year"] = df_month["Tanggal"].dt.year
for L in [1, 2, 3, 6, 12]:
    df_month[f"lag{L}"] = df_month["y"].shift(L)
df_month = df_month.dropna().reset_index(drop=True)

FEATURES_MONTHLY = ["month", "year", "lag1", "lag2", "lag3", "lag6", "lag12"]

model_monthly = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
model_monthly.fit(df_month[FEATURES_MONTHLY], df_month["y"])

# 🔹 Simpan model bulanan
joblib.dump(model_monthly, "model_monthly.pkl")
print("✅ Model bulanan disimpan ke model_monthly.pkl")

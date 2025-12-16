import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

HOUSE = int(input("0 ~ 380: "))
HOUSE_COL = f"MT_{HOUSE:03d}"

CSV_PATH = "benchmark/test/LD2011_2014_converted.csv"

# 데이터 불러오기
df = pd.read_csv(CSV_PATH)
df["Datetime"] = pd.to_datetime(df["Datetime"])

# Prophet 입력(ds,y) 만들기
ts = df[["Datetime", HOUSE_COL]].rename(columns={"Datetime": "ds", HOUSE_COL: "y"}).sort_values("ds")
ts["y"] = pd.to_numeric(ts["y"], errors="coerce")

# 15분 그리드 정리 + 결측 보간
ts = ts.set_index("ds").resample("15min").mean().reset_index()
ts["y"] = ts["y"].interpolate(limit_direction="both")

# val split
split = int(len(ts) * 0.9)
train = ts.iloc[:split]
val   = ts.iloc[split:]

# 모델 학습
m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
m.fit(train)

# validation 구간 예측
future = m.make_future_dataframe(periods=len(val), freq="15min", include_history=False)
fc = m.predict(future)[["ds", "yhat"]]

# 정답 vs 예측 (validation 기준)
y_true = val["y"].to_numpy()
y_pred = fc["yhat"].to_numpy()
t = np.arange(len(y_true))

# 300스텝만 보기
N = 14256
t = t[:N]
y_true = y_true[:N]
y_pred = y_pred[:N]

# 간격 조절
STEP = 1
t = t[::STEP]
y_true = y_true[::STEP]
y_pred = y_pred[::STEP]

plt.figure(figsize=(8,5))
plt.scatter(t, y_true, s=20, c="red", label="ground truth")
plt.scatter(t, y_pred, s=20, c="blue", label="prediction")
plt.xlabel("Time (validation steps)")
plt.ylabel("Value")
plt.title(f"{HOUSE_COL}: Ground Truth (red) vs Prophet (blue)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"imgs/{HOUSE}-household_prophet_comparison.png")
plt.show()

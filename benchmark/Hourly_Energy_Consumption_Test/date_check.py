import pandas as pd

INPUT_CSV = "benchmark/Electricity_Consumption_Prediction_Test/hourly_energy_consumption_combined.csv"

df = pd.read_csv(INPUT_CSV, parse_dates=["Datetime"])
df = df.set_index("Datetime").sort_index()

# 시간당 변화량
diff = df.diff()

# 컬럼별 threshold
threshold = diff.std() * 6

outlier_diff = diff.abs() > threshold

# 결과 확인
print(outlier_diff.sum().sort_values(ascending=False))

col = "NI"
window = 3  # 전후 몇 시간 볼지

# DEOK에서 튀는 지점 Datetime
outlier_times = df.index[outlier_diff[col]]
# df.loc["20011-10-20 14:00:00", "AEP"] = pd.NA

print(f"Total outliers in {col}: {len(outlier_times)}")

# 앞에서 몇 개만 확인 (전체 보려면 [:]로)
for t in outlier_times[:5]:
    print("\n--- Outlier around:", t, "---")
    print(
        df.loc[
            t - pd.Timedelta(hours=window) :
            t + pd.Timedelta(hours=window),
            [col]
        ]
    )



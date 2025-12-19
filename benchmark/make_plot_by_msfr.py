import torch
import matplotlib.pyplot as plt
import numpy as np
import sys, os, time
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # 난 이거 안쓰면 안돌아감
from benchmark.Electricity_Consumption_Prediction_Test.train_msfr import load_dataset, train_val_split, TestModel


HOUSE = int(input("0 ~ 369: "))  # 귀찮아서 인풋으로 바꿈

CKPT_PATH = "./model/msfr_fixed.ckpt"
CSV_PATH = "benchmark/test/LD2011_2014_converted.csv"

# 데이터 불러오기
X, y, mean, std= load_dataset(CSV_PATH)
(X_tr, y_tr), (X_val, y_val) = train_val_split(X, y, val_ratio=0.1)


start = time.time()
# 모델 준비
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = TestModel(input_dim = X_tr.shape[1], output_dim = y_tr.shape[1], n_harmonics = 12).to(device)

# 주기 초기화 및 가중치 로드
with torch.no_grad():
    init_cycles = torch.tensor([96.0, 96.0 * 7.0, 96.0 * 365.0, 1.0], dtype = torch.float32, device = device)
    model.msfr.cycle.copy_(init_cycles)
state = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

# 예측
X_val, y_val = X_val.to(device), y_val.to(device)
with torch.no_grad():
    pred_val = model(X_val)

# 빨강(정답) vs 파랑(예측)
y_true = y_val[:, HOUSE].detach().cpu().numpy()
y_pred = pred_val[:, HOUSE].detach().cpu().numpy()
t = np.arange(len(y_true))

N = 14256
t = t[:N]
y_true = y_true[:N]
y_pred = y_pred[:N]

#간격조절
STEP = 1  # 1이면 전부, 2면 절반, 5면 1/5만 표시
t = t[::STEP]
y_true = y_true[::STEP]
y_pred = y_pred[::STEP]

WINDOW = 96 * 7

true_trend = (
    pd.Series(y_true)
    .rolling(WINDOW, center=True)
    .mean()
    .to_numpy()
)

pred_trend = (
    pd.Series(y_pred)
    .rolling(WINDOW, center=True)
    .mean()
    .to_numpy()
)

# ===== 그래프 =====
plt.figure(figsize=(10,6))
# 원래 산점도 (연하게)
plt.scatter(t, y_true, s=10, c="red", alpha=0.3, label="ground truth")
plt.scatter(t, y_pred, s=10, c="blue", alpha=0.3, label="prediction")
# 추세선
plt.plot(t, true_trend, c="darkred", linewidth=2.5, label="true trend (7-day MA)")
plt.plot(t, pred_trend, c="navy", linewidth=2.5, label="predicted trend (7-day MA)")
plt.xlabel("Time (validation steps)")
plt.ylabel("Value")
plt.title(f"Household {HOUSE}: Prediction with Trend")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f"imgs/{HOUSE}-household_comparison_with_trend.png")
plt.show()
# 와 새롭게 안 사실
# savefig는 show보다 먼저 호출해야함. 무조건!
# how to run: python -m benchmark.Hourly_Energy_Consumption_Test.diagnose_msfr

import os
import math
import torch
import math
import torch
from typing import Dict
from benchmark.Hourly_Energy_Consumption_Test.train_msfr import load_datasets, train_val_split, TestModel

CSV_PATH   = "benchmark/Hourly_Energy_Consumption_Test/hourly_energy_consumption_combined.csv"
CKPT_PATH = "./model/msfr_new.ckpt"

N_HARMONICS = 6
VAL_RATIO   = 0.2
SEASONAL_LAG = 24   # hourly → 하루 전 값

# 학습 코드에서 조건부 손실 적용시켰으니 여기서도 해야지 nan이 안 뜸
def masked_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    mask = torch.isfinite(y_true) & torch.isfinite(y_pred)
    valid_ratio = mask.float().mean().item()

    if mask.sum().item() == 0:
        return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "valid_ratio": 0.0}

    diff = y_pred - y_true
    diff = torch.where(mask, diff, torch.zeros_like(diff))

    mse = (diff * diff).sum() / mask.sum()
    mae = torch.abs(diff).sum() / mask.sum()
    rmse = torch.sqrt(mse)

    return {
        "mse": mse.item(),
        "rmse": rmse.item(),
        "mae": mae.item(),
        "valid_ratio": valid_ratio,
    }

def inverse_normalize(y: torch.Tensor, mean, std) -> torch.Tensor:
    if not torch.is_tensor(mean):
        mean = torch.from_numpy(mean)
    if not torch.is_tensor(std):
        std = torch.from_numpy(std)

    mean = mean.to(y.device)
    std  = std.to(y.device)
    return y * std + mean

def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    mse  = torch.mean((y_pred - y_true) ** 2).item()
    rmse = math.sqrt(mse)
    mae  = torch.mean(torch.abs(y_pred - y_true)).item()
    return {"mse": mse, "rmse": rmse, "mae": mae}

def mean_baseline(y_train: torch.Tensor, n_val: int) -> torch.Tensor:
    mask = torch.isfinite(y_train)
    den = mask.sum(dim=0).clamp(min=1)
    y_sum = torch.where(mask, y_train, torch.zeros_like(y_train)).sum(dim=0)
    mean_per_col = (y_sum / den).unsqueeze(0)  # (1, C)
    return mean_per_col.repeat(n_val, 1)

def seasonal_naive_baseline(y_all: torch.Tensor, split: int, lag: int) -> torch.Tensor:
    if split - lag < 0:
        raise ValueError("Not enough history for seasonal naive baseline")
    n_val = y_all.shape[0] - split
    return y_all[split - lag : split - lag + n_val]

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"cannot find data file: {CSV_PATH}")

    # Datetime -> X
    # 12개 컬럼 -> y
    X, y, mean, std = load_datasets(CSV_PATH)
    (X_tr, y_tr), (X_val, y_val) = train_val_split(X, y, val_ratio = VAL_RATIO)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    model = None
    if CKPT_PATH is not None:
        model = TestModel(input_dim = X_tr.shape[1], output_dim = y_tr.shape[1], n_harmonics = N_HARMONICS).to(device)
        with torch.no_grad():
            init_cycles = torch.tensor([24.0, 24.0 * 7.0, 24.0 * 365.0], dtype = torch.float32, device=device)
            model.msfr.cycle.data = init_cycles
        state = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(state)
        model.eval()
    else:
        raise ValueError("The variable 'CKPT_PATH' is not defined. you need to have a pretrained model to evaluate.")

    # 모델 예측 및 핵심 메트릭
    model.eval()
    with torch.no_grad():
        pred_val = model(X_val)
        pred_val_real = inverse_normalize(pred_val, mean, std)
        y_val_real   = inverse_normalize(y_val, mean, std)
        msfr_real = masked_metrics(y_val_real, pred_val_real)

        mean_pred = mean_baseline(y_tr, y_val.shape[0]).to(device)

        split = X_tr.shape[0]
        seasonal_pred = seasonal_naive_baseline(
            torch.cat([y_tr, y_val], dim=0),
            split,
            SEASONAL_LAG
        ).to(device)

        mean_pred_real = inverse_normalize(mean_pred, mean, std)
        seasonal_pred_real = inverse_normalize(seasonal_pred, mean, std)

        mean_real = masked_metrics(y_val_real, mean_pred_real)
        seasonal_real = masked_metrics(y_val_real, seasonal_pred_real)

    print()
    print(f"[MSFR] RMSE={msfr_real['rmse']:.3f}, MAE={msfr_real['mae']:.3f}, valid={msfr_real['valid_ratio']:.6f}")
    print(f"[Baseline Mean] RMSE={mean_real['rmse']:.3f}, MAE={mean_real['mae']:.3f}, valid={mean_real['valid_ratio']:.6f}")
    print(f"[Baseline Seasonal] RMSE={seasonal_real['rmse']:.3f}, MAE={seasonal_real['mae']:.3f}, valid={seasonal_real['valid_ratio']:.6f}")
    print()

if __name__ == "__main__":
    main()

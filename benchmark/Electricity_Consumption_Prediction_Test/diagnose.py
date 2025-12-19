# how to run: python -m benchmark.Electricity_Consumption_Prediction_Test.diagnose
import os
import math
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict
from benchmark.Electricity_Consumption_Prediction_Test.train_msfr import load_dataset, train_val_split, TestModel

"""
comparison model: MSFR (Multi-Seasonal Fourier Regression), 단순 평균 베이스라인(mean), 계절성 나이브 베이스라인(seasonal naive)
dataset: LD2011_2014_converted.csv (15-minute power consumption data)
Evaluation Metrics: RMSE, MAE -> huber -> revert to (rmse, mae)
"""

CSV_PATH = "benchmark/test/LD2011_2014_converted.csv"
N_HARMONICS = 12
VAL_RATIO = 0.2
SEASONAL_LAG = 96
CKPT_PATH = "./model/msfr_fixed.ckpt"

# 역정규화 함수(안하면 우리가 읽을 수가 없음)
def inverse_normalize(y: torch.Tensor, mean, std) -> torch.Tensor:
    """
    y   : (N, H) normalized torch tensor
    mean: (H,) numpy or torch
    std : (H,) numpy or torch
    """
    if not torch.is_tensor(mean):
        mean = torch.from_numpy(mean)
    if not torch.is_tensor(std):
        std = torch.from_numpy(std)

    mean = mean.to(y.device)
    std = std.to(y.device)

    return y * std + mean

def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    mse = torch.mean((y_pred - y_true) ** 2).item()
    rmse = math.sqrt(mse)
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    return {"mse": mse, "rmse": rmse, "mae": mae}

def per_household_rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # (N, H) -> 각 가구별 RMSE(H,)
    se = (y_pred - y_true) ** 2  # (N, H)
    rmse_h = torch.sqrt(torch.mean(se, dim = 0))  # (H,)
    return rmse_h

def mean_baseline(y_train: torch.Tensor, n_val: int) -> torch.Tensor:
    # 가구별 학습 평균을 그대로 예측
    mean_per_house = torch.mean(y_train, dim = 0, keepdim = True)  # (1, H)
    return mean_per_house.repeat(n_val, 1)  # (N_val, H)

def seasonal_naive_baseline(y_all: torch.Tensor, split: int, lag: int) -> torch.Tensor:
    # 검증 구간의 각 시점은 과거 동일한 계절(lag) 앞의 값을 그대로 복사해서 예측한다.
    # 유효하려면 split - lag >= 0 이어야 함
    n_val = y_all.shape[0] - split
    if split - lag < 0:
        raise ValueError(f"seasonal_naive_baseline: split({split}) - lag({lag}) < 0 으로 사용할 수 없습니다.")
    start = split - lag
    end = start + n_val
    return y_all[start:end]  # (N_val, H)

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"cannot find data file: {CSV_PATH}")

    X, y, mean, std = load_dataset(CSV_PATH)
    (X_tr, y_tr), (X_val, y_val) = train_val_split(X, y, val_ratio=VAL_RATIO)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    input_dim = X_tr.shape[1]
    output_dim = y_tr.shape[1]

    model = None
    if CKPT_PATH is not None:
        # 체크포인트 로드하여 평가만 수행
        model = TestModel(input_dim = input_dim, output_dim = output_dim, n_harmonics = N_HARMONICS).to(device)
        with torch.no_grad():
            init_cycles = torch.tensor([96.0, 96.0 * 7.0, 96.0 * 365.0], dtype = torch.float32, device=device)
            model.msfr.cycle.data = init_cycles
        state = torch.load(CKPT_PATH, map_location = device)
        model.load_state_dict(state)
        model.eval()
    else:
        raise ValueError("The variable 'CKPT_PATH' is not defined. you need to have a pretrained model to evaluate.")
    
    # 모델 예측 및 핵심 메트릭
    model.eval()
    with torch.no_grad():
        pred_tr = model(X_tr)
        pred_val = model(X_val)

    y_tr_real   = inverse_normalize(y_tr, mean, std)
    pred_tr_real = inverse_normalize(pred_tr, mean, std)

    y_val_real   = inverse_normalize(y_val, mean, std)
    pred_val_real = inverse_normalize(pred_val, mean, std)

    train_metrics = compute_metrics(y_tr_real, pred_tr_real)
    val_metrics   = compute_metrics(y_val_real, pred_val_real)
    rmse_h = per_household_rmse(y_val_real, pred_val_real).detach().cpu()


    print(f"model | train RMSE = {train_metrics['rmse']:.3f}, val RMSE = {val_metrics['rmse']:.3f}, val MAE = {val_metrics['mae']:.3f}")
    print(f"per-household RMSE | mean = {rmse_h.mean().item():.3f}, median = {rmse_h.median().item():.3f}, min = {rmse_h.min().item():.3f}, max = {rmse_h.max().item():.3f}")

    # 베이스라인 비교
    with torch.no_grad():
        mean_pred = mean_baseline(y_tr, n_val=y_val.shape[0])
        mean_pred = mean_pred.to(device)
        mean_metrics = compute_metrics(y_val, mean_pred)
    print(f"baseline(mean) | val RMSE = {mean_metrics['rmse']:.3f}, val MAE = {mean_metrics['mae']:.3f}")

    try:
        split = X_tr.shape[0]
        seasonal_pred = seasonal_naive_baseline(torch.cat([y_tr, y_val], dim=0), split=split, lag=SEASONAL_LAG)
        seasonal_pred = seasonal_pred.to(device)
        seasonal_metrics = compute_metrics(y_val, seasonal_pred)
        print(f"baseline(seasonal naive, lag = {SEASONAL_LAG}) | val RMSE = {seasonal_metrics['rmse']:.3f}, val MAE = {seasonal_metrics['mae']:.3f}")
    except ValueError as e:
        print(f"skipped baseline(seasonal lag): {e}")

if __name__ == "__main__":
    main()
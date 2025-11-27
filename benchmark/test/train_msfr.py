#how to run : At root directory -> python -m benchmark.test.train_msfr --save-ckpt ./model/msfr_fixed.ckpt
import os
import argparse
import torch
import pandas as pd
from typing import Tuple
from msfr import MSFR
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR # lr 스케줄러
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

class TestModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, n_harmonics: int = 12) -> None:
        super().__init__()
        self.msfr = MSFR(input_dim = input_dim, 
        output_dim = output_dim, 
        n_harmonics = n_harmonics)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.msfr(x)

def load_dataset(csv_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(csv_path)

    # 첫 컬럼이 날짜/인덱스일 가능성에 대비하여 숫자형 컬럼만 타깃으로 사용
    numeric_cols = df.select_dtypes(include = ["number"]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("there is no numeric columns in the dataset it must be 370 households")

    y = torch.tensor(df[numeric_cols].values, dtype = torch.float32)

    # 15분 간격 정수 시간축 t 생성 (0,1,2,...) -> 입력은 [t, t, t]로 3계절성 공유
    t = torch.arange(y.shape[0], dtype = torch.float32).unsqueeze(1)  # (N, 1)
    X = t.repeat(1, 3)  # (N, 3) = [t, t, t]
    return X, y

def train_val_split(X: torch.Tensor, y: torch.Tensor, val_ratio: float = 0.1) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    n = X.shape[0]
    split = int(n * (1 - val_ratio))
    return (X[:split], y[:split]), (X[split:], y[split:])

def lr_lambda(epoch):
    if epoch < 60:
        return 1.0
    else:
        return 0.95 ** (epoch - 70)

def make_plots(cycle_hist, train_mse_hist, val_mse_hist, bias_hist, args, model):
    # 1) cycle 
    import numpy as np
    cycle_hist = np.stack(cycle_hist, axis = 0)   
    fig1 = plt.figure(figsize = (10, 6))
    plt.plot(cycle_hist[:, 0], label = "day (≈96)")
    plt.xlabel("Epochs")
    plt.ylabel("Cycle Length")
    plt.title("MSFR Cycle Parameter Evolution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    ymin, ymax = plt.ylim()
    yticks = np.arange(ymin, ymax, 0.02)
    plt.yticks(yticks)

    # 2) Train MSE
    fig2 = plt.figure()
    plt.plot(train_mse_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE")
    plt.title("Train MSE over epochs")
    plt.grid(True, alpha=0.3)

    # 3) Validation MSE
    fig3 = plt.figure()
    plt.plot(val_mse_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Val MSE")
    plt.title("Validation MSE over epochs")
    plt.grid(True, alpha=0.3)

    # 4) bias evolution
    fig4 = plt.figure()
    plt.plot(bias_hist)
    plt.xlabel("Epoch")
    plt.ylabel("Bias values")
    plt.title("Bias Parameter Evolution (mean)")
    plt.grid(True, alpha=0.3)

    fig1.savefig("msfr_cycle_evolution.png")
    fig2.savefig("msfr_train_mse.png")
    fig3.savefig("msfr_val_mse.png")
    fig4.savefig("msfr_bias_evolution.png")

    # 체크포인트 저장 (옵션)
    if args.save_ckpt is not None:
        ckpt_path = os.path.abspath(args.save_ckpt)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"checkpoint saved to: {ckpt_path}")

def main():
    parser = argparse.ArgumentParser(description = "Train MSFR and optionally save checkpoint")
    parser.add_argument("--save-ckpt", type = str, default = None)
    args = parser.parse_args()
    csv_path = "benchmark/test/LD2011_2014_converted.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"dataset file not found: {csv_path}")

    X, y = load_dataset(csv_path)
    (X_tr, y_tr), (X_val, y_val) = train_val_split(X, y, val_ratio=0.1)

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    X_tr, y_tr = X_tr.to(device), y_tr.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    input_dim = X_tr.shape[1]  # 3 (일/주/년 계절성 위한 공유 t)
    output_dim = y_tr.shape[1]  # 370 가구 수

    model = TestModel(input_dim = input_dim, output_dim = output_dim, n_harmonics = 12).to(device)

    # 주기 파라미터 초기화 (15분 간격): 일 = 96, 주 = 672, 년 = 35040
    with torch.no_grad():
        init_cycles = torch.tensor([96.0, 96.0 * 7.0, 96.0 * 365.0], dtype = torch.float32, device = device)
        model.msfr.cycle.copy_(init_cycles)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size = 512, shuffle = True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size = 1024, shuffle = False)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda) # lr 스케줄러
    loss_fn = nn.MSELoss()

    cycle_hist = []          # [(day, week, year), ...]
    train_mse_hist = []      # [mse_epoch1, mse_epoch2, ...]
    val_mse_hist = []        # [mse_epoch1, mse_epoch2, ...]
    bias_hist = []

    epochs = 70
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none = True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        train_mse = total_loss / X_tr.size(0)
        train_mse_hist.append(train_mse)
        scheduler.step() # 에폭 끝날 때마다 lr 스케줄러 스텝 호출 (안쓰고 싶으면 주석 처리)
        train_loss = total_loss / X_tr.size(0)

        # 주기 그래프용
        cyc = model.msfr.cycle.detach().cpu().numpy()   # (3,)
        cycle_hist.append(cyc)

        model.eval()
        with torch.no_grad():
            val_total = 0.0
            for xb, yb in val_loader:
                pred = model(xb)
                val_total += loss_fn(pred, yb).item() * xb.size(0)
            val_loss = val_total / X_val.size(0)
            val_mse = val_loss
            val_mse_hist.append(val_mse)

        # 바이어스 그래프용
        b = model.msfr.bias.detach().cpu().numpy()
        bias_hist.append(b.mean())

        print(f"[Epoch {epoch:02d}] bias mean = {b.mean():.3f}, "f"min = {b.min():.3f}, max = {b.max():.3f}")
        print(f"lr = {scheduler.get_last_lr()[0]:.6f}")
        print(f"cycle:", model.msfr.cycle.detach().cpu().numpy())
        print()
    make_plots(cycle_hist, train_mse_hist, val_mse_hist, bias_hist, args, model)
    # print(f"[Epoch {epoch:02d}] train MSE: {train_loss:.6f} | val MSE: {val_loss:.6f}")

if __name__ == "__main__":
    main()
#how to 실행 : root directory에서 python -m benchmark.test.train_msfr --save-ckpt ./ckpts/msfr_fixed.ckpt
import os
import argparse
import torch
import pandas as pd
from typing import Tuple
from msfr import MSFR
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

"""
하모닉스를 기존 3에서 10으로 늘려봐도 성능이 거의 그대로임 (변화가 그냥 없음)
현재 msfr은 모든 출력을 sin cos 조합으로만 표현하고 있는걸로 아는데
sin cos의 평균은 0이니까 평균값을 스스로 만들 수 없잖아?
예를 들어 실제 전력 데이터가 2000Wh 근처에서 살짝 진동한다면, 
MSFR은 진동만 표현할 수 있고 2000이라는 중심값 자체를 못 맞춰서 전체적으로 밀려버림
"""
# --------------------------------
"""
gpt가 그러는데 
baseline(mean) RMSE 2375, MAE 269
baseline(seasonal lag=96) RMSE 505, MAE 60
-> 이 두 숫자의 차이를 보면,

단순 평균으로도 오차가 2천 정도인데,
하루 전 값만 써도 오차가 500 정도로 급감한다.

즉, 데이터의 주요 패턴은 “일주기(96)” 하나면 충분히 설명된다는 뜻.
MSFR은 이걸 못 잡았다는 건, “cycle은 맞지만 위상이 틀렸거나 bias가 없어 중심이 안 맞는다” 쪽이야.

라고 하네요 
지금 데이터의 사이클은 정확히 적은게 맞음 (92번줄 참고)
그럼 bias가 없어서 중심이 안맞는다는게 문제
"""
# 위에쓴건 좀 고민을 해봐야할거 같습니다

class TestModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, n_harmonics: int = 10) -> None:
        super().__init__()
        self.msfr = MSFR(input_dim = input_dim, 
        output_dim = output_dim, 
        n_harmonics = n_harmonics)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.msfr(x)


def load_dataset(csv_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(csv_path)

    # 첫 컬럼이 날짜/인덱스일 가능성에 대비하여 숫자형 컬럼만 타깃으로 사용
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("there is no numeric columns in the dataset it must be 370 households")

    y = torch.tensor(df[numeric_cols].values, dtype=torch.float32)

    # 15분 간격 정수 시간축 t 생성 (0,1,2,...) → 입력은 [t, t, t]로 3계절성 공유
    t = torch.arange(y.shape[0], dtype=torch.float32).unsqueeze(1)  # (N, 1)
    X = t.repeat(1, 3)  # (N, 3) = [t, t, t]
    return X, y


def train_val_split(X: torch.Tensor, y: torch.Tensor, val_ratio: float = 0.1) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    n = X.shape[0]
    split = int(n * (1 - val_ratio))
    return (X[:split], y[:split]), (X[split:], y[split:])


def main():
    parser = argparse.ArgumentParser(description="Train MSFR and optionally save checkpoint")
    parser.add_argument("--save-ckpt", type=str, default=None, help="path to save model state_dict")
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

    model = TestModel(input_dim=input_dim, output_dim=output_dim, n_harmonics=3).to(device)

    # 주기 파라미터 초기화 (15분 간격): 일 = 96, 주 = 672, 년 ≈ 35040
    with torch.no_grad():
        init_cycles = torch.tensor([96.0, 96.0 * 7.0, 96.0 * 365.0],dtype=torch.float32, device=device)
        model.msfr.cycle.copy_(init_cycles)
        print(model.msfr.cycle)


    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=512, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=1024, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    epochs = 33
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
        print(f"[Epoch {epoch:02d}] cycle:", model.msfr.cycle.detach().cpu().numpy())

        train_loss = total_loss / X_tr.size(0)

        model.eval()
        with torch.no_grad():
            val_total = 0.0
            for xb, yb in val_loader:
                pred = model(xb)
                val_total += loss_fn(pred, yb).item() * xb.size(0)
            val_loss = val_total / X_val.size(0)

    print(f"[Epoch {epoch:02d}] train MSE: {train_loss:.6f} | val MSE: {val_loss:.6f}")

    # 최종 검증 샘플 몇 개 출력
    with torch.no_grad():
        sample_pred = model(X_val[:5])
        print("shape of sample prediction:", tuple(sample_pred.shape))

    # 체크포인트 저장 (옵션)
    if args.save_ckpt is not None:
        ckpt_path = os.path.abspath(args.save_ckpt)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print(f"checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
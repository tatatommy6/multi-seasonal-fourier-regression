"""Train MSFR on NOAA verified hourly tide heights.

Run from the repository root:
    python -m benchmark.NOAA_tide_hourly_height.train_msfr

The script uses a chronological train/validation/test split, computes target
normalization from the training split only, restores the best validation model,
and saves diagnostic plots and a checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from msfr import MSFR


DEFAULT_CSV = Path(
    "benchmark/NOAA_tide_hourly_height/battery_hourly_combined.csv"
)
DEFAULT_OUTPUT_DIR = Path("benchmark/NOAA_tide_hourly_height/plots")
DEFAULT_CHECKPOINT = Path("model/noaa_tide_msfr.ckpt")

# Important astronomical tidal constituents, expressed in hours.
# M2: principal lunar semidiurnal, S2: principal solar semidiurnal,
# K1: luni-solar diurnal, O1: principal lunar diurnal, Sa: annual.
INITIAL_CYCLES = (12.4206012, 12.0, 23.934472, 25.819342, 24.0 * 365.25)
CYCLE_LABELS = ("M2", "S2", "K1", "O1", "annual")


class TideMSFR(nn.Module):
    """MSFR seasonal component plus a separately normalized linear trend."""

    def __init__(
        self,
        n_harmonics: int,
        time_center: float,
        time_scale: float,
        init_cycles: torch.Tensor,
    ) -> None:
        super().__init__()
        self.msfr = MSFR(
            input_dim=len(INITIAL_CYCLES),
            output_dim=1,
            n_harmonics=n_harmonics,
            trend=False,
            init_cycle=init_cycles,
        )
        self.trend = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.trend.weight)
        self.register_buffer("time_center", torch.tensor(time_center, dtype=torch.float32))
        self.register_buffer("time_scale", torch.tensor(time_scale, dtype=torch.float32))

    def forward(self, time_hours: torch.Tensor) -> torch.Tensor:
        seasonal_input = time_hours.repeat(1, len(INITIAL_CYCLES))
        normalized_time = (time_hours - self.time_center) / self.time_scale
        return self.msfr(seasonal_input) + self.trend(normalized_time)


def load_dataset(csv_path: Path) -> tuple[pd.DatetimeIndex, torch.Tensor, torch.Tensor]:
    if not csv_path.exists():
        raise FileNotFoundError(f"dataset file not found: {csv_path}")

    frame = pd.read_csv(csv_path, skipinitialspace=True)
    frame.columns = frame.columns.str.strip()
    required = {"Date Time", "Water Level"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns in dataset: {sorted(missing)}")

    frame["Date Time"] = pd.to_datetime(frame["Date Time"], errors="coerce")
    frame["Water Level"] = pd.to_numeric(frame["Water Level"], errors="coerce")
    frame = (
        frame.dropna(subset=["Date Time", "Water Level"])
        .sort_values("Date Time")
        .drop_duplicates("Date Time", keep="last")
        .reset_index(drop=True)
    )
    if len(frame) < 1_000:
        raise ValueError(f"dataset is too small after cleaning: {len(frame)} rows")

    timestamps = pd.DatetimeIndex(frame["Date Time"])
    elapsed_hours = (
        (timestamps - timestamps[0]).total_seconds().to_numpy(dtype=np.float64) / 3600.0
    )
    time = torch.tensor(elapsed_hours, dtype=torch.float32).unsqueeze(1)
    target = torch.tensor(
        frame["Water Level"].to_numpy(dtype=np.float32), dtype=torch.float32
    ).unsqueeze(1)
    return timestamps, time, target


def chronological_split(
    time: torch.Tensor,
    target: torch.Tensor,
    val_ratio: float,
    test_ratio: float,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be positive and sum to less than 1")
    count = len(time)
    train_end = int(count * (1.0 - val_ratio - test_ratio))
    val_end = int(count * (1.0 - test_ratio))
    return (
        (time[:train_end], target[:train_end]),
        (time[train_end:val_end], target[train_end:val_end]),
        (time[val_end:], target[val_end:]),
    )


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_mse(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for time_batch, target_batch in loader:
            time_batch = time_batch.to(device)
            target_batch = target_batch.to(device)
            prediction = model(time_batch)
            total_loss += loss_fn(prediction, target_batch).item() * len(time_batch)
            total_count += len(time_batch)
    return total_loss / total_count


def predict(
    model: nn.Module, time: torch.Tensor, batch_size: int, device: torch.device
) -> torch.Tensor:
    loader = DataLoader(TensorDataset(time), batch_size=batch_size, shuffle=False)
    batches = []
    model.eval()
    with torch.no_grad():
        for (time_batch,) in loader:
            batches.append(model(time_batch.to(device)).cpu())
    return torch.cat(batches)


def save_plots(
    output_dir: Path,
    timestamps: pd.DatetimeIndex,
    test_start: int,
    test_target_real: np.ndarray,
    test_prediction_real: np.ndarray,
    cycle_history: list[np.ndarray],
    train_mse_history: list[float],
    val_mse_history: list[float],
    bias_history: list[float],
    trend_history: list[float],
    weight_norm_history: list[float],
    learning_rate_history: list[tuple[float, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(train_mse_history) + 1)

    cycles = np.stack(cycle_history)
    fig, ax = plt.subplots(figsize=(11, 6))
    for index, label in enumerate(CYCLE_LABELS):
        ax.plot(epochs, cycles[:, index], label=label)
    ax.set_yscale("log")
    ax.set(title="MSFR Cycle Evolution", xlabel="Epoch", ylabel="Cycle (hours)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "msfr_cycle_evolution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_mse_history, label="train")
    ax.plot(epochs, val_mse_history, label="validation")
    ax.set_yscale("log")
    ax.set(title="Normalized MSE", xlabel="Epoch", ylabel="MSE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "msfr_mse.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(epochs, bias_history)
    axes[0].set_ylabel("Bias")
    axes[0].set_title("Parameter Evolution")
    axes[1].plot(epochs, trend_history)
    axes[1].set_ylabel("Trend weight")
    axes[2].plot(epochs, weight_norm_history)
    axes[2].set(xlabel="Epoch", ylabel="Fourier weight norm")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "msfr_parameter_evolution.png", dpi=160)
    plt.close(fig)

    rates = np.asarray(learning_rate_history)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, rates[:, 0], label="Fourier/trend LR")
    ax.plot(epochs, rates[:, 1], label="Cycle LR")
    ax.set_yscale("log")
    ax.set(title="Learning Rate", xlabel="Epoch", ylabel="Learning rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "msfr_learning_rate.png", dpi=160)
    plt.close(fig)

    plot_count = min(24 * 30, len(test_target_real))
    plot_dates = timestamps[test_start : test_start + plot_count]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(plot_dates, test_target_real[:plot_count], label="observed", linewidth=1.2)
    ax.plot(plot_dates, test_prediction_real[:plot_count], label="MSFR", linewidth=1.2)
    ax.set(title="Test Forecast (first 30 days)", xlabel="Date", ylabel="Water level (m)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_dir / "msfr_test_forecast.png", dpi=160)
    plt.close(fig)

    residual = test_prediction_real - test_target_real
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(residual, bins=80, alpha=0.8)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set(title="Test Residual Distribution", xlabel="Prediction - observation (m)", ylabel="Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "msfr_test_residuals.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MSFR on NOAA hourly tide data")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-ckpt", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-harmonics", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cycle-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-cycles", action="store_true")
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.n_harmonics <= 0:
        raise ValueError("epochs, batch_size, and n_harmonics must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    timestamps, time, target_real = load_dataset(args.csv_path)
    (time_train, target_train_real), (time_val, target_val_real), (
        time_test,
        target_test_real,
    ) = chronological_split(time, target_real, args.val_ratio, args.test_ratio)

    target_mean = target_train_real.mean()
    target_std = target_train_real.std().clamp_min(1e-6)
    target_train = (target_train_real - target_mean) / target_std
    target_val = (target_val_real - target_mean) / target_std

    time_center = float(time_train.mean())
    time_scale = max(float(time_train.std()), 1.0)
    init_cycles = torch.tensor(INITIAL_CYCLES, dtype=torch.float32)
    device = choose_device()
    model = TideMSFR(
        args.n_harmonics, time_center, time_scale, init_cycles
    ).to(device)
    model.msfr.log_cycle.requires_grad_(not args.freeze_cycles)

    train_loader = DataLoader(
        TensorDataset(time_train, target_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(time_val, target_val),
        batch_size=args.batch_size * 2,
        shuffle=False,
    )

    parameter_groups = [
        {
            "params": [model.msfr.weight, model.msfr.bias, model.trend.weight],
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
    ]
    if not args.freeze_cycles:
        parameter_groups.append(
            {"params": [model.msfr.log_cycle], "lr": args.cycle_lr, "weight_decay": 0.0}
        )
    optimizer = torch.optim.AdamW(parameter_groups)
    loss_fn = nn.MSELoss()

    cycle_history: list[np.ndarray] = []
    train_mse_history: list[float] = []
    val_mse_history: list[float] = []
    bias_history: list[float] = []
    trend_history: list[float] = []
    weight_norm_history: list[float] = []
    learning_rate_history: list[tuple[float, float]] = []
    best_val_mse = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        for time_batch, target_batch in train_loader:
            time_batch = time_batch.to(device)
            target_batch = target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(time_batch)
            loss = loss_fn(prediction, target_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            train_total += loss.item() * len(time_batch)

        train_mse = train_total / len(time_train)
        val_mse = evaluate_mse(model, val_loader, loss_fn, device)
        train_mse_history.append(train_mse)
        val_mse_history.append(val_mse)
        cycle_history.append(model.msfr.cycle.detach().cpu().numpy().copy())
        bias_history.append(float(model.msfr.bias.detach().cpu().item()))
        trend_history.append(float(model.trend.weight.detach().cpu().item()))
        weight_norm_history.append(float(model.msfr.weight.detach().norm().cpu()))
        main_lr = optimizer.param_groups[0]["lr"]
        current_cycle_lr = (
            optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0
        )
        learning_rate_history.append((main_lr, current_cycle_lr))

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        print(
            f"[Epoch {epoch:03d}] train MSE={train_mse:.6f} | "
            f"val MSE={val_mse:.6f} | bias={bias_history[-1]:.4f} | "
            f"cycles={np.round(cycle_history[-1], 4)}"
        )
        if epoch - best_epoch >= args.patience:
            print(f"early stopping: no validation improvement for {args.patience} epochs")
            break

    if best_state is None:
        raise RuntimeError("training did not produce a valid model state")
    model.load_state_dict(best_state)

    test_prediction_normalized = predict(model, time_test, args.batch_size * 2, device)
    test_prediction_real = test_prediction_normalized * target_std + target_mean
    test_error = test_prediction_real - target_test_real
    test_mae = float(test_error.abs().mean())
    test_rmse = float(torch.sqrt((test_error**2).mean()))
    test_start = len(time_train) + len(time_val)

    save_plots(
        args.output_dir,
        timestamps,
        test_start,
        target_test_real.squeeze(1).numpy(),
        test_prediction_real.squeeze(1).numpy(),
        cycle_history,
        train_mse_history,
        val_mse_history,
        bias_history,
        trend_history,
        weight_norm_history,
        learning_rate_history,
    )

    args.save_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_harmonics": args.n_harmonics,
            "initial_cycles": INITIAL_CYCLES,
            "target_mean": float(target_mean),
            "target_std": float(target_std),
            "time_origin": str(timestamps[0]),
            "time_center": time_center,
            "time_scale": time_scale,
            "best_epoch": best_epoch,
            "best_val_mse": best_val_mse,
            "test_mae_meters": test_mae,
            "test_rmse_meters": test_rmse,
            "cycles_frozen": args.freeze_cycles,
            "seed": args.seed,
        },
        args.save_ckpt,
    )

    print()
    print(f"device: {device}")
    print(
        f"split dates: train={timestamps[0]}..{timestamps[len(time_train) - 1]}, "
        f"validation={timestamps[len(time_train)]}..{timestamps[test_start - 1]}, "
        f"test={timestamps[test_start]}..{timestamps[-1]}"
    )
    print(f"best epoch: {best_epoch} | best validation MSE: {best_val_mse:.6f}")
    print(f"test MAE: {test_mae:.6f} m | test RMSE: {test_rmse:.6f} m")
    print(f"learned cycles (hours): {model.msfr.cycle.detach().cpu().numpy()}")
    print(f"plots saved to: {args.output_dir.resolve()}")
    print(f"checkpoint saved to: {args.save_ckpt.resolve()}")


if __name__ == "__main__":
    main()

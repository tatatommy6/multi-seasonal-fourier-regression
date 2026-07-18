"""Compare learned/fixed-cycle MSFR with fair rolling-origin tide baselines.

Run from the repository root after producing both checkpoints:
    python -m benchmark.NOAA_tide_hourly_height.compare_msfr
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from benchmark.NOAA_tide_hourly_height.train_msfr import (
    DEFAULT_CSV,
    TideMSFR,
    choose_device,
    load_dataset,
    predict,
)


DEFAULT_LEARNED = Path("model/noaa_tide_msfr_learned_seed42.ckpt")
DEFAULT_FIXED = Path("model/noaa_tide_msfr_fixed_seed42.ckpt")
DEFAULT_OUTPUT = Path(
    "benchmark/NOAA_tide_hourly_height/msfr_baseline_comparison.csv"
)
HORIZONS = (24, 168, 720)


def load_checkpoint_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[TideMSFR, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {
        "model_state_dict",
        "n_harmonics",
        "initial_cycles",
        "target_mean",
        "target_std",
        "time_center",
        "time_scale",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"checkpoint {checkpoint_path} is missing: {sorted(missing)}")

    initial_cycles = torch.tensor(checkpoint["initial_cycles"], dtype=torch.float32)
    model = TideMSFR(
        int(checkpoint["n_harmonics"]),
        float(checkpoint["time_center"]),
        float(checkpoint["time_scale"]),
        initial_cycles,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


def repeat_last_pattern(
    series: pd.Series, origin: pd.Timestamp, period: int, horizon: int
) -> np.ndarray:
    history_index = pd.date_range(
        origin - pd.Timedelta(hours=period - 1), origin, freq="h"
    )
    history = series.reindex(history_index).to_numpy(dtype=float)
    return np.resize(history, horizon)


def calculate_metrics(errors: list[float]) -> tuple[int, float, float]:
    values = np.asarray(errors, dtype=float)
    return len(values), float(np.mean(np.abs(values))), float(np.sqrt(np.mean(values**2)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare tide MSFR models and baselines")
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--learned-ckpt", type=Path, default=DEFAULT_LEARNED)
    parser.add_argument("--fixed-ckpt", type=Path, default=DEFAULT_FIXED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--origin-step-hours", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    if not 0 < args.test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    if args.origin_step_hours <= 0:
        raise ValueError("origin_step_hours must be positive")

    timestamps, time, target = load_dataset(args.csv_path)
    series = pd.Series(target.squeeze(1).numpy(), index=timestamps)
    test_start_index = int(len(series) * (1.0 - args.test_ratio))
    test_start = timestamps[test_start_index]
    test_end = timestamps[-1]
    pretest_mean = float(series.iloc[:test_start_index].mean())

    device = choose_device()
    learned_model, learned_checkpoint = load_checkpoint_model(args.learned_ckpt, device)
    fixed_model, fixed_checkpoint = load_checkpoint_model(args.fixed_ckpt, device)

    learned_all = (
        predict(learned_model, time, args.batch_size, device).squeeze(1).numpy()
        * float(learned_checkpoint["target_std"])
        + float(learned_checkpoint["target_mean"])
    )
    fixed_all = (
        predict(fixed_model, time, args.batch_size, device).squeeze(1).numpy()
        * float(fixed_checkpoint["target_std"])
        + float(fixed_checkpoint["target_mean"])
    )
    learned_series = pd.Series(learned_all, index=timestamps)
    fixed_series = pd.Series(fixed_all, index=timestamps)

    model_names = (
        "mean",
        "last",
        "12h_naive",
        "24h_naive",
        "12h_24h_average",
        "fixed_cycle_msfr",
        "learned_cycle_msfr",
    )
    errors = {(horizon, name): [] for horizon in HORIZONS for name in model_names}
    origins = pd.date_range(
        test_start.ceil("D"), test_end.floor("D"), freq=f"{args.origin_step_hours}h"
    )

    for origin in origins:
        if origin not in series.index or not np.isfinite(series.get(origin, np.nan)):
            continue
        for horizon in HORIZONS:
            target_index = pd.date_range(
                origin + pd.Timedelta(hours=1), periods=horizon, freq="h"
            )
            if target_index[-1] > test_end:
                continue
            truth = series.reindex(target_index).to_numpy(dtype=float)
            naive_12 = repeat_last_pattern(series, origin, 12, horizon)
            naive_24 = repeat_last_pattern(series, origin, 24, horizon)
            forecasts = {
                "mean": np.full(horizon, pretest_mean),
                "last": np.full(horizon, float(series.loc[origin])),
                "12h_naive": naive_12,
                "24h_naive": naive_24,
                "12h_24h_average": (naive_12 + naive_24) / 2.0,
                "fixed_cycle_msfr": fixed_series.reindex(target_index).to_numpy(dtype=float),
                "learned_cycle_msfr": learned_series.reindex(target_index).to_numpy(dtype=float),
            }
            for name, forecast in forecasts.items():
                valid = np.isfinite(truth) & np.isfinite(forecast)
                errors[(horizon, name)].extend((forecast[valid] - truth[valid]).tolist())

    rows = []
    for horizon in HORIZONS:
        for name in model_names:
            count, mae, rmse = calculate_metrics(errors[(horizon, name)])
            rows.append(
                {"horizon": horizon, "model": name, "n": count, "mae_m": mae, "rmse_m": rmse}
            )
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print(f"test: {test_start} .. {test_end} | origins every {args.origin_step_hours}h")
    print(
        result.to_string(
            index=False,
            formatters={"mae_m": "{:.6f}".format, "rmse_m": "{:.6f}".format},
        )
    )
    print(f"results saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()

"""Fair rolling-origin comparison of naive and fixed-Fourier baselines on PJME.

Experiment A contains simple forecasting baselines. Experiment B fits a ridge
regression on fixed daily, weekly, and yearly Fourier features. Hyperparameters
are selected on validation data; the final test year is not touched until the
selection is complete.

Run from the repository root:
    python -m benchmark.Hourly_Energy_Consumption_Test.experiment_a_b
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CSV_PATH = Path(
    "benchmark/Hourly_Energy_Consumption_Test/"
    "hourly_energy_consumption_combined.csv"
)
SERIES = "PJME"
PERIODS = (24.0, 168.0, 24.0 * 365.25)
HORIZONS = (1, 24, 168)
PRIMARY_HORIZON = 168
ORDER_CANDIDATES = ((3, 2, 1), (6, 3, 2), (12, 6, 3))
ALPHA_CANDIDATES = (1e-8, 1e-6, 1e-4, 1e-2, 1.0)


@dataclass(frozen=True)
class FourierConfig:
    orders: tuple[int, int, int]
    alpha: float


@dataclass
class RidgeFourier:
    config: FourierConfig
    time_center: float
    time_scale: float
    target_mean: float
    target_scale: float
    coefficients: np.ndarray

    def predict(self, index: pd.DatetimeIndex) -> np.ndarray:
        hours = to_hours(index)
        features = make_features(
            hours, self.config.orders, self.time_center, self.time_scale
        )
        normalized = features @ self.coefficients
        return normalized * self.target_scale + self.target_mean


def load_series() -> pd.Series:
    frame = pd.read_csv(
        CSV_PATH, parse_dates=["Datetime"], usecols=["Datetime", SERIES]
    )
    series = frame.set_index("Datetime")[SERIES].sort_index().astype(float)
    if series.index.has_duplicates:
        raise ValueError("Datetime contains duplicate values")
    return series


def to_hours(index: pd.DatetimeIndex) -> np.ndarray:
    return index.asi8.astype(np.float64) / (3_600 * 1e9)


def make_features(
    hours: np.ndarray,
    orders: tuple[int, int, int],
    time_center: float,
    time_scale: float,
) -> np.ndarray:
    columns = [np.ones_like(hours), (hours - time_center) / time_scale]
    for period, order in zip(PERIODS, orders):
        for harmonic in range(1, order + 1):
            angle = 2.0 * np.pi * harmonic * hours / period
            columns.extend((np.sin(angle), np.cos(angle)))
    return np.column_stack(columns)


def fit_fourier(series: pd.Series, config: FourierConfig) -> RidgeFourier:
    observed = series.dropna()
    hours = to_hours(observed.index)
    time_center = float(hours.mean())
    time_scale = float(hours.std())
    target = observed.to_numpy(dtype=np.float64)
    target_mean = float(target.mean())
    target_scale = float(target.std())

    features = make_features(hours, config.orders, time_center, time_scale)
    normalized_target = (target - target_mean) / target_scale
    gram = features.T @ features / len(features)
    rhs = features.T @ normalized_target / len(features)
    penalty = np.eye(features.shape[1]) * config.alpha
    penalty[0, 0] = 0.0  # Do not regularize the intercept.
    coefficients = np.linalg.solve(gram + penalty, rhs)
    return RidgeFourier(
        config,
        time_center,
        time_scale,
        target_mean,
        target_scale,
        coefficients,
    )


def forecast_naive(
    series: pd.Series,
    origin: pd.Timestamp,
    horizon: int,
    train_mean: float,
) -> dict[str, np.ndarray]:
    last_day = series.reindex(
        pd.date_range(origin - pd.Timedelta(hours=23), origin, freq="h")
    ).to_numpy(dtype=float)
    last_week = series.reindex(
        pd.date_range(origin - pd.Timedelta(hours=167), origin, freq="h")
    ).to_numpy(dtype=float)
    daily = np.resize(last_day, horizon)
    weekly = np.resize(last_week, horizon)
    return {
        "mean": np.full(horizon, train_mean),
        "last": np.full(horizon, float(series.loc[origin])),
        "daily_naive": daily,
        "weekly_naive": weekly,
        "daily_weekly_avg": (daily + weekly) / 2.0,
    }


def evaluate(
    series: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_mean: float,
    model: RidgeFourier | None,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    names = ["mean", "last", "daily_naive", "weekly_naive", "daily_weekly_avg"]
    if model is not None:
        names.append("fixed_fourier_ridge")
    errors = {(h, name): [] for h in horizons for name in names}
    origins = pd.date_range(start.ceil("D"), end.floor("D"), freq="24h")

    for origin in origins:
        if origin not in series.index or not np.isfinite(series.get(origin, np.nan)):
            continue
        for horizon in horizons:
            targets = pd.date_range(
                origin + pd.Timedelta(hours=1), periods=horizon, freq="h"
            )
            if targets[-1] > end:
                continue
            truth = series.reindex(targets).to_numpy(dtype=float)
            forecasts = forecast_naive(series, origin, horizon, train_mean)
            if model is not None:
                forecasts["fixed_fourier_ridge"] = model.predict(targets)
            for name, prediction in forecasts.items():
                valid = np.isfinite(truth) & np.isfinite(prediction)
                errors[(horizon, name)].extend((prediction[valid] - truth[valid]).tolist())

    rows = []
    for horizon in horizons:
        for name in names:
            error = np.asarray(errors[(horizon, name)], dtype=float)
            rows.append(
                {
                    "horizon": horizon,
                    "model": name,
                    "n": len(error),
                    "mae": np.mean(np.abs(error)),
                    "rmse": np.sqrt(np.mean(error**2)),
                }
            )
    return pd.DataFrame(rows)


def select_config(
    train: pd.Series,
    full_series: pd.Series,
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> tuple[FourierConfig, pd.DataFrame]:
    rows = []
    train_mean = float(train.mean())
    for orders in ORDER_CANDIDATES:
        for alpha in ALPHA_CANDIDATES:
            config = FourierConfig(orders, alpha)
            model = fit_fourier(train, config)
            result = evaluate(
                full_series,
                validation_start,
                validation_end,
                train_mean,
                model,
                horizons=(PRIMARY_HORIZON,),
            )
            score = result.loc[
                result["model"] == "fixed_fourier_ridge", "mae"
            ].item()
            rows.append({"orders": orders, "alpha": alpha, "validation_mae": score})
    tuning = pd.DataFrame(rows).sort_values("validation_mae", ignore_index=True)
    best = tuning.iloc[0]
    return FourierConfig(tuple(best["orders"]), float(best["alpha"])), tuning


def main() -> None:
    series = load_series()
    observed_end = series.dropna().index.max()
    test_start = observed_end - pd.DateOffset(years=1)
    validation_start = test_start - pd.DateOffset(years=1)

    train = series.loc[series.index < validation_start].dropna()
    best_config, tuning = select_config(
        train, series, validation_start, test_start - pd.Timedelta(hours=1)
    )

    train_and_validation = series.loc[series.index < test_start].dropna()
    final_model = fit_fourier(train_and_validation, best_config)
    result = evaluate(
        series,
        test_start,
        observed_end,
        float(train_and_validation.mean()),
        final_model,
    )

    print(f"series: {SERIES}")
    print(f"train: < {validation_start}")
    print(f"validation: {validation_start} .. {test_start}")
    print(f"test: {test_start} .. {observed_end}")
    print(
        "selected config: "
        f"orders={best_config.orders}, alpha={best_config.alpha:g}, "
        f"validation {PRIMARY_HORIZON}h MAE={tuning.iloc[0]['validation_mae']:.3f}"
    )
    print("\nTop validation configurations:")
    print(tuning.head(5).to_string(index=False))
    print("\nTest metrics:")
    print(
        result.to_string(
            index=False,
            formatters={"mae": "{:.3f}".format, "rmse": "{:.3f}".format},
        )
    )


if __name__ == "__main__":
    main()

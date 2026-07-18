# PJME experiments A and B

## Protocol

- Series: `PJME`, hourly observations
- Train: before 2016-08-03
- Validation: 2016-08-03 through 2017-08-02
- Test: 2017-08-03 through 2018-08-03
- Forecast origins: once per day at 00:00
- Horizons: 1, 24, and 168 hours
- No observation after a forecast origin is available to a model
- Metrics are calculated in the original load unit

Experiment A compares mean, last-value, daily-naive, weekly-naive, and the
average of daily and weekly naive forecasts. Experiment B uses ridge regression
with fixed Fourier periods of 24, 168, and 8766 hours plus a linear trend.
Fourier orders and ridge alpha are selected only by 168-hour validation MAE.

Selected Experiment B configuration:

- Fourier orders: `(12, 6, 3)` for daily, weekly, and yearly periods
- Ridge alpha: `0.01`
- Validation 168-hour MAE: `3159.304`

## Test results

| Horizon | Model | MAE | RMSE |
| ---: | --- | ---: | ---: |
| 1 | Mean | 5667.650 | 6355.238 |
| 1 | Last value | 1729.833 | **1820.267** |
| 1 | Daily naive | **1715.523** | 2316.082 |
| 1 | Weekly naive | 3164.384 | 4179.641 |
| 1 | Daily-weekly average | 2058.060 | 2687.217 |
| 1 | Fixed Fourier ridge | 2647.042 | 3453.101 |
| 24 | Mean | 4903.975 | 6129.213 |
| 24 | Last value | 4067.305 | 5079.453 |
| 24 | Daily naive | **2307.173** | **3146.479** |
| 24 | Weekly naive | 3582.400 | 4852.846 |
| 24 | Daily-weekly average | 2435.677 | 3226.173 |
| 24 | Fixed Fourier ridge | 3164.587 | 4035.763 |
| 168 | Mean | 4881.195 | 6097.915 |
| 168 | Last value | 4694.322 | 6031.436 |
| 168 | Daily naive | 3458.898 | 4680.699 |
| 168 | Weekly naive | 3594.127 | 4869.662 |
| 168 | Daily-weekly average | **3141.767** | 4256.836 |
| 168 | Fixed Fourier ridge | 3148.654 | **4023.386** |

For the 168-hour horizon, fixed Fourier ridge reduces RMSE by about 5.5%
relative to the strongest naive RMSE while its MAE is about 0.2% higher. This
suggests that fixed Fourier features reduce large errors, but do not yet improve
typical absolute error. The next useful experiment is to fit Fourier ridge to
the residual of the daily-weekly baseline rather than replacing that baseline.

## Reproduction

Run from the repository root:

```shell
python -m benchmark.Hourly_Energy_Consumption_Test.experiment_a_b
```

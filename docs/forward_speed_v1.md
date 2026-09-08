# Learned forward-speed experiment with uncertainty

## Decision

**Do not integrate this model into the navigation pipeline yet.** It fits the S1 development interval but fails the untouched S4 confirmation interval. This demonstrates session-specific learning rather than reliable generalization.

The model is a two-layer causal GRU. It consumes gyroscope, gravity-removed accelerometer, last trusted GNSS speed and elapsed outage time. At every sample it emits a non-negative mean forward speed and a positive standard deviation. It has no access to GNSS after the interval begins.

## Split and anti-leakage rule

| Role | Sessions |
|---|---|
| Training | Vta2, Vta16, S3c |
| Validation / checkpoint selection | Vta08 |
| Development test | S1 |
| Untouched final confirmation | S4 |

The S4 interval was the first deterministic 20-second qualifying interval (indices 348–548, 34.8–54.8 s) and was selected before inference. S4 did not influence training, hyperparameters or checkpoint selection.

The safe comparison baseline holds the last trusted GNSS speed constant. This deliberately avoids the unsupported gyroscope-body-axis and accelerometer projection assumptions found in the calculation audit.

## Results

| Session | Status | Method | Speed MAE | Speed RMSE | Final distance error | 95% coverage |
|---|---|---|---:|---:|---:|---:|
| S1 | Development | Constant-speed baseline | 5.40 m/s | 5.90 m/s | -108.47 m | — |
| S1 | Development | Learned GRU | **1.24 m/s** | **1.51 m/s** | **+2.23 m** | 95.5% |
| S4 | Untouched confirmation | Constant-speed baseline | **1.61 m/s** | **1.97 m/s** | **-13.10 m** | — |
| S4 | Untouched confirmation | Learned GRU | 2.94 m/s | 3.30 m/s | +58.66 m | 72.6% |

The S4 result rejects the model for production. Its uncertainty is also overconfident: a nominal 95% interval covers only 72.6% of reference samples.

## Evidence

- [S1 development evaluation](../results/experiments/forward_speed_v1/evaluation_S1.json)
- [S4 untouched evaluation](../results/experiments/forward_speed_v1/evaluation_S4.json)
- [S4 plot](../results/plots/forward_speed_v1/fresh_S4_speed_and_distance.png)
- [Training history and exact sampled windows](../results/experiments/forward_speed_v1/training.json)
- [Checkpoint](../results/models/forward_speed_v1.pt)

This negative result is useful: it prevents a model that looks strong on one recording from silently replacing a safer baseline.

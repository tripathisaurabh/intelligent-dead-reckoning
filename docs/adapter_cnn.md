# Step 4: Dilated CNN IEKF Adapter

## Outcome

Step 4 adds a small causal CNN that changes only IEKF measurement covariance and bounded IMU
calibration terms. It is evaluated on top of the Step 3b yaw-aligned filter because Step 3b proved
that mounting yaw was a structural source of the trajectory bend.

The model never predicts position, velocity, or attitude. All navigation states still come from the
same IEKF propagation and non-holonomic update.

## Architecture

- Input: 32 IMU samples x 6 channels (bias-centred XYZ gyro and XYZ accelerometer)
- Context: 3.2 seconds at the measured IO-VNBD 10 Hz rate
- Four causal Conv1d blocks with dilations 1, 2, 4, 8
- Every block: Conv1d -> BatchNorm1d -> GELU -> Dropout
- Hidden width: 32
- Parameters: 10,411
- Output: 11 tanh-bounded adapter values
  - two non-holonomic measurement-covariance multipliers in `[0.1, 10]`
  - three gyro-bias corrections bounded to `+/-0.02 rad/s`
  - three accelerometer-bias corrections bounded to `+/-0.5 m/s^2`
  - three accelerometer scales in `[0.9, 1.1]`

The original build prompt described 112 samples as roughly one second, but this dataset is 10 Hz;
112 samples would actually be 11.2 seconds. A 32-sample window was selected because the four
kernel-3 dilation blocks have a 31-sample receptive field, giving 3.2 seconds of causal context.

## Differentiable training

`train.py` propagates batches through the 21-state IEKF, including the SE2(3) state correction and
the non-holonomic Kalman update. Huber losses are applied to one-second relative position,
velocity, and rotation increments. Gradients therefore flow from navigation increments through the
filter update into the adapter. Training is not performed on isolated position-regression labels.

The model head is zero-initialized. Before training, it exactly reproduces the fixed-covariance
Step 3b filter: covariance multipliers are one, bias deltas are zero, and acceleration scales are one.

## Drive split

| Role | Session | Windows | Use |
|---|---|---:|---|
| Train | Vta2 | 512 | Gradient updates |
| Validation | Vta16 | 128 | Whole-drive validation and epoch selection |
| Test | Vta08 | fixed 20 s blackout | Final named evaluation only |

There is no within-drive train/validation split. Vw4 and Vtb5 were requested as preferred sessions,
but their CSVs are Git-LFS pointer files in this checkout and the external data download was not
authorized. Complete local Driver-E sessions Vta2 and Vta16 were used instead. This is a prototype
split, not strong evidence of generalization across drivers, vehicles, or phone mountings.

The smartphone GPS speed column is labelled `Kmh` but behaves as m/s on both training and
validation drives. The loader records that assessment. Training labels use the VBOX speed column,
which is explicitly converted from km/h to m/s.

## Training result

Best checkpoint: epoch 7 of 8.

| Loss | Epoch 1 | Best/final |
|---|---:|---:|
| Train weighted Huber | 0.15376 | 0.10127 (epoch 8) |
| Validation weighted Huber | 0.17670 | 0.13872 (epoch 7) |

The small train/validation gap and flattening validation curve are visible in
`results/plots/adapter_cnn/training_loss.png`.

## Vta08 identical-blackout comparison

Locked window: 91.6-111.6 seconds, 20 seconds, 141.995 m reference distance.

| Metric | Step 1 | Step 2 | Step 3 tilt | Step 3b yaw | Step 4 CNN |
|---|---:|---:|---:|---:|---:|
| Final position error | 136.731 m | 78.723 m | 78.747 m | 32.478 m | **29.770 m** |
| Position RMSE | 65.015 m | 41.069 m | 41.128 m | 35.013 m | **27.845 m** |
| Mean position error | 50.903 m | 34.108 m | 34.156 m | 29.177 m | **23.114 m** |
| Maximum position error | 136.731 m | 78.723 m | 78.747 m | **66.952 m** | 66.960 m |
| Drift | 96.293% | 55.441% | 55.458% | 22.872% | **20.966%** |

Relative to Step 3b, the CNN improves endpoint error by 2.707 m, RMSE by 7.167 m, and drift by
1.907 percentage points. The structural yaw correction remains the much larger improvement.
Visually, the CNN reduces error but the trajectory still contains nonphysical lateral excursions;
it has not solved dead reckoning or reached the eventual 10% target.

## Validation-session blackout

Vta16, 413.7-433.7 seconds, 20 seconds, 145.174 m reference distance:

| Metric | Fixed yaw-aligned IEKF | CNN-adapted IEKF |
|---|---:|---:|
| Final position error | 143.266 m | **73.397 m** |
| Position RMSE | 82.288 m | **68.090 m** |
| Drift | 98.686% | **50.558%** |

Vta16 was excluded from gradient updates but used to select the best epoch. These numbers are
validation performance, not a blind held-out test.

## Runtime

- Checkpoint size: 52,386 bytes
- CPU inference latency per 32-sample window: median 0.131 ms, p95 0.230 ms (200 runs)
- Dynamic inference rate required by the current runner: 10 Hz

Latency measures the CNN only, not the NumPy IEKF, data loading, or eventual mobile runtime.

## Reproduce

```bash
MPLCONFIGDIR=/tmp/idr-matplotlib .venv/bin/python train.py
MPLCONFIGDIR=/tmp/idr-matplotlib .venv/bin/python -m scripts.run_adapter_vta08
```

Outputs:

- `results/models/adapter_cnn_v1.pt`
- `results/experiments/adapter_cnn/training_history.json`
- `results/experiments/adapter_cnn/evaluation.json`
- `results/metrics/adapter_cnn_vta08_metrics.csv`
- `results/plots/adapter_cnn/training_loss.png`
- `results/plots/adapter_cnn/vta08_trajectory_comparison.png`
- `results/plots/adapter_cnn/vta08_error_comparison.png`

The implementation follows the hybrid model/learning direction of Brossard, Barrau and Bonnabel's
AI-IMU IEKF and Guo et al.'s model-based deep-learning IMU correction approach. The local design
uses a smaller causal CNN and explicit bounded outputs suitable for this 10 Hz smartphone dataset.

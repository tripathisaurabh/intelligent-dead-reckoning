# Step 3: Classical Smartphone-to-Vehicle Mounting Alignment

## Scope

This step adds a causal, non-learned mounting-tilt estimate before the existing fixed-covariance
IEKF. It does not add a CNN, GNSS updates, map matching, or any evaluation reference data to the
filter input.

Run from the repository root:

```bash
MPLCONFIGDIR=/tmp/idr-matplotlib .venv/bin/python -m scripts.run_alignment_vta08
```

## Method and observability

`src/preprocessing/alignment.py` searches the first 50 seconds for the earliest 30-second window
where at least 90% of bias-corrected phone yaw-rate samples are below 0.03 rad/s and gravity has a
valid magnitude. The window selection uses only IMU signals and is therefore independent of the
blackout error.

For mean phone-frame gravity `g_i`, vehicle-frame gravity is defined as `[0, 0, g]`. The estimated
rotation maps vehicle coordinates to phone IMU coordinates:

```text
R_car_to_imu = R_y(pitch) R_x(roll)
R_car_to_imu [0, 0, 1]^T = g_i / ||g_i||
roll  = asin(-g_i_y / ||g_i||)
pitch = atan2(g_i_x, g_i_z)
```

Gravity observes pitch and roll, but not mounting yaw. A low-yaw-rate gyroscope window confirms
that the car is not turning; its average cannot physically recover static yaw. This step therefore
leaves mounting yaw at zero instead of inventing an unobservable estimate.

The IEKF now accepts the estimated `R_car_to_imu`. Its initial IMU navigation attitude is
`R_car_to_ENU R_car_to_imu^T`, while its car extrinsic state starts at the estimated rotation and
continues to participate in the existing IEKF updates.

## Vta08 estimate

The automatically selected window was 0.0-30.0 seconds:

| Quantity | Estimate |
|---|---:|
| Mounting roll | +0.006394 deg |
| Mounting pitch | +0.009630 deg |
| Window yaw-rate RMS | 0.017432 rad/s |
| Mean gravity (phone XYZ) | `[0.001648, -0.001094, 9.806585]` m/s^2 |

This recording was already mechanically level to within 0.01 degree, so the estimated correction
is almost identity. During the blackout, the online IEKF extrinsic state changed rotation by
0.004412 degree and translation by 0.009904 m, confirming that the state remains active rather
than being overwritten or frozen by the initializer.

## Identical blackout result

Vta08, 91.6-111.6 seconds, 20 seconds, 141.995 m reference distance:

| Metric | Step 1 strapdown | Step 2 fixed IEKF | Step 3 aligned IEKF |
|---|---:|---:|---:|
| Final position error | 136.731 m | 78.723 m | 78.747 m |
| Position RMSE | 65.015 m | 41.069 m | 41.128 m |
| Mean position error | 50.903 m | 34.108 m | 34.156 m |
| Maximum position error | 136.731 m | 78.723 m | 78.747 m |
| Drift | 96.293% | 55.441% | 55.458% |

Step 3 is effectively flat and numerically worsens final error by 0.024 m, RMSE by 0.058 m, and
drift by 0.017 percentage points relative to Step 2. This is not evidence that alignment is harmful:
the Vta08 phone was already level, leaving no material pitch/roll error for this classical stage to
remove. No angle or window was tuned using VBOX blackout truth.

The remaining large trajectory error is therefore dominated by factors other than static phone
pitch/roll, such as inertial bias/noise, heading evolution, fixed covariance, and the lack of aiding.

## Outputs

- `results/experiments/alignment_vta08/alignment_estimate.json`
- `results/experiments/alignment_vta08/aligned_iekf_trajectory.csv`
- `results/experiments/alignment_vta08/comparison.json`
- `results/metrics/alignment_vta08_metrics.csv`
- `results/plots/alignment_vta08/trajectory_comparison.png`
- `results/plots/alignment_vta08/position_error_comparison.png`

## Tests

The alignment tests cover a synthetic known gravity tilt, earliest-window detection, rejection of a
continuously turning trace, and IEKF use of a supplied mounting rotation.

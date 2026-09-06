# Step 3b: Causal GPS-Heading Mounting Yaw

## Purpose and isolation

Step 3 established roll and pitch from gravity but correctly left mounting yaw unobserved. Step 3b
uses only samples before the locked Vta08 blackout to observe yaw during a short, straight
acceleration. It does not inspect blackout ground truth, tune an angle against endpoint error, or
modify the IEKF.

Run from the repository root:

```bash
MPLCONFIGDIR=/tmp/idr-matplotlib .venv/bin/python -m scripts.run_alignment_yaw_vta08
```

## Frame convention and estimator

The existing convention is vehicle `X=forward, Y=left, Z=up`, and
`R_car_to_imu` maps a car-frame vector into phone IMU coordinates. After gravity and the existing
stationary accelerometer bias are removed, positive straight-line vehicle acceleration is

```text
a_imu_horizontal ~= acceleration * R_car_to_imu [1, 0, 0]^T
yaw_car_to_imu = atan2(mean(a_imu_y), mean(a_imu_x))
R_car_to_imu = R_z(yaw_car_to_imu) R_tilt
```

This direction is accepted only when GPS/VBOX establishes that the vehicle is moving forward,
accelerating, and travelling straight. The deterministic policy chooses the latest qualifying
two-second window that ends before the blackout. Requirements are:

- starting speed at least 0.5 m/s and ending speed at least 4.0 m/s;
- speed gain at least 3.0 m/s and displacement at least 4.0 m;
- displacement course within 5 degrees of circular-mean GPS heading;
- bias-corrected phone yaw-rate RMS no more than 0.08 rad/s.

The smartphone `GPS SPEED (Kmh)` field is not used for this estimator. The loader's independent
unit check finds that its numerical values behave as m/s despite the label (MAE 1.330 m/s when
treated as m/s versus 8.903 m/s when divided as km/h). VBOX `Velocity (km/hr)` is explicitly
converted to m/s and is the speed source used here.

The phone Euler orientation channel has median pitch -88.85 degrees in the selected interval.
Consequently its yaw/roll decomposition is at gimbal lock and is not a safe absolute device
heading. Step 3b records this diagnostic and does not subtract that Euler yaw from GPS heading.
Instead, GPS course and GPS heading define/validate vehicle-forward motion, phone acceleration
observes that forward axis in phone coordinates, and the phone gyroscope rejects turning.

## Automatically selected pre-blackout window

| Quantity | Value |
|---|---:|
| Window | 89.5-91.5 s (indices 895-915) |
| First blackout sample | 91.6 s (index 916) |
| Estimated car-to-phone yaw | **+13.537105 deg** |
| Mean phone horizontal acceleration | `[1.268227, 0.305343]` m/s^2 |
| Speed | 0.532 to 4.878 m/s |
| Speed gain | 4.346 m/s |
| Displacement | 6.002 m |
| Displacement course | 14.558 deg |
| Circular-mean VBOX heading | 14.859 deg |
| Course-heading difference | -0.301 deg |
| Phone yaw-rate RMS | 0.043818 rad/s |

The last estimator sample is index 915. The first blackout sample is index 916, so the leakage
check passes exactly at the boundary.

## Identical blackout result

Vta08, 91.6-111.6 seconds, 20 seconds, 141.995 m reference distance:

| Metric | Step 2 IEKF | Step 3 tilt | Step 3b tilt + yaw |
|---|---:|---:|---:|
| Final position error | 78.723 m | 78.747 m | **32.478 m** |
| Position RMSE | 41.069 m | 41.128 m | **35.013 m** |
| Mean position error | 34.108 m | 34.156 m | **29.177 m** |
| Maximum position error | 78.723 m | 78.747 m | **66.952 m** |
| Drift | 55.441% | 55.458% | **22.872%** |

Relative to Step 2, Step 3b reduces final error by 46.246 m (58.74%), RMSE by 6.056 m
(14.75%), and drift by 32.569 percentage points. Mounting yaw was therefore a major contributor
to the bend. It was not the only contributor: the trajectory still develops a loop and the final
22.87% drift remains above the project target. The CNN adapter should be evaluated on top of this
structural correction, not treated as a substitute for it.

This is a single-drive prototype result. Cross-drive validation is required before treating the
window thresholds or achieved improvement as general.

## Outputs

- `results/experiments/alignment_yaw_vta08/yaw_alignment_estimate.json`
- `results/experiments/alignment_yaw_vta08/yaw_aligned_iekf_trajectory.csv`
- `results/experiments/alignment_yaw_vta08/comparison.json`
- `results/metrics/alignment_yaw_vta08_metrics.csv`
- `results/plots/alignment_yaw_vta08/trajectory_comparison.png`
- `results/plots/alignment_yaw_vta08/position_error_comparison.png`

## Tests

Synthetic tests verify recovery of a known car-to-phone yaw, strict pre-blackout selection, and
rejection of windows without sufficient speed gain or with excessive turning.

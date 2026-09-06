# Step 2 - Fixed-Covariance IEKF on Vta08

## Scope

This step adds a 21-dimensional right-invariant error-state filter for the same 91.6-111.6 s
Vta08 blackout used by Step 1. It does not use GNSS after initialization and contains no learned
covariance, mounting-alignment estimator, mode arbitration, or map matching.

Implementation references:

- Brossard, Barrau, Bonnabel, *AI-IMU Dead-Reckoning*, equations 5-7, 14-15 and Appendix A/B:
  <https://arxiv.org/abs/1904.06064>
- Authors' public NumPy reference implementation:
  <https://github.com/mbrossar/ai-imu-dr/blob/master/src/utils_numpy_filter.py>

## State and error ordering

```text
nominal = (R_imu, v_imu, p_imu, b_gyro, b_accel, R_car_to_imu, p_car_to_imu)
error   = (xi_R, xi_v, xi_p, e_bgyro, e_baccel, xi_Rcar, e_pcar)
```

`(R_imu, v_imu, p_imu)` is embedded in a 5x5 `SE_2(3)` extended-pose matrix. The closed-form
`SO(3)` exponential, left Jacobian, and `SE_2(3)` exponential are implemented directly in
`src/ins/iekf.py`.

The pseudo-measurement is the car-frame lateral and upward velocity:

```text
y = [v_lateral, v_upward] = [0, 0] + noise
```

## Fixed covariance selection

Process and initial covariance values start from the public KITTI implementation. The published
measurement diagonal `[0.2, 300]` was unstable on the 10 Hz Vta08 smartphone IMU, producing
311.53 m final error and 219.4% drift. Its extremely loose upward constraint allowed attitude and
gravity-projection errors to grow.

A small manual stability sweep was recorded in
`results/experiments/iekf_vta08/covariance_trials.json`. The locked value is:

```text
N = diag(10, 10) (m/s)^2
```

This value was chosen over `[1,1]` because the tighter constraint produced a looping trajectory and
worse whole-trajectory RMSE, despite a better endpoint by chance. The selected covariance is still
Vta08-specific and must be validated on another drive before it can be treated as general.

## Result

| Metric | Step 1 strapdown | Step 2 fixed IEKF | Change |
|---|---:|---:|---:|
| Final position error | 136.731 m | 78.723 m | 42.42% lower |
| Position RMSE | 65.015 m | 41.069 m | 36.83% lower |
| Mean position error | 50.903 m | 34.108 m | 32.99% lower |
| Maximum position error | 136.731 m | 78.723 m | 42.42% lower |
| Drift | 96.293% | 55.441% | 40.852 percentage points lower |

The fixed-covariance IEKF meaningfully reduces both endpoint drift and trajectory RMSE on this
specific blackout. It does not yet produce a physically satisfactory path: the trajectory bends and
temporarily reverses direction. This is consistent with uncertain phone-to-car mounting and weakly
observed forward velocity. Step 2 therefore demonstrates improvement, not completion or the target
10% drift.

## Reproduction

```bash
python -m pytest -q
python -m scripts.run_iekf_vta08 --config configs/iekf_vta08.yaml
```

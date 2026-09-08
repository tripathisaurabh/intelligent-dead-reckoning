# Baseline v1 calculation audit

## Result

Most numerical conventions are internally consistent, but one important frame assumption is not validated and one evaluation dependency was mislabeled.

| Check | Result | Evidence |
|---|---|---|
| Time and units | Pass | Both sessions are monotonic at 10 Hz. VBOX km/h is divided by 3.6. Phone speed behaves as m/s despite its CSV label. |
| Accelerometer and gravity | Pass | Accelerometer includes gravity. Bias calibration uses `accelerometer - gravity`; propagation subtracts residual bias and adds ENU gravity once. Mean stationary navigation acceleration is 0.17 m/s² on Vta08 and 0.28 m/s² on S3c. |
| Mounting matrix algebra | Pass | Both matrices have determinant 1 and orthogonality error below 5e-16. `R_imu_to_enu = R_car_to_enu @ R_car_to_imu.T` is consistent with the documented car-to-IMU convention. |
| Initialization | Pass, oracle | Position starts at local ENU zero; VBOX speed and heading at the first blackout sample initialize the state. This isolates outage drift but is not a cold-start test. |
| Training/runtime adaptive IEKF | Pass | A new non-zero adapter parity test compares position, velocity, attitude, biases and covariance. It passes to numerical precision. |
| Gyroscope body axes | **Fail / unsupported** | The dataset documents semantic `Yaw, Pitch, Roll` columns, not Cartesian `X, Y, Z`. The code treats `[Roll, Pitch, Yaw]` as body `[x,y,z]`. On S3c, the assumed z channel correlates only 0.058 with VBOX yaw rate while the assumed y channel correlates 0.953. |
| Mounting-yaw data source | **Evaluation-only dependency** | The routine is named GPS-heading alignment, but the runners pass VBOX position, speed and heading. Vta08 has sufficient phone GNSS in that window; S3c has one stale phone fix, zero phone displacement and zero phone speed in the selected window. |

The detailed measurements are in [baseline_v1_calculations.json](../results/audits/baseline_v1_calculations.json).

## What was corrected now

1. The original model, checkpoint, configs and intervals remain immutable under the `baseline-v1` tag.
2. A non-zero learned-correction parity test now prevents the PyTorch training IEKF and NumPy runtime IEKF from silently diverging.
3. Reports now call the yaw calibration **VBOX-assisted reference alignment**, not phone-only GPS calibration.
4. The unsupported gyroscope-axis assumption is explicitly failed rather than marked correct.

The exact Vta08 and S3c intervals were also rerun with an axis-independent forward baseline that
holds the trusted entry speed constant. On Vta08 its speed MAE is 2.05 m/s and final distance error
is -41.22 m; on S3c its speed MAE is 6.35 m/s and final distance error is -128.14 m. These are not
complete navigation solutions, but they are the corrected comparison for forward motion because
they make neither the invalid gyro-body mapping nor an unverified accelerometer-axis projection.

It would be scientifically wrong to choose a new gyro permutation from Vta08 or S3c endpoint error and call that a fix. The correct next implementation is a planar vehicle model that calibrates its yaw channel/sign from GNSS-aided training data, plus an independently evaluated forward-speed estimator. Until that is validated, Step 6's 11.15% Vta08 result remains a Baseline v1 pipeline result, not evidence that the underlying inertial trajectory is correct.

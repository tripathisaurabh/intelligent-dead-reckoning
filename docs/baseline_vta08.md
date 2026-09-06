# Vta08 Unfiltered Calibrated INS Baseline

## Experiment question

Given a VBOX-initialized navigation state and only smartphone inertial measurements for 20 seconds,
how far does a basic unfiltered strapdown INS drift from the synchronized VBOX reference?

This experiment is a baseline. It intentionally excludes denoising, trajectory smoothing, ZUPT,
ESKF updates, non-holonomic constraints, machine learning, and map matching.

## Data contract

- Smartphone source: synchronized `S-Vta8.csv`.
- Reference source: synchronized `V-vta8.csv`.
- Smartphone timestamp: `TIME SINCE START (ms)`, converted to elapsed seconds.
- VBOX timestamp: `Time Since Start of Day (seconds)`, converted to elapsed seconds.
- VBOX speed: km/h converted to m/s.
- VBOX heading: degrees clockwise from North. This convention is checked against the course calculated
  from consecutive VBOX latitude/longitude positions.
- The smartphone speed field is retained for unit inspection and is not used by the estimator.

## Frames

The IO-VNBD mounting diagram is used for the first baseline:

- Device/vehicle X: forward.
- Device/vehicle Y: left.
- Device/vehicle Z: up.
- Navigation frame: East-North-Up (ENU).

The source gyroscope order is semantic yaw, pitch, roll. INS body angular velocity is ordered as
roll, pitch, yaw before integration. The phone-to-vehicle mounting assumption is a known limitation;
automated straight-line alignment is a later ablation.

For a navigation bearing `psi`, the initial velocity is:

```text
v_east  = speed * sin(psi)
v_north = speed * cos(psi)
```

## Calibration and propagation

The longest qualifying stationary run is selected before the outage. Gyroscope bias is the stationary
mean. Device-frame residual accelerometer bias is estimated only after subtracting the dataset-provided
device gravity vector:

```text
a_linear_device = a_measured_device - gravity_device - accel_bias_device
```

The initial vehicle-to-ENU attitude is created from the VBOX heading. During blackout, attitude is
propagated exclusively from bias-corrected smartphone gyroscope measurements:

```text
R[k] = R[k-1] * Exp((omega[k] - gyro_bias) * dt)
```

Acceleration, velocity, and position propagation are:

```text
a_enu[k] = R[k] * a_linear_device[k]
v[k] = v[k-1] + 0.5 * (a[k-1] + a[k]) * dt
p[k] = p[k-1] + 0.5 * (v[k-1] + v[k]) * dt
```

The INS input type contains no GNSS, VBOX, or reference measurement fields. Reference values enter only
at initialization and are used again after propagation for evaluation.

## Blackout selection

The code selects the earliest interval after calibration that satisfies the configured duration, boundary
speed, moving-sample fraction, timestamp-gap, and surrounding-context rules. Selection is completed
before seeing INS error, preventing selection of an artificially favorable interval.

## Reproduction

From the repository root:

```bash
python -m pytest -q
python -m scripts.run_baseline_vta08 --config configs/baseline_vta08.yaml
```

Generated experiment data, metrics, metadata, and plots are written under `results/experiments/`,
`results/metrics/`, and `results/plots/`.

## Interpretation limitation

VBOX is treated as the benchmark reference, not as an assertion of perfect absolute ground truth. The
baseline also assumes the documented phone mounting is exact. Its result must not be presented as the
performance of the final IDR architecture.

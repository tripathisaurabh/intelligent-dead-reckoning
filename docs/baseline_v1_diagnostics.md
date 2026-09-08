# Baseline v1 diagnostic report (map matching disabled)

## Direct answer

The two failures are different:

- **Vta08 fails in speed first.** Absolute speed error stays above 2 m/s from 1.0 s, heading error stays above 10° from 4.2 s, and position error stays above 5 m from 2.5 s.
- **S3c fails in both direction and speed almost immediately.** Heading fails first at 0.401 s, speed follows at 0.700 s, and position exceeds 5 m at 1.600 s.

These are diagnostic thresholds, not accuracy scores. A crossing is counted only when it remains beyond the threshold for a full second.

| Session | First failure | Speed MAE | Final speed error | Heading MAE | Final heading error | Final distance error | Final position error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vta08 | Speed | 3.44 m/s | +10.69 m/s | 24.58° | -22.70° | +115.38 m | 29.77 m |
| S3c | Direction, then speed | 20.57 m/s | +35.49 m/s | 116.57° | -107.37° | +331.42 m | 242.50 m |

The distance error is cumulative path length error. It can be much larger than endpoint position error when the estimated path bends or loops.

## Judge-ready plots

- [Vta08 four-panel diagnostic](../results/diagnostics/baseline_v1/Vta08/kinematics_diagnostic.png)
- [S3c four-panel diagnostic](../results/diagnostics/baseline_v1/S3c/kinematics_diagnostic.png)
- [Machine-readable summary](../results/diagnostics/baseline_v1/summary.json)

Each plot contains estimated/reference speed, cumulative distance, vehicle heading and horizontal position error for the exact Baseline v1 interval. The plotted estimator is the Step 4 CNN-adapted IEKF. GNSS updates and Step 6 map matching are disabled.

## Interpretation

Vta08 initially underestimates speed, then sharply overestimates it after about 12 s. Its heading also develops a large temporary bend. S3c is more severe: heading turns opposite to the reference almost immediately while speed grows to more than 50 m/s. Therefore another map-matching layer cannot repair the root cause. Speed and direction must be validated independently before map matching.

Run the diagnostic again with:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib-cache .venv/bin/python scripts/run_baseline_v1_diagnostics.py
```

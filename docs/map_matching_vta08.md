# Step 6 — OSM HMM Map Matching

## Implementation

Step 6 is a downstream map constraint and does not modify or feed corrections
back into the IEKF. The matcher:

- parses a local OpenStreetMap XML extract into a drivable road graph;
- projects each Step 5 position onto twelve candidate road segments;
- scores candidates using perpendicular distance and heading agreement;
- scores transitions using graph path distance, direction consistency, and road-turn continuity;
- applies fixed-lag Viterbi decoding over a five-second sliding window.

Heading comes from bias- and CNN-corrected gyro integration initialized at the
blackout boundary. VBOX heading is not used after initialization. This matters:
differentiating the drifting position trace produced false headings when the
trajectory looped and caused plausible but incorrect side-road matches.

The OSM extract contains 1,279 drivable segments, 1,263 graph nodes, and 1,279
graph edges. OpenStreetMap data is licensed under ODbL.

## Locked Vta08 blackout result

Interval: 91.6–111.6 s, 201 samples, 141.995 m reference distance.

| Metric | Step 5 input | Step 6 OSM HMM | Change |
|---|---:|---:|---:|
| Final position error | 29.770 m | 15.827 m | -13.944 m |
| Position RMSE | 27.845 m | 19.243 m | -8.602 m |
| Mean position error | 23.114 m | 16.235 m | -6.879 m |
| Maximum position error | 66.960 m | 42.437 m | -24.522 m |
| Drift | 20.966% | 11.146% | -9.820 percentage points |

The decoded path stays on one OSM way and uses seven connected segments. Mean
filter-to-road snap distance is 14.375 m and mean heading disagreement is 6.662°.

Map matching removes cross-road drift and the false loop, but it cannot fully
observe distance travelled along a straight road. The remaining 15.827 m is
primarily along-road error. Reducing that further requires a better inertial
speed estimate or another landmark/range observation, not stronger snapping.

## Cross-session validation: Vta16

The exact matcher parameters above were frozen and applied without Vta16 tuning
to the Step 4 Vta16 validation blackout at 413.7–433.7 s.

| Metric | Vta16 Step 4 | Vta16 Step 6 | Change |
|---|---:|---:|---:|
| Final position error | 73.397 m | 82.821 m | +9.423 m |
| Position RMSE | 68.090 m | 63.177 m | -4.913 m |
| Mean position error | 52.445 m | 48.651 m | -3.794 m |
| Maximum position error | 144.261 m | 141.728 m | -2.533 m |
| Drift | 50.558% | 57.049% | +6.491 percentage points |

This is a mixed and unsuccessful generalization result: average trajectory
error improves, but the judge-facing endpoint drift metric gets worse. The HMM
selects one consistent road and its final point is only 2.143 m from the driven
reference path, but it corresponds to approximately 7.8 s into the 20 s route,
leaving 90.751 m of reference road progress unaccounted for. The map matcher can
identify the road but cannot recover longitudinal distance after the upstream
Vta16 inertial trajectory loops.

Vta16 was excluded from gradient updates but was used for Step 4 epoch selection,
so this is a cross-session validation result rather than a pristine held-out test.
Step 6 is implemented, but broad generalization is **not established**. A third
untouched drive and a better along-road speed/progress model are required before
making a generalization claim.

## Reproduce

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib-step6 \
  .venv/bin/python scripts/run_map_matching_vta08.py
```

Outputs:

- `results/experiments/map_matching_vta08/evaluation.json`
- `results/experiments/map_matching_vta08/matched_trajectory.csv`
- `results/metrics/map_matching_vta08_metrics.csv`
- `results/plots/map_matching_vta08/osm_hmm_comparison.png`
- `results/experiments/map_matching_vta16/evaluation.json`
- `results/plots/map_matching_vta16/osm_hmm_comparison.png`

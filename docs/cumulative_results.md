# Cumulative Vta08 Blackout Results

Locked evaluation interval: Vta08, 91.6-111.6 s, 20 seconds, 141.995 m reference distance.

| Step | Method | Drift % | Final error | RMSE | Notes |
|---|---|---:|---:|---:|---|
| 1 | Plain strapdown integration | 96.293% | 136.731 m | 65.015 m | Complete |
| 2 | Fixed-covariance IEKF + NHC | 55.441% | 78.723 m | 41.069 m | `N=diag(10,10)`; Vta08-specific manual stability selection |
| 3 | + gravity mounting alignment | 55.458% | 78.747 m | 41.128 m | Complete; Vta08 tilt <0.01 deg, effectively flat vs Step 2 |
| 3b | + causal GPS-heading mounting yaw | 22.872% | 32.478 m | 35.013 m | +13.537 deg yaw; major structural bend reduction |
| 4 | + dilated CNN IEKF adapter | 20.966% | 29.770 m | 27.845 m | 10,411 params; Vta2 train, Vta16 validation, Vta08 test |
| 5 | + GNSS fusion/arbitration | 20.966% | 29.770 m | 27.845 m | Fresh oracle-initialized Step 5 run; Δ endpoint vs Step 4 = 0.0000065 m; recovery blend verified separately |
| 6 | + downstream OSM HMM map matching | 11.146% | 15.827 m | 19.243 m | Fixed-lag Viterbi; one connected OSM way; no filter feedback |

No value in this table should be interpreted as held-out or cross-drive performance until a later
step explicitly reports that experiment.

## Cross-session Step 6 validation

The Vta08 Step 6 parameters were frozen and reused unchanged on Vta16's existing
Step 4 validation blackout (413.7–433.7 s).

| Session | Before map matching | After map matching | Endpoint change | RMSE change | Conclusion |
|---|---:|---:|---:|---:|---|
| Vta08 | 20.966% | 11.146% | -13.944 m | -8.602 m | Improved |
| Vta16 | 50.558% | 57.049% | +9.423 m | -4.913 m | Mixed; endpoint drift worsened |

Vta16 demonstrates road identification but not longitudinal recovery: the final
matched point is close to the correct driven road approximately 90.751 m behind
the reference endpoint. Generalization is therefore not established.

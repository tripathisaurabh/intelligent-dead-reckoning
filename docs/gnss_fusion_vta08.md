# Step 5 — GNSS Fusion, Mode Arbitration, and Reacquisition

## Scope

Step 5 adds horizontal phone-GNSS position/velocity updates to the existing
CNN-adapted right-invariant IEKF. It includes:

- satellite-count, reported-accuracy, freshness, and chi-square innovation gates;
- `GNSS_AIDED`, `DEAD_RECKONING`, and `TRANSITIONING` modes;
- immediate dead reckoning during the synthetic 91.6–111.6 s outage;
- a three-second recovery blend, capped at 1 m of position correction per sample;
- a 90.102–140.0 s Vta08 evaluation covering the outage and real reacquisition.

Map matching is deliberately excluded.

The synchronized phone log repeats each physical GNSS fix at the 10 Hz IMU
rate. Step 5 detects changes in the GNSS tuple and fuses each distinct source fix
once. In this Vta08 section only six distinct fixes exist; fixes inside the
synthetic blackout are discarded. The first usable post-outage fix is at 117.1 s.

## Results

The isolated locked blackout was re-run through Step 5 with the same oracle VBOX
initialization used by Steps 1–4. It produced **20.965747% drift**, **29.770286 m**
endpoint error, and **27.845303 m RMSE**. Its endpoint differs from Step 4 by only
0.0000065 m (floating-point/geodetic-origin rounding). This fresh run confirms
that Step 5 does not change blackout propagation when GNSS is unavailable.

On the separate causal extended trace, phone GNSS gives a 1.389 m horizontal
position error and 0.082 m/s horizontal velocity error at blackout entry. The
first post-outage fix enters `TRANSITIONING` at 117.1 s with exactly 0 m immediate
position correction. This is intentional: reacquisition starts at blend weight
zero to guarantee no position jump. At 117.201 s the blend weight reaches
0.033667 and the first non-zero correction is applied. The largest subsequent
transition correction is capped at 1 m/sample.

The extended-trace endpoint error is 196.203 m with GNSS arbitration versus
216.941 m for dead reckoning from the same phone-GNSS initialization. This is a
recovery improvement, not evidence that Step 5 reduces inertial drift in the
blackout. The causal blackout propagation is also materially worse than the
locked Step 4 benchmark because Step 4 uses an oracle VBOX speed/bearing state at
91.6 s, whereas this trace initializes from the sparse phone fix at 90.102 s.
Those two experiments must not be compared as if they used the same initializer.

Two later distinct fixes are rejected by the innovation gate. That result is
useful: Vta08's sparse phone GNSS is not sufficient to demonstrate continuously
GNSS-aided navigation, but the outage and non-jarring recovery state machine are
directly exercised.

## Reproduce

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib-step5 \
  .venv/bin/python scripts/run_gnss_fusion_vta08.py
```

Outputs:

- `results/experiments/gnss_fusion_vta08/evaluation.json`
- `results/experiments/gnss_fusion_vta08/extended_trace.csv`
- `results/experiments/gnss_fusion_vta08/oracle_initialized_blackout.csv`
- `results/metrics/gnss_fusion_vta08_metrics.csv`
- `results/plots/gnss_fusion_vta08/extended_trace.png`

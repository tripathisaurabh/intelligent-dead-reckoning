# Intelligent Dead Reckoning

Smartphone-based GNSS-denied navigation system for SIH.

The aim of this project is to keep estimating a vehicle's position, velocity and heading when GNSS/GPS becomes weak or completely unavailable, using only the sensors available inside a smartphone.

The system is being designed around a hybrid navigation pipeline using smartphone IMU data, classical inertial navigation, sensor fusion, machine learning, vehicle constraints and road-network map matching.

## Why this project exists

Normal navigation apps work well while GNSS is available. The problem starts when a vehicle enters places such as tunnels, underground parking areas, dense urban roads or other environments where satellite signals are blocked or unreliable.

When that happens, the location may freeze, jump or become inaccurate.

Our goal is to continue navigation during these temporary outages without requiring OBD-II, wheel-speed sensors or any other physical connection to the vehicle.

## Main target

We are working toward keeping positional drift below roughly 10% of the distance travelled during a GNSS blackout wherever the available smartphone sensor quality makes that achievable.

This will be treated as an experimental target. We will not claim accuracy until it is measured on held-out data and controlled GNSS-outage tests.

## Current system pipeline

```text
Smartphone Sensors
        |
        +------------------+
        |                  |
       GNSS               IMU
                    Accel / Gyro / Mag
        |                  |
        |           Preprocessing
        |                  |
        |        Orientation + Alignment
        |                  |
        |             CNN / GRU
        |                  |
        |            INS Propagation
        |                  |
        +--------- ESKF ---+
                    |
                  NHC
                    |
              Map Matching
             HMM / Viterbi
                    |
             Navigation State
                    |
             Android App
```

## Build order

We are deliberately not trying to build everything at once.

### Phase 1 — Dataset understanding and preprocessing

- Inspect IO-VNBD dataset structure
- Verify sensor columns and units
- Measure actual sampling frequencies from timestamps
- Check missing values and timestamp gaps
- Plot raw IMU and GNSS data
- Identify stationary segments
- Estimate initial IMU bias
- Build preprocessing pipeline
- Validate orientation and gravity removal
- Export the first clean drive

### Phase 2 — Basic INS baseline

- Implement simple inertial propagation
- Test using synthetic motion first
- Run on processed IO-VNBD data
- Measure how quickly pure inertial navigation drifts

### Phase 3 — ESKF fusion

- Implement INS error-state model
- Add GNSS measurement updates
- Verify stable fused navigation while GNSS is available
- Inject controlled GNSS outages and measure drift

### Phase 4 — Machine learning

Current direction:

- Temporal IMU windows as input
- CNN-GRU / GRU-based motion model
- Predict short-window displacement or correction
- Use model output as an additional measurement/correction inside the fusion pipeline

The exact ML target will be finalized after the dataset and baseline experiments are complete.

### Phase 5 — Vehicle constraints

Implement Non-Holonomic Constraints (NHC) to reduce unrealistic lateral and vertical motion for normal four-wheel vehicle movement.

### Phase 6 — Map matching

Use OpenStreetMap road-network data with HMM/Viterbi map matching to constrain the estimated trajectory to plausible roads.

### Phase 7 — Android deployment

Once the Python pipeline is validated:

- Read smartphone sensors in real time
- Run the lightweight model locally
- Execute dead reckoning / fusion on-device
- Display continuous navigation on an offline-capable map
- Show navigation mode and uncertainty

## Repository structure

```text
intelligent-dead-reckoning/
|
|-- data/
|   |-- raw/                 # Original datasets. Not committed to GitHub.
|   |-- processed/           # Cleaned/synchronized data. Not committed.
|   `-- splits/              # Train/validation/test drive lists
|
|-- notebooks/               # Exploration only
|
|-- src/
|   |-- preprocessing/       # Loading, filtering, timing, bias, orientation
|   |-- ins/                 # Strapdown INS / dead-reckoning core
|   |-- fusion/              # ESKF and GNSS fusion
|   |-- models/              # GRU / CNN-GRU / TCN experiments
|   |-- constraints/         # NHC and vehicle-motion constraints
|   |-- map_matching/        # OSM + HMM/Viterbi
|   |-- evaluation/          # Metrics, outage simulation and experiments
|   `-- utils/               # Shared math and coordinate utilities
|
|-- configs/                 # Experiment and preprocessing settings
|-- tests/                   # Unit and numerical tests
|
|-- results/
|   |-- phase1/
|   |-- plots/
|   |-- metrics/
|   |-- models/
|   `-- experiments/
|
|-- android/                 # Final Android application
|-- scripts/                 # Repeatable CLI scripts
|-- docs/                    # Architecture, plans and technical decisions
`-- team/                    # Short team handoff / working notes
```

## Dataset

Primary dataset for the project:

**IO-VNBD — Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning**

Raw datasets must not be uploaded directly to this repository. Keep them locally under `data/raw/`.

Train, validation and test data must be separated by complete drives/trajectories to avoid leakage between windows from the same journey.

## Development rules

A few rules are important for this project:

1. Do not push raw datasets or large trained models to GitHub.
2. Do not change coordinate-frame conventions without documenting the change.
3. Keep internal navigation calculations in consistent SI units.
4. Do not use future GNSS information inside a simulated GNSS blackout.
5. Do not train using test trajectories.
6. Do not call an algorithm successful just because the code runs.
7. Save metrics and comparisons for every meaningful experiment.
8. Core navigation code should remain independent of the Android UI.
9. Validate navigation mathematics in Python before rewriting it in Kotlin.
10. Avoid direct experimental commits to `main`; use branches and pull requests once team development starts.

## Evaluation approach

We will inject controlled GNSS outages into held-out drives, for example:

- 10 seconds
- 30 seconds
- 60 seconds
- longer intervals where useful

Important metrics include:

- Position RMSE
- Mean position error
- Final displacement error
- Drift percentage
- Velocity error
- Heading error
- GNSS reacquisition jump
- Runtime / inference latency

The full system should eventually be compared through an ablation sequence:

```text
Raw double integration
        ->
Basic INS
        ->
INS + GNSS fusion
        ->
INS + ESKF
        ->
INS + ESKF + ML
        ->
INS + ESKF + ML + NHC
        ->
Full system + map matching
```

## Current milestone

The first milestone is intentionally small:

```text
RAW IO-VNBD DRIVE
        ->
VERIFIED TIMESTAMPS
        ->
CLEAN IMU
        ->
PROCESSED DRIVE
        ->
BASIC INS INPUT
```

Once this works reliably, we move to measured INS drift and then begin improving it one component at a time.

## Team workflow

For the first development cycle:

- **Duo 1:** Dataset + preprocessing
- **Duo 2:** INS baseline
- **Duo 3:** ESKF preparation

GitHub Projects will be used to track tasks through:

```text
To Do -> In Progress -> Done
```

As implementation becomes more active, meaningful coding tasks should be linked to GitHub Issues and completed through feature branches / pull requests.

## Project status

**Current stage:** Phase 1 — Dataset Understanding & Preprocessing

The architecture is decided for the first implementation cycle, but individual algorithm choices may still change if experiments show a better option.

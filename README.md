# Intelligent Dead Reckoning

Smartphone-based GNSS-denied navigation system for SIH.

The aim of this project is to keep estimating a vehicle's position, velocity and heading when GNSS/GPS becomes weak or completely unavailable, using only the sensors available inside a smartphone.

The system is being built as a hybrid navigation engine using smartphone IMU data, inertial navigation, GNSS/INS fusion, machine learning, vehicle-motion constraints and road-network map matching.

## Problem we are solving

Normal navigation apps work well while GNSS is available. The problem starts when a vehicle enters a tunnel, underground parking area, dense urban road or another environment where satellite signals are blocked or unreliable.

When this happens, location may freeze, jump or become inaccurate.

Our goal is to continue estimating the vehicle's motion during these temporary GNSS outages without requiring OBD-II, wheel-speed sensors or any physical connection to the vehicle.

## Main target

We are working toward restricting positional drift to roughly 10% of the distance travelled during GNSS blackout wherever smartphone sensor quality and driving conditions make that achievable.

This is an experimental target. Accuracy will only be claimed after it is measured on held-out data and controlled GNSS-outage tests.

---

# Development approach

We are not starting by building the complete application.

The order is:

```text
Understand the dataset
        ↓
Preprocess and validate the sensor data
        ↓
Build a basic INS baseline
        ↓
Build GNSS + INS fusion using ESKF
        ↓
Add the learned motion/displacement model
        ↓
Add vehicle constraints (NHC)
        ↓
Add road-network map matching
        ↓
Evaluate the complete pipeline
        ↓
Deploy the validated engine on Android
```

---

# Correct system architecture

The IMU, GNSS and learned motion model are not simply arranged in one straight chain. They provide different information to the navigation estimator.

```text
                          SMARTPHONE SENSORS
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                   GNSS                        IMU
             position / speed          accel / gyro / mag
                    │                           │
                    │                    PREPROCESSING
                    │                           │
                    │              ┌────────────┴────────────┐
                    │              │                         │
                    │       Orientation +               ML Motion Model
                    │       Vehicle Alignment          CNN-GRU / GRU
                    │              │                         │
                    │              │                  displacement /
                    │              │                   correction
                    │              │                         │
                    │        STRAPDOWN INS                    │
                    │              │                         │
                    │        state propagation               │
                    │              │                         │
                    └──────────────┼──────────────┬──────────┘
                                   │              │
                                   ▼              ▼
                              GNSS / INS / ML
                                ESKF FUSION
                                   ▲
                                   │
                          Non-Holonomic Constraint
                          lateral / vertical velocity
                          pseudo-measurements
                                   │
                                   ▼
                          FUSED NAVIGATION STATE
                                   │
                                   ▼
                             MAP MATCHING
                             HMM / Viterbi
                                   │
                                   ▼
                         FINAL NAVIGATION STATE
                                   │
                                   ▼
                             ANDROID APP
```

A GNSS availability state machine will decide whether GNSS measurements are trusted, degraded, unavailable or being reacquired.

During GNSS availability, GNSS measurements correct the inertial state through the ESKF.

During GNSS outage, the INS continues propagating the state using IMU measurements while ML and vehicle constraints help reduce drift.

When GNSS returns, it is fused back into the existing state instead of hard-resetting the position.

---

# Phase 1 — Dataset Understanding and Preprocessing

This is the current project stage.

We first analyse IO-VNBD together before fixing preprocessing parameters or training any model.

## Step 1 — Dataset analysis

Before cleaning anything, we need to understand what is actually present in the dataset.

We will inspect:

- file and drive structure
- all available columns
- sensor units
- timestamp format
- actual accelerometer sampling rate
- actual gyroscope sampling rate
- GNSS update rate
- missing values
- duplicated timestamps
- timestamp gaps
- sensor spikes and unusual values
- stationary sections
- real GNSS dropouts already present in the data
- available orientation / gravity information
- GNSS accuracy, speed and heading fields
- available reference / ground-truth signals

We will not assume a 100 Hz IMU rate until it is measured from timestamps.

## Step 2 — Preprocessing pipeline

After the dataset has been inspected, the first preprocessing pipeline will be built in this order:

```text
RAW IO-VNBD DATA
       │
       ▼
Data validation
       │
       ▼
Timestamp cleaning
       │
       ▼
Actual sampling-rate estimation
       │
       ▼
Sensor synchronization / resampling
       │
       ▼
Outlier and isolated-spike detection
       │
       ▼
IMU denoising
       │
       ▼
Stationary-segment detection
       │
       ▼
Initial IMU bias estimation
       │
       ▼
Orientation estimation
       │
       ▼
Gravity removal
       │
       ▼
Phone-to-vehicle alignment
       │
       ▼
Coordinate-frame conversion
       │
       ▼
Unit standardization
       │
       ▼
Feature generation / normalization
       │
       ▼
CLEAN IMU + GNSS DATA
```

## Techniques planned for preprocessing

### 1. Timestamp validation

Calculate consecutive time differences and determine the actual sensor frequency from the data.

We will detect:

- duplicate timestamps
- non-monotonic timestamps
- large gaps
- irregular sampling

The median time difference will be used as the first robust estimate of nominal sampling frequency.

### 2. Sensor synchronization and resampling

Different sensor streams may arrive at different rates.

Continuous sensor measurements can initially be aligned using timestamp-based interpolation to a common time grid.

The target rate will be chosen after inspecting the real dataset. Interpolating low-rate data to 100 Hz will not be treated as creating new sensor information.

### 3. Outlier / isolated-spike detection

Use robust statistics such as:

- Median Absolute Deviation (MAD)
- Hampel filtering

These will mainly be used to identify isolated abnormal samples.

Real vehicle events such as braking, turns, potholes and shocks must not be blindly removed because they contain useful motion information.

### 4. IMU denoising

Primary first method:

**Butterworth low-pass filtering**

Starting configuration to investigate:

- 4th-order Butterworth
- accelerometer cutoff roughly 15–20 Hz
- gyroscope cutoff roughly around 10 Hz

These values are not final and will be selected after inspecting the frequency content and actual sampling rate.

For offline analysis, zero-phase filtering can be used to study the signal without phase delay. The later real-time phone implementation must use a causal version.

### 5. Stationary detection

Stationary segments can be detected from a combination of:

- very low GNSS speed
- low gyroscope magnitude
- low acceleration variance
- acceleration magnitude close to gravity

These regions are useful for IMU calibration.

### 6. Initial accelerometer and gyroscope bias estimation

Use stationary-window statistics to estimate the initial sensor biases.

Gyroscope bias can be estimated from the mean angular-rate reading while the system is stationary.

Accelerometer calibration must account for gravity rather than simply subtracting the XYZ mean blindly.

The ESKF can later continue estimating slowly changing bias during navigation.

### 7. Orientation estimation

Primary first method:

**Madgwick AHRS**

Inputs:

- accelerometer
- gyroscope
- magnetometer, when reliable

Output:

- orientation quaternion

Because magnetometers can be disturbed inside vehicles, magnetometer measurements should be checked before being trusted strongly.

### 8. Gravity removal

After orientation is known, acceleration is rotated from the device frame into a level/navigation frame and the gravity vector is removed.

A basic validation test is:

```text
Stationary vehicle
      ↓
gravity removed
      ↓
linear acceleration should remain close to zero
```

### 9. Phone-to-vehicle alignment

Initial approach:

**PCA + GNSS heading/course cross-check**

Straight, non-zero-speed driving sections can be used to estimate the dominant vehicle-forward direction and determine the fixed rotation between the phone frame and vehicle frame.

Continuous re-alignment while the phone moves on its mount is not part of the first implementation.

### 10. Coordinate frames

The project must explicitly distinguish:

```text
Device Frame
     ↓
Vehicle Frame
     ↓
Navigation Frame
```

A local ENU (East, North, Up) navigation frame is the current preferred representation for short ground-vehicle trajectories.

### 11. Unit standardization

Internal calculations will use consistent SI units where possible:

- distance → metres
- velocity → metres/second
- acceleration → metres/second²
- angular velocity → radians/second
- time → seconds
- angles → radians internally

### 12. ML feature preparation

After physical preprocessing is validated, ML input features may include:

- accel x/y/z
- gyro x/y/z
- acceleration magnitude
- gyroscope magnitude

This gives 8 features if only these values are used.

Normalization parameters must be calculated using training data only, not the complete dataset.

---

# What we are NOT doing yet

During Phase 1 we are not trying to finish:

- CNN-GRU training
- final ESKF tuning
- NHC tuning
- map matching
- Android UI
- final accuracy claims

The purpose of Phase 1 is to make sure the data entering those modules is physically and numerically trustworthy.

---

# Repository structure

```text
intelligent-dead-reckoning/
│
├── data/
│   ├── raw/                 # Original datasets. Not committed to GitHub.
│   ├── processed/           # Cleaned/synchronized data. Not committed.
│   └── splits/              # Train/validation/test drive lists
│
├── notebooks/               # Dataset exploration and temporary experiments
│
├── src/
│   ├── preprocessing/       # Loading, timing, filtering, bias, orientation
│   ├── ins/                 # Strapdown INS / dead reckoning
│   ├── fusion/              # ESKF and GNSS fusion
│   ├── models/              # GRU / CNN-GRU / TCN experiments
│   ├── constraints/         # NHC and other vehicle-motion constraints
│   ├── map_matching/        # OSM + HMM/Viterbi
│   ├── evaluation/          # Metrics and GNSS-outage experiments
│   └── utils/               # Shared mathematics and coordinate utilities
│
├── configs/                 # Experiment and preprocessing settings
├── tests/                   # Unit and numerical tests
│
├── results/
│   ├── phase1/
│   ├── plots/
│   ├── metrics/
│   ├── models/
│   └── experiments/
│
├── android/                 # Final Android application
├── scripts/                 # Repeatable command-line scripts
└── docs/                    # Plans, architecture and technical decisions
```

---

# Dataset

Primary dataset:

**IO-VNBD — Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning**

Raw datasets should remain local under `data/raw/` and should not be committed to GitHub.

Train, validation and test data must later be separated by complete drives/trajectories so windows from the same journey do not leak between splits.

---

# Evaluation approach

After the baseline navigation pipeline works, controlled GNSS outages will be injected into held-out drives, for example:

- 10 seconds
- 30 seconds
- 60 seconds
- longer intervals where useful

Important metrics include:

- position RMSE
- mean position error
- final displacement error
- drift percentage
- velocity error
- heading error
- GNSS reacquisition behaviour
- inference/runtime performance

The project will eventually compare progressively stronger configurations:

```text
Raw double integration
        ↓
Basic INS
        ↓
INS + GNSS fusion
        ↓
INS + ESKF
        ↓
INS + ESKF + ML
        ↓
INS + ESKF + ML + NHC
        ↓
Full system + map matching
```

---

# Current milestone

```text
RAW IO-VNBD DRIVE
        ↓
DATASET UNDERSTOOD
        ↓
TIMESTAMPS + UNITS VERIFIED
        ↓
PREPROCESSING VALIDATED
        ↓
CLEAN IMU + GNSS
        ↓
READY FOR BASIC INS
```

The entire team is currently focused on understanding the dataset and preprocessing decisions before splitting into separate implementation modules.

**Current stage: Phase 1 — Dataset Understanding & Preprocessing**

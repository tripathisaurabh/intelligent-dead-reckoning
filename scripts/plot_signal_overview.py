from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocessing.analysis import read_table


def pick_column(df: pd.DataFrame, text: str) -> str:
    matches = [c for c in df.columns if text.lower() in str(c).lower()]
    if not matches:
        raise ValueError(f"Could not find column containing: {text}")
    return matches[0]


def save_plot(x, ys, labels, title, ylabel, path, vertical_lines=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    for y, label in zip(ys, labels):
        ax.plot(x, y, label=label, linewidth=0.8)

    if vertical_lines:
        for xpos, label in vertical_lines:
            ax.axvline(x=xpos, linestyle="--", linewidth=1.0, label=label)

    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_route_plot(lon, lat, path):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(lon, lat, linewidth=0.8)
    ax.set_title("GPS Route Overview")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Phase 1 overview plots for one IO-VNBD CSV file.")
    parser.add_argument("csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase1/signal_overview"),
    )
    args = parser.parse_args()

    df = read_table(args.csv)
    x = np.arange(len(df))

    time_col = pick_column(df, "time since start")
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    dt = np.diff(t)
    reset_rows = (np.where(dt <= 0)[0] + 1).tolist()
    gap_rows = (np.where(dt > 250)[0] + 1).tolist()

    markers = []
    for i, row in enumerate(reset_rows):
        markers.append((row, "timestamp reset" if i == 0 else None))
    for i, row in enumerate(gap_rows):
        markers.append((row, "large time gap" if i == 0 else None))

    accel_cols = [
        pick_column(df, "accelerometer x"),
        pick_column(df, "accelerometer y"),
        pick_column(df, "accelerometer z"),
    ]
    gyro_cols = [
        pick_column(df, "gyroscope yaw"),
        pick_column(df, "gyroscope pitch"),
        pick_column(df, "gyroscope roll"),
    ]
    gravity_cols = [
        pick_column(df, "gravity x"),
        pick_column(df, "gravity y"),
        pick_column(df, "gravity z"),
    ]

    speed_col = pick_column(df, "gps speed")
    accuracy_col = pick_column(df, "gps accuracy")
    lat_col = pick_column(df, "gps latitude")
    lon_col = pick_column(df, "gps longitude")

    out_dir = args.output / args.csv.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    save_plot(
        x,
        [df[c] for c in accel_cols],
        ["X", "Y", "Z"],
        "Raw Accelerometer - Full Recording",
        "Acceleration (m/s²)",
        out_dir / "01_accelerometer_full.png",
        markers,
    )

    save_plot(
        x,
        [df[c] for c in gyro_cols],
        ["Yaw", "Pitch", "Roll"],
        "Raw Gyroscope - Full Recording",
        "Angular velocity (rad/s)",
        out_dir / "02_gyroscope_full.png",
        markers,
    )

    save_plot(
        x,
        [df[c] for c in gravity_cols],
        ["X", "Y", "Z"],
        "Gravity Vector - Full Recording",
        "Gravity (m/s²)",
        out_dir / "03_gravity_full.png",
        markers,
    )

    save_plot(
        x,
        [df[speed_col]],
        ["GPS speed"],
        "GPS Speed - Full Recording",
        "Speed (km/h)",
        out_dir / "04_gps_speed.png",
        markers,
    )

    save_plot(
        x,
        [df[accuracy_col]],
        ["GPS accuracy"],
        "GPS Accuracy - Full Recording",
        "Accuracy (m)",
        out_dir / "05_gps_accuracy.png",
        markers,
    )

    save_route_plot(df[lon_col], df[lat_col], out_dir / "06_gps_route.png")

    # Zoom around every detected reset/gap so suspicious regions are easy to inspect.
    suspicious_rows = [(r, "reset") for r in reset_rows] + [(r, "gap") for r in gap_rows]
    window = 150  # roughly 15 seconds on each side at 10 Hz

    for row, kind in suspicious_rows:
        lo = max(0, row - window)
        hi = min(len(df), row + window)
        local_x = x[lo:hi]

        save_plot(
            local_x,
            [df[c].iloc[lo:hi] for c in accel_cols],
            ["X", "Y", "Z"],
            f"Accelerometer Around {kind.title()} at Row {row}",
            "Acceleration (m/s²)",
            out_dir / f"07_accel_{kind}_row_{row}.png",
            [(row, kind)],
        )

        save_plot(
            local_x,
            [df[c].iloc[lo:hi] for c in gyro_cols],
            ["Yaw", "Pitch", "Roll"],
            f"Gyroscope Around {kind.title()} at Row {row}",
            "Angular velocity (rad/s)",
            out_dir / f"08_gyro_{kind}_row_{row}.png",
            [(row, kind)],
        )

    print(f"Created signal overview plots in: {out_dir}")
    print(f"Timestamp reset rows: {reset_rows}")
    print(f"Large gap rows (>0.25 s): {gap_rows}")


if __name__ == "__main__":
    main()

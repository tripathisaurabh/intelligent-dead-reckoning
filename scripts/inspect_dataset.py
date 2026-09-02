from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.preprocessing.analysis import (
    column_report,
    find_data_files,
    find_time_columns,
    keyword_columns,
    read_table,
    sampling_summary,
)


def save_plot(df: pd.DataFrame, columns: list[str], output: Path, title: str) -> None:
    numeric_cols = []
    for col in columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            numeric_cols.append(col)

    if not numeric_cols:
        return

    plt.figure(figsize=(12, 5))
    for col in numeric_cols[:8]:
        plt.plot(pd.to_numeric(df[col], errors="coerce").to_numpy(), label=col, linewidth=0.8)
    plt.title(title)
    plt.xlabel("Sample index")
    plt.ylabel("Raw value")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def inspect_file(path: Path, output_root: Path) -> None:
    print(f"\nInspecting: {path}")
    df = read_table(path)

    safe_name = path.stem.replace(" ", "_")
    out = output_root / safe_name
    out.mkdir(parents=True, exist_ok=True)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumn names:")
    for col in df.columns:
        print(f"  - {col}")

    report = column_report(df)
    report.to_csv(out / "column_report.csv", index=False)

    time_cols = find_time_columns(df)
    sampling = []
    for col in time_cols:
        summary = sampling_summary(df, col)
        sampling.append(asdict(summary))
        print(
            f"Time candidate '{col}': "
            f"median dt={summary.median_dt_seconds}, "
            f"estimated Hz={summary.estimated_hz}, "
            f"duplicates={summary.duplicate_timestamps}, "
            f"non-monotonic={summary.non_monotonic_steps}, "
            f"max gap={summary.max_gap_seconds}"
        )

    with open(out / "sampling_report.json", "w", encoding="utf-8") as f:
        json.dump(sampling, f, indent=2)

    accel_cols = keyword_columns(df, ["accelerometer", "accel"])
    gyro_cols = keyword_columns(df, ["gyroscope", "gyro"])
    gps_cols = keyword_columns(df, ["gps", "gnss", "latitude", "longitude", "speed", "accuracy"])

    save_plot(df, accel_cols, out / "accelerometer_raw.png", "Raw accelerometer-related columns")
    save_plot(df, gyro_cols, out / "gyroscope_raw.png", "Raw gyroscope-related columns")
    save_plot(df, gps_cols, out / "gnss_raw.png", "Raw GNSS/GPS-related columns")

    summary = {
        "file": str(path),
        "rows": len(df),
        "columns": list(map(str, df.columns)),
        "time_candidates": time_cols,
        "accelerometer_candidates": accel_cols,
        "gyroscope_candidates": gyro_cols,
        "gnss_candidates": gps_cols,
    }
    with open(out / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved inspection output to: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 IO-VNBD dataset inspection")
    parser.add_argument("path", type=Path, help="Path to one dataset file or a directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase1/dataset_inspection"),
        help="Directory for reports and plots",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="If path is a directory, inspect all CSV/TXT files instead of only the first one",
    )
    args = parser.parse_args()

    if args.path.is_file():
        files = [args.path]
    elif args.path.is_dir():
        files = find_data_files(args.path)
        if not args.all:
            files = files[:1]
    else:
        raise FileNotFoundError(args.path)

    if not files:
        raise RuntimeError("No CSV/TXT data files found")

    args.output.mkdir(parents=True, exist_ok=True)
    for file in files:
        inspect_file(file, args.output)


if __name__ == "__main__":
    main()

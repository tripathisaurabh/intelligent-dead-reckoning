from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


OUTPUT = Path("results/final_summary/judge_vta08_result.png")


def main() -> None:
    baseline = pd.read_csv("results/experiments/baseline_vta08/basic_ins_trajectory.csv")
    final = pd.read_csv("results/experiments/map_matching_vta08/matched_trajectory.csv")

    stages = [
        "Basic sensor\ncalculation",
        "+ vehicle\nmovement rules",
        "+ phone direction\ncorrection",
        "+ trained CNN\ncorrection",
        "+ road-map\nmatching",
    ]
    drift = np.array([96.292881, 55.441020, 22.872485, 20.965754, 11.145913])
    colors = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]

    figure = plt.figure(figsize=(16, 9), facecolor="#f7f9fc")
    layout = GridSpec(
        2, 2, figure=figure, width_ratios=(1.35, 1.0), height_ratios=(1.1, 0.9),
        left=0.055, right=0.965, bottom=0.09, top=0.82, wspace=0.25, hspace=0.36,
    )
    figure.suptitle(
        "When GPS was switched off, our complete system stayed much closer to the actual route",
        fontsize=23, fontweight="bold", color="#16324f", y=0.955,
    )
    figure.text(
        0.5, 0.885,
        "Vta08 test: 20 seconds without GPS | Vehicle travelled 141.995 metres",
        ha="center", fontsize=15, color="#425466",
    )

    trajectory = figure.add_subplot(layout[:, 0])
    trajectory.plot(
        final["reference_east_m"], final["reference_north_m"],
        color="#17202a", linewidth=4.2, label="Where the vehicle actually went", zorder=4,
    )
    trajectory.plot(
        baseline["east_m"], baseline["north_m"],
        color="#d73027", linewidth=2.6, linestyle="--",
        label="Basic phone-sensor estimate", zorder=2,
    )
    trajectory.plot(
        final["matched_east_m"], final["matched_north_m"],
        color="#1a9850", linewidth=4.0, label="Complete system estimate", zorder=3,
    )
    trajectory.scatter([0], [0], s=95, color="#1565c0", edgecolor="white", linewidth=1.5,
                       label="GPS blackout starts", zorder=6)
    trajectory.scatter(
        [final["reference_east_m"].iloc[-1]], [final["reference_north_m"].iloc[-1]],
        marker="*", s=220, color="#17202a", zorder=7,
    )
    trajectory.scatter(
        [final["matched_east_m"].iloc[-1]], [final["matched_north_m"].iloc[-1]],
        marker="*", s=220, color="#1a9850", zorder=7,
    )
    trajectory.set_title("Estimated route during the GPS blackout", fontsize=17, fontweight="bold")
    trajectory.set_xlabel("East distance from blackout start (metres)", fontsize=12)
    trajectory.set_ylabel("North distance from blackout start (metres)", fontsize=12)
    trajectory.grid(alpha=0.20)
    trajectory.axis("equal")
    trajectory.legend(loc="best", fontsize=11, frameon=True)

    bars = figure.add_subplot(layout[0, 1])
    positions = np.arange(len(stages))
    rectangles = bars.barh(positions, drift, color=colors, height=0.68)
    bars.set_yticks(positions, stages, fontsize=11)
    bars.invert_yaxis()
    bars.set_xlim(0, 105)
    bars.set_xlabel("Final error ÷ distance travelled (%)", fontsize=11)
    bars.set_title("Final error falls as each technique is added", fontsize=17, fontweight="bold")
    bars.grid(axis="x", alpha=0.20)
    bars.set_axisbelow(True)
    for rectangle, value in zip(rectangles, drift):
        bars.text(
            min(value + 1.5, 98), rectangle.get_y() + rectangle.get_height() / 2,
            f"{value:.1f}%", va="center", fontsize=11, fontweight="bold", color="#263238",
        )
    bars.text(
        103, 4.65, "Lower is better", ha="right", va="bottom", fontsize=10,
        color="#607d8b", fontstyle="italic",
    )

    summary = figure.add_subplot(layout[1, 1])
    summary.axis("off")
    summary.text(0.0, 0.92, "Measured outcome", fontsize=17, fontweight="bold", color="#16324f")
    summary.text(0.02, 0.66, "136.73 m", fontsize=27, fontweight="bold", color="#d73027")
    summary.text(0.43, 0.67, "→", fontsize=28, fontweight="bold", color="#607d8b")
    summary.text(0.58, 0.66, "15.83 m", fontsize=27, fontweight="bold", color="#1a9850")
    summary.text(0.02, 0.53, "basic final error", fontsize=11, color="#607d8b")
    summary.text(0.58, 0.53, "complete-system final error", fontsize=11, color="#607d8b")
    summary.text(
        0.02, 0.28, "88.4% reduction in endpoint error", fontsize=19,
        fontweight="bold", color="#1a9850",
    )
    summary.text(
        0.02, 0.08,
        "Final drift: 11.146%\nThis is one controlled Vta08 test, not a universal accuracy claim.",
        fontsize=11.5, color="#425466", linespacing=1.5,
    )

    figure.text(
        0.5, 0.025,
        "Complete system = calibration + mounting alignment + invariant Kalman filter + vehicle-motion constraints + learned correction + road matching",
        ha="center", fontsize=11, color="#607d8b",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=200, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()

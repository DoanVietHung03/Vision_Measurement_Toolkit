from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATA = {
    "real_distance": [1.44, 2.79, 2.65, 2.60],
    "homography_distance": [1.58, 2.49, 2.39, 3.20],
    "depth_distance": [0.88, 2.90, 3.01, 2.50],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Homography and Depth distance errors.")
    parser.add_argument("--csv", help="CSV with real_distance, homography_distance and depth_distance columns.")
    parser.add_argument("--output", default="experiments/error_chart.png", help="Output chart path.")
    return parser


def calculate_metrics(real, measured):
    absolute_error = np.abs(measured - real)
    relative_error = np.divide(
        absolute_error,
        real,
        out=np.zeros_like(absolute_error, dtype=float),
        where=np.asarray(real) != 0,
    ) * 100.0
    return absolute_error, relative_error


def main() -> None:
    args = build_parser().parse_args()
    dataframe = pd.read_csv(args.csv) if args.csv else pd.DataFrame(DEFAULT_DATA)
    required = {"real_distance", "homography_distance", "depth_distance"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    dataframe["homography_ae"], dataframe["homography_re"] = calculate_metrics(
        dataframe["real_distance"],
        dataframe["homography_distance"],
    )
    dataframe["depth_ae"], dataframe["depth_re"] = calculate_metrics(
        dataframe["real_distance"],
        dataframe["depth_distance"],
    )

    print(dataframe.round(3).to_string(index=False))
    print(f"Homography MAPE: {dataframe['homography_re'].mean():.2f}%")
    print(f"Depth MAPE: {dataframe['depth_re'].mean():.2f}%")

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        dataframe["real_distance"],
        dataframe["homography_re"],
        marker="o",
        label="Homography",
        linewidth=2,
    )
    axis.plot(
        dataframe["real_distance"],
        dataframe["depth_re"],
        marker="s",
        label="Depth Anything",
        linewidth=2,
    )
    axis.axhline(y=5, color="red", linestyle="--", alpha=0.5, label="5% threshold")
    axis.set_title("Distance error comparison")
    axis.set_xlabel("Real distance (m)")
    axis.set_ylabel("Relative error (%)")
    axis.grid(True, linestyle="--", alpha=0.7)
    axis.legend()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_trajectory(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Trajectory CSV does not exist: {csv_path}")

    points: list[list[float]] = []
    timestamps: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            timestamps.append(float(row["timestamp"]))
            points.append(
                [
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                ]
            )

    if len(points) == 0:
        raise ValueError(f"Trajectory CSV is empty: {csv_path}")

    return (
        np.asarray(timestamps, dtype=np.float64),
        np.asarray(points, dtype=np.float64),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    timestamps, points = _load_trajectory(args.csv_path)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(points[:, 0], points[:, 1], points[:, 2], linewidth=2.0, label="agent")
    ax.scatter(
        points[0, 0],
        points[0, 1],
        points[0, 2],
        c="green",
        s=50,
        label="start",
    )
    ax.scatter(
        points[-1, 0],
        points[-1, 1],
        points[-1, 2],
        c="red",
        s=50,
        label="end",
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(0.0, 3.5)
    ax.set_zlim(0.0, 2.0)
    ax.set_title(f"Agent 3D Trajectory ({timestamps[0]:.3f}s -> {timestamps[-1]:.3f}s)")
    ax.legend()
    ax.grid(True)

    output_path = args.output
    if output_path is None:
        output_path = args.csv_path.with_suffix(".png")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved figure to: {output_path}", flush=True)

    plt.show()
    # if args.show:
    #     plt.show()
    # else:
    #     plt.close(fig)


if __name__ == "__main__":
    main()

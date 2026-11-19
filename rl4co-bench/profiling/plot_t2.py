"""Plot host attribution and CUDA activity across the three T2 sweep axes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SLICES = (
    ("n", "Cities", {"b": 16, "k": 1}),
    ("b", "Batch size", {"n": 100, "k": 1}),
    ("k", "Decode starts", {"n": 100, "b": 16}),
)
HOST_SERIES = (
    ("host_sync_pct", "CUDA synchronisation", "#cc6644"),
    ("host_launch_pct", "CUDA launch", "#eab76c"),
    ("host_torch_runtime_pct", "PyTorch ops and runtime", "#5379a8"),
    ("host_python_pct", "Project Python frames", "#689a89"),
    ("host_unattributed_pct", "Unattributed", "#a0a4ad"),
)


def plot(cases: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with cases.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == "ok" and row.get("device_busy_pct")]
    if not rows:
        raise ValueError("No successful CUDA case records")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), sharey="row")
    fig.suptitle("Attention Model inference on one RTX 4090", fontsize=16, x=0.08, ha="left")
    for col, (key, label, fixed) in enumerate(SLICES):
        selected = sorted(
            (row for row in rows if all(int(row[field]) == value for field, value in fixed.items())),
            key=lambda row: int(row[key]),
        )
        if not selected:
            raise ValueError(f"No CUDA cases for the {key} sweep")
        x = list(range(len(selected)))
        labels = [row[key] for row in selected]

        gpu, host = axes[:, col]
        gpu.set_title(label, loc="left", fontweight="bold")
        for metric, legend, colour in (
            ("device_busy_pct", "Any captured device activity", "#1b6b66"),
            ("device_math_hint_pct", "Named math/attention kernels", "#8657a4"),
        ):
            values = [float(row[metric]) for row in selected]
            # The K=1 case uses ordinary greedy decoding. Connect only the
            # multistart cases; the isolated point shows the reference case.
            offset = int(key == "k" and labels[0] == "1")
            gpu.plot(x[offset:], values[offset:], marker="o", color=colour, label=legend)
            if offset:
                gpu.scatter(x[0], values[0], color=colour, marker="o", zorder=3)
        gpu.set_ylim(bottom=0)
        lower = [0.0] * len(x)
        for metric, legend, colour in HOST_SERIES:
            values = [float(row[metric]) for row in selected]
            host.bar(x, values, bottom=lower, width=0.63, color=colour, label=legend)
            lower = [bottom + value for bottom, value in zip(lower, values, strict=True)]
        host.set_ylim(0, 100)
        for ax in (gpu, host):
            ax.set_xticks(x, labels)
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        if key == "k" and labels[0] == "1":
            host.set_xticks(x, ["1\ngreedy", *labels[1:]])
    axes[0, 0].set_ylabel("Profiled GPU window (%)")
    axes[1, 0].set_ylabel("Profiled host thread (%)")
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(0, 1.48), ncols=2, frameon=False)
    axes[1, 0].legend(loc="upper left", bbox_to_anchor=(0, -0.24), ncols=3, frameon=False)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.81, bottom=0.19, wspace=0.23, hspace=0.36)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, help="cases.csv from a CUDA T2 report")
    parser.add_argument("--output", required=True, type=Path, help="Image file to write")
    args = parser.parse_args()
    try:
        plot(args.cases, args.output)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()

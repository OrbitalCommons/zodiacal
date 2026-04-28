#!/usr/bin/env python3
"""Plot solve time distributions from attitude_hint benchmark.

Usage:
    cargo bench --bench attitude_hint -- --trials 100 > bench_results.json
    python3 scripts/plot_attitude_bench.py bench_results.json
"""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <bench_results.json>", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1]) as f:
        results = json.load(f)

    conditions = [
        ("no_hint", "Blind (no hint)", "#e74c3c"),
        ("hint_10deg", "10° hint", "#3498db"),
        ("hint_1deg", "1° hint", "#2ecc71"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Histogram of solve times ---
    ax = axes[0]
    for key, label, color in conditions:
        times_us = [r[f"{key}_us"] for r in results if r[f"{key}_solved"]]
        if not times_us:
            continue
        times_ms = np.array(times_us) / 1000.0
        ax.hist(
            times_ms,
            bins=30,
            alpha=0.5,
            label=f"{label} (n={len(times_ms)})",
            color=color,
            edgecolor=color,
            linewidth=0.8,
        )

    ax.set_xlabel("Solve time (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Solve Time Distribution")
    ax.legend()

    # --- Box plot ---
    ax = axes[1]
    box_data = []
    box_labels = []
    box_colors = []
    for key, label, color in conditions:
        times_us = [r[f"{key}_us"] for r in results if r[f"{key}_solved"]]
        if times_us:
            box_data.append(np.array(times_us) / 1000.0)
            box_labels.append(label)
            box_colors.append(color)

    if box_data:
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

    ax.set_ylabel("Solve time (ms)")
    ax.set_title("Solve Time by Attitude Hint Accuracy")

    # Print summary stats
    print("\n--- Summary ---")
    for key, label, _ in conditions:
        times_us = [r[f"{key}_us"] for r in results if r[f"{key}_solved"]]
        n_solved = len(times_us)
        n_total = len(results)
        if times_us:
            arr = np.array(times_us) / 1000.0
            print(
                f"{label:20s}: solved {n_solved}/{n_total}, "
                f"median={np.median(arr):.2f}ms, "
                f"mean={np.mean(arr):.2f}ms, "
                f"p95={np.percentile(arr, 95):.2f}ms"
            )
        else:
            print(f"{label:20s}: solved 0/{n_total}")

    plt.tight_layout()
    out_path = sys.argv[1].replace(".json", ".png")
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()

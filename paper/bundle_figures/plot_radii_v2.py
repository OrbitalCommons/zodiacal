#!/usr/bin/env python3
"""Solve-time CDF + load-time distribution table across position-hint radii.

Picks the best available CSV per radius:
- v2 file (post-LRU) if 1000 rows present.
- v1 file otherwise (a few were truncated by the pre-LRU OOM at large radii).

Re-run any time more data lands.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/home/meawoppl/scratch/bundle-build/bench")

# Per radius: try v2 first, then fall back to a v1 CSV if present.
SOURCES = [
    ("0.5°", ["d9-mag20-set2-r0.5-v2.csv"], "tab:purple"),
    ("1.0°", ["d9-mag20-set2-r1.0-v2.csv", "d9-mag20-set2-r1.0.csv"], "tab:blue"),
    ("2.0°", ["d9-mag20-set2-r2.0-v2.csv", "d9-mag20-set2-r2.0.csv"], "tab:orange"),
    ("4.0°", ["d9-mag20-set2-r4.0-v2.csv"],                            "tab:red"),
]

# Depth-9 bundle: 12 * 4^9 = 3,145,728 cells over the 4π-sterradian
# sphere → 3,145,728 / 41,253 ≈ 76.3 cells per square degree.
# `bundle.load_region(R)` pulls every cell whose center is within R°
# of the truth; ~π R² × 76.3 cells per query (each cell = 1 .zqd + 1
# .zga shard, so total file reads is 2× the cell count).
CELLS_PER_SQ_DEG_DEPTH9 = 3_145_728 / (4.0 * np.pi * (180.0 / np.pi) ** 2)

def cells_for_radius_deg(r: float) -> int:
    return int(round(np.pi * r * r * CELLS_PER_SQ_DEG_DEPTH9))

datasets = []
for label, fnames, color in SOURCES:
    chosen = None
    for fname in fnames:
        p = OUT / fname
        if not p.exists():
            continue
        rows = list(csv.DictReader(p.open()))
        # Prefer the first source that has 1000 rows; otherwise hold the
        # widest one we've seen so partial in-progress files still plot.
        if chosen is None or len(rows) > len(chosen[1]):
            chosen = (p, rows)
        if len(rows) >= 1000:
            break
    if chosen is None:
        print(f"  skip {label}: no CSV yet")
        continue
    p, rows = chosen
    solved = np.array([r["solved"] == "1" for r in rows])
    solve_ms = np.array([float(r["solve_ms"]) for r in rows if r["solved"] == "1"])
    load_ms = np.array([float(r["load_ms"]) for r in rows])
    error_arcsec = np.array(
        [float(r["error_arcsec"]) for r in rows if r["solved"] == "1"]
    )
    # Re-classify: solved cases with > WRONG_THRESHOLD_ARCSEC error are
    # "solved wrong" (false-positive WCS that nonetheless passed
    # log-odds verification at large search radii). Real successes are
    # error <= threshold.
    WRONG_THRESHOLD_ARCSEC = 1.0
    n_correct = int((error_arcsec <= WRONG_THRESHOLD_ARCSEC).sum())
    n_wrong = int((error_arcsec > WRONG_THRESHOLD_ARCSEC).sum())
    n_total = len(rows)
    n_solved = int(solved.sum())
    # Numeric radius parsed back out of the human label ("1.41°" → 1.41).
    radius_deg = float(label.rstrip("°"))
    cells = cells_for_radius_deg(radius_deg)
    datasets.append({
        "label": label, "color": color, "solve_ms": solve_ms, "load_ms": load_ms,
        "error_arcsec": error_arcsec,
        "n_total": n_total, "n_solved": n_solved,
        "n_correct": n_correct, "n_wrong": n_wrong,
        "src": p.name,
        "radius_deg": radius_deg, "cells_loaded": cells,
    })
    print(f"  {label}: n={n_total} solved={n_solved} "
          f"(solve median={np.median(solve_ms):.2f}ms, load median={np.median(load_ms):.1f}ms) "
          f"[{p.name}]")

# ---- Solve-time CDF, p50 + p99 annotated ---------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
# p50 callouts alternate above/below the y=0.5 line, pulled in toward
# the centre (y=0.30 / y=0.70). p99 callouts sit above the curve at
# y=0.85 with short vertical leaders to (p99, 0.99); they're well
# separated on a log-x axis so a single row suffices.
p50_above_y, p50_below_y = 0.70, 0.30
p99_callout_ys = (0.78, 0.88)  # alternates so adjacent labels don't collide; both below the y=0.99 marker
for i, d in enumerate(datasets):
    ms = np.sort(d["solve_ms"])
    cdf = np.arange(1, ms.size + 1) / ms.size
    p50 = float(np.median(d["solve_ms"]))
    p99 = float(np.percentile(d["solve_ms"], 99))
    label = (
        f"r={d['label']}  correct={d['n_correct']}  wrong={d['n_wrong']} (>1″)  "
        f"(~{d['cells_loaded']:,} cells/region)"
    )
    ax.plot(ms, cdf, color=d["color"], lw=1.8, label=label)
    ax.scatter([p50, p99], [0.5, 0.99], color=d["color"], s=50, zorder=5,
               edgecolor="black", linewidths=0.7)
    p50_y = p50_above_y if i % 2 == 1 else p50_below_y
    ax.annotate(
        f"r={d['label']}\np50 = {p50:.2f} ms",
        xy=(p50, 0.5),
        xycoords="data",
        xytext=(p50, p50_y),
        textcoords="data",
        color=d["color"],
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=d["color"], lw=0.8),
        arrowprops=dict(arrowstyle="-", color=d["color"], lw=0.7, alpha=0.7),
    )
    ax.annotate(
        f"p99 {p99:.0f} ms",
        xy=(p99, 0.99),
        xycoords="data",
        xytext=(p99, p99_callout_ys[i % len(p99_callout_ys)]),
        textcoords="data",
        color=d["color"],
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=d["color"], lw=0.7),
        arrowprops=dict(arrowstyle="-", color=d["color"], lw=0.6, alpha=0.6),
    )
ax.axhline(0.5, color="gray", lw=0.8, alpha=0.4, ls="--")
ax.set_xscale("log")
ax.set_xlabel("solve time (ms, log scale)")
ax.set_ylabel("cumulative fraction of solved cases")
ax.set_title(
    "d9-mag20 / set2-dr3-mag19 — solve-time CDF by position-hint radius\n"
    "(scale_hint=True; bundles built per `bundle-build-lessons.md`)",
)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "solve_time_by_radius_cdf_v2.png", dpi=140, bbox_inches="tight")
print(f"wrote {OUT / 'solve_time_by_radius_cdf_v2.png'}")

# ---- Load-time CDF, p50 + p99 annotated ----------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
p50_above_y, p50_below_y = 0.70, 0.30
p99_callout_ys = (0.78, 0.88)  # alternates so adjacent labels don't collide; both below the y=0.99 marker
for i, d in enumerate(datasets):
    ms = np.sort(d["load_ms"])
    cdf = np.arange(1, ms.size + 1) / ms.size
    p50 = float(np.median(d["load_ms"]))
    p99 = float(np.percentile(d["load_ms"], 99))
    label = (
        f"r={d['label']}  correct={d['n_correct']}  wrong={d['n_wrong']} (>1″)  "
        f"(~{d['cells_loaded']:,} cells/region)"
    )
    ax.plot(ms, cdf, color=d["color"], lw=1.8, label=label)
    ax.scatter([p50, p99], [0.5, 0.99], color=d["color"], s=50, zorder=5,
               edgecolor="black", linewidths=0.7)
    p50_y = p50_above_y if i % 2 == 1 else p50_below_y
    ax.annotate(
        f"r={d['label']}\np50 = {p50:.0f} ms",
        xy=(p50, 0.5),
        xycoords="data",
        xytext=(p50, p50_y),
        textcoords="data",
        color=d["color"],
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=d["color"], lw=0.8),
        arrowprops=dict(arrowstyle="-", color=d["color"], lw=0.7, alpha=0.7),
    )
    ax.annotate(
        f"p99 {p99:.0f} ms",
        xy=(p99, 0.99),
        xycoords="data",
        xytext=(p99, p99_callout_ys[i % len(p99_callout_ys)]),
        textcoords="data",
        color=d["color"],
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=d["color"], lw=0.7),
        arrowprops=dict(arrowstyle="-", color=d["color"], lw=0.6, alpha=0.6),
    )
ax.axhline(0.5, color="gray", lw=0.8, alpha=0.4, ls="--")
ax.set_xscale("log")
ax.set_xlabel("region-load time (ms, log scale)")
ax.set_ylabel("cumulative fraction of cases")
ax.set_title(
    "d9-mag20 / set2-dr3-mag19 — region-load-time CDF by position-hint radius\n"
    "(time = `bundle.load_region(&region)` — every overlapping shard)",
)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "load_time_by_radius_cdf_v2.png", dpi=140, bbox_inches="tight")
print(f"wrote {OUT / 'load_time_by_radius_cdf_v2.png'}")

# ---- Solve-accuracy CDF, p50 + p99 annotated -----------------------------
# error_arcsec is the angular distance between the solver's WCS field
# centre and the truth field centre. Smaller is better.
fig, ax = plt.subplots(figsize=(10, 6))
p50_above_y, p50_below_y = 0.70, 0.30
p99_callout_ys = (0.78, 0.88)
for i, d in enumerate(datasets):
    err = d["error_arcsec"]
    if err.size == 0:
        continue
    # CSV rounds error to 3 decimals → exact-0 entries are really
    # sub-mas residuals. Clip to 1e-3″ so log-scale shows them at the
    # left edge instead of dropping them and skewing percentiles.
    err = np.maximum(err, 1e-3)
    err_sorted = np.sort(err)
    cdf = np.arange(1, err_sorted.size + 1) / err_sorted.size
    p50 = float(np.median(err))
    p99 = float(np.percentile(err, 99))
    label = (
        f"r={d['label']}  correct={d['n_correct']}  wrong={d['n_wrong']} (>1″)  "
        f"(~{d['cells_loaded']:,} cells/region)"
    )
    ax.plot(err_sorted, cdf, color=d["color"], lw=1.8, label=label)
    ax.scatter([p50, p99], [0.5, 0.99], color=d["color"], s=50, zorder=5,
               edgecolor="black", linewidths=0.7)
    p50_y = p50_above_y if i % 2 == 1 else p50_below_y
    ax.annotate(
        f"r={d['label']}\np50 = {p50:.3f}″",
        xy=(p50, 0.5),
        xycoords="data",
        xytext=(p50, p50_y),
        textcoords="data",
        color=d["color"],
        fontsize=10,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=d["color"], lw=0.8),
        arrowprops=dict(arrowstyle="-", color=d["color"], lw=0.7, alpha=0.7),
    )
    ax.annotate(
        f"p99 {p99:.3f}″",
        xy=(p99, 0.99),
        xycoords="data",
        xytext=(p99, p99_callout_ys[i % len(p99_callout_ys)]),
        textcoords="data",
        color=d["color"],
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=d["color"], lw=0.7),
        arrowprops=dict(arrowstyle="-", color=d["color"], lw=0.6, alpha=0.6),
    )
ax.axhline(0.5, color="gray", lw=0.8, alpha=0.4, ls="--")
ax.axvline(1.0, color="black", lw=1, alpha=0.5, ls=":")
ax.text(1.0, 0.04, "  ↑ 1″ wrong-solve threshold",
        color="black", fontsize=9, ha="left", va="bottom")
ax.set_xscale("log")
ax.set_xlabel("solve accuracy: angular error vs truth (arcsec, log scale)")
ax.set_ylabel("cumulative fraction of solved cases")
ax.set_title(
    "d9-mag20 / set2-dr3-mag19 — solve-accuracy CDF by position-hint radius\n"
    "(error = angular distance between solver field-centre WCS and truth)",
)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "accuracy_by_radius_cdf_v2.png", dpi=140, bbox_inches="tight")
print(f"wrote {OUT / 'accuracy_by_radius_cdf_v2.png'}")

# ---- Load-time distribution table (markdown) -----------------------------
md = ["# Load-time distribution by hint radius",
      "",
      f"Bundle: depth-9 G≤20 ({SOURCES[0][1]}). All 1000 cases @ scale_hint=True.",
      "",
      "| radius | n | min | p10 | p25 | **p50 (median)** | p75 | p90 | p95 | p99 | max | mean |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for d in datasets:
    lm = d["load_ms"]
    if lm.size == 0:
        continue
    pcts = np.percentile(lm, [10, 25, 50, 75, 90, 95, 99])
    md.append(
        f"| {d['label']} | {d['n_total']} | "
        f"{lm.min():.1f} | {pcts[0]:.1f} | {pcts[1]:.1f} | "
        f"**{pcts[2]:.1f}** | {pcts[3]:.1f} | {pcts[4]:.1f} | "
        f"{pcts[5]:.1f} | {pcts[6]:.1f} | {lm.max():.1f} | {lm.mean():.1f} |"
    )
md.append("")
md.append("All times in milliseconds. Load time = `bundle.load_region(&region)` "
          "per case (loads + parses every overlapping cell shard).")

(OUT / "load_time_table.md").write_text("\n".join(md))
print(f"wrote {OUT / 'load_time_table.md'}")
print()
print("\n".join(md))

#!/usr/bin/env python3
"""Mollweide map of wrong-solve case positions across all four radii.

A "wrong solve" is one where the solver returned solved=1 but the
WCS centre was more than 1″ from the truth. Catalog quad density is
overlaid as a faint background (from the cached scan) so you can see
whether failures cluster in low-density regions or are uniform.
"""
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy_healpix import HEALPix
from astropy.coordinates import ICRS, SkyCoord

OUT = Path("/home/meawoppl/scratch/bundle-build/bench")
TC_DIR = Path("/home/meawoppl/repos/zodiacal-test-cases/set2-dr3-mag19")
CACHE = OUT / "quads_per_cell_scan.npz"
RADII = [
    ("0.5°", "d9-mag20-set2-r0.5-v2.csv", "tab:purple"),
    ("1.0°", "d9-mag20-set2-r1.0-v2.csv", "tab:blue"),
    ("2.0°", "d9-mag20-set2-r2.0-v2.csv", "tab:orange"),
    ("4.0°", "d9-mag20-set2-r4.0-v2.csv", "tab:red"),
]
WRONG_THRESHOLD = 1.0  # arcsec

# ---- collect failures across all radii -----------------------------------
failures = []  # list of dicts {radius, color, ra, dec, error}
truth_cache: dict[str, tuple[float, float]] = {}
def truth_for(case: str) -> tuple[float, float]:
    if case not in truth_cache:
        tc = json.load((TC_DIR / f"{case}.json").open())
        truth_cache[case] = (tc["ra_deg"], tc["dec_deg"])
    return truth_cache[case]

per_radius_counts = []
for label, fname, color in RADII:
    p = OUT / fname
    if not p.exists():
        per_radius_counts.append((label, 0, 0, color))
        continue
    n_total = 0
    n_wrong = 0
    for row in csv.DictReader(p.open()):
        n_total += 1
        if row["solved"] != "1":
            continue
        try:
            err = float(row["error_arcsec"])
        except ValueError:
            continue
        if err > WRONG_THRESHOLD:
            ra, dec = truth_for(row["case"])
            failures.append({
                "radius": label, "color": color,
                "ra_deg": ra, "dec_deg": dec, "error_arcsec": err,
            })
            n_wrong += 1
    per_radius_counts.append((label, n_total, n_wrong, color))

print("wrong-solve counts:")
for label, n_total, n_wrong, _ in per_radius_counts:
    print(f"  r={label}: {n_wrong}/{n_total}")

# ---- background quads-per-cell map (light alpha) -------------------------
def load_or_scan_quads():
    if CACHE.exists():
        z = np.load(CACHE)
        return z["cell_ids"], z["sizes"]
    return None, None

cell_ids, sizes = load_or_scan_quads()
HEADER_BYTES = 32 + 7 * 24
BYTES_PER_QUAD = 16 + 32

# ---- Mollweide -----------------------------------------------------------
fig = plt.figure(figsize=(13, 7))
ax = fig.add_subplot(111, projection="mollweide")

if cell_ids is not None:
    n_quads = np.clip((sizes - HEADER_BYTES) // BYTES_PER_QUAD, 0, None)
    hp = HEALPix(nside=2 ** 9, order="nested", frame=ICRS())
    coords = hp.healpix_to_skycoord(cell_ids)
    ra_bg = coords.ra.deg
    dec_bg = coords.dec.deg
    ra_bg = np.where(ra_bg > 180, ra_bg - 360, ra_bg)
    ax.hexbin(
        np.radians(ra_bg), np.radians(dec_bg),
        C=n_quads, reduce_C_function=np.mean,
        gridsize=180, cmap="Greys", alpha=0.45, edgecolors="none",
    )

# ---- failure scatter -----------------------------------------------------
# Larger marker for larger radius so they read distinguishably even when
# colocated. Slight transparency so overlapping points stack visibly.
sizes_by_radius = {"0.5°": 40, "1.0°": 50, "2.0°": 60, "4.0°": 70}
plotted_labels = set()
for f in failures:
    ra = f["ra_deg"]
    if ra > 180:
        ra -= 360
    x = np.radians(ra)
    y = np.radians(f["dec_deg"])
    label = None
    if f["radius"] not in plotted_labels:
        # Find the per-radius count for the legend.
        for lbl, _ntot, nw, _c in per_radius_counts:
            if lbl == f["radius"]:
                label = f"r={lbl}  wrong = {nw}"
        plotted_labels.add(f["radius"])
    ax.scatter(
        [x], [y],
        s=sizes_by_radius[f["radius"]],
        marker="x", color=f["color"], linewidths=1.6,
        zorder=10, label=label, alpha=0.9,
    )

# Galactic poles for sanity orientation.
for label, ra_pole, dec_pole in [("NGP", 192.85948, 27.12825),
                                 ("SGP", 12.85948, -27.12825)]:
    rp = ra_pole - 360 if ra_pole > 180 else ra_pole
    ax.scatter(np.radians(rp), np.radians(dec_pole),
               marker="+", color="black", s=110, linewidths=2.0, zorder=11)
    ax.annotate(
        label, xy=(np.radians(rp), np.radians(dec_pole)),
        xytext=(8, 6), textcoords="offset points", fontsize=9, color="black",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5, alpha=0.85),
    )

ax.grid(True, alpha=0.3)
ax.set_title(
    f"Wrong-solve positions (>{WRONG_THRESHOLD:.0f}″ error) — d9-mag20 / set2-dr3-mag19\n"
    "background: mean quads/cell (greyscale); markers: failure positions colored by hint radius"
)
ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
fig.tight_layout()
fig.savefig(OUT / "failures_mollweide.png", dpi=140, bbox_inches="tight")
print(f"wrote {OUT / 'failures_mollweide.png'}")

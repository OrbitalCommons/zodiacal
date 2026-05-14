#!/usr/bin/env python3
"""(Δra·cos(dec), Δdec) residual scatter for the d9-mag20 / set2-dr3-mag19
1° bench run (v3 CSV with solved_ra_deg/solved_dec_deg columns)."""
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUT = Path("/home/meawoppl/scratch/bundle-build/bench")
TC_DIR = Path("/home/meawoppl/repos/zodiacal-test-cases/set2-dr3-mag19")

# Read solved (ra, dec) from the v3 CSV.
csv_path = OUT / "d9-mag20-set2-r1.0-v3.csv"
solved = []
with csv_path.open() as f:
    for row in csv.DictReader(f):
        if row["solved"] != "1":
            continue
        case = row["case"]
        solved.append({
            "case": case,
            "solved_ra": float(row["solved_ra_deg"]),
            "solved_dec": float(row["solved_dec_deg"]),
            "error_arcsec": float(row["error_arcsec"]),
        })

# Pull truth (ra, dec) from each test-case JSON.
import json
for d in solved:
    tc = json.load((TC_DIR / f"{d['case']}.json").open())
    d["truth_ra"] = tc["ra_deg"]
    d["truth_dec"] = tc["dec_deg"]

# Signed residuals. Δra·cos(dec) projects onto a tangent plane so the
# horizontal axis is in arcseconds-of-arc (true sky distance), not
# arcseconds-of-RA which shrink toward the poles.
dra = []
ddec = []
err = []
case_ids = []
for d in solved:
    # Wrap solved_ra to be near truth_ra (handles 360° wrap).
    diff_ra = (d["solved_ra"] - d["truth_ra"] + 540.0) % 360.0 - 180.0
    cos_dec = np.cos(np.radians(d["truth_dec"]))
    dra.append(diff_ra * 3600.0 * cos_dec)
    ddec.append((d["solved_dec"] - d["truth_dec"]) * 3600.0)
    err.append(d["error_arcsec"])
    case_ids.append(d["case"])
dra = np.array(dra)
ddec = np.array(ddec)
err = np.array(err)

WRONG = err > 1.0
n_correct = int((~WRONG).sum())
n_wrong = int(WRONG.sum())
print(f"loaded {len(solved)} solved cases  correct={n_correct}  wrong (>1″)={n_wrong}")

# Two panels: zoomed (sub-arcsec, the "correct" cluster) and full
# (entire range, log-spaced — the wrong-cluster outliers live decimally
# far away and would dominate any single-axis view).
fig, (ax_zoom, ax_full) = plt.subplots(1, 2, figsize=(14, 7))

# Zoomed: sub-arcsec residual cloud.
ax_zoom.scatter(dra[~WRONG], ddec[~WRONG], s=14, alpha=0.55,
                color="tab:blue", edgecolor="none",
                label=f"correct ≤1″ (n={n_correct})")
ax_zoom.scatter(dra[WRONG], ddec[WRONG], s=24, alpha=0.85,
                color="crimson", edgecolor="black", linewidths=0.5,
                marker="x", label=f"wrong >1″ (n={n_wrong})")
ax_zoom.axhline(0, color="gray", lw=0.6, alpha=0.5)
ax_zoom.axvline(0, color="gray", lw=0.6, alpha=0.5)
# Draw 1″ circle for reference.
theta = np.linspace(0, 2*np.pi, 200)
ax_zoom.plot(np.cos(theta), np.sin(theta), color="black", lw=0.8, ls=":",
             label="1″ wrong-solve threshold")
ax_zoom.set_xlim(-2.0, 2.0)
ax_zoom.set_ylim(-2.0, 2.0)
ax_zoom.set_aspect("equal")
ax_zoom.set_xlabel("Δra · cos(dec) (arcsec)")
ax_zoom.set_ylabel("Δdec (arcsec)")
ax_zoom.set_title(f"Zoomed (±2″) — sub-arcsec residual cloud")
ax_zoom.legend(loc="upper right", fontsize=9)
ax_zoom.grid(True, alpha=0.3)

# Full: signed-symlog over the whole error range (so wrong-solve
# outliers also visible without losing the "correct" cluster at zero).
ax_full.scatter(dra[~WRONG], ddec[~WRONG], s=14, alpha=0.55,
                color="tab:blue", edgecolor="none",
                label=f"correct ≤1″ (n={n_correct})")
ax_full.scatter(dra[WRONG], ddec[WRONG], s=36, alpha=0.9,
                color="crimson", edgecolor="black", linewidths=0.7,
                marker="x", label=f"wrong >1″ (n={n_wrong})")
ax_full.axhline(0, color="gray", lw=0.6, alpha=0.5)
ax_full.axvline(0, color="gray", lw=0.6, alpha=0.5)
ax_full.set_xscale("symlog", linthresh=1.0)
ax_full.set_yscale("symlog", linthresh=1.0)
ax_full.set_aspect("equal")
ax_full.set_xlabel("Δra · cos(dec) (arcsec, symlog)")
ax_full.set_ylabel("Δdec (arcsec, symlog)")
ax_full.set_title("Full (symlog) — wrong-solve outliers")
ax_full.legend(loc="upper right", fontsize=9)
ax_full.grid(True, alpha=0.3)

fig.suptitle(
    "RA/Dec residuals — d9-mag20 / set2-dr3-mag19, r=1.0° + scale-hint\n"
    f"residual = solver field-centre WCS minus truth (n={len(solved)} solved)"
)
fig.tight_layout()
fig.savefig(OUT / "radec_residuals_r1.png", dpi=140, bbox_inches="tight")
print(f"wrote {OUT / 'radec_residuals_r1.png'}")

# ---- Component histograms (correct cluster only) -------------------------
dra_correct = dra[~WRONG]
ddec_correct = ddec[~WRONG]

# Common bins to make the two panels visually comparable.
combined = np.concatenate([dra_correct, ddec_correct])
lo, hi = np.percentile(combined, [0.5, 99.5])
abs_lim = max(abs(lo), abs(hi)) * 1.05
bins = np.linspace(-abs_lim, abs_lim, 81)

fig, (ax_ra, ax_dec) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, data, title, color in [
    (ax_ra, dra_correct, "Δra · cos(dec)", "tab:blue"),
    (ax_dec, ddec_correct, "Δdec", "tab:purple"),
]:
    ax.hist(data, bins=bins, color=color, edgecolor="white", alpha=0.85)
    mean = float(np.mean(data))
    median = float(np.median(data))
    std = float(np.std(data, ddof=1))
    ax.axvline(0, color="black", lw=0.7, alpha=0.6)
    ax.axvline(median, color="crimson", lw=1.2,
               label=f"median {median:+.4f}″")
    ax.axvline(mean, color="darkorange", lw=1.2, ls="--",
               label=f"mean {mean:+.4f}″")
    ax.set_xlabel(f"{title} (arcsec)")
    ax.set_title(f"{title}  (n={dra_correct.size} correct cases)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02, 0.97,
        f"σ = {std:.4f}″\n"
        f"min = {data.min():+.3f}″\n"
        f"max = {data.max():+.3f}″",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.6, alpha=0.85),
    )
ax_ra.set_ylabel("# cases")

fig.suptitle(
    "Residual component histograms (correct cluster only) — "
    "d9-mag20 / set2-dr3-mag19, r=1.0° + scale-hint",
    y=1.02,
)
fig.tight_layout()
fig.savefig(OUT / "radec_residual_hists_r1.png", dpi=140, bbox_inches="tight")
print(f"wrote {OUT / 'radec_residual_hists_r1.png'}")

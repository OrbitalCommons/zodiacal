#!/usr/bin/env python3
"""Generate plots analyzing failed plate solves."""

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def equatorial_to_galactic(ra_deg, dec_deg):
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    ra_ngp = math.radians(192.85948)
    dec_ngp = math.radians(27.12825)
    l_ncp = math.radians(122.93192)
    sin_b = (math.sin(dec) * math.sin(dec_ngp)
             + math.cos(dec) * math.cos(dec_ngp) * math.cos(ra - ra_ngp))
    b = math.asin(max(-1, min(1, sin_b)))
    y = math.cos(dec) * math.sin(ra - ra_ngp)
    x = (math.sin(dec) * math.cos(dec_ngp)
         - math.cos(dec) * math.sin(dec_ngp) * math.cos(ra - ra_ngp))
    l = l_ncp - math.atan2(y, x)
    return math.degrees(l) % 360, math.degrees(b)


def main():
    failures_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/failures")
    test_cases_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("test_cases")
    out_dir = failures_dir

    # Load failure data
    analysis = json.load(open(failures_dir / "analysis.json"))

    # Load ALL test cases for comparison
    all_cases = []
    for f in sorted(test_cases_dir.glob("*.json")):
        d = json.load(open(f))
        ra = d.get("ra_deg", 0)
        dec = d.get("dec_deg", 0)
        l, b = equatorial_to_galactic(ra, dec)
        all_cases.append({
            "file": f.stem,
            "ra": ra, "dec": dec, "l": l, "b": b,
            "n_sources": len(d.get("sources", [])),
        })

    fail_files = {f["file"] for f in analysis}
    solved = [c for c in all_cases if c["file"] not in fail_files]
    failed = [c for c in all_cases if c["file"] in fail_files]

    fig = plt.figure(figsize=(18, 14))

    # --- Plot 1: Sky map (equatorial) ---
    ax1 = fig.add_subplot(2, 2, 1, projection="aitoff")
    ax1.set_title("Failure Locations (Equatorial)", fontsize=12, pad=15)
    ax1.grid(True, alpha=0.3)

    # Convert RA to range [-pi, pi] for aitoff
    def to_aitoff(ra, dec):
        ra_rad = math.radians(ra)
        if ra_rad > math.pi:
            ra_rad -= 2 * math.pi
        return ra_rad, math.radians(dec)

    s_ra = [to_aitoff(c["ra"], c["dec"])[0] for c in solved]
    s_dec = [to_aitoff(c["ra"], c["dec"])[1] for c in solved]
    f_ra = [to_aitoff(c["ra"], c["dec"])[0] for c in failed]
    f_dec = [to_aitoff(c["ra"], c["dec"])[1] for c in failed]

    ax1.scatter(s_ra, s_dec, s=3, c="steelblue", alpha=0.3, label=f"Solved ({len(solved)})")
    ax1.scatter(f_ra, f_dec, s=40, c="red", marker="x", linewidths=1.5, label=f"Failed ({len(failed)})")
    ax1.legend(loc="lower right", fontsize=9)

    # --- Plot 2: Sky map (galactic) ---
    ax2 = fig.add_subplot(2, 2, 2, projection="aitoff")
    ax2.set_title("Failure Locations (Galactic)", fontsize=12, pad=15)
    ax2.grid(True, alpha=0.3)

    def to_aitoff_gal(l, b):
        l_rad = math.radians(l)
        if l_rad > math.pi:
            l_rad -= 2 * math.pi
        return l_rad, math.radians(b)

    s_l = [to_aitoff_gal(c["l"], c["b"])[0] for c in solved]
    s_b = [to_aitoff_gal(c["l"], c["b"])[1] for c in solved]
    f_l = [to_aitoff_gal(c["l"], c["b"])[0] for c in failed]
    f_b = [to_aitoff_gal(c["l"], c["b"])[1] for c in failed]

    ax2.scatter(s_l, s_b, s=3, c="steelblue", alpha=0.3, label=f"Solved ({len(solved)})")
    ax2.scatter(f_l, f_b, s=40, c="red", marker="x", linewidths=1.5, label=f"Failed ({len(failed)})")
    ax2.legend(loc="lower right", fontsize=9)

    # --- Plot 3: Source count distribution ---
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("Source Count Distribution", fontsize=12)

    solved_srcs = [c["n_sources"] for c in solved]
    failed_srcs = [c["n_sources"] for c in failed]

    bins = np.arange(0, 220, 10)
    ax3.hist(solved_srcs, bins=bins, alpha=0.5, color="steelblue", label="Solved", density=True)
    ax3.hist(failed_srcs, bins=bins, alpha=0.7, color="red", label="Failed", density=True)
    ax3.set_xlabel("Number of extracted sources")
    ax3.set_ylabel("Density")
    ax3.legend()
    ax3.axvline(x=np.median(failed_srcs), color="red", linestyle="--", alpha=0.5, label=f"Failed median={np.median(failed_srcs):.0f}")

    # --- Plot 4: Galactic latitude distribution ---
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("|b| Distribution: Failed vs All", fontsize=12)

    all_abs_b = [abs(c["b"]) for c in all_cases]
    fail_abs_b = [abs(c["b"]) for c in failed]

    bins_b = np.arange(0, 95, 5)
    ax4.hist(all_abs_b, bins=bins_b, alpha=0.4, color="steelblue", label=f"All ({len(all_cases)})", density=True)
    ax4.hist(fail_abs_b, bins=bins_b, alpha=0.7, color="red", label=f"Failed ({len(failed)})", density=True)
    ax4.set_xlabel("|Galactic latitude| (degrees)")
    ax4.set_ylabel("Density")
    ax4.legend()

    plt.tight_layout()
    out_path = out_dir / "failure_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")

    # --- Extra plot: Verification count vs source count ---
    fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 5))

    ax5.set_title("N_verified vs N_sources (failures only)", fontsize=12)
    n_src = [f["n_sources"] for f in analysis]
    n_ver = [f["n_verified"] for f in analysis]
    best_lo = [f["best_lo"] for f in analysis]

    sc = ax5.scatter(n_src, n_ver, c=best_lo, cmap="RdYlGn", s=60, edgecolors="black", linewidths=0.5)
    ax5.set_xlabel("N sources")
    ax5.set_ylabel("N verified")
    plt.colorbar(sc, ax=ax5, label="Best log-odds")
    for f in analysis:
        ax5.annotate(f["file"], (f["n_sources"], f["n_verified"]), fontsize=6, alpha=0.7)

    # Best log-odds histogram
    ax6.set_title("Best Rejected Log-Odds (failures)", fontsize=12)
    ax6.hist(best_lo, bins=15, color="red", alpha=0.7, edgecolor="black")
    ax6.set_xlabel("Best rejected log-odds")
    ax6.set_ylabel("Count")
    ax6.axvline(x=20.0, color="green", linestyle="--", label="Accept threshold (20)")
    ax6.legend()

    plt.tight_layout()
    out_path2 = out_dir / "failure_details.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path2}")


if __name__ == "__main__":
    main()

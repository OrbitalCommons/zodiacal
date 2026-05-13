#!/usr/bin/env python3
"""Render a zoomed grid of every top-N verification miss, classified
against the bundle catalog.

Each top-N detection that the solver did **not** match goes into its
own panel: a small crop of the image, an open-centre × at the
reported (x, y), and a rank label. Each miss is bucketed as either:

  • **in our catalog** — a bundle Gaia projection sits within
    `--catalog-near-px` pixels of the detection. The × is yellow and
    the nearby catalog projection is drawn as an open ○. This means
    the catalog has it but the solver/verifier didn't accept the
    match (typically because the local WCS error exceeds the
    `match_radius_pix` tolerance, or the Bayesian distractor
    density beats the Gaussian foreground at this radius).
  • **expected, but missing from our catalog** — no bundle Gaia
    projection within `--catalog-near-px`. The × is crimson. The
    detection is a real point source but our G≤20 bundle has no
    matching entry at its sky position (faint MW star, M101 globular
    cluster, background galaxy / AGN, centroid-offset bloom, etc.).

A second figure plots Gaia G magnitude against extractor flux for the
trace's matched stars (hits), with a linear fit
`log10(flux) = slope·G + intercept`. Misses are overplotted: in-catalog
ones at the nearest catalog projection's G, not-in-catalog ones at a
fictitious G off the bottom of the panel.

The bundle catalog dump (`trace.projected_catalog`) is required;
re-run `bench-bundle --trace-out` against a recent build (the field was
added in mid-2026). FITS is auto-downloaded from MAST when the case
JSON carries an `hst` block.

Example
-------
    python scripts/render_miss_grid.py \\
        --case-json /tmp/m101_solve/0000.json \\
        --trace-json /tmp/m101_pm/on/0000.trace.json \\
        --out-grid /tmp/m101_miss_grid.png \\
        --out-magflux /tmp/m101_mag_flux.png
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from _debugutil import (  # noqa: E402
    case_inst_blurb,
    detection_display_xy,
    draw_open_x,
    implied_g,
    photometric_fit,
    pick_image_hdu,
    resolve_fits_path,
)


COL_IN_CAT = "#FFFF00"        # yellow — bundle has a star near this miss
COL_NOT_IN_CAT = "crimson"    # red    — bundle has nothing near this miss
COL_HIT = "lime"              # green  — verifier matched this detection


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--case-json", type=Path, required=True)
    p.add_argument("--trace-json", type=Path, required=True)
    p.add_argument("--fits", type=Path, default=None,
                   help="FITS path. Auto-downloaded from MAST if omitted.")
    p.add_argument("--cache-dir", type=Path, default=Path("/tmp/fits_cache"))
    p.add_argument("--hdu", type=int, default=None)
    p.add_argument("--top-n", type=int, default=50,
                   help="Number of brightest detections to scan. (default: 50)")
    p.add_argument("--half-box", type=int, default=32,
                   help="Half-side of each crop in pixels. (default: 32)")
    p.add_argument("--cols", type=int, default=7,
                   help="Grid columns. (default: 7)")
    p.add_argument("--catalog-near-px", type=float, default=5.0)
    p.add_argument("--stretch-low-pct", type=float, default=25.0)
    p.add_argument("--stretch-high-pct", type=float, default=99.5)
    p.add_argument("--out-grid", type=Path, required=True,
                   help="Output PNG for the miss-grid figure.")
    p.add_argument("--out-magflux", type=Path, default=None,
                   help="Output PNG for the mag-vs-flux companion figure. "
                        "Skipped if omitted.")
    p.add_argument("--dpi", type=int, default=140)
    return p.parse_args()


class _OpenXHandler(HandlerBase):
    """Render the open-centre × as a legend swatch."""

    def __init__(self, color):
        super().__init__()
        self.color = color

    def create_artists(self, legend, orig, xd, yd, width, height, fs, trans):
        cx, cy = xd + width / 2, yd + height / 2
        r = min(width, height) * 0.40
        gap = r * 0.35
        arms = []
        for ang_deg in (45.0, 135.0, 225.0, 315.0):
            a = math.radians(ang_deg)
            ca, sa = math.cos(a), math.sin(a)
            arms.append(Line2D(
                [cx + gap * ca, cx + r * ca],
                [cy + gap * sa, cy + r * sa],
                color=self.color, lw=1.6, solid_capstyle="round",
                transform=trans,
            ))
        return arms


def render_grid(args, case, trace, img, pc_xy, pc_mag, misses, lo, hi):
    n = len(misses)
    cols = args.cols
    rows = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.7))
    axes = np.atleast_2d(axes)

    ny, nx = img.shape
    disp_vmin = 0.0
    disp_vmax = float(np.arcsinh(hi - lo))

    for idx, m in enumerate(misses):
        rank, flux, dx, dy, in_cat, nd, nm = m
        r_i, c_i = divmod(idx, cols)
        ax = axes[r_i, c_i]
        color = COL_IN_CAT if in_cat else COL_NOT_IN_CAT
        x0, x1 = int(dx - args.half_box), int(dx + args.half_box)
        y0, y1 = int(dy - args.half_box), int(dy + args.half_box)
        x0c, x1c = max(0, x0), min(nx, x1)
        y0c, y1c = max(0, y0), min(ny, y1)
        crop = img[y0c:y1c, x0c:x1c]
        disp = np.arcsinh(np.clip(crop, lo, hi) - lo)
        ax.imshow(disp, origin="lower", cmap="gray_r",
                  interpolation="nearest",
                  vmin=disp_vmin, vmax=disp_vmax,
                  extent=[x0c, x1c, y0c, y1c])
        ax.plot([x0, dx - 5], [dy, dy], color=color, lw=0.4, alpha=0.25)
        ax.plot([dx + 5, x1], [dy, dy], color=color, lw=0.4, alpha=0.25)
        ax.plot([dx, dx], [y0, dy - 5], color=color, lw=0.4, alpha=0.25)
        ax.plot([dx, dx], [dy + 5, y1], color=color, lw=0.4, alpha=0.25)
        draw_open_x(ax, dx, dy, color=color, arm_len=12.0, gap=4.5, lw=1.4)
        ax.text(dx + 14, dy + 14, f"#{rank}",
                color=color, fontsize=8, fontweight="bold",
                ha="left", va="bottom",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.7, pad=0.6))
        if in_cat:
            d = np.hypot(pc_xy[:, 0] - dx, pc_xy[:, 1] - dy)
            ci = int(np.argmin(d))
            cx, cy = pc_xy[ci]
            ax.add_patch(plt.Circle((cx, cy), 4, fill=False,
                                    edgecolor=COL_IN_CAT, lw=1.2, alpha=0.9))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xticks([])
        ax.set_yticks([])
        title = f"#{rank}  f={flux:.0f}"
        if in_cat:
            title += f"\nGaia G={nm:.1f} ({nd:.1f}px)"
        else:
            title += f"\nno cat ≤{args.catalog_near_px:.0f}px"
        ax.set_title(title, fontsize=7, pad=2,
                     color="black" if in_cat else "#600")

    # Hide leftover cells.
    for idx in range(n, rows * cols):
        r_i, c_i = divmod(idx, cols)
        axes[r_i, c_i].axis("off")

    n_in = sum(1 for m in misses if m[4])
    n_out = n - n_in
    in_cat_proxy = Line2D([], [], lw=0)
    not_in_cat_proxy = Line2D([], [], lw=0)
    leg_handles = [
        in_cat_proxy,
        not_in_cat_proxy,
        Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor=COL_IN_CAT, markersize=10, lw=0),
    ]
    leg_labels = [
        f"in our catalog (≤{args.catalog_near_px:.0f} px from a bundle Gaia projection): {n_in}",
        f"expected, but missing from our catalog: {n_out}",
        "open ○ = bundle Gaia projection position",
    ]
    fig.legend(handles=leg_handles, labels=leg_labels,
               handler_map={in_cat_proxy: _OpenXHandler(COL_IN_CAT),
                            not_in_cat_proxy: _OpenXHandler(COL_NOT_IN_CAT)},
               loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.02))

    case_id = trace.get("case") or args.case_json.stem
    n_quad = 4
    n_hits = len(trace["verification"]["matches"])
    n_proj = len(trace.get("projected_catalog") or [])
    err_arcsec = math.sqrt(
        (trace["solved"]["ra_deg"] - trace["truth"]["ra_deg"]) ** 2
        + (trace["solved"]["dec_deg"] - trace["truth"]["dec_deg"]) ** 2
    ) * 3600
    fig.suptitle(
        f"{case_id} top-{args.top_n} verification misses{case_inst_blurb(case)}\n"
        f"WCS error {err_arcsec:.2f}″,  "
        f"{n_proj} bundle catalog stars project on-image "
        f"({n_quad} quad + {n_hits} verification hits used)",
        fontsize=10, y=0.995,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    args.out_grid.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_grid, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out_grid}")
    plt.close(fig)


def render_magflux(args, case, trace, sources_sorted, misses):
    """Companion plot: matched-hit fit + miss overlay."""
    hits = []
    for h in trace["verification"]["matches"]:
        fi = h["field_idx"]
        if "mag_g" in h and fi < len(sources_sorted):
            hits.append((sources_sorted[fi]["flux"], h["mag_g"], fi + 1))
    for i, cat in enumerate(trace["quad"]["catalog"]):
        if "mag_g" in cat:
            fi = trace["quad"]["field_indices"][i]
            if fi < len(sources_sorted):
                hits.append((sources_sorted[fi]["flux"], cat["mag_g"], fi + 1))
    if not hits:
        print("no hits to plot; skipping mag-vs-flux figure", file=sys.stderr)
        return

    xs = np.array([h[0] for h in hits])
    ys = np.array([h[1] for h in hits])
    slope, intercept = np.polyfit(ys, np.log10(xs), 1)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(xs, ys, s=80, color=COL_HIT, edgecolor="black",
               linewidth=0.5, alpha=0.9, label=f"matched hits ({len(hits)})")
    for f, g, r in hits:
        ax.annotate(f"#{r}", xy=(f, g), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, alpha=0.8)

    miss_xs_in = [m[1] for m in misses if m[4]]
    miss_ys_in = [m[6] for m in misses if m[4]]
    miss_xs_not = [m[1] for m in misses if not m[4]]
    if miss_xs_in:
        ax.scatter(miss_xs_in, miss_ys_in, s=50, marker="x",
                   color=COL_IN_CAT, edgecolor="black", linewidths=1.2,
                   alpha=0.85,
                   label=f"misses, in catalog ({len(miss_xs_in)})")
    if miss_xs_not:
        base_y = max(ys.max() if len(ys) else 20.0, 22.0)
        ax.scatter(miss_xs_not, [base_y + 0.5] * len(miss_xs_not),
                   s=50, marker="x", color=COL_NOT_IN_CAT, alpha=0.7,
                   label=f"misses, NOT in catalog ({len(miss_xs_not)}) — "
                         f"plotted at fictitious G={base_y+0.5:.1f}")

    fit_g = np.linspace(ys.min() - 1, max(22.0, ys.max() + 1), 100)
    fit_flux = 10 ** (slope * fit_g + intercept)
    ax.plot(fit_flux, fit_g, "--", color="0.4", lw=1.2,
            label=f"hits fit: log10(flux) = {slope:+.2f}·G {intercept:+.2f}")

    ax.set_xscale("log")
    ax.set_xlabel("Instrumental flux (extractor units)")
    ax.set_ylabel("Gaia G magnitude")
    ax.invert_yaxis()
    case_id = trace.get("case") or args.case_json.stem
    ax.set_title(f"{case_id} — instrumental flux vs Gaia G magnitude{case_inst_blurb(case)}\n"
                 "matched hits define the photometric zero-point; misses overlaid",
                 fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    args.out_magflux.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_magflux, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out_magflux}")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    case = json.loads(args.case_json.read_text())
    trace = json.loads(args.trace_json.read_text())

    sources_sorted = sorted(case["sources"], key=lambda s: -s["flux"])
    if not sources_sorted:
        sys.exit("Case JSON has no detected sources.")
    if not (trace.get("projected_catalog") or []):
        sys.exit("Trace JSON has no `projected_catalog`. Re-run bench-bundle "
                 "with a build that emits this field (mid-2026 or later).")

    fits_path = resolve_fits_path(case, args.fits, args.cache_dir)
    hdul = fits.open(fits_path)
    sci = pick_image_hdu(hdul, args.hdu)
    img = np.squeeze(sci.data).astype(np.float32)
    ny, nx = img.shape

    pc = trace["projected_catalog"]
    pc_xy = np.array([[s["px"], s["py"]] for s in pc])
    pc_mag = np.array([s["mag_g"] for s in pc])
    print(f"projected_catalog: {len(pc)} stars on-image, "
          f"mag_g {pc_mag.min():.2f}..{pc_mag.max():.2f}")

    matched_field_idx = {m["field_idx"] for m in trace["verification"]["matches"]}
    quad_field_set = set(trace["quad"]["field_indices"])

    misses = []
    for rank, s in enumerate(sources_sorted[:args.top_n], 1):
        fid = rank - 1
        if fid in quad_field_set or fid in matched_field_idx:
            continue
        dx, dy = detection_display_xy(s, ny)
        if len(pc_xy):
            d = np.hypot(pc_xy[:, 0] - dx, pc_xy[:, 1] - dy)
            nd = float(d.min())
            ni = int(np.argmin(d))
            nm = float(pc_mag[ni])
        else:
            nd, nm = float("inf"), float("nan")
        in_cat = nd <= args.catalog_near_px
        misses.append((rank, s["flux"], dx, dy, in_cat, nd, nm))

    n_in = sum(1 for m in misses if m[4])
    print(f"\n{len(misses)} top-{args.top_n} misses:")
    print(f"  in our catalog (≤{args.catalog_near_px:.0f} px): {n_in}")
    print(f"  NOT in our catalog:                        {len(misses) - n_in}")

    finite = img[np.isfinite(img)]
    lo, hi = np.percentile(finite, [args.stretch_low_pct, args.stretch_high_pct])
    render_grid(args, case, trace, img, pc_xy, pc_mag, misses, lo, hi)
    if args.out_magflux:
        render_magflux(args, case, trace, sources_sorted, misses)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

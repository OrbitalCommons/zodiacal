#!/usr/bin/env python3
"""Diagnose a single detection on a plate-solver bench case.

Given a `bench-bundle --trace-out` sidecar JSON and its companion
extraction JSON, this tool zooms in on one detection (by brightness
rank, or by field-source index) and renders a multi-radius diagnostic
panel showing:

  • a tight box around the reported (x, y) of the detection
  • a mid-range box revealing the local PSF / saturation environment
  • a wide-area box showing the surrounding plate context

All panels share the same arcsinh-stretched white→black colorbar so
brightness is directly comparable across zoom levels. The reported
extractor centroid is marked with an open-centre × (so it never covers
the source) and a gold + marks the brightest pixel found in each crop.
The figure title carries the case ID + instrument, the reported (x, y)
+ flux, an implied Gaia G magnitude from a linear fit through the
trace's matched stars, and the nearest bundle catalog projection.

The FITS image is fetched automatically when the extraction JSON
carries an `hst` block (cached under `--cache-dir`); pass `--fits` to
override.

Example
-------
    python scripts/probe_detection.py \\
        --case-json /tmp/m101_solve/0000.json \\
        --trace-json /tmp/m101_pm/on/0000.trace.json \\
        --rank 2 \\
        --out /tmp/m101_rank2.png

    # Hubble case via MAST auto-download:
    python scripts/probe_detection.py \\
        --case-json ../zodiacal-test-cases/hubble-f606w/jbrv14010.json \\
        --trace-json /tmp/hubble_sample/on/jbrv14010.trace.json \\
        --rank 5 --out /tmp/jbrv_rank5.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--case-json", type=Path, required=True,
                   help="Extraction sources JSON.")
    p.add_argument("--trace-json", type=Path, required=True,
                   help="bench-bundle --trace-out sidecar.")
    target = p.add_mutually_exclusive_group()
    target.add_argument("--rank", type=int, default=None,
                        help="1-based brightness rank (1 = brightest).")
    target.add_argument("--field-idx", type=int, default=None,
                        help="0-based field source index.")
    p.add_argument("--fits", type=Path, default=None,
                   help="FITS image path. Auto-downloaded from MAST if omitted.")
    p.add_argument("--cache-dir", type=Path, default=Path("/tmp/fits_cache"))
    p.add_argument("--hdu", type=int, default=None)
    p.add_argument("--zoom-radii", type=str, default="32,200,600",
                   help="Comma-separated half-box sizes in pixels. (default: 32,200,600)")
    p.add_argument("--stretch-low-pct", type=float, default=10.0)
    p.add_argument("--stretch-high-pct", type=float, default=99.7)
    p.add_argument("--catalog-near-px", type=float, default=5.0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dpi", type=int, default=140)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    case = json.loads(args.case_json.read_text())
    trace = json.loads(args.trace_json.read_text())
    case_id = trace.get("case") or args.case_json.stem
    sources_sorted = sorted(case["sources"], key=lambda s: -s["flux"])
    if not sources_sorted:
        sys.exit("Case JSON has no detected sources.")

    if args.rank is not None:
        if args.rank < 1 or args.rank > len(sources_sorted):
            sys.exit(f"--rank {args.rank} out of range 1..{len(sources_sorted)}")
        field_idx = args.rank - 1
    elif args.field_idx is not None:
        if args.field_idx < 0 or args.field_idx >= len(sources_sorted):
            sys.exit(f"--field-idx out of range")
        field_idx = args.field_idx
    else:
        matched = {m["field_idx"] for m in trace["verification"]["matches"]}
        quad_set = set(trace["quad"]["field_indices"])
        field_idx = next(
            (i for i, _ in enumerate(sources_sorted)
             if i not in matched and i not in quad_set),
            0,
        )
    rank = field_idx + 1
    det = sources_sorted[field_idx]
    flux = det["flux"]

    fits_path = resolve_fits_path(case, args.fits, args.cache_dir)
    hdul = fits.open(fits_path)
    sci = pick_image_hdu(hdul, args.hdu)
    img = np.squeeze(sci.data).astype(np.float32)
    if img.ndim != 2:
        sys.exit(f"Selected HDU has shape {img.shape}; need 2-D image.")
    ny, nx = img.shape
    dx, dy = detection_display_xy(det, ny)

    slope, intercept, n_hits, g_lo, g_hi = photometric_fit(trace, sources_sorted)
    g_est = implied_g(flux, slope, intercept)
    if slope is not None:
        photo_blurb = (
            f"implied Gaia G ≈ {g_est:.2f} from {n_hits}-hit fit "
            f"log10(flux)={slope:+.3f}·G {intercept:+.3f} "
            f"(G range {g_lo:.1f}…{g_hi:.1f})"
        )
    else:
        photo_blurb = "(no photometric fit available — too few hits)"

    pc = trace.get("projected_catalog") or []
    in_cat_blurb = ""
    if pc:
        pc_xy = np.array([[s["px"], s["py"]] for s in pc])
        d_arr = np.hypot(pc_xy[:, 0] - dx, pc_xy[:, 1] - dy)
        j = int(np.argmin(d_arr))
        d_near = float(d_arr[j])
        verdict = ("in our catalog" if d_near <= args.catalog_near_px
                   else f"NOT in our catalog within {args.catalog_near_px:.0f} px")
        in_cat_blurb = f"  · nearest bundle Gaia: G={pc[j]['mag_g']:.1f} at {d_near:.1f} px → {verdict}"

    finite = img[np.isfinite(img)]
    lo, hi = np.percentile(finite, [args.stretch_low_pct, args.stretch_high_pct])
    disp_vmin = 0.0
    disp_vmax = float(np.arcsinh(hi - lo))
    radii = [int(r.strip()) for r in args.zoom_radii.split(",")]

    fig, axes = plt.subplots(1, len(radii), figsize=(5 * len(radii) + 1, 5))
    if len(radii) == 1:
        axes = [axes]
    ims = []
    for ax, half in zip(axes, radii):
        x0, x1 = max(0, int(dx - half)), min(nx, int(dx + half))
        y0, y1 = max(0, int(dy - half)), min(ny, int(dy + half))
        crop = img[y0:y1, x0:x1]
        disp = np.arcsinh(np.clip(crop, lo, hi) - lo)
        im = ax.imshow(disp, origin="lower", cmap="gray_r",
                       interpolation="nearest",
                       vmin=disp_vmin, vmax=disp_vmax,
                       extent=[x0, x1, y0, y1])
        ims.append(im)
        arm = max(8.0, half * 0.05)
        gap = max(3.0, half * 0.018)
        draw_open_x(ax, dx, dy, arm_len=arm, gap=gap, lw=1.6)
        ax.text(dx + arm + 4, dy + arm + 4, f"#{rank}",
                color="crimson", fontsize=10, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.75, pad=1))
        sub_filled = np.nan_to_num(crop, nan=0.0)
        if sub_filled.size:
            py, px = np.unravel_index(int(np.nanargmax(sub_filled)), sub_filled.shape)
            peak_x, peak_y = x0 + px, y0 + py
            peak_val = float(sub_filled[py, px])
            ax.scatter([peak_x], [peak_y], s=120, marker="+",
                       color="gold", linewidths=1.4, alpha=0.9)
            ax.set_title(f"±{half} px (peak={peak_val:.0f} at ({peak_x},{peak_y}))",
                         fontsize=10)
        ax.set_xlabel("display x (px)")
        if ax is axes[0]:
            ax.set_ylabel("display y (px)")

    fig.tight_layout(rect=(0, 0, 0.93, 1))
    cbar_ax = fig.add_axes((0.94, 0.12, 0.012, 0.78))
    cb = fig.colorbar(ims[0], cax=cbar_ax)
    tick_arcsinh = np.linspace(disp_vmin, disp_vmax, 6)
    tick_counts = np.sinh(tick_arcsinh) + lo
    cb.set_ticks(tick_arcsinh)
    cb.set_ticklabels([f"{v:.1f}" for v in tick_counts])
    cb.set_label(
        f"pixel counts (global p{args.stretch_low_pct:g}..p{args.stretch_high_pct:g}, "
        "arcsinh-stretched)", fontsize=9)

    fig.suptitle(
        f"{case_id} rank-{rank} detection diagnosis{case_inst_blurb(case)} — "
        f"display ({dx:.0f},{dy:.0f}), extractor flux={flux:.0f}\n"
        f"{photo_blurb}{in_cat_blurb}\n"
        "Red ×: extractor centroid.  Gold +: local peak pixel in each box.",
        fontsize=10, y=0.99,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

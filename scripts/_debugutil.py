"""Shared helpers for the bench-debugging scripts in `scripts/`.

These tools — `probe_detection.py`, `render_miss_grid.py`, and
`render_solver_trace.py` — all need the same handful of utilities:

* resolve a FITS path from a case JSON (with optional MAST auto-download)
* fit `log10(flux) = slope·G + intercept` through a trace's matched stars
* draw the open-centre × that lets the source pixel show through

Keeping them here avoids drift between the three callers.
"""
from __future__ import annotations

import math
import sys
import urllib.request
from pathlib import Path

import numpy as np
from astropy.io import fits


def resolve_fits_path(case_json: dict, override: Path | None,
                      cache_dir: Path) -> Path:
    """Return a path to the case's FITS image, downloading via MAST if
    no local file is supplied.

    The case JSON's optional `hst` block carries the URL (see
    starfield-datasources' MAST helper, which writes
    `hst.image_url` or `hst.product_uri`). Downloads are cached
    by filename under `cache_dir` so re-runs are instant.
    """
    if override is not None:
        if not override.exists():
            sys.exit(f"--fits {override} not found")
        return override
    hst = case_json.get("hst") or {}
    url = hst.get("image_url")
    if not url:
        uri = hst.get("product_uri")
        if uri and uri.startswith("mast:"):
            url = ("https://mast.stsci.edu/api/v0.1/Download/file?uri=" + uri)
    if not url:
        sys.exit("Case JSON has no `hst.image_url` or `hst.product_uri`; "
                 "pass --fits explicitly.")
    filename = hst.get("product_filename") or url.split("/")[-1].split("?")[0]
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / filename
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    print(f"downloading {url}\n  -> {dst}", file=sys.stderr)
    urllib.request.urlretrieve(url, dst)
    return dst


def pick_image_hdu(hdul: fits.HDUList, idx: int | None):
    """Return the requested HDU or the first 2-D image HDU."""
    if idx is not None:
        return hdul[idx]
    for h in hdul:
        if h.data is not None and h.data.ndim >= 2:
            return h
    sys.exit("No 2-D image HDU found in FITS file.")


def photometric_fit(trace: dict, sources_sorted: list):
    """Linear fit through the trace's matched (flux, G) hits.

    Returns `(slope, intercept, n_hits, g_min, g_max)` for
    `log10(flux) = slope·G + intercept`. `slope` is `None` when there
    aren't enough hits to fit (need ≥2).
    """
    hits = []
    for h in trace["verification"]["matches"]:
        fi = h["field_idx"]
        if "mag_g" not in h or fi >= len(sources_sorted):
            continue
        hits.append((sources_sorted[fi]["flux"], h["mag_g"]))
    for i, cat in enumerate(trace["quad"]["catalog"]):
        if "mag_g" not in cat:
            continue
        fi = trace["quad"]["field_indices"][i]
        if fi < len(sources_sorted):
            hits.append((sources_sorted[fi]["flux"], cat["mag_g"]))
    if len(hits) < 2:
        return None, None, len(hits), float("nan"), float("nan")
    flux = np.array([h[0] for h in hits])
    g = np.array([h[1] for h in hits])
    slope, intercept = np.polyfit(g, np.log10(flux), 1)
    return float(slope), float(intercept), len(hits), float(g.min()), float(g.max())


def implied_g(flux: float, slope: float | None, intercept: float | None) -> float:
    """G magnitude implied by `flux` under the linear hits fit."""
    if slope is None or slope == 0:
        return float("nan")
    return (math.log10(flux) - intercept) / slope


def draw_open_x(ax, dx, dy, *, color="crimson", arm_len=14.0,
                gap=4.0, lw=1.4, alpha=0.85):
    """× whose centre is left empty so the source pixel under it
    remains visible. Four short line segments meeting at a `gap`-radius
    boundary around (dx, dy)."""
    for ang_deg in (45.0, 135.0, 225.0, 315.0):
        a = math.radians(ang_deg)
        ca, sa = math.cos(a), math.sin(a)
        ax.plot([dx + gap * ca, dx + arm_len * ca],
                [dy + gap * sa, dy + arm_len * sa],
                color=color, lw=lw, alpha=alpha, solid_capstyle="round")


def detection_display_xy(det: dict, ny: int) -> tuple:
    """Convert an extractor (x, y) — which is y-from-top because
    zodiacal flips rows on load — to display (y-from-bottom) coords."""
    return float(det["x"]), float(ny - 1 - det["y"])


def case_inst_blurb(case: dict) -> str:
    """One-line target/instrument summary for figure titles when the
    case carries an `hst` block; empty string otherwise."""
    hst = case.get("hst") or {}
    target = hst.get("target_name") or ""
    instr = hst.get("instrument_name") or ""
    filt = hst.get("filters") or ""
    if not target:
        return ""
    return f" {target} ({instr}/{filt})".rstrip("/ ")

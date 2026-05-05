"""Visual analysis of a built `.zdcl` v3 index.

Loads a `.zdcl` file via mmap-backed numpy views (no full-file copy) and
emits a set of self-contained figures characterising the index's quad
coverage, sky distribution, code-space density, and star reuse.

Usage
-----
    python3 scripts/analyze_index.py scratch/gaia_g14.zdcl

By default, figures are written to `scripts/out/<index-stem>/`. Pass
`--out-dir` to override.

Each figure answers one question about the index:

  scale_coverage.png : What scales / magnitudes does the index cover?
                       What's the per-cell quad density spread?
  sky_density.png    : Where on the sky are the stars and quads? Are
                       there dead zones or galactic-plane oversampling?
  code_space.png     : Do the canonical 4-D codes spread out (good for
                       matching) or pile up in a corner (bad)?
  star_reuse.png     : Does any one star dominate? Is the `max_reuse`
                       cap hitting?
  fov_coverage.png   : For a typical FOV at random sky positions, how
                       many quads land inside? (this is the metric the
                       solver actually cares about)

The point of these plots is to give a fast intuition for whether a
freshly-built index is well-behaved before trusting it on real data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure


STAR_DTYPE = np.dtype([
    ("catalog_id", "<u8"),
    ("ra", "<f8"),
    ("dec", "<f8"),
    ("mag", "<f8"),
])
QUAD_DTYPE = np.dtype([("star_ids", "<u4", (4,))])
CODE_DTYPE = np.dtype([("code", "<f8", (4,))])

ARCSEC_PER_RAD = (180.0 / math.pi) * 3600.0


@dataclass
class IndexHeader:
    version: int
    cell_depth: int
    n_cells: int
    n_stars: int
    n_quads: int
    scale_lower_rad: float
    scale_upper_rad: float
    metadata: Optional[dict]


class ZdclV3:
    """Memory-mapped view of a `.zdcl` v3 file.

    Star, quad, and code blocks are exposed as numpy structured arrays
    backed by the mmap — no copy until you index into them.
    """

    def __init__(self, path: Path):
        self.path = path
        self.size_bytes = path.stat().st_size

        with path.open("rb") as f:
            magic = f.read(4)
            if magic != b"ZDCL":
                raise ValueError(f"bad magic {magic!r} in {path}")
            (version,) = struct.unpack("<I", f.read(4))
            if version != 3:
                raise ValueError(f"only v3 is supported; got v{version}")
            (meta_len,) = struct.unpack("<Q", f.read(8))
            meta_bytes = f.read(meta_len)
            metadata = json.loads(meta_bytes) if meta_bytes else None
            (cell_depth,) = struct.unpack("<B", f.read(1))
            f.read(7)  # reserved
            (n_cells,) = struct.unpack("<Q", f.read(8))

            cell_table_bytes = n_cells * (8 + 8 + 4)
            cell_table_buf = f.read(cell_table_bytes)

            consumed = 4 + 4 + 8 + meta_len + 1 + 7 + 8 + cell_table_bytes
            pad = (-consumed) & 7
            if pad:
                f.read(pad)
            consumed += pad

            (n_stars,) = struct.unpack("<Q", f.read(8))
            (n_quads,) = struct.unpack("<Q", f.read(8))
            (scale_lower,) = struct.unpack("<d", f.read(8))
            (scale_upper,) = struct.unpack("<d", f.read(8))
            consumed += 8 * 4
            stars_offset = consumed

        self.header = IndexHeader(
            version=version,
            cell_depth=cell_depth,
            n_cells=n_cells,
            n_stars=n_stars,
            n_quads=n_quads,
            scale_lower_rad=scale_lower,
            scale_upper_rad=scale_upper,
            metadata=metadata,
        )

        # Cell table as structured array.
        cell_dtype = np.dtype([("cell_id", "<u8"), ("star_offset", "<u8"), ("star_count", "<u4")])
        self.cell_table = np.frombuffer(cell_table_buf, dtype=cell_dtype)

        # mmap the whole file once for the bulk arrays.
        self._mm = np.memmap(path, dtype=np.uint8, mode="r")
        stars_size = n_stars * STAR_DTYPE.itemsize
        quads_offset = stars_offset + stars_size
        quads_size = n_quads * QUAD_DTYPE.itemsize
        codes_offset = quads_offset + quads_size
        codes_size = n_quads * CODE_DTYPE.itemsize
        self.stars = np.frombuffer(
            self._mm[stars_offset:stars_offset + stars_size].tobytes(),
            dtype=STAR_DTYPE,
        )
        self.quads = np.frombuffer(
            self._mm[quads_offset:quads_offset + quads_size].tobytes(),
            dtype=QUAD_DTYPE,
        )["star_ids"]
        self.codes = np.frombuffer(
            self._mm[codes_offset:codes_offset + codes_size].tobytes(),
            dtype=CODE_DTYPE,
        )["code"]


# --- analyses -----------------------------------------------------------


def angular_distance_rad(ra1, dec1, ra2, dec2):
    """Vectorised great-circle distance in radians."""
    sin_dec = np.sin(dec1) * np.sin(dec2)
    cos_dec = np.cos(dec1) * np.cos(dec2)
    cos_sep = sin_dec + cos_dec * np.cos(ra1 - ra2)
    return np.arccos(np.clip(cos_sep, -1.0, 1.0))


def quad_backbone_arcsec(z: ZdclV3) -> np.ndarray:
    """Maximum angular separation (arcsec) over all 6 pairs of stars in
    each quad.

    Note: `canonical_quad_order` puts the longest pair at indices [0],[1]
    pre-encoding, but `compute_canonical_code` may then reorder the
    members during canonical encoding — so reading just `star_ids[0]`
    vs `star_ids[1]` from disk does NOT always give the longest pair.
    We take the max over all 6 pairwise distances to recover the
    defining backbone scale unambiguously.
    """
    sids = z.quads
    rs = z.stars["ra"][sids]   # (N, 4)
    ds = z.stars["dec"][sids]  # (N, 4)
    sin_d = np.sin(ds)
    cos_d = np.cos(ds)
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    n = sids.shape[0]
    max_sep = np.zeros(n, dtype=np.float64)
    for i, j in pairs:
        cos_sep = sin_d[:, i] * sin_d[:, j] + cos_d[:, i] * cos_d[:, j] * np.cos(rs[:, i] - rs[:, j])
        sep = np.arccos(np.clip(cos_sep, -1.0, 1.0))
        np.maximum(max_sep, sep, out=max_sep)
    return max_sep * ARCSEC_PER_RAD


def star_reuse_counts(z: ZdclV3) -> np.ndarray:
    """How many quads each star participates in, indexed by star id."""
    flat = z.quads.reshape(-1)
    counts = np.bincount(flat, minlength=z.header.n_stars)
    return counts


def quads_per_cell(z: ZdclV3) -> np.ndarray:
    """Quads per HEALPix cell (anchor-star-defined): count quads whose
    `star_ids[0]` falls in each cell.
    """
    # cell_table is (cell_id, star_offset, star_count). Build a bin
    # edge array so np.searchsorted maps each anchor's star idx to a
    # cell bucket.
    bin_edges = np.empty(len(z.cell_table) + 1, dtype=np.int64)
    bin_edges[:-1] = z.cell_table["star_offset"]
    bin_edges[-1] = z.header.n_stars
    anchor_star = z.quads[:, 0].astype(np.int64)
    bucket = np.searchsorted(bin_edges, anchor_star, side="right") - 1
    counts = np.bincount(bucket, minlength=len(z.cell_table))
    return counts


def quad_centers(z: ZdclV3) -> tuple[np.ndarray, np.ndarray]:
    """Approximate (ra, dec) of each quad's centroid (mean of 3-D unit
    vectors of its 4 stars). Returns radians.
    """
    sids = z.quads
    rs = z.stars["ra"]
    ds = z.stars["dec"]
    # Stack (N, 4) → 4 unit vectors per quad.
    cos_d = np.cos(ds)
    x = cos_d * np.cos(rs)
    y = cos_d * np.sin(rs)
    zc = np.sin(ds)
    # Sum across the 4 members for each quad.
    sx = x[sids].sum(axis=1)
    sy = y[sids].sum(axis=1)
    sz = zc[sids].sum(axis=1)
    norm = np.sqrt(sx * sx + sy * sy + sz * sz)
    sx /= norm
    sy /= norm
    sz /= norm
    ra = np.arctan2(sy, sx)
    ra = np.where(ra < 0, ra + 2 * np.pi, ra)  # [0, 2π)
    dec = np.arcsin(np.clip(sz, -1.0, 1.0))
    return ra, dec


def fov_quad_count(
    z: ZdclV3,
    n_samples: int = 2000,
    fov_radius_deg: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """For `n_samples` random sky pointings, count quads whose centroid
    lies within `fov_radius_deg`. Sampling is uniform on the sphere.

    Implemented as a chunked BLAS GEMM over XYZ unit vectors:
    cos(separation) = sample_xyz · quad_xyz. One pass, ~seconds for
    1 M quads × 2 k samples on a single core.
    """
    rng = rng or np.random.default_rng(20260504)
    u = rng.uniform(size=n_samples)
    v = rng.uniform(size=n_samples)
    sample_ra = 2 * np.pi * u
    sample_dec = np.arcsin(2 * v - 1)
    cos_sd = np.cos(sample_dec)
    sample_xyz = np.column_stack([
        cos_sd * np.cos(sample_ra),
        cos_sd * np.sin(sample_ra),
        np.sin(sample_dec),
    ])  # (n_samples, 3)

    quad_ra, quad_dec = quad_centers(z)
    cos_qd = np.cos(quad_dec)
    quad_xyz = np.column_stack([
        cos_qd * np.cos(quad_ra),
        cos_qd * np.sin(quad_ra),
        np.sin(quad_dec),
    ])  # (n_quads, 3)

    cos_radius = math.cos(math.radians(fov_radius_deg))

    counts = np.zeros(n_samples, dtype=np.int64)
    # Chunk over samples so peak (chunk × n_quads) stays under ~1.5 GB.
    chunk = max(1, int(2_000_000 // max(quad_xyz.shape[0], 1)))
    chunk = max(min(chunk, n_samples), 1)
    for i in range(0, n_samples, chunk):
        block = sample_xyz[i:i + chunk]
        cos_sep = block @ quad_xyz.T  # (chunk, n_quads)
        counts[i:i + chunk] = np.sum(cos_sep >= cos_radius, axis=1)
    return counts


# --- figures ------------------------------------------------------------


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def fig_scale_coverage(z: ZdclV3) -> Figure:
    backbone = quad_backbone_arcsec(z)
    qpc = quads_per_cell(z)
    mags = z.stars["mag"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    fig.suptitle("Scale coverage & per-cell density", fontsize=13, weight="bold", y=1.02)

    # Backbone.
    ax = axes[0]
    lo = z.header.scale_lower_rad * ARCSEC_PER_RAD
    hi = z.header.scale_upper_rad * ARCSEC_PER_RAD
    bins = np.geomspace(max(lo * 0.8, 1.0), hi * 1.2, 80)
    ax.hist(backbone, bins=bins, color="#4c72b0", edgecolor="white", linewidth=0.3)
    ax.axvline(lo, color="#c44", ls="--", lw=1, label=f"scale_lower = {lo:.0f}″")
    ax.axvline(hi, color="#c44", ls="--", lw=1, label=f"scale_upper = {hi:.0f}″")
    ax.set_xscale("log")
    ax.set_xlabel("backbone angular distance (arcsec)")
    ax.set_ylabel("quads")
    ax.set_title(f"Quad backbone scale\n(N = {len(backbone):,})")
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    # Magnitudes.
    ax = axes[1]
    ax.hist(mags, bins=80, color="#55a868", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Gaia G magnitude")
    ax.set_ylabel("stars")
    median_mag = float(np.median(mags))
    p99 = float(np.percentile(mags, 99))
    ax.axvline(median_mag, color="#444", ls=":", lw=1)
    ax.text(
        median_mag, ax.get_ylim()[1] * 0.92,
        f"  median {median_mag:.2f}", color="#444", fontsize=9,
    )
    ax.set_title(f"Star magnitude depth\n(N = {len(mags):,}, p99 = {p99:.2f})")

    # Quads per cell.
    ax = axes[2]
    populated = qpc[qpc > 0]
    ax.hist(populated, bins=60, color="#c44e52", edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("quads per HEALPix cell (anchor-defined)")
    ax.set_ylabel("cells (log)")
    n_pop = int(np.sum(qpc > 0))
    n_total = len(qpc)
    median_q = float(np.median(populated)) if populated.size else 0.0
    ax.set_title(
        f"Per-cell quad density\n"
        f"({n_pop:,}/{n_total:,} cells populated, median {median_q:.0f})"
    )
    return fig


def fig_sky_density(z: ZdclV3) -> Figure:
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), subplot_kw={"projection": "mollweide"})
    fig.suptitle("Sky density (Mollweide, log-scale)", fontsize=14, weight="bold", y=0.94)

    # Mollweide expects RA in [-π, π], Dec in [-π/2, π/2].
    star_ra = np.where(z.stars["ra"] > np.pi, z.stars["ra"] - 2 * np.pi, z.stars["ra"])
    star_dec = z.stars["dec"]

    quad_ra_full, quad_dec = quad_centers(z)
    quad_ra = np.where(quad_ra_full > np.pi, quad_ra_full - 2 * np.pi, quad_ra_full)

    # Subsample stars for hexbin to keep memory & runtime sane on 16 M.
    if star_ra.size > 2_000_000:
        idx = np.random.default_rng(0).choice(star_ra.size, size=2_000_000, replace=False)
        sra = star_ra[idx]
        sdec = star_dec[idx]
        sub_factor = star_ra.size / sra.size
    else:
        sra, sdec = star_ra, star_dec
        sub_factor = 1.0

    for ax, x, y, title, factor in [
        (axes[0], sra, sdec, f"Star density (subsampled to {sra.size:,}; full = {star_ra.size:,})", sub_factor),
        (axes[1], quad_ra, quad_dec, f"Quad-centroid density (N = {quad_ra.size:,})", 1.0),
    ]:
        hb = ax.hexbin(
            x, y,
            gridsize=(120, 60),
            mincnt=1,
            cmap="viridis",
            norm=LogNorm(),
        )
        ax.set_title(title)
        ax.set_xticklabels([])  # default ticks overlap with the projection edges
        cbar = fig.colorbar(hb, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label(f"count / hex (×{factor:.1f} for full-set scaling)")
        ax.grid(True, alpha=0.3, color="white", linewidth=0.4)

    return fig


def fig_code_space(z: ZdclV3, n_max: int = 250_000) -> Figure:
    codes = z.codes
    if codes.shape[0] > n_max:
        idx = np.random.default_rng(1).choice(codes.shape[0], size=n_max, replace=False)
        codes_sub = codes[idx]
    else:
        codes_sub = codes

    fig, axes = plt.subplots(4, 4, figsize=(11, 10))
    fig.suptitle(
        f"Quad code distribution (4-D corner plot, {codes_sub.shape[0]:,} samples)",
        fontsize=13, weight="bold",
    )

    labels = [r"$c_x$", r"$c_y$", r"$d_x$", r"$d_y$"]
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            ax.grid(True, alpha=0.2)
            if i == j:
                ax.hist(codes_sub[:, i], bins=80, color="#4c72b0", edgecolor="white", linewidth=0.2)
                ax.set_yticklabels([])
                if i == 3:
                    ax.set_xlabel(labels[i])
            elif i > j:
                ax.hexbin(
                    codes_sub[:, j], codes_sub[:, i],
                    gridsize=40, cmap="magma", mincnt=1, norm=LogNorm(),
                )
                if j == 0:
                    ax.set_ylabel(labels[i])
                if i == 3:
                    ax.set_xlabel(labels[j])
            else:
                ax.set_visible(False)
    fig.text(
        0.6, 0.92,
        "Codes should fill the unit interval roughly uniformly.\n"
        "Hot spots near corners or along axes indicate the canonical\n"
        "ordering is biased — bad for matching robustness.",
        fontsize=9, color="#444", ha="left", va="top",
    )
    return fig


def fig_star_reuse(z: ZdclV3) -> Figure:
    counts = star_reuse_counts(z)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    fig.suptitle("Star reuse across quads", fontsize=13, weight="bold")

    used = counts[counts > 0]
    unused = int(np.sum(counts == 0))
    fraction_unused = unused / len(counts) if len(counts) else 0.0

    ax = axes[0]
    bin_cap = 200  # Beyond this, switch to outlier annotation; otherwise
                   # an extreme reuse value (e.g. 250 k+ for an
                   # un-uniformized index) makes matplotlib draw
                   # hundreds of thousands of bars.
    p99_used = int(np.percentile(used, 99)) if used.size else 0
    n_outliers = int(np.sum(used > bin_cap))
    if used.size:
        max_used = int(used.max())
        bins = np.arange(0.5, min(max_used, bin_cap) + 1.5)
        ax.hist(np.minimum(used, bin_cap), bins=bins, color="#937860", edgecolor="white", linewidth=0.3)
        if n_outliers > 0:
            ax.axvline(bin_cap, color="#c44", ls="--", lw=1)
            ax.text(
                bin_cap, ax.get_ylim()[1] * 0.4,
                f"  {n_outliers:,} outliers > {bin_cap}\n"
                f"  (max = {max_used:,})",
                color="#c44", fontsize=8, va="top",
            )
    ax.set_xlabel("quads referencing a given star (clipped at 200)")
    ax.set_ylabel("stars")
    ax.set_yscale("log")
    median_used = float(np.median(used)) if used.size else 0.0
    ax.set_title(
        f"Reuse histogram (used stars only)\n"
        f"unused = {unused:,} ({fraction_unused*100:.1f} %), median = {median_used:.0f}, p99 = {p99_used}"
    )

    ax = axes[1]
    top_n = min(20, len(counts))
    if top_n:
        top_idx = np.argpartition(counts, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(counts[top_idx])[::-1]]
        cids = z.stars["catalog_id"][top_idx]
        mags = z.stars["mag"][top_idx]
        vals = counts[top_idx]
        y = np.arange(top_n)
        bars = ax.barh(y, vals, color="#937860")
        labels = [f"G={m:.1f}" for m in mags]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f" {val:,}", va="center", fontsize=7, color="#444")
        ax.set_xlim(0, vals.max() * 1.18)
    ax.set_xlabel("quads referencing this star")
    ax.set_title(f"Top {top_n} most-reused stars (by Gaia G mag)")
    return fig


def per_cell_max_used_mag(z: ZdclV3) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For every populated HEALPix cell, return (cell_ra_rad, cell_dec_rad,
    max_used_mag, used_count) — one entry per cell that has at least one
    star in the index.

    `max_used_mag` is the dimmest (highest-mag) star in the cell that
    appears in at least one quad — the depth the index reaches in that
    cell. NaN for cells whose stars are all unused. The cell center is
    the unit-vector centroid of its stars.
    """
    counts = star_reuse_counts(z)  # quads-using-star, indexed by star id
    mags = z.stars["mag"]
    ras = z.stars["ra"]
    decs = z.stars["dec"]

    cell_offsets = z.cell_table["star_offset"].astype(np.int64)
    cell_counts = z.cell_table["star_count"].astype(np.int64)

    # Pre-compute unit vectors so cell centroids are correct across the
    # RA=0 wrap.
    cos_d = np.cos(decs)
    x = cos_d * np.cos(ras)
    y = cos_d * np.sin(ras)
    zc = np.sin(decs)

    n_cells = len(z.cell_table)
    cell_ra = np.empty(n_cells)
    cell_dec = np.empty(n_cells)
    cell_max_mag = np.full(n_cells, np.nan)
    cell_used = np.zeros(n_cells, dtype=np.int64)

    for i in range(n_cells):
        off = cell_offsets[i]
        n = cell_counts[i]
        if n == 0:
            cell_ra[i] = np.nan
            cell_dec[i] = np.nan
            continue
        sl = slice(off, off + n)
        sx = x[sl].mean()
        sy = y[sl].mean()
        sz = zc[sl].mean()
        norm = math.sqrt(sx * sx + sy * sy + sz * sz)
        if norm > 0:
            sx /= norm; sy /= norm; sz /= norm
        ra = math.atan2(sy, sx)
        if ra < 0:
            ra += 2 * math.pi
        dec = math.asin(max(-1.0, min(1.0, sz)))
        cell_ra[i] = ra
        cell_dec[i] = dec

        used_mask = counts[sl] > 0
        n_used = int(used_mask.sum())
        cell_used[i] = n_used
        if n_used > 0:
            cell_max_mag[i] = float(mags[sl][used_mask].max())

    keep = ~np.isnan(cell_ra)
    return cell_ra[keep], cell_dec[keep], cell_max_mag[keep], cell_used[keep]


def fig_per_cell_depth(z: ZdclV3) -> Figure:
    """Two-panel sky map: the dimmest (highest-magnitude) used star
    per HEALPix cell, plus a histogram of those values.

    The dimmest used star in a cell tells you how deep into the catalog
    the index actually reached there — bright values mean the cell only
    used the few brightest stars; dim values mean the index pulled in
    fainter stars to build out quads.
    """
    cell_ra, cell_dec, max_mag, used = per_cell_max_used_mag(z)

    # Wrap RA to (-π, π) for Mollweide.
    ra_proj = np.where(cell_ra > np.pi, cell_ra - 2 * np.pi, cell_ra)
    has_used = used > 0

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(
        f"Index depth per HEALPix cell  "
        f"(depth {z.header.cell_depth}, {int(np.sum(has_used)):,} of {len(max_mag):,} cells contain quads)",
        fontsize=13, weight="bold", y=0.98,
    )
    gs = fig.add_gridspec(1, 3, width_ratios=[2.4, 1.0, 0.05], wspace=0.25)
    ax_map = fig.add_subplot(gs[0, 0], projection="mollweide")
    ax_hist = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[0, 2])

    if has_used.any():
        sc = ax_map.scatter(
            ra_proj[has_used], cell_dec[has_used],
            c=max_mag[has_used],
            s=42, cmap="viridis",  # bright (low mag) → dark; faint (high mag) → yellow
            vmin=float(np.nanpercentile(max_mag, 1)),
            vmax=float(np.nanpercentile(max_mag, 99)),
            edgecolors="#222", linewidths=0.3,
        )
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label("dimmest used G mag (higher = deeper index)")

    ax_map.grid(True, alpha=0.3)
    ax_map.set_xticklabels([])
    ax_map.set_title(
        f"Mollweide sky map  "
        f"({len(max_mag) - int(np.sum(has_used)):,} populated cells with no quads omitted)"
    )

    used_max = max_mag[has_used]
    ax_hist.hist(used_max, bins=60, color="#4c72b0", edgecolor="white", linewidth=0.3)
    median = float(np.median(used_max)) if used_max.size else 0.0
    p10 = float(np.percentile(used_max, 10)) if used_max.size else 0.0
    p90 = float(np.percentile(used_max, 90)) if used_max.size else 0.0
    ax_hist.axvline(median, color="#444", ls=":", lw=1)
    ax_hist.text(
        median, ax_hist.get_ylim()[1] * 0.95,
        f"  median = {median:.2f}", color="#444", fontsize=9, va="top",
    )
    ax_hist.set_xlabel("dimmest used G mag in cell")
    ax_hist.set_ylabel("cells")
    ax_hist.set_title(
        f"Distribution across {used_max.size:,} cells with quads\n"
        f"p10 / median / p90 = {p10:.2f} / {median:.2f} / {p90:.2f}"
    )
    return fig


def fig_fov_coverage(z: ZdclV3, fov_radius_deg: float = 0.5) -> Figure:
    counts = fov_quad_count(z, n_samples=2000, fov_radius_deg=fov_radius_deg)
    sky_area_sqdeg = 4 * math.pi * (180 / math.pi) ** 2  # ≈ 41,253
    fov_area_sqdeg = math.pi * fov_radius_deg ** 2

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    fig.suptitle(
        f"FOV quad coverage  ({fov_radius_deg}° radius ≈ {fov_area_sqdeg:.2f} sq.deg, "
        f"2000 random pointings)",
        fontsize=12, weight="bold",
    )

    ax = axes[0]
    # Use log y so the empty-FOV bar (often dominant for non-uniform indexes)
    # doesn't squash the rest of the distribution.
    cap = max(int(np.percentile(counts, 99) + 1), 20)
    bins = np.arange(0, cap + 2)
    ax.hist(np.minimum(counts, cap), bins=bins, color="#4c72b0", edgecolor="white", linewidth=0.3)
    n_outliers = int(np.sum(counts > cap))
    if n_outliers > 0:
        ax.axvline(cap, color="#c44", ls="--", lw=1)
        ax.text(cap, ax.get_ylim()[1] * 0.5, f"  {n_outliers} > {cap}\n  (max {int(counts.max())})",
                color="#c44", fontsize=8, va="top")
    median_c = int(np.median(counts))
    n_empty = int(np.sum(counts == 0))
    ax.set_yscale("log")
    ax.set_xlabel("quads in FOV")
    ax.set_ylabel("samples (log)")
    ax.set_title(
        f"Per-pointing quad count\n"
        f"empty FOVs = {n_empty:,} / {len(counts):,} "
        f"({n_empty/len(counts)*100:.1f} %), median = {median_c}"
    )

    ax = axes[1]
    # Survival curve: P(count ≥ N).
    sorted_counts = np.sort(counts)
    survival = 1.0 - np.arange(len(sorted_counts)) / len(sorted_counts)
    ax.step(sorted_counts, survival, where="post", color="#c44e52", lw=1.5)
    for thr in (1, 5, 10, 20):
        frac = float(np.mean(counts >= thr))
        ax.axhline(frac, color="#444", ls=":", lw=0.7)
        ax.text(
            sorted_counts.max() * 0.95, frac, f"  ≥{thr}: {frac*100:.1f}%",
            fontsize=8, color="#444", va="bottom", ha="right",
        )
    ax.set_ylim(0, 1)
    ax.set_xlabel("threshold N")
    ax.set_ylabel("P(quads in FOV ≥ N)")
    ax.set_title("Coverage survival curve")
    expected_density = z.header.n_quads / sky_area_sqdeg
    fig.text(
        0.5, -0.04,
        f"Uniform expectation: {expected_density * fov_area_sqdeg:.1f} quads / FOV  "
        f"(global density {expected_density:.1f} quads/sq.deg)",
        ha="center", fontsize=9, color="#666",
    )
    return fig


# --- driver -------------------------------------------------------------


def text_summary(z: ZdclV3) -> str:
    h = z.header
    lo_arcsec = h.scale_lower_rad * ARCSEC_PER_RAD
    hi_arcsec = h.scale_upper_rad * ARCSEC_PER_RAD
    populated = int(np.sum(z.cell_table["star_count"] > 0))
    counts = star_reuse_counts(z)
    used = counts[counts > 0]
    backbone = quad_backbone_arcsec(z)
    lines = [
        f"index file       : {z.path}",
        f"size on disk     : {z.size_bytes/1024/1024:.1f} MB",
        f"version          : v{h.version}  cell_depth={h.cell_depth}",
        f"cells            : {h.n_cells:,}  ({populated:,} populated, "
        f"{populated/h.n_cells*100:.1f} %)",
        f"stars            : {h.n_stars:,}",
        f"quads            : {h.n_quads:,}",
        f"scale band       : [{lo_arcsec:.1f}″, {hi_arcsec:.1f}″]",
        f"backbone p10/p50/p90 : "
        f"{np.percentile(backbone, 10):.1f}″ / {np.percentile(backbone, 50):.1f}″ / "
        f"{np.percentile(backbone, 90):.1f}″",
        f"star magnitudes  : "
        f"min {z.stars['mag'].min():.2f}, "
        f"median {np.median(z.stars['mag']):.2f}, "
        f"max {z.stars['mag'].max():.2f}",
        f"unused stars     : "
        f"{int(np.sum(counts == 0)):,} ({np.mean(counts == 0)*100:.1f} %)",
        f"used-star reuse  : "
        f"min {int(used.min()) if used.size else 0}, "
        f"median {int(np.median(used)) if used.size else 0}, "
        f"max {int(used.max()) if used.size else 0}",
        f"metadata         : {h.metadata}",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("zdcl_path", type=Path, help="path to a .zdcl v3 index")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="output directory (default: scripts/out/<index-stem>/)")
    p.add_argument("--fov-deg", type=float, default=0.5, help="FOV radius for coverage plot")
    p.add_argument("--no-plots", action="store_true", help="skip figure generation, print summary only")
    args = p.parse_args()

    if not args.zdcl_path.exists():
        sys.exit(f"no such file: {args.zdcl_path}")

    setup_style()

    out = args.out_dir or Path(__file__).resolve().parent / "out" / args.zdcl_path.stem
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    z = ZdclV3(args.zdcl_path)
    t_load = time.time() - t0

    summary = text_summary(z)
    print(summary)
    print(f"\n(load + summary: {t_load:.2f}s)")

    summary_path = out / "summary.txt"
    summary_path.write_text(summary + "\n")

    if args.no_plots:
        return

    figs = [
        ("scale_coverage.png", lambda: fig_scale_coverage(z)),
        ("sky_density.png", lambda: fig_sky_density(z)),
        ("code_space.png", lambda: fig_code_space(z)),
        ("star_reuse.png", lambda: fig_star_reuse(z)),
        ("per_cell_depth.png", lambda: fig_per_cell_depth(z)),
        ("fov_coverage.png", lambda: fig_fov_coverage(z, fov_radius_deg=args.fov_deg)),
    ]
    for name, builder in figs:
        t0 = time.time()
        fig = builder()
        fig.savefig(out / name)
        plt.close(fig)
        print(f"  wrote {out/name} ({time.time()-t0:.1f}s)")

    print(f"\noutputs: {out}")


if __name__ == "__main__":
    main()

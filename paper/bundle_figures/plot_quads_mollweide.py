#!/usr/bin/env python3
"""Mollweide map of quads-per-cell for the depth-9 G≤20 bundle.

We don't need to load any catalog — the .zqd file size is an exact
linear function of the per-cell quad count:

    file_size = HEADER_SIZE (32) + n_bands × BAND_ENTRY_SIZE (24)
              + n_quads × (QUAD_RECORD_SIZE (16) + CODE_RECORD_SIZE (32))

For 7 bands: file_size = 200 + 48 × n_quads.

Walks the bundle's quads/ directory once with os.scandir (fast — just
stat() per entry) and builds a per-cell quad-count map keyed by HEALPix
nested-scheme cell id. cdshealpix → (ra, dec) of each cell centre.
"""
import os
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
from astropy_healpix import HEALPix
from astropy.coordinates import ICRS, SkyCoord
import astropy.units as u

BUNDLE = Path("/home/meawoppl/scratch/bundle-build/d9-mag20.zdcl.bundle")
OUT = Path("/home/meawoppl/scratch/bundle-build/bench")
CACHE = OUT / "quads_per_cell_scan.npz"

HEADER_BYTES = 32 + 7 * 24   # 7 bands always emitted
BYTES_PER_QUAD = 16 + 32

CELL_RE = re.compile(r"^cell_(\d+)\.zqd$")


def load_or_scan() -> tuple[np.ndarray, np.ndarray]:
    """Return (cell_ids, sizes), reusing the on-disk cache when present."""
    if CACHE.exists():
        z = np.load(CACHE)
        print(f"loaded cached scan from {CACHE} ({z['cell_ids'].size:,} cells)")
        return z["cell_ids"], z["sizes"]
    print(f"scanning {BUNDLE / 'quads'} (no cache yet) …")
    cell_ids: list[int] = []
    sizes: list[int] = []
    for entry in os.scandir(BUNDLE / "quads"):
        m = CELL_RE.match(entry.name)
        if not m:
            continue
        cell_ids.append(int(m.group(1)))
        sizes.append(entry.stat().st_size)
    cell_ids_arr = np.array(cell_ids, dtype=np.int64)
    sizes_arr = np.array(sizes, dtype=np.int64)
    np.savez_compressed(CACHE, cell_ids=cell_ids_arr, sizes=sizes_arr)
    print(f"  scanned {cell_ids_arr.size:,} cells; cached to {CACHE}")
    return cell_ids_arr, sizes_arr


cell_ids, sizes = load_or_scan()
n_quads = (sizes - HEADER_BYTES) // BYTES_PER_QUAD
n_quads = np.clip(n_quads, 0, None)
print(f"  quads min={n_quads.min()} median={int(np.median(n_quads))} max={n_quads.max()} "
      f"mean={n_quads.mean():.1f} sum={n_quads.sum():,}")

# Resolve cell_id → (RA, Dec) via cdshealpix nested scheme.
hp = HEALPix(nside=2 ** 9, order="nested", frame=ICRS())
coords = hp.healpix_to_skycoord(cell_ids)
ra_deg = coords.ra.deg
dec_deg = coords.dec.deg

# Mollweide expects RA in [-π, π].
ra_wrapped = np.where(ra_deg > 180.0, ra_deg - 360.0, ra_deg)
ra_rad = np.radians(ra_wrapped)
dec_rad = np.radians(dec_deg)

# Hexbin in Mollweide, linear color scale.
fig = plt.figure(figsize=(13, 7))
ax = fig.add_subplot(111, projection="mollweide")
hb = ax.hexbin(
    ra_rad, dec_rad,
    C=n_quads,
    reduce_C_function=np.mean,
    gridsize=180,
    cmap="viridis",
    edgecolors="none",
)
ax.grid(True, alpha=0.3)

# Annotate galactic poles (J2000 equatorial).
ngp = SkyCoord(192.85948, 27.12825, unit="deg", frame="icrs")
sgp = SkyCoord(12.85948, -27.12825, unit="deg", frame="icrs")
for label, coord in [("NGP", ngp), ("SGP", sgp)]:
    ra_pole = coord.ra.deg
    if ra_pole > 180.0:
        ra_pole -= 360.0
    x = np.radians(ra_pole)
    y = np.radians(coord.dec.deg)
    ax.scatter([x], [y], marker="x", color="white", s=120, zorder=10, linewidths=2.5)
    ax.scatter([x], [y], marker="x", color="black", s=80, zorder=11, linewidths=1.5)
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(8, 8),
        textcoords="offset points",
        color="black",
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.6, alpha=0.9),
        zorder=12,
    )

# User-supplied marker: RA 17h 22m 38.2s, Dec -23° 49' 34" (J2000).
target_ra = (17 + 22 / 60.0 + 38.2 / 3600.0) * 15.0    # 260.65917°
target_dec = -(23 + 49 / 60.0 + 34 / 3600.0)            # -23.82611°
ra_target = target_ra - 360.0 if target_ra > 180.0 else target_ra
x_t = np.radians(ra_target)
y_t = np.radians(target_dec)
ax.scatter([x_t], [y_t], marker="o", color="lime", s=120, zorder=12,
           edgecolor="black", linewidths=1.2)
ax.annotate(
    f"{target_ra:.3f}°, {target_dec:+.3f}°",
    xy=(x_t, y_t),
    xytext=(10, -14),
    textcoords="offset points",
    color="black",
    fontsize=10,
    fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="lime", lw=0.8, alpha=0.95),
    zorder=13,
)

ax.set_title(
    "Quads per HEALPix cell — d9-mag20 bundle "
    f"(N={cell_ids.size:,} cells, total quads = {n_quads.sum():,})"
)
cb = fig.colorbar(hb, ax=ax, fraction=0.025, pad=0.04, location="bottom")
cb.set_label("mean quads per cell in hex bin")
fig.tight_layout()
fig.savefig(OUT / "quads_per_cell_mollweide.png", dpi=140, bbox_inches="tight")
print(f"wrote {OUT / 'quads_per_cell_mollweide.png'}")

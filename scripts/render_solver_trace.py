#!/usr/bin/env python3
"""Render a FITS plate with the matched quad overlaid using the manim
default-palette convention.

Inputs:
  --case        case id (used to look up FITS / JSON / trace)
  --fits        full path to the FITS frame
  --src-json    extracted-sources JSON (zodiacal extract output)
  --trace       bench-bundle --trace-out sidecar
  --out         output PNG path
  --hdu         HDU index (default = first 2-D image HDU)

Colour convention (manim default palette):
  A (anchor)            #FC6255   AB-backbone line          #FFFF00 (yellow)
  B (backbone tip)      #83C167   AC, AD reference lines    #58C4DD α≈0.55
  C (interior 1)        #58C4DD   AB-diameter circle        GREY_C dashed
  D (interior 2)        #FF862F   detected (non top-50)     cyan ×

Quad identification:
  A, B = the pair with the longest pixel-distance in {p0..p3}.
  C, D = the remaining two, ordered so C has the smaller x (matches
         astrometry.net's `cx ≤ dx` disambiguation).

Lines never overlap a vertex's marker disk: segments are clipped at
each endpoint's circle boundary.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from astropy.io import fits

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True)
ap.add_argument("--fits", required=True, type=Path)
ap.add_argument("--src-json", required=True, type=Path)
ap.add_argument("--trace", required=True, type=Path)
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--hdu", type=int, default=None,
                help="HDU index. Default: first 2-D image HDU.")
ap.add_argument("--top-n", type=int, default=50)
args = ap.parse_args()

# --- palette ---
COL_A = "#FC6255"   # anchor
COL_B = "#83C167"   # backbone tip
COL_C = "#58C4DD"   # interior 1
COL_D = "#FF862F"   # interior 2
COL_AB = "#FFFF00"  # AB backbone segment
COL_AC = "#58C4DD"  # AC/AD reference (semi-saturated)
COL_VAL_CIRCLE = "0.55"  # grey_c dashed
COL_HIT = "lime"
COL_MISS = "crimson"
COL_NONTOP = "cyan"

trace = json.loads(args.trace.read_text())
src_json = json.loads(args.src_json.read_text())

hdul = fits.open(args.fits)
if args.hdu is not None:
    sci = hdul[args.hdu]
else:
    sci = next((h for h in hdul if h.data is not None and h.data.ndim >= 2))
img = np.squeeze(sci.data).astype(np.float32)
ny, nx = img.shape


def y_flip(y):
    return (ny - 1) - y


sources_sorted = sorted(src_json["sources"], key=lambda s: -s["flux"])
all_xy = np.array([(s["x"], y_flip(s["y"])) for s in sources_sorted])
top_xy = all_xy[:args.top_n]
rest_xy = all_xy[args.top_n:]

quad = trace["quad"]
quad_field_idx = quad["field_indices"]
# Quad stars in trace order (matches solver's reordered_orig / index_indices).
quad_xy = np.array([(c["px"], y_flip(c["py"])) for c in quad["catalog"]])

# --- identify A, B (longest pair), then C, D --------------------------------
ij_pairs = list(itertools.combinations(range(4), 2))
dists = [np.hypot(quad_xy[i, 0] - quad_xy[j, 0],
                  quad_xy[i, 1] - quad_xy[j, 1]) for i, j in ij_pairs]
ai, bi = ij_pairs[int(np.argmax(dists))]
others = [k for k in range(4) if k not in (ai, bi)]
# astrometry.net uses cx ≤ dx (smaller x is C). Use the image-pixel x
# as the disambiguation since we haven't normalised to code space.
others_sorted = sorted(others, key=lambda k: quad_xy[k, 0])
ci, di = others_sorted

A = quad_xy[ai]; B = quad_xy[bi]; C = quad_xy[ci]; D = quad_xy[di]
quad_field_id = {"A": quad_field_idx[ai], "B": quad_field_idx[bi],
                 "C": quad_field_idx[ci], "D": quad_field_idx[di]}

verif = trace["verification"]
matched_field_idx = {m["field_idx"] for m in verif["matches"]}
matched_xy_cat = np.array([(m["px"], y_flip(m["py"])) for m in verif["matches"]]) \
    if verif["matches"] else np.zeros((0, 2))

# --- figure ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 11))
finite = img[np.isfinite(img)]
lo, hi = np.percentile(finite, [10, 99.7])
disp = np.arcsinh(np.clip(img, lo, hi) - lo)
ax.imshow(disp, origin="lower", cmap="gray_r", interpolation="nearest")

# Non-top-50 detections — cyan ×s underneath everything else.
if rest_xy.size:
    ax.scatter(rest_xy[:, 0], rest_xy[:, 1], s=18, marker="x",
               color=COL_NONTOP, linewidths=0.9, alpha=0.7,
               label=f"detected, not used for solve (ranks {args.top_n+1}-{len(all_xy)})")

# Top-50 verification hits / misses (excluding the 4 quad vertices
# themselves — those get their own bigger A/B/C/D markers below).
hit_mask = np.array([i in matched_field_idx for i in range(min(args.top_n, len(top_xy)))])
quad_field_set = set(quad_field_idx)
miss_xy = []
hit_xy_top = []
for i in range(len(hit_mask)):
    if i in quad_field_set:
        continue  # drawn as A/B/C/D below
    (hit_xy_top if hit_mask[i] else miss_xy).append(top_xy[i])
miss_xy = np.array(miss_xy) if miss_xy else np.zeros((0, 2))
hit_xy_top = np.array(hit_xy_top) if hit_xy_top else np.zeros((0, 2))

ax.scatter(miss_xy[:, 0], miss_xy[:, 1], s=40, marker="x",
           color=COL_MISS, linewidths=1.0, alpha=0.85,
           label=f"top-{args.top_n} verification miss ({len(miss_xy)})")
ax.scatter(hit_xy_top[:, 0], hit_xy_top[:, 1], s=120, marker="o",
           facecolors="none", edgecolors=COL_HIT, linewidths=2.0,
           label=f"top-{args.top_n} verification hit ({len(hit_xy_top)})")
# Catalog projection for each verification match — open circles
# (smaller than the detection hit circle) so the underlying source is
# never covered. Drawn as data-space Circle patches for consistent
# pixel-radius sizing.
hit_r = max(20.0, np.hypot(nx, ny) * 0.005)
for mxy in matched_xy_cat:
    ax.add_patch(Circle(mxy, hit_r, fill=False, edgecolor=COL_HIT,
                        lw=1.4, alpha=0.9, zorder=9))

# --- AB-diameter validity circle (dashed grey) -----------------------------
# Drawn as two arcs that stop where the circle crosses the A and B vertex
# marker disks, so the dashed line never enters either vertex circle.
ab_mid = 0.5 * (A + B)
ab_radius = 0.5 * np.linalg.norm(B - A)
# We'll need vertex_r below; compute the angular half-gap each vertex
# carves out of the AB-diameter circle. The vertex disk has radius
# vertex_r centred on A (or B), and the AB-circle passes through A; the
# intersection half-angle (as seen from ab_mid) is arccos(1 - r²/(2R²)).
def _ab_circle_arcs(M, R, A_pt, B_pt, vr):
    ang_A = np.arctan2(A_pt[1] - M[1], A_pt[0] - M[0])
    if vr >= R * np.sqrt(2):  # gap exceeds 180°, nothing to draw
        return []
    delta = np.arccos(1.0 - (vr * vr) / (2.0 * R * R))
    # Arc 1: from A+δ to B-δ (= A+π-δ).
    # Arc 2: from B+δ to A+2π-δ.
    arc1 = (ang_A + delta, ang_A + np.pi - delta)
    arc2 = (ang_A + np.pi + delta, ang_A + 2.0 * np.pi - delta)
    return [arc1, arc2]

# vertex_r is defined just below; compute it inline here to avoid a forward ref.
_vertex_r_for_arc = max(35.0, np.hypot(nx, ny) * 0.012)
for (a0, a1) in _ab_circle_arcs(ab_mid, ab_radius, A, B, _vertex_r_for_arc):
    th = np.linspace(a0, a1, 256)
    xs = ab_mid[0] + ab_radius * np.cos(th)
    ys = ab_mid[1] + ab_radius * np.sin(th)
    ax.plot(xs, ys, color=COL_VAL_CIRCLE, lw=1.4, linestyle="--",
            alpha=0.85, zorder=8)

# --- quad vertex circles (data-space, so we can clip lines exactly) ---------
# Marker radius in image pixels — large enough to read at 4k, scaled to
# the image diagonal so it adapts to plate scale.
vertex_r = max(35.0, np.hypot(nx, ny) * 0.012)

def add_vertex(p, label, colour):
    ax.add_patch(Circle(p, vertex_r, fill=False, edgecolor=colour, lw=2.6,
                        zorder=11))
    # Annotation just outside the circle so it never covers the source.
    ang = np.deg2rad({"A": 45, "B": 135, "C": 225, "D": 315}[label])
    off = np.array([np.cos(ang), np.sin(ang)]) * vertex_r * 1.5
    ax.annotate(f"{label}\nf{quad_field_id[label]}",
                xy=p, xytext=p + off, textcoords="data",
                ha="center", va="center", color=colour,
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=colour, lw=0.9, alpha=0.95),
                zorder=12)

add_vertex(A, "A", COL_A)
add_vertex(B, "B", COL_B)
add_vertex(C, "C", COL_C)
add_vertex(D, "D", COL_D)


def clipped_segment(p1, p2, r1, r2):
    """Endpoints of the segment p1→p2 trimmed so it does not enter the
    disks of radius r1/r2 around p1/p2."""
    u = p2 - p1
    L = np.linalg.norm(u)
    if L <= r1 + r2:
        return None  # circles touch or overlap; no segment to draw
    u_hat = u / L
    return (p1 + r1 * u_hat, p2 - r2 * u_hat)


def draw_edge(p1, p2, colour, *, lw, alpha):
    seg = clipped_segment(p1, p2, vertex_r, vertex_r)
    if seg is None:
        return
    (q1, q2) = seg
    ax.plot([q1[0], q2[0]], [q1[1], q2[1]], color=colour, lw=lw, alpha=alpha,
            zorder=10)

# AB backbone — yellow.
draw_edge(A, B, COL_AB, lw=2.4, alpha=0.95)
# AC, AD reference lines — semi-saturated cyan.
draw_edge(A, C, COL_AC, lw=1.6, alpha=0.55)
draw_edge(A, D, COL_AC, lw=1.6, alpha=0.55)

inst = src_json.get("hst", {}).get("instrument_name", "?")
target = src_json.get("hst", {}).get("target_name", "")
err = ((trace["solved"]["ra_deg"]-trace["truth"]["ra_deg"])**2 +
       (trace["solved"]["dec_deg"]-trace["truth"]["dec_deg"])**2)**0.5 * 3600

ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_aspect("equal")
ax.set_title(
    f"{args.case} — {target} ({inst})  {nx}×{ny} px @ "
    f"{src_json['plate_scale_arcsec']:.4f}\"/px\n"
    f"quad band {quad['band_idx']} | log_odds={verif['log_odds']:.1f} | "
    f"n_matched={verif['n_matched']} | n_distractor={verif['n_distractor']} | "
    f"WCS error {err:.2f}\""
)
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(args.out, dpi=140, bbox_inches="tight")
print(f"wrote {args.out}")

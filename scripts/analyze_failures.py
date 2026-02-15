#!/usr/bin/env python3
"""Analyze failed plate solves from batch-solve diagnostics."""

import json
import sys
import math
from pathlib import Path


def equatorial_to_galactic(ra_deg, dec_deg):
    """Convert equatorial (J2000) to galactic coordinates."""
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

    diag_files = sorted(failures_dir.glob("*.diag.json"))
    if not diag_files:
        print("No diagnostic files found in", failures_dir)
        return

    print(f"Found {len(diag_files)} failure diagnostics\n")

    failures = []
    for df in diag_files:
        diag = json.load(open(df))
        stem = df.stem.replace(".diag", "")
        src_file = test_cases_dir / f"{stem}.json"
        src = json.load(open(src_file)) if src_file.exists() else {}

        ra = src.get("ra_deg", 0)
        dec = src.get("dec_deg", 0)
        l, b = equatorial_to_galactic(ra, dec)
        n_sources = len(src.get("sources", []))
        fluxes = sorted([s["flux"] for s in src.get("sources", [])], reverse=True)

        best_rej = diag.get("best_rejected") or {}
        best_lo = best_rej.get("log_odds", 0)
        best_nm = best_rej.get("n_matched", 0)
        n_verified = diag.get("n_verified", 0)
        elapsed = diag.get("elapsed_s", 0)
        timed_out = diag.get("timed_out", False)

        wcs = diag.get("best_rejected_wcs")
        wcs_ra_deg = wcs_dec_deg = wcs_scale = None
        if wcs:
            wcs_ra_deg = math.degrees(wcs["crval"][0])
            wcs_dec_deg = math.degrees(wcs["crval"][1])
            cd = wcs["cd"]
            wcs_scale = math.sqrt(abs(cd[0][0]*cd[1][1] - cd[0][1]*cd[1][0])) * 3600 * 180 / math.pi

        failures.append({
            "file": stem, "ra": ra, "dec": dec, "l": l, "b": b,
            "n_sources": n_sources, "fluxes": fluxes,
            "best_lo": best_lo, "best_nm": best_nm,
            "n_verified": n_verified, "elapsed": elapsed, "timed_out": timed_out,
            "wcs_ra": wcs_ra_deg, "wcs_dec": wcs_dec_deg, "wcs_scale": wcs_scale,
        })

    # Table
    print(f"{'File':>8}  {'RA':>7} {'Dec':>7}  {'l':>6} {'b':>6}  {'Src':>3}  {'BestLO':>6} {'Nm':>2}  {'Verified':>8}  {'Time':>5}  {'WCS_scale':>9}")
    print("-" * 105)
    for f in failures:
        wcs_s = f"{f['wcs_scale']:.4f}" if f["wcs_scale"] else "   N/A"
        print(f"{f['file']:>8}  {f['ra']:7.2f} {f['dec']:+7.2f}  {f['l']:6.1f} {f['b']:+6.1f}  {f['n_sources']:3d}  {f['best_lo']:6.1f} {f['best_nm']:2d}  {f['n_verified']:8d}  {f['elapsed']:5.1f}  {wcs_s:>9}")

    # Summary
    bs = [f["b"] for f in failures]
    srcs = [f["n_sources"] for f in failures]
    los = [f["best_lo"] for f in failures]
    ras = [f["ra"] for f in failures]
    decs = [f["dec"] for f in failures]

    print(f"\n--- Summary ---")
    print(f"RA range:  {min(ras):.1f} - {max(ras):.1f} deg")
    print(f"Dec range: {min(decs):.1f} - {max(decs):.1f} deg")
    abs_bs = sorted(abs(b) for b in bs)
    print(f"|b| stats: min={abs_bs[0]:.1f}, median={abs_bs[len(abs_bs)//2]:.1f}, max={abs_bs[-1]:.1f}")
    srcs_s = sorted(srcs)
    print(f"Sources:   min={srcs_s[0]}, median={srcs_s[len(srcs_s)//2]}, max={srcs_s[-1]}")
    los_s = sorted(los)
    print(f"Best LO:   min={los_s[0]:.1f}, median={los_s[len(los_s)//2]:.1f}, max={los_s[-1]:.1f}")

    low_lat = sum(1 for b in bs if abs(b) < 15)
    mid_lat = sum(1 for b in bs if 15 <= abs(b) < 30)
    high_lat = sum(1 for b in bs if abs(b) >= 30)
    n = len(failures)
    print(f"\nGalactic latitude distribution:")
    print(f"  |b| < 15° (plane): {low_lat}/{n} ({100*low_lat/n:.0f}%)")
    print(f"  15° <= |b| < 30°:  {mid_lat}/{n} ({100*mid_lat/n:.0f}%)")
    print(f"  |b| >= 30°:        {high_lat}/{n} ({100*high_lat/n:.0f}%)")

    # Check if best rejected WCS is near truth
    print(f"\n--- Best rejected WCS vs truth ---")
    n_near = 0
    for f in failures:
        if f["wcs_ra"] is not None:
            dra = abs(f["wcs_ra"] - f["ra"])
            if dra > 180:
                dra = 360 - dra
            dra *= math.cos(math.radians(f["dec"]))
            ddec = abs(f["wcs_dec"] - f["dec"])
            sep = math.sqrt(dra**2 + ddec**2)
            near = "NEAR" if sep < 1.0 else "FAR"
            if sep < 1.0:
                n_near += 1
            known_scale = 0.1296
            scale_ok = "OK" if f["wcs_scale"] and abs(f["wcs_scale"] - known_scale) / known_scale < 0.1 else "BAD"
            print(f"  {f['file']}: sep={sep:6.2f}° scale={f['wcs_scale'] or 0:.4f}\"/px  [{near}] [{scale_ok}]")
    print(f"\n  {n_near}/{len(failures)} best rejections are near the true location")

    # Write JSON for plotting
    out = [{k: v for k, v in f.items() if k != "fluxes"} for f in failures]
    out_path = failures_dir / "analysis.json"
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

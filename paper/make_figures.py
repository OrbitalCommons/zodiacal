"""Single entry point that regenerates every figure in the paper.

Usage:
    python make_figures.py                    # regenerate all figures
    python make_figures.py --only solve_cdf   # regenerate one
    python make_figures.py --list             # list known figures

Inputs (paths are relative to the repo root by default):
    scratch/4band_results.log     - 4-band batch-solve log
    scratch/12band_results.log    - 12-band batch-solve log
    scratch/12band_hires.log      - 12-band rerun with 8-decimal RA/Dec
    test_cases/*.json             - 1000 synthetic field JSONs (truth + sources)

Outputs are written to paper/figures/ as <name>.pdf and <name>.png; that
directory is gitignored. .tex files should reference figures/<name>.pdf.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = PAPER_DIR / "figures"

SOLVED_RE = re.compile(
    r"\]\s+(\S+)\.json:\s+SOLVED in (\d+\.\d+)s.*?RA=([\-\d\.]+)\s+Dec=([+\-\d\.]+)"
)
FAILED_RE = re.compile(r"\]\s+(\S+)\.json:\s+FAILED in (\d+\.\d+)s")


def parse_log(path):
    """Return (solved, failed) where each is a list of dicts."""
    solved, failed = [], []
    for line in open(path):
        m = SOLVED_RE.search(line)
        if m:
            solved.append({
                "name": m.group(1),
                "elapsed": float(m.group(2)),
                "ra": float(m.group(3)),
                "dec": float(m.group(4)),
            })
            continue
        m = FAILED_RE.search(line)
        if m:
            failed.append({"name": m.group(1), "elapsed": float(m.group(2))})
    return solved, failed


def angular_sep_arcsec(ra1, dec1, ra2, dec2):
    r1, d1 = np.radians(ra1), np.radians(dec1)
    r2, d2 = np.radians(ra2), np.radians(dec2)
    sd = np.sin((d2 - d1) / 2.0) ** 2
    sr = np.sin((r2 - r1) / 2.0) ** 2
    a = sd + np.cos(d1) * np.cos(d2) * sr
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))) * 3600.0


def residuals_arcsec(solved, cases_dir):
    """Per-field (ΔRA·cos(δ), Δδ) in arcsec for solved fields with truth."""
    dra_cos, ddec = [], []
    for s in solved:
        p = cases_dir / f"{s['name']}.json"
        if not p.exists():
            continue
        tc = json.load(open(p))
        cd = np.cos(np.radians(tc["dec_deg"]))
        dra_cos.append((s["ra"] - tc["ra_deg"]) * cd * 3600.0)
        ddec.append((s["dec"] - tc["dec_deg"]) * 3600.0)
    return np.array(dra_cos), np.array(ddec)


def save(fig, name, out_dir):
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {name}.pdf, {name}.png")


def fig_solve_time_hist(name, log_path, title, out_dir):
    """Solve-time histogram (log-x) with timeout overlay."""
    solved, failed = parse_log(log_path)
    s = np.array([x["elapsed"] for x in solved])
    f = np.array([x["elapsed"] for x in failed])
    upper = max(s.max() if len(s) else 80, f.max() if len(f) else 80, 80) * 1.2
    bins = np.logspace(-1.5, np.log10(upper), 35)

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    if len(s):
        ax.hist(s, bins=bins, color="#4878CF", alpha=0.85,
                edgecolor="white", linewidth=0.3, label="Solved")
    if len(f):
        ax.hist(f, bins=bins, color="#D65F5F", alpha=0.85,
                edgecolor="white", linewidth=0.3, label="Timeout")
    if len(s):
        ax.axvline(np.median(s), color="black", ls="--", lw=0.8,
                   label=f"Median {np.median(s):.2f}s")
    ax.set_xscale("log")
    ax.set_xlim(0.02, upper)
    ax.set_xlabel("Solve time (s)")
    ax.set_ylabel("Number of fields")
    ax.set_title(f"{title}: {len(s)}/{len(s)+len(f)} solved")
    ax.legend(fontsize=7, loc="upper right")
    save(fig, name, out_dir)


def fig_solve_cdf(name, log_paths_with_labels, out_dir):
    """Cumulative solve-time distribution across multiple runs."""
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    colors = ["#D65F5F", "#4878CF", "#6ACC65"]
    for (path, label), color in zip(log_paths_with_labels, colors):
        solved, failed = parse_log(path)
        if not solved:
            continue
        s = np.sort([x["elapsed"] for x in solved])
        total = len(solved) + len(failed)
        cdf = np.arange(1, len(s) + 1) / total
        ax.plot(s, cdf * 100, lw=1.5, color=color,
                label=f"{label} ({len(solved)}/{total})")
    ax.set_xscale("log")
    ax.set_xlim(0.02, 70)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Solve time (s)")
    ax.set_ylabel("Cumulative fields solved (%)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    save(fig, name, out_dir)


def fig_residuals_raw(name, log_path, cases_dir, out_dir):
    """Raw pointing residual magnitude histogram."""
    solved, _ = parse_log(log_path)
    dra_cos, ddec = residuals_arcsec(solved, cases_dir)
    mag = np.hypot(dra_cos, ddec)

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    pos = mag[mag > 0]
    if len(pos) and pos.max() / pos.min() > 30:
        bins = np.logspace(np.log10(pos.min() / 2), np.log10(mag.max() * 1.2), 40)
        ax.set_xscale("log")
    else:
        bins = np.linspace(max(0, mag.min() - 0.01), mag.max() + 0.01, 40)
    ax.hist(mag, bins=bins, color="#4878CF", alpha=0.85,
            edgecolor="white", linewidth=0.3)
    ax.axvline(np.median(mag), color="black", ls="--", lw=0.8,
               label=f"Median {np.median(mag):.3f}\"")
    ax.set_xlabel("Pointing residual (arcsec)")
    ax.set_ylabel("Number of fields")
    ax.set_title(f"Raw pointing residual, n={len(mag)}")
    ax.legend(fontsize=8, loc="upper right")
    save(fig, name, out_dir)


def fig_residuals_corrected(name, log_path, cases_dir, out_dir):
    """Two-panel: corrected residual histogram + direction scatter (offset removed).

    Subtracting the median (ΔRA·cos δ, Δδ) takes out the systematic
    ~1-pixel Dec offset documented in zodiacal#67 and exposes the
    underlying solver precision.
    """
    solved, _ = parse_log(log_path)
    dra_cos, ddec = residuals_arcsec(solved, cases_dir)
    ra_off, dec_off = np.median(dra_cos), np.median(ddec)
    dra_c = dra_cos - ra_off
    ddec_c = ddec - dec_off
    mag = np.hypot(dra_c, ddec_c)
    mag_mas = mag * 1000

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.2))

    p99 = np.percentile(mag_mas, 99)
    bins = np.linspace(0, p99 * 1.1, 40)
    nover = int(np.sum(mag_mas > p99 * 1.1))
    ax1.hist(mag_mas, bins=bins, color="#4878CF", alpha=0.85,
             edgecolor="white", linewidth=0.3)
    ax1.axvline(np.median(mag_mas), color="black", ls="--", lw=0.8,
                label=f"Median {np.median(mag_mas):.1f} mas")
    ax1.set_xlabel("Pointing residual after offset removal (mas)")
    ax1.set_ylabel("Number of fields")
    title_extra = f"  ({nover} > {p99*1.1:.0f} mas off-plot)" if nover else ""
    ax1.set_title(f"Residual magnitude, n={len(mag)}{title_extra}")
    ax1.legend(fontsize=8)

    ax2.scatter(dra_c * 1000, ddec_c * 1000, s=6, alpha=0.5, color="#4878CF")
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.axvline(0, color="gray", lw=0.5)
    ax2.set_xlabel(r"$\Delta\alpha\cos\delta - $median (mas)")
    ax2.set_ylabel(r"$\Delta\delta - $median (mas)")
    ax2.set_aspect("equal")
    lim = np.percentile(
        np.maximum(np.abs(dra_c * 1000), np.abs(ddec_c * 1000)), 98
    ) * 1.4
    ax2.set_xlim(-lim, lim)
    ax2.set_ylim(-lim, lim)
    ax2.set_title("Direction (offset removed)")

    save(fig, name, out_dir)


def fig_residual_scatter(name, log_path, cases_dir, out_dir):
    """Diagnostic: raw residual direction in (ΔRA·cos δ, Δδ) space.

    Shows the systematic ~1-pixel offset is along Dec (not isotropic).
    """
    solved, _ = parse_log(log_path)
    dra_cos, ddec = residuals_arcsec(solved, cases_dir)

    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.scatter(dra_cos, ddec, s=8, alpha=0.6, color="#4878CF")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    theta = np.linspace(0, 2 * np.pi, 200)
    px = 0.1296  # IMX455 plate scale (arcsec/px)
    ax.plot(px * np.cos(theta), px * np.sin(theta), "--", color="#D65F5F",
            lw=0.8, label=f'1-px ({px}")')
    ax.set_xlabel(r"$\Delta\alpha\cos\delta$ (arcsec)")
    ax.set_ylabel(r"$\Delta\delta$ (arcsec)")
    ax.set_aspect("equal")
    lim = max(0.18, np.abs(dra_cos).max() * 1.1, np.abs(ddec).max() * 1.1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.legend(fontsize=8)
    ax.set_title(f"Residual direction (raw), n={len(dra_cos)}")
    save(fig, name, out_dir)


# Registry: figure name -> (callable, *args from REPO_ROOT)
def figures(scratch_dir, cases_dir, out_dir):
    return {
        "hist_4band": lambda: fig_solve_time_hist(
            "hist_4band", scratch_dir / "4band_results.log",
            r"4 bands ($\alpha=3$)", out_dir,
        ),
        "hist_12band": lambda: fig_solve_time_hist(
            "hist_12band", scratch_dir / "12band_results.log",
            r"12 bands ($\alpha=\sqrt{2}$)", out_dir,
        ),
        "hist_12band_hires": lambda: fig_solve_time_hist(
            "hist_12band_hires", scratch_dir / "12band_hires.log",
            r"12 bands ($\alpha=\sqrt{2}$), rerun", out_dir,
        ),
        "solve_cdf": lambda: fig_solve_cdf(
            "solve_cdf",
            [
                (scratch_dir / "4band_results.log", "4 bands"),
                (scratch_dir / "12band_results.log", "12 bands"),
            ],
            out_dir,
        ),
        "hist_residuals": lambda: fig_residuals_raw(
            "hist_residuals", scratch_dir / "12band_hires.log",
            cases_dir, out_dir,
        ),
        "hist_residuals_corrected": lambda: fig_residuals_corrected(
            "hist_residuals_corrected", scratch_dir / "12band_hires.log",
            cases_dir, out_dir,
        ),
        "residual_scatter": lambda: fig_residual_scatter(
            "residual_scatter", scratch_dir / "12band_hires.log",
            cases_dir, out_dir,
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch-dir", default=REPO_ROOT / "scratch", type=Path)
    ap.add_argument("--cases-dir", default=REPO_ROOT / "test_cases", type=Path)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    ap.add_argument("--only", default=None,
                    help="Comma-separated subset of figure names to build.")
    ap.add_argument("--list", action="store_true",
                    help="List known figures and exit.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    figs = figures(args.scratch_dir, args.cases_dir, args.out_dir)

    if args.list:
        for name in figs:
            print(name)
        return

    targets = list(figs.keys())
    if args.only:
        targets = [n.strip() for n in args.only.split(",")]
        unknown = [n for n in targets if n not in figs]
        if unknown:
            print(f"Unknown figure(s): {', '.join(unknown)}", file=sys.stderr)
            print(f"Known: {', '.join(figs.keys())}", file=sys.stderr)
            sys.exit(2)

    for name in targets:
        try:
            figs[name]()
        except FileNotFoundError as e:
            print(f"skip {name}: missing input ({e.filename})", file=sys.stderr)


if __name__ == "__main__":
    main()

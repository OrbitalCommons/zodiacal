#!/usr/bin/env python3
"""Build-and-benchmark loop for index strategies against a test-case set.

Subcommands:

  prepare  Pick a stratified dev sample of solvable cases (using a
           gaia_census.py CSV), and build a reduced excerpt directory that
           symlinks only the HEALPix-5 excerpt cells within the bench radius
           of those cases, so each candidate bundle builds in minutes.

  run      Build one bundle from the reduced excerpt with the given
           `build-from-excerpt-series` arguments, run `bench-bundle` on the
           dev sample, score it, and append a row to results.md.

  score    Re-score an existing run directory.

A case counts as *correct* if it solved and the solved centre is within
--correct-arcsec of the header pointing; a solve farther away is *wrong*.

Example:
    scripts/index_strategy_loop.py prepare --loop-dir ~/scratch/index-loop \
        --cases ~/scratch/hst-reextract/hubble-f606w \
        --census ~/scratch/hst-reextract/_gaia_census.csv \
        --excerpt ~/.cache/starfield/gaia-excerpts/dr3-mag20-bycell
    scripts/index_strategy_loop.py run --loop-dir ~/scratch/index-loop \
        --tools ~/scratch/index-loop/bin/zodiacal-tools-76081ac \
        --name d9-b8-150-q200 -- --cell-depth 9 --bands 8 --scale-lower 8 \
        --scale-upper 150 --quads-per-cell 200 --max-stars-per-cell 5000 --mag-limit 20
"""

import argparse
import csv
import json
import math
import os
import random
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path



def prepare(args):
    loop = args.loop_dir
    dev = loop / "dev-cases"
    sub = loop / "excerpt-subset"
    for d in (dev, sub):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    rows = list(csv.DictReader(open(args.census)))
    rng = random.Random(args.seed)
    by_inst = defaultdict(list)
    for r in rows:
        if int(r["n_sources"]) >= args.min_sources and int(r["in_g20"]) >= args.min_gaia:
            by_inst[r["instrument"]].append(r)
    picked = []
    for inst, cand in sorted(by_inst.items()):
        rng.shuffle(cand)
        take = cand[: args.per_inst]
        picked += take
        print(f"{inst:10} solvable={len(cand):6} picked={len(take)}")
    for r in picked:
        (dev / r["case"]).symlink_to(args.cases / r["case"])

    # Cells overlapping the bench region (bench radius + field radius) of each case;
    # cone_search already returns every pixel the disc touches.
    from astropy import units as u
    from astropy.coordinates import Latitude, Longitude
    from astropy_healpix import HEALPix

    hp = HEALPix(nside=32, order="nested")
    cells = set()
    for r in picked:
        d = json.loads((args.cases / r["case"]).read_text())
        rad = args.radius_deg + float(r["r_out_arcsec"]) / 3600.0
        cells.update(int(c) for c in hp.cone_search_lonlat(
            Longitude(d["ra_deg"] * u.deg), Latitude(d["dec_deg"] * u.deg), radius=rad * u.deg))
    n = 0
    for c in sorted(cells):
        src = args.excerpt / f"shard_{c:05d}.csv.gz"
        if src.exists():
            (sub / src.name).symlink_to(src)
            n += 1
    manifest = args.excerpt / ".gaia-excerpt-manifest.json"
    if manifest.exists():
        # The bundle builder reads the excerpt manifest for depth/release metadata.
        (sub / manifest.name).symlink_to(manifest)
    meta = {"n_cases": len(picked), "n_cells": n, "radius_deg": args.radius_deg,
            "min_sources": args.min_sources, "min_gaia_in_g20": args.min_gaia, "seed": args.seed,
            "per_inst": args.per_inst}
    (loop / "prepare.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


def run(args, build_args):
    loop = args.loop_dir
    rd = loop / "runs" / args.name
    rd.mkdir(parents=True, exist_ok=True)
    prep = json.loads((loop / "prepare.json").read_text())
    bundle_prefix = rd / "bundle"
    bundle = Path(str(bundle_prefix) + ".zdcl.bundle")
    t0 = time.time()
    if not bundle.exists():
        cmd = [str(args.tools), "build-from-excerpt-series", "--excerpt-dir", str(loop / "excerpt-subset"),
               "--work-dir", str(rd / "work"), "--output-prefix", str(bundle_prefix),
               "--prune-work-dir", "--experiment", args.name, *build_args]
        (rd / "build.cmd").write_text(" ".join(cmd) + "\n")
        with open(rd / "build.log", "w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    build_s = time.time() - t0
    t1 = time.time()
    bench = [str(args.tools), "bench-bundle", "--bundle-path", str(bundle), "--test-cases-dir",
             str(loop / "dev-cases"), "--radius-deg", str(prep["radius_deg"]), "--timeout-secs",
             str(args.timeout_secs)]
    if not args.no_scale_hint:
        bench.append("--scale-hint")
    bench_extra = shlex.split(args.bench_extra)
    bench += bench_extra
    (rd / "bench.cmd").write_text(" ".join(bench) + "\n")
    with open(rd / "bench.csv", "w") as out, open(rd / "bench.log", "w") as log:
        subprocess.run(bench, stdout=out, stderr=log, check=True)
    bench_s = time.time() - t1
    size = sum(f.stat().st_size for f in bundle.rglob("*") if f.is_file())
    meta = {"name": args.name, "build_args": build_args, "bench_extra": bench_extra,
            "build_s": round(build_s, 1), "bench_s": round(bench_s, 1), "bundle_bytes": size,
            "note": args.note}
    (rd / "run.json").write_text(json.dumps(meta, indent=2))
    score(args.loop_dir, rd, args.cases, args.correct_arcsec)
    if not args.keep_bundle:
        # Bundles are large and cheap to rebuild from build.cmd; keep results only.
        shutil.rmtree(bundle, ignore_errors=True)


def epoch(case):
    h = case.get("hst", {})
    a, b = h.get("t_min_mjd"), h.get("t_max_mjd")
    if a is None:
        return None
    return 2000.0 + (0.5 * (a + (b or a)) - 51544.5) / 365.25


def score(loop, rd, cases_dir, correct_arcsec):
    meta = json.loads((rd / "run.json").read_text())
    rows = list(csv.DictReader(open(rd / "bench.csv")))
    agg = defaultdict(lambda: {"n": 0, "ok": 0, "wrong": 0, "ms": []})
    for r in rows:
        case = json.loads((cases_dir / f"{r['case']}.json").read_text())
        inst = case.get("hst", {}).get("instrument_name", "?")
        ep = epoch(case)
        dt = "?" if ep is None else ("<5y" if abs(ep - 2016) < 5 else ("5-15y" if abs(ep - 2016) < 15 else ">15y"))
        solved = r["solved"] == "1"
        err = float(r["error_arcsec"]) if r["error_arcsec"] else math.inf
        for key in (("all",), ("inst", inst), ("dt", dt)):
            a = agg[key]
            a["n"] += 1
            if solved and err <= correct_arcsec:
                a["ok"] += 1
            elif solved:
                a["wrong"] += 1
            a["ms"].append(float(r["solve_ms"]))
    out = {}
    for key, a in agg.items():
        out["/".join(key)] = {"n": a["n"], "correct": a["ok"], "wrong": a["wrong"],
                              "rate": round(a["ok"] / a["n"], 4) if a["n"] else 0,
                              "median_ms": round(statistics.median(a["ms"]), 1) if a["ms"] else None}
    (rd / "score.json").write_text(json.dumps(out, indent=2))

    def pct(k):
        s = out.get(k)
        return f"{100 * s['rate']:.1f}%" if s else "–"

    res = loop / "results.md"
    if not res.exists() or res.stat().st_size == 0:
        res.write_text("| run | all | wrong | ACS/WFC | WFC3/UVIS | WFPC2/PC | ACS/HRC | <5y | 5-15y | >15y "
                       "| median ms | bundle GB | build min | args |\n"
                       "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    a = out["all"]
    line = (f"| {meta['name']} | {pct('all')} ({a['correct']}/{a['n']}) | {a['wrong']} | {pct('inst/ACS/WFC')} | "
            f"{pct('inst/WFC3/UVIS')} | {pct('inst/WFPC2/PC')} | {pct('inst/ACS/HRC')} | {pct('dt/<5y')} | "
            f"{pct('dt/5-15y')} | {pct('dt/>15y')} | {a['median_ms']} | {meta['bundle_bytes'] / 1e9:.2f} | "
            f"{meta['build_s'] / 60:.1f} | `{' '.join(meta['build_args'] + meta['bench_extra'])}` |\n")
    with open(res, "a") as f:
        f.write(line)
    print(line, end="")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--loop-dir", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--census", type=Path, required=True)
    p.add_argument("--excerpt", type=Path, required=True)
    p.add_argument("--radius-deg", type=float, default=1.0)
    p.add_argument("--min-sources", type=int, default=10)
    p.add_argument("--min-gaia", type=int, default=5, help="min Gaia G<=20 stars in the inscribed circle")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-inst", type=int, default=200, help="cases sampled per instrument")
    r = sub.add_parser("run")
    r.add_argument("--loop-dir", type=Path, required=True)
    r.add_argument("--cases", type=Path, required=True)
    r.add_argument("--tools", type=Path, required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--note", default="")
    r.add_argument("--timeout-secs", type=int, default=20)
    r.add_argument("--correct-arcsec", type=float, default=2.0)
    r.add_argument("--no-scale-hint", action="store_true")
    r.add_argument("--bench-extra", default="",
                   help='extra bench-bundle flags as one string, e.g. --bench-extra="--min-matches 8"')
    r.add_argument("--keep-bundle", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--loop-dir", type=Path, required=True)
    s.add_argument("--cases", type=Path, required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--correct-arcsec", type=float, default=2.0)
    argv = sys.argv[1:]
    build_args = []
    if "--" in argv:
        i = argv.index("--")
        argv, build_args = argv[:i], argv[i + 1:]
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "run":
        run(args, build_args)
    else:
        score(args.loop_dir, args.loop_dir / "runs" / args.name, args.cases, args.correct_arcsec)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate test case JSONs from simulated star field images.

Runs meter-sim's sensor_shootout to render synthetic star fields, then uses
zodiacal's extract command to pull detected sources into JSON files with
known RA/Dec metadata for authoritative accuracy testing.

Generates images in batches to limit disk usage (~60GB of FITS per batch).

Requirements:
    - meter-sim built: cargo build --release (in meter-sim repo)
    - zodiacal built:  cargo build --release --features fits (in zodiacal repo)
    - A starfield binary catalog (e.g. to_mag_19.bin from star-cats)

Usage:
    python3 scripts/generate_test_cases.py \\
        --catalog /path/to/star-cats/to_mag_19.bin \\
        --meter-sim /path/to/meter-sim \\
        --count 1000 \\
        --batch-size 100
"""

import argparse
import csv
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path


PIXEL_PITCH_UM = 3.76  # IMX455 pixel pitch in microns


def find_timestamped_dir(base_pattern):
    """Find the most recent timestamped output directory matching a glob pattern."""
    dirs = sorted(glob.glob(f"{base_pattern}_*"), key=os.path.getmtime, reverse=True)
    return Path(dirs[0]) if dirs else None


def compute_plate_scale(pixel_pitch_um, focal_length_m):
    """Compute plate scale in arcsec/pixel from pixel pitch and focal length."""
    return pixel_pitch_um / focal_length_m * 206265.0 / 1e6


def generate_batch(
    batch_num,
    batch_size,
    offset,
    shootout_bin,
    extract_bin,
    catalog,
    scratch_dir,
    output_dir,
    threads,
):
    """Generate one batch of test cases. Returns number of JSONs written."""
    batch_dir = scratch_dir / f"batch_{batch_num}"
    csv_file = scratch_dir / f"batch_{batch_num}.csv"

    # Clean previous artifacts
    for d in glob.glob(f"{batch_dir}_*"):
        shutil.rmtree(d, ignore_errors=True)
    csv_file.unlink(missing_ok=True)

    # Run sensor_shootout
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(threads)
    result = subprocess.run(
        [
            str(shootout_bin),
            "--experiments", str(batch_size),
            "--exposure-range-ms", "10000",
            "--seed", str(batch_num),
            "--catalog", str(catalog),
            "--output-dir", str(batch_dir),
            "--output-csv", str(csv_file),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ERROR: sensor_shootout failed:\n{result.stderr[-500:]}", file=sys.stderr)
        return 0

    # Find timestamped output dir
    fits_dir = find_timestamped_dir(str(batch_dir))
    if not fits_dir:
        print(f"  ERROR: No output directory found for batch {batch_num}", file=sys.stderr)
        return 0

    # Parse CSV and extract each experiment
    n_written = 0
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            exp_num = int(row["experiment_num"])
            ra = row["ra"]
            dec = row["dec"]
            focal_length_m = float(row["focal_length_m"])

            fits_name = f"{exp_num:04d}_IMX455_10000ms_data_raw.fits"
            fits_file = fits_dir / fits_name
            if not fits_file.exists():
                print(f"  WARN: Missing {fits_file}", file=sys.stderr)
                continue

            plate_scale = compute_plate_scale(PIXEL_PITCH_UM, focal_length_m)
            global_num = offset + exp_num
            json_file = output_dir / f"{global_num:04d}.json"

            subprocess.run(
                [
                    str(extract_bin), "extract", str(fits_file),
                    "--output", str(json_file),
                    f"--ra={ra}",
                    f"--dec={dec}",
                    f"--plate-scale={plate_scale:.6f}",
                ],
                capture_output=True,
            )
            n_written += 1

    # Clean up FITS to save disk
    shutil.rmtree(fits_dir, ignore_errors=True)
    csv_file.unlink(missing_ok=True)
    return n_written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", required=True, type=Path, help="Path to starfield binary catalog")
    parser.add_argument("--meter-sim", type=Path, default=Path(__file__).resolve().parents[2] / "meter-sim",
                        help="Path to meter-sim repo (default: sibling of zodiacal)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "test_cases",
                        help="Output directory for JSON test cases")
    parser.add_argument("--count", type=int, default=1000, help="Total number of test cases to generate")
    parser.add_argument("--batch-size", type=int, default=100, help="Experiments per batch (limits disk usage)")
    parser.add_argument("--threads", type=int, default=32, help="Number of threads for sensor_shootout")
    parser.add_argument("--start-batch", type=int, default=0, help="Batch number to start from (for resuming)")
    args = parser.parse_args()

    zodiacal_root = Path(__file__).resolve().parents[1]
    shootout_bin = args.meter_sim / "target" / "release" / "sensor_shootout"
    extract_bin = zodiacal_root / "target" / "release" / "zodiacal"
    scratch_dir = args.meter_sim / ".batch_scratch"

    # Validate paths
    for name, path in [("catalog", args.catalog), ("sensor_shootout", shootout_bin), ("zodiacal", extract_bin)]:
        if not path.exists():
            print(f"ERROR: {name} not found at {path}", file=sys.stderr)
            sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    num_batches = (args.count + args.batch_size - 1) // args.batch_size
    total_written = 0

    for batch in range(args.start_batch, num_batches):
        offset = batch * args.batch_size
        end = min(offset + args.batch_size - 1, args.count - 1)
        print(f"=== Batch {batch + 1}/{num_batches} (seed={batch}, experiments {offset}-{end}) ===")

        n = generate_batch(
            batch_num=batch,
            batch_size=args.batch_size,
            offset=offset,
            shootout_bin=shootout_bin,
            extract_bin=extract_bin,
            catalog=args.catalog,
            scratch_dir=scratch_dir,
            output_dir=args.output_dir,
            threads=args.threads,
        )
        total_written += n
        total_on_disk = len(list(args.output_dir.glob("*.json")))
        print(f"  Batch done ({n} written). Total JSONs on disk: {total_on_disk}")

    # Clean up scratch
    shutil.rmtree(scratch_dir, ignore_errors=True)

    total_on_disk = len(list(args.output_dir.glob("*.json")))
    print(f"\n=== DONE ===")
    print(f"Generated {total_on_disk} test case JSONs in {args.output_dir}")


if __name__ == "__main__":
    main()

#!/bin/bash
set -euo pipefail

# Generate 1000 test case JSON blobs from simulated star field images.
# Runs sensor_shootout in batches, extracts sources with zodiacal, injects RA/Dec.

METER_SIM="/home/meawoppl/repos/meter-sim"
ZODIACAL="/home/meawoppl/repos/zodiacal"
CATALOG="/home/meawoppl/repos/star-cats/gaia_mag16_multi.bin"
SHOOTOUT="$METER_SIM/target/release/sensor_shootout"
EXTRACT="$ZODIACAL/target/release/zodiacal"
OUTPUT_DIR="$ZODIACAL/test_cases"
BATCH_SIZE=100
NUM_BATCHES=10
PIXEL_PITCH_UM=3.76

mkdir -p "$OUTPUT_DIR"

for batch in $(seq 0 $((NUM_BATCHES - 1))); do
    seed=$batch
    offset=$((batch * BATCH_SIZE))
    batch_dir="/tmp/zodiacal_batch_${batch}"
    csv_file="/tmp/zodiacal_batch_${batch}.csv"

    echo "=== Batch $((batch + 1))/$NUM_BATCHES (seed=$seed, experiments $offset-$((offset + BATCH_SIZE - 1))) ==="

    # Clean any previous batch artifacts
    rm -rf "${batch_dir}"_* "$csv_file"

    # Generate FITS images (single 10s exposure)
    RAYON_NUM_THREADS=32 "$SHOOTOUT" \
        --experiments "$BATCH_SIZE" \
        --exposure-range-ms "10000" \
        --seed "$seed" \
        --catalog "$CATALOG" \
        --output-dir "$batch_dir" \
        --output-csv "$csv_file" 2>&1 | tail -3

    # Find the timestamped output directory
    fits_dir=$(ls -dt "${batch_dir}_"* 2>/dev/null | head -1)
    if [ -z "$fits_dir" ]; then
        echo "ERROR: No output directory found for batch $batch"
        exit 1
    fi

    echo "  FITS dir: $fits_dir"

    # Parse CSV and extract each experiment to JSON
    # CSV header: experiment_num,trial_num,ra,dec,focal_length_m,sensor,exposure_ms,...
    tail -n +2 "$csv_file" | while IFS=',' read -r exp_num trial_num ra dec focal_length_m sensor exposure_ms rest; do
        fits_file="$fits_dir/$(printf '%04d' "$exp_num")_IMX455_10000ms_data_raw.fits"
        if [ ! -f "$fits_file" ]; then
            echo "  WARN: Missing $fits_file"
            continue
        fi

        # plate_scale = pixel_pitch_m / focal_length_m * 206265
        plate_scale=$(echo "scale=6; $PIXEL_PITCH_UM * 0.000001 / $focal_length_m * 206265" | bc)

        global_num=$((offset + exp_num))
        json_file="$OUTPUT_DIR/$(printf '%04d' "$global_num").json"

        "$EXTRACT" extract "$fits_file" \
            --output "$json_file" \
            --ra="$ra" \
            --dec="$dec" \
            --plate-scale="$plate_scale" 2>/dev/null

    done

    generated=$(ls "$OUTPUT_DIR"/*.json 2>/dev/null | wc -l)
    echo "  Batch done. Total JSONs: $generated"

    # Clean up FITS/PNGs to save disk space
    rm -rf "$fits_dir"
    rm -f "$csv_file"
done

echo ""
echo "=== DONE ==="
echo "Generated $(ls "$OUTPUT_DIR"/*.json | wc -l) test case JSONs in $OUTPUT_DIR"

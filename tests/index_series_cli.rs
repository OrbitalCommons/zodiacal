#![cfg(feature = "cli")]

use std::process::Command;

#[test]
fn rejects_invalid_scales_before_loading_catalog() {
    for (lower, upper, factor) in [
        ("10", "600", "1"),
        ("10", "600", "0.5"),
        ("0", "600", "2"),
        ("-1", "600", "2"),
        ("600", "10", "2"),
        ("10", "10", "2"),
        ("NaN", "600", "2"),
        ("10", "inf", "2"),
        ("10", "600", "NaN"),
        ("10", "600", "inf"),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_zodiacal"))
            .args([
                "build-index-series",
                "--catalog=nonexistent-catalog.bin",
                "--output-prefix=unused",
                &format!("--scale-lower={lower}"),
                &format!("--scale-upper={upper}"),
                &format!("--scale-factor={factor}"),
            ])
            .output()
            .unwrap();
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("Invalid scales:"), "{stderr}");
        assert!(!stderr.contains("Failed to load catalog"), "{stderr}");
    }
}

#[test]
fn rejects_band_bounds_that_cannot_advance() {
    let output = Command::new(env!("CARGO_BIN_EXE_zodiacal"))
        .args([
            "build-index-series",
            "--catalog=nonexistent-catalog.bin",
            "--output-prefix=unused",
            "--scale-lower=5e-324",
            "--scale-upper=1",
            "--scale-factor=1.0000000000000002",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("too close to 1"), "{stderr}");
}

#[test]
fn default_scales_produce_twelve_bands() {
    let output = Command::new(env!("CARGO_BIN_EXE_zodiacal"))
        .args([
            "build-index-series",
            "--catalog=nonexistent-catalog.bin",
            "--output-prefix=unused",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Building 12 narrow-band indexes"),
        "{stderr}"
    );
    assert!(stderr.contains("Failed to load catalog"), "{stderr}");
}

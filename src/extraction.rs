use ndarray::Array2;
use std::collections::VecDeque;

/// A detected source in an image.
#[derive(Debug, Clone)]
pub struct DetectedSource {
    pub x: f64,
    pub y: f64,
    pub flux: f64,
}

/// Configuration for source extraction.
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    /// Gaussian PSF sigma in pixels (not FWHM).
    pub psf_sigma: f64,
    /// Detection threshold in units of background sigma.
    pub threshold_sigma: f64,
    /// Maximum number of sources to return (brightest first).
    pub max_sources: usize,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            psf_sigma: 1.5,
            threshold_sigma: 5.0,
            max_sources: 200,
        }
    }
}

/// Compute the median of a slice of f32 values.
fn median(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Estimate background level and noise from an image.
///
/// Returns `(background, sigma)` where background is the median pixel value
/// and sigma is estimated from the median absolute deviation (MAD):
/// `sigma = 1.4826 * MAD`.
fn estimate_background(image: &Array2<f32>) -> (f32, f32) {
    let pixels: Vec<f32> = image.iter().copied().collect();
    if pixels.is_empty() {
        return (0.0, 0.0);
    }
    let med = median(&pixels);
    let abs_devs: Vec<f32> = pixels.iter().map(|&v| (v - med).abs()).collect();
    let mad = median(&abs_devs);
    let sigma = 1.4826 * mad;
    (med, sigma)
}

/// Find connected components of pixels above threshold using 4-connectivity flood fill.
fn find_connected_components(mask: &Array2<bool>) -> Vec<Vec<(usize, usize)>> {
    let (ny, nx) = mask.dim();
    let mut visited = Array2::<bool>::default((ny, nx));
    let mut components = Vec::new();

    for y in 0..ny {
        for x in 0..nx {
            if mask[[y, x]] && !visited[[y, x]] {
                let mut component = Vec::new();
                let mut queue = VecDeque::new();
                queue.push_back((y, x));
                visited[[y, x]] = true;

                while let Some((cy, cx)) = queue.pop_front() {
                    component.push((cy, cx));

                    for (dy, dx) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                        let ny2 = cy as i64 + dy;
                        let nx2 = cx as i64 + dx;
                        if ny2 >= 0 && ny2 < ny as i64 && nx2 >= 0 && nx2 < nx as i64 {
                            let ny2 = ny2 as usize;
                            let nx2 = nx2 as usize;
                            if mask[[ny2, nx2]] && !visited[[ny2, nx2]] {
                                visited[[ny2, nx2]] = true;
                                queue.push_back((ny2, nx2));
                            }
                        }
                    }
                }

                component.sort();
                components.push(component);
            }
        }
    }

    components
}

/// Extract point sources from a 2D image.
///
/// Algorithm:
/// 1. Estimate background level and noise (median and MAD-based sigma)
/// 2. Find pixels above threshold (background + threshold_sigma * noise)
/// 3. Connected-component labeling to group adjacent bright pixels
/// 4. For each component, compute flux-weighted centroid
/// 5. Sort by flux (brightest first), truncate to max_sources
pub fn extract_sources(image: &Array2<f32>, config: &ExtractionConfig) -> Vec<DetectedSource> {
    let (ny, nx) = image.dim();
    if ny == 0 || nx == 0 {
        return Vec::new();
    }

    let (background, sigma) = estimate_background(image);

    // When sigma is zero the image is uniform; any pixel above
    // background is a real source so use a tiny threshold offset.
    let threshold = if sigma > 0.0 {
        background + config.threshold_sigma as f32 * sigma
    } else {
        background + f32::EPSILON
    };

    let mask = image.mapv(|v| v > threshold);
    let components = find_connected_components(&mask);

    let mut sources: Vec<DetectedSource> = components
        .into_iter()
        .filter(|comp| comp.len() >= 3)
        .filter_map(|comp| {
            let mut sum_flux = 0.0_f64;
            let mut sum_x = 0.0_f64;
            let mut sum_y = 0.0_f64;

            for &(y, x) in &comp {
                let flux = (image[[y, x]] - background) as f64;
                if flux > 0.0 {
                    sum_flux += flux;
                    sum_x += x as f64 * flux;
                    sum_y += y as f64 * flux;
                }
            }

            if sum_flux <= 0.0 {
                return None;
            }

            Some(DetectedSource {
                x: sum_x / sum_flux,
                y: sum_y / sum_flux,
                flux: sum_flux,
            })
        })
        .collect();

    sources.sort_by(|a, b| {
        b.flux
            .partial_cmp(&a.flux)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sources.truncate(config.max_sources);
    sources
}

/// Create sources from a list of (x, y) pixel coordinates.
///
/// All sources get flux = 1.0. Useful when source positions are already known.
pub fn sources_from_points(points: &[(f64, f64)]) -> Vec<DetectedSource> {
    points
        .iter()
        .map(|&(x, y)| DetectedSource { x, y, flux: 1.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gaussian(image: &mut Array2<f32>, cx: f64, cy: f64, sigma: f64, amplitude: f32) {
        let (ny, nx) = image.dim();
        for y in 0..ny {
            for x in 0..nx {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let val = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
                image[[y, x]] += amplitude * val as f32;
            }
        }
    }

    #[test]
    fn empty_image() {
        let image = Array2::<f32>::zeros((100, 100));
        let config = ExtractionConfig::default();
        let sources = extract_sources(&image, &config);
        assert!(sources.is_empty());
    }

    #[test]
    fn single_bright_point() {
        let mut image = Array2::<f32>::zeros((64, 64));
        make_gaussian(&mut image, 30.0, 25.0, 2.0, 1000.0);

        let config = ExtractionConfig {
            psf_sigma: 2.0,
            threshold_sigma: 3.0,
            max_sources: 10,
        };

        let sources = extract_sources(&image, &config);
        assert_eq!(sources.len(), 1);

        let s = &sources[0];
        assert!((s.x - 30.0).abs() < 0.5, "x centroid off: {}", s.x);
        assert!((s.y - 25.0).abs() < 0.5, "y centroid off: {}", s.y);
        assert!(s.flux > 0.0);
    }

    #[test]
    fn multiple_sources_ordered_by_brightness() {
        let mut image = Array2::<f32>::zeros((128, 128));

        make_gaussian(&mut image, 20.0, 20.0, 2.0, 500.0);
        make_gaussian(&mut image, 80.0, 80.0, 2.0, 1000.0);
        make_gaussian(&mut image, 50.0, 100.0, 2.0, 200.0);

        let config = ExtractionConfig {
            psf_sigma: 2.0,
            threshold_sigma: 3.0,
            max_sources: 10,
        };

        let sources = extract_sources(&image, &config);
        assert_eq!(sources.len(), 3);

        assert!(sources[0].flux > sources[1].flux);
        assert!(sources[1].flux > sources[2].flux);

        assert!((sources[0].x - 80.0).abs() < 1.0);
        assert!((sources[0].y - 80.0).abs() < 1.0);
    }

    #[test]
    fn subpixel_centroid_accuracy() {
        let mut image = Array2::<f32>::zeros((32, 32));
        let true_x = 15.3;
        let true_y = 12.7;
        make_gaussian(&mut image, true_x, true_y, 1.5, 500.0);

        let config = ExtractionConfig {
            psf_sigma: 1.5,
            threshold_sigma: 3.0,
            max_sources: 10,
        };

        let sources = extract_sources(&image, &config);
        assert_eq!(sources.len(), 1);

        let s = &sources[0];
        assert!(
            (s.x - true_x).abs() < 0.15,
            "subpixel x off: {} vs {}",
            s.x,
            true_x
        );
        assert!(
            (s.y - true_y).abs() < 0.15,
            "subpixel y off: {} vs {}",
            s.y,
            true_y
        );
    }

    #[test]
    fn sources_from_points_roundtrip() {
        let points = vec![(10.5, 20.3), (30.1, 40.7), (50.0, 60.0)];
        let sources = sources_from_points(&points);

        assert_eq!(sources.len(), 3);
        for (i, (px, py)) in points.iter().enumerate() {
            assert_eq!(sources[i].x, *px);
            assert_eq!(sources[i].y, *py);
            assert_eq!(sources[i].flux, 1.0);
        }
    }

    #[test]
    fn max_sources_limit() {
        let mut image = Array2::<f32>::zeros((512, 512));

        // Place 10 well-separated sources in a grid pattern
        for i in 0..10 {
            let x = 50.0 + (i as f64 % 5.0) * 90.0;
            let y = 150.0 + (i as f64 / 5.0).floor() * 200.0;
            make_gaussian(&mut image, x, y, 2.0, 500.0 + (i as f32) * 100.0);
        }

        let config = ExtractionConfig {
            psf_sigma: 2.0,
            threshold_sigma: 3.0,
            max_sources: 5,
        };

        let sources = extract_sources(&image, &config);
        assert_eq!(sources.len(), 5);

        for i in 0..4 {
            assert!(sources[i].flux >= sources[i + 1].flux);
        }
    }

    #[test]
    fn noisy_background_with_bright_sources() {
        let mut image = Array2::<f32>::zeros((128, 128));

        let (ny, nx) = image.dim();
        for y in 0..ny {
            for x in 0..nx {
                let hash = ((x * 7919 + y * 104729 + 1) % 1000) as f32 / 1000.0;
                image[[y, x]] = hash * 10.0;
            }
        }

        make_gaussian(&mut image, 40.0, 40.0, 2.0, 500.0);
        make_gaussian(&mut image, 90.0, 90.0, 2.0, 800.0);

        let config = ExtractionConfig {
            psf_sigma: 2.0,
            threshold_sigma: 5.0,
            max_sources: 10,
        };

        let sources = extract_sources(&image, &config);

        assert!(
            sources.len() >= 2,
            "expected at least 2 sources, got {}",
            sources.len()
        );

        let has_source_near_40_40 = sources
            .iter()
            .any(|s| (s.x - 40.0).abs() < 2.0 && (s.y - 40.0).abs() < 2.0);
        let has_source_near_90_90 = sources
            .iter()
            .any(|s| (s.x - 90.0).abs() < 2.0 && (s.y - 90.0).abs() < 2.0);

        assert!(has_source_near_40_40, "missing source near (40, 40)");
        assert!(has_source_near_90_90, "missing source near (90, 90)");
    }
}

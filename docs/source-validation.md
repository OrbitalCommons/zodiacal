# Checked detected sources

`DetectedSource::new(x, y, flux)` returns `Result<DetectedSource, SourceError>`.
Coordinates must be finite and flux must be finite and strictly positive.
Use `x()`, `y()`, and `flux()` to read values. Replace struct literals with the
constructor; replace a source to change its values. JSON retains the same
`x`, `y`, `flux` object shape, but deserialization uses the checked constructor.
`sources_from_points` now returns a `Result` as well.

Image dimensions are positive integer pixel counts represented as `f64`.
Integer coordinates denote zero-based pixel centers. Sources must lie in the
inclusive rectangle `[0, width - 1] × [0, height - 1]`. The solver checks the
entire source list before searching, including sources beyond `max_field_stars`.
Invalid input returns no solution and sets `SolveStats::invalid_input`; no
sources are discarded or renumbered. `validate_sources` exposes the same check
for callers that want to report errors before solving.

Aperture1 adapters can remove duplicate validity formulas, but must retain their
policy of dropping invalid detections before applying the mission source cap.
Use checked construction to discard intrinsic invalidity, then
`source.validate_in_image(image_size).is_ok()` to filter out-of-frame centroids.
Passing an unfiltered out-of-frame source to the solver rejects the whole frame.
Keep mission source-count limits and ordering policy downstream. These changes
intentionally break unchecked source construction while retaining the legacy
solver tuple and JSON wire format.

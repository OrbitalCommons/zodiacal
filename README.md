# zodiacal
A blind astrometry library written in Rust.

## Sources JSON Format

Zodiacal uses a JSON format for exchanging detected source lists between tools. The `extract` command produces this format, and the solver can consume it.

### Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image_width` | number | yes | Image width in pixels |
| `image_height` | number | yes | Image height in pixels |
| `ra_deg` | number | no | Known RA of field center (degrees) |
| `dec_deg` | number | no | Known Dec of field center (degrees) |
| `plate_scale_arcsec` | number | no | Known plate scale (arcsec/pixel) |
| `sources` | array | yes | Detected sources |
| `sources[].x` | number | yes | Source x position (pixels) |
| `sources[].y` | number | yes | Source y position (pixels) |
| `sources[].flux` | number | yes | Source brightness (ADU) |

### Example

```json
{
  "image_width": 9568,
  "image_height": 6380,
  "ra_deg": 265.47,
  "dec_deg": 44.31,
  "plate_scale_arcsec": 0.1296,
  "sources": [
    { "x": 4821.3, "y": 3190.7, "flux": 54210.0 },
    { "x": 1023.5, "y": 892.1, "flux": 38450.0 }
  ]
}
```

The `ra_deg`, `dec_deg`, and `plate_scale_arcsec` fields are optional hints. When present they can speed up solving by constraining the search space. When absent they are omitted from the JSON entirely.

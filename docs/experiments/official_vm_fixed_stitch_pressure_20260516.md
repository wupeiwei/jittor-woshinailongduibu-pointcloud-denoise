# Official VM Fixed-Stitch Pressure

Source cluster:

- `analysis/large_cloud_pressure_official_vm_fixed_stitch_20260516/pressure_report.md`
- `analysis/large_cloud_pressure_official_vm_fixed_stitch_20260516/prototype_streaming_fixed_stitch_jittor_report.md`

## Result

The streaming fixed-stitch prototype preserved output count and stayed finite on
small tests, 100k regression, and 500k pressure runs.

## Boundary decision

- Keep the streaming path as engineering evidence.
- Keep the dense fixed-stitch path as the historical baseline.
- Do not treat the prototype as a score improvement by itself.

## Why it matters

This report is the cleanest explanation for why the official VM path was
patched and why the streaming variant is present in the repository.

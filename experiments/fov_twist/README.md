# fov_twist (superseded)

This experiment measured camera FOV twist from star fields against an earlier
navigation API. Its maintained successor lives in
[`util/fov_distortion/`](../../util/fov_distortion/), which measures both the
camera twist and the lateral residual distortion per instrument, decouples the
two, aggregates a per-instrument twist-consistency verdict and rotation-fitting
recommendation, and publishes the
[FOV Distortion and Camera-Twist Report](../../docs/fov_distortion_report/fov_distortion_report.rst).

The star-frame lists in `config/` were ported to the cohort YAML files in
`util/fov_distortion/configs/`. The code here is retained only for reference and
is not maintained.

# Image library

Operator-curated images with verified ground-truth offsets.  This directory
is the registry — adding a sidecar under `images/<class>/<image_id>.yaml`
enrolls the image in the structural-invariants test
(`tests/integration/test_image_library.py`) and the per-image regression
test (`tests/integration/test_autonomous_nav.py`).

The schema, scene-class list, and library layout are documented in
`docs/developer_guide_image_library.rst`; the validator lives in
`tests/integration/sidecar.py`.

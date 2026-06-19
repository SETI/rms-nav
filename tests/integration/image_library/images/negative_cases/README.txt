negative_cases
==============

Scenes the pipeline must correctly refuse — expected.status='failed'
and expected.confidence_tier='failed'. Tests that the orchestrator
fails closed rather than reporting a spurious offset.

Required:
- One of the following:
    * Distant tiny body (under 15 px) plus all catalog stars below
      the SNR floor — nothing navigable.
    * Empty interplanetary frame — no body, no rings, no usable
      stars.
    * Majority-dropout image (sensor missing-data fraction above the
      configured threshold).
    * Fully overexposed frame.
    * Saturated bloom dominates the entire FOV.
- The frame must be genuinely unnavigable, not merely difficult.

Excluded:
- Scenes that *barely* navigate (use a body_* / ring_* / star_*
  class with confidence_tier='low' instead).
- Frames where any single technique would legitimately produce a
  result with non-zero confidence.

Typical sources:
- Any mission, any camera. Pick deliberately bad frames spread
  across missions so the failure path is exercised on every code
  branch.

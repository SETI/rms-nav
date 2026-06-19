ring_only_curved
================

Ring edge with measurable 2-D curvature, no body in FOV. Exercises
the full-rank RingEdgeNav fit.

Required:
- At least one ring-edge polyline visible in the FOV.
- Polyline maximum deviation greater than 0.5 px from any straight-
  line fit (the "curved" qualifier).
- No body silhouette anywhere in the FOV.

Excluded:
- Polyline that fits within 0.5 px of a straight line (use
  ring_only_flat).
- Any body in FOV (use ring_plus_body).

Typical sources:
- Cassini Saturn rings, mid-range geometry where the ansa curvature
  is well-resolved.

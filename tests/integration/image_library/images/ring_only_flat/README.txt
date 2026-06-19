ring_only_flat
==============

Ring edge with negligible 2-D curvature, no body in FOV. Exercises
the rank-1 RingEdgeNav fallback (along-edge component is observable;
across-edge requires another feature).

Required:
- At least one ring-edge polyline visible in the FOV.
- Polyline curvature less than 0.5 px (essentially straight across
  the FOV).
- No body silhouette anywhere in the FOV.
- expected.status will typically be 'ok' with confidence_tier
  'medium' or 'low' since rank-1 only constrains one axis.

Excluded:
- Polyline with measurable curvature greater than 0.5 px (use
  ring_only_curved).
- Any body in FOV (use ring_plus_body).

Typical sources:
- Cassini Saturn ansa shots.
- Long-range Saturn ring frames where the ring plane is nearly
  edge-on across the FOV.

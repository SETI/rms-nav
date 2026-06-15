body_irregular
==============

One highly irregular body close enough that the predicted ellipsoidal
limb does not match the actual shape — the BLOB regime.

Required:
- Single irregularly shaped body (Phoebe, Hyperion, Prometheus,
  Pandora, Atlas, Pan, Janus, Epimetheus, etc.).
- Range close enough that the predicted limb has uncertainty greater
  than ~3 px (limb fitting becomes unreliable).
- Body diameter greater than 15 px (otherwise the resolution gate
  forces use of below_resolution_body).
- Brightness sufficient that a brightness-weighted centroid is
  meaningful.

Excluded:
- Regular ellipsoidal body (use body_full_fov,
  body_partial_overflow, or body_mostly_offscreen).
- Body smaller than 15 px (use below_resolution_body).
- More than one body (use multi_body).

Typical sources:
- Cassini ISS Phoebe encounter.
- Cassini ISS Hyperion close flyby.
- Cassini ISS ring-shepherd / co-orbital moon close-ups.

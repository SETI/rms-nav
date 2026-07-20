"""Cross-technique covariance-components estimator (truth-free solve).

Estimates per-technique 2x2 error covariance matrices from nothing but the
techniques' own offsets on shared frames -- the generalized three-cornered-hat
solve the agreement study consumes.  Given per-frame offsets from two or more
estimator instances, every pairwise difference supplies moment equations

    E[(o_i - o_j)(o_i - o_j)^T] = C_i + C_j - 2 * S_ij

(with ``S_ij`` the symmetric part of the cross-covariance, zero unless the
pair is explicitly declared suspect), and the module solves the resulting
linear system for the unknown covariance elements by least squares.

The solve is carried in full matrix form, never scalar sigmas:

- A ``full`` estimator instance carries a symmetric 2x2 covariance --
  three unknowns ``(c_11, c_12, c_22)`` -- expressed in its own *basis
  frame*.  The basis frame is either the fixed image frame or a per-frame
  rotating frame (e.g. a limb fit's covariance is anisotropic in the
  arc-aligned frame, and that frame rotates from image to image); the
  per-frame basis angle rotates the unknown into image coordinates inside
  the design matrix, so anisotropy that rotates frame to frame is handled
  by the solve itself rather than by orientation binning.
- A ``rank1`` estimator instance (a straight ring edge) measures the offset
  only along a per-frame axis; it carries a single variance unknown, and
  every equation involving it is projected onto that frame's axis before
  entering the system.

Instances may share a parameter ``group`` (two bodies navigated by the same
technique in one frame), which asserts the technique's error covariance is
common across the group's instances within the cohort -- an explicit
stationarity assumption the caller must be able to defend.

Identifiability is part of the output, not an assumption: the solve reports
the design matrix's singular spectrum, the (numerical) null space mapped
back to parameter names, and a per-parameter identifiability score.  A
degenerate cohort (e.g. limb+disc alone: one matrix equation, two unknown
matrices) yields an explicit null space instead of a silently arbitrary
answer; the minimum-norm solution is returned with those directions flagged
unidentifiable.

All angles follow the (v, u) image convention: an angle ``alpha`` denotes
the unit direction ``(cos(alpha), sin(alpha))`` in (v, u) coordinates.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    'EstimatorSpec',
    'FrameSample',
    'PairKey',
    'SolveResult',
    'solve_covariance_components',
]

PairKey = tuple[str, str]

# Two rank-1 instances can only be differenced when their measurement axes
# are parallel to within this |cos| tolerance; otherwise the pair carries no
# common measured component and is skipped for that frame.
_RANK1_PARALLEL_MIN_ABS_COS = 0.99


@dataclass(frozen=True)
class EstimatorSpec:
    """One estimator instance participating in the solve.

    Parameters:
        name: Instance name, unique within the solve (e.g. ``'limb'`` or
            ``'limb@RHEA'``).
        kind: ``'full'`` for a 2-D offset estimate with a full 2x2
            covariance; ``'rank1'`` for an estimate constrained only along
            a per-frame axis (a straight ring edge), carrying a single
            variance unknown.
        basis: ``'image'`` when the covariance is stationary in image
            coordinates; ``'rotating'`` when it is stationary in a
            per-frame frame whose angle each :class:`FrameSample` supplies
            (only meaningful for ``kind='full'``).
        group: Parameter-sharing key.  Instances with the same group share
            one set of covariance unknowns (asserting a common error
            covariance across the instances); defaults to ``name``.

    Raises:
        ValueError: if ``kind`` / ``basis`` combinations are inconsistent.
    """

    name: str
    kind: Literal['full', 'rank1']
    basis: Literal['image', 'rotating'] = 'image'
    group: str | None = None

    def __post_init__(self) -> None:
        """Validate the kind/basis combination."""
        if self.kind not in ('full', 'rank1'):
            raise ValueError(f'kind must be full or rank1; got {self.kind!r}')
        if self.basis not in ('image', 'rotating'):
            raise ValueError(f'basis must be image or rotating; got {self.basis!r}')
        if self.kind == 'rank1' and self.basis == 'rotating':
            raise ValueError('rank1 instances take their axis per frame; basis must be image')

    @property
    def group_key(self) -> str:
        """The parameter-sharing key (``group`` or the instance name)."""
        return self.group if self.group is not None else self.name


@dataclass(frozen=True)
class FrameSample:
    """Per-frame technique offsets plus the frame's geometry angles.

    Parameters:
        offsets: Mapping from instance name to its measured ``(dv, du)``
            offset on this frame.  Instances absent from a frame simply
            contribute no equations there.
        basis_angle_rad: Per-instance basis-frame angle (radians, image
            (v, u) convention) for ``basis='rotating'`` instances; ignored
            for image-basis instances.
        axis_angle_rad: Per-instance measurement-axis angle (radians) for
            ``rank1`` instances; required for each rank1 instance present
            in ``offsets``.
    """

    offsets: Mapping[str, tuple[float, float]]
    basis_angle_rad: Mapping[str, float] = field(default_factory=dict)
    axis_angle_rad: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SolveResult:
    """Output of :func:`solve_covariance_components`.

    Parameters:
        param_names: Ordered unknown names.  Covariance elements are
            ``'<group>:c11'`` / ``'<group>:c12'`` / ``'<group>:c22'`` (in
            the group's basis frame), rank-1 variances ``'<group>:s2'``,
            and declared pair covariances ``'cov(<i>,<j>):<elem>'``.
        params: Minimum-norm least-squares solution, aligned with
            ``param_names``.  Values along unidentifiable directions are
            arbitrary (minimum-norm); consult ``identifiability``.
        covariances: Per-group symmetric 2x2 covariance (basis frame)
            assembled from ``params`` for every ``full`` group.
        rank1_variances: Per-group scalar variance for every rank1 group.
        pair_covariances: Recovered symmetric cross-covariance per declared
            pair (2x2 image-frame matrix for full/full pairs, scalar for
            pairs involving a rank1 instance).
        pair_mean_diff: Per differenced pair, the cohort mean difference
            that was subtracted before forming second moments (the bias
            channel; 2-vector for full/full pairs, scalar otherwise).
        singular_values: Singular values of the design matrix, descending.
        condition_number: Ratio of largest to smallest singular value
            (``inf`` for an exactly rank-deficient system).
        null_space: ``(k, n_params)`` orthonormal rows spanning the
            numerical null space at ``practical_sv_ratio``; empty when the
            system is fully identifiable.
        identifiability: Per-parameter score in [0, 1]: the squared
            projection of the parameter axis onto the identifiable row
            space.  1.0 means fully determined; ~0 means the parameter
            lives in the null space and its returned value is arbitrary.
        n_frames: Frames consumed.
        n_equations: Scalar equations assembled.
        residual_rms: Root-mean-square residual of the fitted equations.
        bootstrap_ci: Optional per-parameter (lo, hi) percentile interval
            from frame-resampling; empty when bootstrap was not requested.
            Intervals for unidentifiable parameters are not meaningful.
    """

    param_names: tuple[str, ...]
    params: NDArray[np.float64]
    covariances: dict[str, NDArray[np.float64]]
    rank1_variances: dict[str, float]
    pair_covariances: dict[PairKey, NDArray[np.float64] | float]
    pair_mean_diff: dict[PairKey, NDArray[np.float64] | float]
    singular_values: NDArray[np.float64]
    condition_number: float
    null_space: NDArray[np.float64]
    identifiability: dict[str, float]
    n_frames: int
    n_equations: int
    residual_rms: float
    bootstrap_ci: dict[str, tuple[float, float]]


def _vech_rotation(theta: float) -> NDArray[np.float64]:
    """Return M with ``vech(R C R^T) = M @ vech(C)`` for a rotation by theta.

    ``R`` maps basis coordinates into image coordinates; its columns are the
    basis vectors ``(cos t, sin t)`` and ``(-sin t, cos t)``.  ``vech`` packs
    a symmetric 2x2 as ``(c11, c12, c22)``.

    Parameters:
        theta: Basis-frame angle in radians.

    Returns:
        The 3x3 rotation-action matrix on vech coordinates.
    """
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [
            [c * c, -2.0 * c * s, s * s],
            [c * s, c * c - s * s, -c * s],
            [s * s, 2.0 * c * s, c * c],
        ],
        dtype=np.float64,
    )


def _vech_quadratic(alpha: float) -> NDArray[np.float64]:
    """Return q with ``a^T C a = q @ vech(C)`` for the unit axis at alpha.

    Parameters:
        alpha: Axis angle in radians (image (v, u) convention).

    Returns:
        Length-3 coefficient row on image-frame vech coordinates.
    """
    a1 = math.cos(alpha)
    a2 = math.sin(alpha)
    return np.array([a1 * a1, 2.0 * a1 * a2, a2 * a2], dtype=np.float64)


@dataclass(frozen=True)
class _ParamLayout:
    """Internal parameter bookkeeping for the linear system."""

    names: tuple[str, ...]
    group_slices: dict[str, slice]
    group_kind: dict[str, str]
    pair_slices: dict[PairKey, slice]
    n_params: int


def _build_layout(
    specs: Sequence[EstimatorSpec], pair_covariances: Sequence[PairKey]
) -> _ParamLayout:
    """Assign parameter-vector slices to groups and declared pairs.

    Parameters:
        specs: Estimator instances (validated by the caller).
        pair_covariances: Declared suspect pairs, as instance-name tuples.

    Returns:
        The parameter layout.

    Raises:
        ValueError: on group kind conflicts or unknown pair members.
    """
    by_name = {spec.name: spec for spec in specs}
    names: list[str] = []
    group_slices: dict[str, slice] = {}
    group_kind: dict[str, str] = {}
    for spec in specs:
        key = spec.group_key
        if key in group_kind:
            if group_kind[key] != spec.kind:
                raise ValueError(f'group {key!r} mixes full and rank1 instances')
            continue
        group_kind[key] = spec.kind
        start = len(names)
        if spec.kind == 'full':
            names.extend([f'{key}:c11', f'{key}:c12', f'{key}:c22'])
        else:
            names.append(f'{key}:s2')
        group_slices[key] = slice(start, len(names))
    pair_slices: dict[PairKey, slice] = {}
    for pair in pair_covariances:
        i_name, j_name = pair
        if i_name not in by_name or j_name not in by_name:
            raise ValueError(f'pair {pair!r} names an unknown instance')
        start = len(names)
        if by_name[i_name].kind == 'full' and by_name[j_name].kind == 'full':
            names.extend(
                [
                    f'cov({i_name},{j_name}):s11',
                    f'cov({i_name},{j_name}):s12',
                    f'cov({i_name},{j_name}):s22',
                ]
            )
        else:
            names.append(f'cov({i_name},{j_name}):gamma')
        pair_slices[pair] = slice(start, len(names))
    return _ParamLayout(
        names=tuple(names),
        group_slices=group_slices,
        group_kind=group_kind,
        pair_slices=pair_slices,
        n_params=len(names),
    )


def _pair_slice(layout: _ParamLayout, i_name: str, j_name: str) -> slice | None:
    """Return the declared-pair slice for (i, j) in either order, or None."""
    for key in ((i_name, j_name), (j_name, i_name)):
        if key in layout.pair_slices:
            return layout.pair_slices[key]
    return None


def _basis_angle(spec: EstimatorSpec, frame: FrameSample) -> float:
    """Return the basis angle for a full instance on a frame (0 for image)."""
    if spec.basis == 'rotating':
        if spec.name not in frame.basis_angle_rad:
            raise ValueError(
                f'frame is missing basis_angle_rad for rotating instance {spec.name!r}'
            )
        return float(frame.basis_angle_rad[spec.name])
    return 0.0


def _axis_angle(spec: EstimatorSpec, frame: FrameSample) -> float:
    """Return the measurement-axis angle for a rank1 instance on a frame."""
    if spec.name not in frame.axis_angle_rad:
        raise ValueError(f'frame is missing axis_angle_rad for rank1 instance {spec.name!r}')
    return float(frame.axis_angle_rad[spec.name])


@dataclass(frozen=True)
class _PairEquations:
    """Per-frame design/target blocks for one instance pair.

    ``rows[f]`` and ``targets[f]`` hold that frame's equations (1 or 3 rows).
    ``frame_index[f]`` maps back to the cohort frame index for bootstrap
    resampling.
    """

    rows: list[NDArray[np.float64]]
    targets: list[NDArray[np.float64]]
    frame_index: list[int]


def _difference_samples(
    frames: Sequence[FrameSample],
    spec_i: EstimatorSpec,
    spec_j: EstimatorSpec,
) -> tuple[list[int], NDArray[np.float64], list[tuple[float, float]]]:
    """Collect raw per-frame differences for one pair.

    For a full/full pair the difference is the 2-vector ``o_i - o_j``.  When
    either member is rank1 the difference is the scalar projection of both
    offsets onto the frame's measurement axis (a rank1 instance's offset is
    only meaningful along that axis).  A rank1/rank1 pair contributes only on
    frames where the two axes are parallel within tolerance.

    Parameters:
        frames: The cohort.
        spec_i: First pair member.
        spec_j: Second pair member.

    Returns:
        ``(frame_indices, diffs, angles)`` where ``diffs`` is ``(n, 2)`` for
        full/full pairs and ``(n, 1)`` otherwise, and ``angles`` carries the
        per-sample ``(basis_or_axis_i, basis_or_axis_j)`` angles needed to
        rebuild design rows.
    """
    indices: list[int] = []
    diffs: list[list[float]] = []
    angles: list[tuple[float, float]] = []
    for f_idx, frame in enumerate(frames):
        if spec_i.name not in frame.offsets or spec_j.name not in frame.offsets:
            continue
        o_i = frame.offsets[spec_i.name]
        o_j = frame.offsets[spec_j.name]
        if spec_i.kind == 'full' and spec_j.kind == 'full':
            indices.append(f_idx)
            diffs.append([o_i[0] - o_j[0], o_i[1] - o_j[1]])
            angles.append((_basis_angle(spec_i, frame), _basis_angle(spec_j, frame)))
        elif spec_i.kind == 'rank1' and spec_j.kind == 'rank1':
            alpha_i = _axis_angle(spec_i, frame)
            alpha_j = _axis_angle(spec_j, frame)
            if abs(math.cos(alpha_i - alpha_j)) < _RANK1_PARALLEL_MIN_ABS_COS:
                continue
            a_i = (math.cos(alpha_i), math.sin(alpha_i))
            a_j = (math.cos(alpha_j), math.sin(alpha_j))
            s_i = a_i[0] * o_i[0] + a_i[1] * o_i[1]
            s_j = a_j[0] * o_j[0] + a_j[1] * o_j[1]
            # Express both along axis i (they are parallel within tolerance;
            # a sign flip between the axes flips s_j's sign).
            sign = 1.0 if math.cos(alpha_i - alpha_j) >= 0 else -1.0
            indices.append(f_idx)
            diffs.append([s_i - sign * s_j])
            angles.append((alpha_i, alpha_j))
        else:
            # Exactly one member is rank1: project both onto its axis.  The
            # angle tuple stays positional -- slot 0 for spec_i, slot 1 for
            # spec_j -- holding the axis angle for the rank1 member and the
            # basis angle for the full member.
            rank1_spec = spec_i if spec_i.kind == 'rank1' else spec_j
            alpha = _axis_angle(rank1_spec, frame)
            a = (math.cos(alpha), math.sin(alpha))
            s_i = a[0] * o_i[0] + a[1] * o_i[1]
            s_j = a[0] * o_j[0] + a[1] * o_j[1]
            indices.append(f_idx)
            diffs.append([s_i - s_j])
            angle_i = alpha if spec_i.kind == 'rank1' else _basis_angle(spec_i, frame)
            angle_j = alpha if spec_j.kind == 'rank1' else _basis_angle(spec_j, frame)
            angles.append((angle_i, angle_j))
    if not indices:
        return [], np.zeros((0, 1), dtype=np.float64), []
    return indices, np.asarray(diffs, dtype=np.float64).reshape(len(indices), -1), angles


def _pair_equations(
    frames: Sequence[FrameSample],
    spec_i: EstimatorSpec,
    spec_j: EstimatorSpec,
    layout: _ParamLayout,
) -> tuple[_PairEquations, NDArray[np.float64] | float] | None:
    """Build the moment equations for one instance pair.

    The cohort mean difference is subtracted before squaring, so the
    second moments are central: a deterministic shared bias between the two
    instances lands in the returned mean, not in the covariance system.

    Parameters:
        frames: The cohort.
        spec_i: First pair member.
        spec_j: Second pair member.
        layout: Parameter layout.

    Returns:
        ``(equations, mean_diff)`` or ``None`` when the pair never co-occurs.
    """
    indices, diffs, angles = _difference_samples(frames, spec_i, spec_j)
    if len(indices) == 0:
        return None
    mean = diffs.mean(axis=0)
    centered = diffs - mean
    pair_slice = _pair_slice(layout, spec_i.name, spec_j.name)
    rows: list[NDArray[np.float64]] = []
    targets: list[NDArray[np.float64]] = []
    frame_index: list[int] = []
    full_full = spec_i.kind == 'full' and spec_j.kind == 'full'
    for k, f_idx in enumerate(indices):
        if full_full:
            d = centered[k]
            target = np.array([d[0] * d[0], d[0] * d[1], d[1] * d[1]], dtype=np.float64)
            block = np.zeros((3, layout.n_params), dtype=np.float64)
            block[:, layout.group_slices[spec_i.group_key]] += _vech_rotation(angles[k][0])
            block[:, layout.group_slices[spec_j.group_key]] += _vech_rotation(angles[k][1])
            if pair_slice is not None:
                block[:, pair_slice] += -2.0 * np.eye(3)
        else:
            s = centered[k, 0]
            target = np.array([s * s], dtype=np.float64)
            block = np.zeros((1, layout.n_params), dtype=np.float64)
            # The projection axis is the rank1 member's axis angle, held in
            # that member's positional slot (for rank1/rank1 pairs slot 0).
            axis = angles[k][0] if spec_i.kind == 'rank1' else angles[k][1]
            for spec, angle_pos in ((spec_i, 0), (spec_j, 1)):
                sl = layout.group_slices[spec.group_key]
                if spec.kind == 'rank1':
                    block[0, sl] += 1.0
                else:
                    theta = angles[k][angle_pos]
                    block[0, sl] += _vech_quadratic(axis) @ _vech_rotation(theta)
            if pair_slice is not None:
                block[0, pair_slice] += -2.0
        rows.append(block)
        targets.append(target)
        frame_index.append(f_idx)
    mean_out: NDArray[np.float64] | float = mean if full_full else float(mean[0])
    return _PairEquations(rows=rows, targets=targets, frame_index=frame_index), mean_out


def _assemble(
    pair_eqs: Iterable[_PairEquations],
    n_params: int,
    keep_frames: NDArray[np.intp] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Stack per-pair equation blocks into a single (A, y) system.

    Parameters:
        pair_eqs: Equation blocks from every pair.
        n_params: Width of the design matrix.
        keep_frames: Optional multiset of frame indices (bootstrap resample);
            a frame's equations are repeated per occurrence.  ``None`` keeps
            every equation once.

    Returns:
        Design matrix ``A`` and target vector ``y``.
    """
    counts: dict[int, int] | None = None
    if keep_frames is not None:
        counts = {}
        for idx in keep_frames:
            counts[int(idx)] = counts.get(int(idx), 0) + 1
    a_blocks: list[NDArray[np.float64]] = []
    y_blocks: list[NDArray[np.float64]] = []
    for eqs in pair_eqs:
        for block, target, f_idx in zip(eqs.rows, eqs.targets, eqs.frame_index, strict=True):
            repeat = 1 if counts is None else counts.get(f_idx, 0)
            for _ in range(repeat):
                a_blocks.append(block)
                y_blocks.append(target)
    if not a_blocks:
        return np.zeros((0, n_params), dtype=np.float64), np.zeros(0, dtype=np.float64)
    return np.vstack(a_blocks), np.concatenate(y_blocks)


def solve_covariance_components(
    frames: Sequence[FrameSample],
    specs: Sequence[EstimatorSpec],
    *,
    pair_covariances: Sequence[PairKey] = (),
    practical_sv_ratio: float = 1e-6,
    n_bootstrap: int = 0,
    bootstrap_seed: int = 0,
) -> SolveResult:
    """Solve the covariance-components system for a cohort of frames.

    Every co-occurring instance pair contributes central second-moment
    equations; the stacked linear system is solved by minimum-norm least
    squares.  Identifiability is diagnosed from the design matrix's singular
    spectrum and reported per parameter -- a degenerate composition returns
    its null space explicitly rather than failing.

    Parameters:
        frames: Cohort of per-frame samples.
        specs: Estimator instances (unique names).
        pair_covariances: Instance-name pairs whose symmetric
            cross-covariance should be solved for instead of assumed zero.
            Declaring a pair adds unknowns; the cohort must be
            over-determined enough to carry them (check the returned
            identifiability scores).
        practical_sv_ratio: Singular values below this fraction of the
            largest are treated as null directions.
        n_bootstrap: Number of frame-resampling bootstrap replicates for
            percentile confidence intervals (0 disables).
        bootstrap_seed: Seed for the bootstrap resampler.

    Returns:
        The :class:`SolveResult`.

    Raises:
        ValueError: on duplicate instance names, an empty cohort, or a
            cohort that yields no equations.
    """
    if len(frames) == 0:
        raise ValueError('frames must be non-empty')
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f'instance names must be unique; got {names!r}')
    if len(specs) < 2:
        raise ValueError('at least two estimator instances are required')
    layout = _build_layout(specs, pair_covariances)
    pair_blocks: dict[PairKey, _PairEquations] = {}
    pair_means: dict[PairKey, NDArray[np.float64] | float] = {}
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            built = _pair_equations(frames, specs[i], specs[j], layout)
            if built is None:
                continue
            eqs, mean = built
            pair_blocks[(specs[i].name, specs[j].name)] = eqs
            pair_means[(specs[i].name, specs[j].name)] = mean
    a_mat, y_vec = _assemble(pair_blocks.values(), layout.n_params)
    if a_mat.shape[0] == 0:
        raise ValueError('no pair of instances ever co-occurs; nothing to solve')
    params, _, _, sv = np.linalg.lstsq(a_mat, y_vec, rcond=None)
    sv = np.sort(np.asarray(sv, dtype=np.float64))[::-1]
    # np.linalg.lstsq returns only nonzero singular values for rank-deficient
    # systems on some backends; recompute the full spectrum for diagnostics.
    _, full_sv, vt = np.linalg.svd(a_mat, full_matrices=False)
    sv_max = float(full_sv[0]) if full_sv.size else 0.0
    null_mask = full_sv < practical_sv_ratio * sv_max
    null_space = vt[null_mask]
    sv_min = float(full_sv[~null_mask][-1]) if (~null_mask).any() else 0.0
    condition = math.inf if null_mask.any() or sv_min == 0.0 else sv_max / sv_min
    row_space = vt[~null_mask]
    identifiability = {
        name: float(np.sum(row_space[:, k] ** 2)) for k, name in enumerate(layout.names)
    }
    residual = a_mat @ params - y_vec
    residual_rms = float(np.sqrt(np.mean(residual**2))) if residual.size else 0.0

    covariances: dict[str, NDArray[np.float64]] = {}
    rank1_variances: dict[str, float] = {}
    for key, sl in layout.group_slices.items():
        if layout.group_kind[key] == 'full':
            c11, c12, c22 = params[sl]
            covariances[key] = np.array([[c11, c12], [c12, c22]], dtype=np.float64)
        else:
            rank1_variances[key] = float(params[sl][0])
    pair_cov_out: dict[PairKey, NDArray[np.float64] | float] = {}
    for pair, sl in layout.pair_slices.items():
        vals = params[sl]
        if vals.size == 3:
            pair_cov_out[pair] = np.array(
                [[vals[0], vals[1]], [vals[1], vals[2]]], dtype=np.float64
            )
        else:
            pair_cov_out[pair] = float(vals[0])

    bootstrap_ci: dict[str, tuple[float, float]] = {}
    if n_bootstrap > 0:
        rng = np.random.default_rng(bootstrap_seed)
        n_frames = len(frames)
        samples = np.empty((n_bootstrap, layout.n_params), dtype=np.float64)
        for b in range(n_bootstrap):
            resample = rng.integers(0, n_frames, size=n_frames).astype(np.intp)
            a_b, y_b = _assemble(pair_blocks.values(), layout.n_params, keep_frames=resample)
            if a_b.shape[0] == 0:
                samples[b] = np.nan
                continue
            samples[b], _, _, _ = np.linalg.lstsq(a_b, y_b, rcond=None)
        lo = np.nanpercentile(samples, 2.5, axis=0)
        hi = np.nanpercentile(samples, 97.5, axis=0)
        bootstrap_ci = {name: (float(lo[k]), float(hi[k])) for k, name in enumerate(layout.names)}

    return SolveResult(
        param_names=layout.names,
        params=np.asarray(params, dtype=np.float64),
        covariances=covariances,
        rank1_variances=rank1_variances,
        pair_covariances=pair_cov_out,
        pair_mean_diff=pair_means,
        singular_values=np.asarray(full_sv, dtype=np.float64),
        condition_number=condition,
        null_space=np.asarray(null_space, dtype=np.float64),
        identifiability=identifiability,
        n_frames=len(frames),
        n_equations=int(a_mat.shape[0]),
        residual_rms=residual_rms,
        bootstrap_ci=bootstrap_ci,
    )

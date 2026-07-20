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

Bias handling: before the second moments are formed, each pair's
differences are centered by a fitted *mean model*, not just a constant.
The model always carries the constant (image-frame) term, and for every
``basis='rotating'`` member it additionally carries that member's
rotating-frame mean components ``R(theta) @ mu`` -- so a technique bias
that is constant in the technique's own rotating frame (a limb fit pulled
radially toward its body) is absorbed instead of aliasing into the
recovered covariance.  The hazard this guards against is real: an
*undeclared* rotating-frame bias has image-frame mean ~0 over an
orientation-diverse cohort, the constant channel sees nothing, and
``R mu mu^T R^T`` enters every second-moment equation as if it were
covariance -- silently, with the system well-conditioned.  Declaring
``basis='rotating'`` for a technique therefore does two jobs: it makes the
covariance stationary in the right frame *and* it arms the rotating mean
columns for that technique.  Biases locked to geometry the model does not
carry (e.g. illumination direction) are NOT absorbed and still alias.

Declared pair covariances carry two model restrictions the caller must
accept: a full/full pair's ``S_ij`` is a single image-frame-constant
symmetric matrix (there is no rotating option, even when a member is
``basis='rotating'``), and a pair involving a rank1 instance carries one
scalar ``gamma`` -- the projected cross-covariance is assumed independent
of the projection axis, i.e. ``S_ij = gamma * I`` (exact for an isotropic
shared bias, an approximation otherwise).

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
from collections.abc import Mapping, Sequence
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
# are parallel to within this |cos| tolerance.  The tolerance must be tight:
# the two techniques share the frame's true offset t, and the difference of
# the projections carries a truth leak (a_i - sign * a_j) . t that does NOT
# cancel -- at |cos| = 0.99 (about 8 deg) a 4 px true offset leaks ~0.5 px
# into the "difference".  At 0.9999 (0.8 deg) the leak is under ~0.06 px.
_RANK1_PARALLEL_MIN_ABS_COS = 0.9999

# The per-pair mean model falls back to constant-only columns when the pair
# has fewer than this many samples per fitted column (guards against
# soaking up second-moment signal with an overfitted mean on tiny cohorts).
_MEAN_MODEL_MIN_SAMPLES_PER_COLUMN = 4


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
            (only meaningful for ``kind='full'``).  Declaring
            ``'rotating'`` also arms the instance's rotating-frame mean
            columns in the pair mean model, so a bias constant in that
            frame is absorbed rather than aliased into the covariance.
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
            pairs involving a rank1 instance; see the module docstring for
            the model restrictions those forms carry).
        pair_mean_diff: Per differenced pair, the *raw image-frame* cohort
            mean of the differences (2-vector for full/full pairs, scalar
            otherwise).  This is the constant bias channel only: a bias
            constant in a rotating frame averages toward zero here and is
            instead captured by the fitted mean model
            (``pair_mean_model``).
        pair_mean_model: Per differenced pair, the fitted mean-model
            coefficients as ``{column_label: value}`` (labels: ``const_v``
            / ``const_u`` or ``const`` / ``axis_v`` / ``axis_u``, plus
            ``<name>:mu1`` / ``<name>:mu2`` rotating-frame mean components
            per rotating member).  When two members share one rotation
            angle the split between their mu columns is minimum-norm; only
            the combined fitted mean function is meaningful then.
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
            from frame-resampling (the pair mean models are re-fitted per
            replicate); empty when bootstrap was not requested.  Intervals
            for unidentifiable parameters are not meaningful.
    """

    param_names: tuple[str, ...]
    params: NDArray[np.float64]
    covariances: dict[str, NDArray[np.float64]]
    rank1_variances: dict[str, float]
    pair_covariances: dict[PairKey, NDArray[np.float64] | float]
    pair_mean_diff: dict[PairKey, NDArray[np.float64] | float]
    pair_mean_model: dict[PairKey, dict[str, float]]
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


def _rot(theta: float) -> NDArray[np.float64]:
    """Rotation matrix whose columns are the basis vectors at ``theta``."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


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
class _PairData:
    """Precomputed per-pair samples: raw diffs, mean design, cov design.

    ``diffs`` is ``(n, 2)`` for full/full pairs and ``(n, 1)`` otherwise.
    ``mean_design`` is ``(n, d, k)`` (per-sample design block of the mean
    model), ``cov_rows`` is ``(n, d_eq, n_params)`` (per-sample block of
    covariance equations; ``d_eq`` is 3 for full/full, 1 for scalar).
    """

    full_full: bool
    frame_index: list[int]
    diffs: NDArray[np.float64]
    mean_design: NDArray[np.float64]
    mean_columns: tuple[str, ...]
    cov_rows: NDArray[np.float64]


def _build_pair_data(
    frames: Sequence[FrameSample],
    spec_i: EstimatorSpec,
    spec_j: EstimatorSpec,
    layout: _ParamLayout,
    mean_model: str,
) -> _PairData | None:
    """Collect one pair's samples, mean-model design, and covariance rows.

    For a full/full pair the difference is the 2-vector ``o_i - o_j``.  When
    either member is rank1 the difference is the scalar projection of both
    offsets onto the frame's measurement axis (a rank1 instance's offset is
    only meaningful along that axis).  A rank1/rank1 pair contributes only on
    frames where the two axes are parallel within tolerance.

    The mean model's columns are the constant (image-frame) terms plus, for
    every ``basis='rotating'`` member, that member's rotating-frame mean
    components; with ``mean_model='constant'`` only the constant terms are
    kept (the pre-rotating-mean behavior, retained for comparison).  An
    image-frame-constant shared bias is absorbed by the constant columns; a
    rotating-frame bias is absorbed only when its member is declared
    rotating -- an undeclared one aliases into the covariance equations
    (see the module docstring).

    Parameters:
        frames: The cohort.
        spec_i: First pair member.
        spec_j: Second pair member.
        layout: Parameter layout.
        mean_model: ``'auto'`` or ``'constant'``.

    Returns:
        The pair data, or ``None`` when the pair never co-occurs.
    """
    pair_slice = _pair_slice(layout, spec_i.name, spec_j.name)
    full_full = spec_i.kind == 'full' and spec_j.kind == 'full'
    rotating_members = [
        (spec, sign)
        for spec, sign in ((spec_i, 1.0), (spec_j, -1.0))
        if spec.kind == 'full' and spec.basis == 'rotating'
    ]
    # 'constant' reproduces plain mean-centering (image-frame vector for
    # full/full pairs, scalar for projected pairs); 'auto' adds the
    # image-frame constant vector projected onto the varying axis (scalar
    # pairs) and the rotating-frame mean columns per rotating member.
    if full_full:
        mean_columns = ['const_v', 'const_u']
    elif mean_model == 'auto':
        mean_columns = ['const', 'axis_v', 'axis_u']
    else:
        mean_columns = ['const']
    if mean_model == 'auto':
        for spec, _ in rotating_members:
            mean_columns.extend([f'{spec.name}:mu1', f'{spec.name}:mu2'])
    n_cols = len(mean_columns)

    frame_index: list[int] = []
    diffs: list[NDArray[np.float64]] = []
    designs: list[NDArray[np.float64]] = []
    cov_blocks: list[NDArray[np.float64]] = []
    for f_idx, frame in enumerate(frames):
        if spec_i.name not in frame.offsets or spec_j.name not in frame.offsets:
            continue
        o_i = np.asarray(frame.offsets[spec_i.name], dtype=np.float64)
        o_j = np.asarray(frame.offsets[spec_j.name], dtype=np.float64)
        if full_full:
            d = o_i - o_j
            design = np.zeros((2, n_cols), dtype=np.float64)
            design[:, 0:2] = np.eye(2)
            col = 2
            for spec, sign in rotating_members if mean_model == 'auto' else []:
                design[:, col : col + 2] = sign * _rot(_basis_angle(spec, frame))
                col += 2
            block = np.zeros((3, layout.n_params), dtype=np.float64)
            block[:, layout.group_slices[spec_i.group_key]] += _vech_rotation(
                _basis_angle(spec_i, frame)
            )
            block[:, layout.group_slices[spec_j.group_key]] += _vech_rotation(
                _basis_angle(spec_j, frame)
            )
            if pair_slice is not None:
                block[:, pair_slice] += -2.0 * np.eye(3)
            frame_index.append(f_idx)
            diffs.append(d)
            designs.append(design)
            cov_blocks.append(block)
            continue
        if spec_i.kind == 'rank1' and spec_j.kind == 'rank1':
            alpha_i = _axis_angle(spec_i, frame)
            alpha_j = _axis_angle(spec_j, frame)
            cos_between = math.cos(alpha_i - alpha_j)
            if abs(cos_between) < _RANK1_PARALLEL_MIN_ABS_COS:
                continue
            axis = alpha_i
            a_i = np.array([math.cos(alpha_i), math.sin(alpha_i)])
            a_j = np.array([math.cos(alpha_j), math.sin(alpha_j)])
            sign_j = 1.0 if cos_between >= 0 else -1.0
            s = float(a_i @ o_i) - sign_j * float(a_j @ o_j)
        else:
            rank1_spec = spec_i if spec_i.kind == 'rank1' else spec_j
            axis = _axis_angle(rank1_spec, frame)
            a = np.array([math.cos(axis), math.sin(axis)])
            s = float(a @ o_i) - float(a @ o_j)
        design = np.zeros((1, n_cols), dtype=np.float64)
        design[0, 0] = 1.0
        if mean_model == 'auto':
            design[0, 1] = math.cos(axis)
            design[0, 2] = math.sin(axis)
            col = 3
            for spec, sign in rotating_members:
                a_row = np.array([math.cos(axis), math.sin(axis)]) @ _rot(_basis_angle(spec, frame))
                design[0, col : col + 2] = sign * a_row
                col += 2
        block = np.zeros((1, layout.n_params), dtype=np.float64)
        for spec in (spec_i, spec_j):
            sl = layout.group_slices[spec.group_key]
            if spec.kind == 'rank1':
                block[0, sl] += 1.0
            else:
                theta = _basis_angle(spec, frame)
                block[0, sl] += _vech_quadratic(axis) @ _vech_rotation(theta)
        if pair_slice is not None:
            block[0, pair_slice] += -2.0
        frame_index.append(f_idx)
        diffs.append(np.array([s], dtype=np.float64))
        designs.append(design)
        cov_blocks.append(block)
    if not frame_index:
        return None
    diffs_arr = np.stack(diffs)
    design_arr = np.stack(designs)
    cov_arr = np.stack(cov_blocks)
    # Overfit guard: with too few samples per mean-model column, fall back
    # to the constant-only columns so the mean fit cannot soak up
    # second-moment signal.
    min_needed = _MEAN_MODEL_MIN_SAMPLES_PER_COLUMN * n_cols
    if diffs_arr.shape[0] < min_needed:
        keep = 2 if full_full else 1
        design_arr = design_arr[:, :, :keep]
        mean_columns = mean_columns[:keep]
    return _PairData(
        full_full=full_full,
        frame_index=frame_index,
        diffs=diffs_arr,
        mean_design=design_arr,
        mean_columns=tuple(mean_columns),
        cov_rows=cov_arr,
    )


def _fit_mean(pair: _PairData, counts: dict[int, int] | None) -> NDArray[np.float64]:
    """Fit the pair's mean-model coefficients by (weighted) least squares.

    Parameters:
        pair: The pair data.
        counts: Optional frame multiset (bootstrap replicate); ``None``
            weights every sample once.

    Returns:
        Coefficient vector aligned with ``pair.mean_columns`` (minimum-norm
        where the design is collinear, e.g. two members sharing one
        rotation angle).
    """
    n, _, k = pair.mean_design.shape
    if counts is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.array(
            [float(counts.get(f_idx, 0)) for f_idx in pair.frame_index], dtype=np.float64
        )
    mask = weights > 0
    if not mask.any():
        return np.zeros(k, dtype=np.float64)
    sqrt_w = np.sqrt(weights[mask])[:, None]
    rows = (pair.mean_design[mask] * sqrt_w[:, :, None]).reshape(-1, k)
    target = (pair.diffs[mask] * sqrt_w).reshape(-1)
    beta, _, _, _ = np.linalg.lstsq(rows, target, rcond=None)
    return np.asarray(beta, dtype=np.float64)


def _pair_targets(pair: _PairData, beta: NDArray[np.float64]) -> NDArray[np.float64]:
    """Centered second-moment targets per sample given mean coefficients.

    Parameters:
        pair: The pair data.
        beta: Mean-model coefficients.

    Returns:
        ``(n, 3)`` vech outer-product targets for full/full pairs,
        ``(n, 1)`` squared-residual targets otherwise.
    """
    residuals = pair.diffs - pair.mean_design @ beta
    if pair.full_full:
        r0 = residuals[:, 0]
        r1 = residuals[:, 1]
        return np.stack([r0 * r0, r0 * r1, r1 * r1], axis=1)
    r = residuals[:, 0]
    return (r * r)[:, None]


def _assemble(
    pairs: Sequence[_PairData],
    n_params: int,
    counts: dict[int, int] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Assemble the (A, y) system: fit means, center, stack equations.

    The pair mean models are (re-)fitted on exactly the frames present in
    ``counts`` (with multiplicity), so bootstrap replicates re-center on
    their own resample rather than reusing the full-cohort mean.

    Parameters:
        pairs: Pair data blocks.
        n_params: Width of the design matrix.
        counts: Optional frame multiset; ``None`` keeps every equation once.

    Returns:
        Design matrix ``A`` and target vector ``y``.
    """
    a_blocks: list[NDArray[np.float64]] = []
    y_blocks: list[NDArray[np.float64]] = []
    for pair in pairs:
        beta = _fit_mean(pair, counts)
        targets = _pair_targets(pair, beta)
        if counts is None:
            repeats = np.ones(len(pair.frame_index), dtype=np.intp)
        else:
            repeats = np.array([counts.get(f_idx, 0) for f_idx in pair.frame_index], dtype=np.intp)
        mask = repeats > 0
        if not mask.any():
            continue
        a_blocks.append(np.repeat(pair.cov_rows[mask], repeats[mask], axis=0).reshape(-1, n_params))
        y_blocks.append(np.repeat(targets[mask], repeats[mask], axis=0).reshape(-1))
    if not a_blocks:
        return np.zeros((0, n_params), dtype=np.float64), np.zeros(0, dtype=np.float64)
    return np.vstack(a_blocks), np.concatenate(y_blocks)


def solve_covariance_components(
    frames: Sequence[FrameSample],
    specs: Sequence[EstimatorSpec],
    *,
    pair_covariances: Sequence[PairKey] = (),
    mean_model: Literal['auto', 'constant'] = 'auto',
    practical_sv_ratio: float = 1e-6,
    n_bootstrap: int = 0,
    bootstrap_seed: int = 0,
) -> SolveResult:
    """Solve the covariance-components system for a cohort of frames.

    Every co-occurring instance pair contributes central second-moment
    equations (centered by the fitted per-pair mean model; see the module
    docstring for what the mean model does and does not absorb); the
    stacked linear system is solved by minimum-norm least squares.
    Identifiability is diagnosed from the design matrix's singular
    spectrum and reported per parameter -- a degenerate composition
    returns its null space explicitly rather than failing.

    Parameters:
        frames: Cohort of per-frame samples.
        specs: Estimator instances (unique names).
        pair_covariances: Instance-name pairs whose symmetric
            cross-covariance should be solved for instead of assumed zero.
            Declaring a pair adds unknowns; the cohort must be
            over-determined enough to carry them (check the returned
            identifiability scores).  Model restrictions: a full/full
            pair's matrix is image-frame constant (no rotating option),
            and a rank1-involving pair's scalar ``gamma`` assumes the
            projected cross-covariance is axis-independent
            (``S = gamma * I``).
        mean_model: ``'auto'`` (default) fits constant plus rotating-frame
            mean columns for rotating members; ``'constant'`` fits the
            constant term only (retained to demonstrate the aliasing an
            unmodeled rotating bias produces).
        practical_sv_ratio: Singular values below this fraction of the
            largest are treated as null directions.
        n_bootstrap: Number of frame-resampling bootstrap replicates for
            percentile confidence intervals (0 disables).  Each replicate
            re-fits the pair mean models on its own resample.
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
    pair_data: dict[PairKey, _PairData] = {}
    pair_means: dict[PairKey, NDArray[np.float64] | float] = {}
    pair_mean_models: dict[PairKey, dict[str, float]] = {}
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            built = _build_pair_data(frames, specs[i], specs[j], layout, mean_model)
            if built is None:
                continue
            key = (specs[i].name, specs[j].name)
            pair_data[key] = built
            raw_mean = built.diffs.mean(axis=0)
            pair_means[key] = raw_mean if built.full_full else float(raw_mean[0])
            beta_full = _fit_mean(built, None)
            pair_mean_models[key] = {
                label: float(beta_full[k]) for k, label in enumerate(built.mean_columns)
            }
    pairs = list(pair_data.values())
    a_mat, y_vec = _assemble(pairs, layout.n_params, None)
    if a_mat.shape[0] == 0:
        raise ValueError('no pair of instances ever co-occurs; nothing to solve')
    params, _, _, _ = np.linalg.lstsq(a_mat, y_vec, rcond=None)
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
    for key_str, sl in layout.group_slices.items():
        if layout.group_kind[key_str] == 'full':
            c11, c12, c22 = params[sl]
            covariances[key_str] = np.array([[c11, c12], [c12, c22]], dtype=np.float64)
        else:
            rank1_variances[key_str] = float(params[sl][0])
    pair_cov_out: dict[PairKey, NDArray[np.float64] | float] = {}
    for pair_key, sl in layout.pair_slices.items():
        vals = params[sl]
        if vals.size == 3:
            pair_cov_out[pair_key] = np.array(
                [[vals[0], vals[1]], [vals[1], vals[2]]], dtype=np.float64
            )
        else:
            pair_cov_out[pair_key] = float(vals[0])

    bootstrap_ci: dict[str, tuple[float, float]] = {}
    if n_bootstrap > 0:
        rng = np.random.default_rng(bootstrap_seed)
        n_frames = len(frames)
        samples = np.empty((n_bootstrap, layout.n_params), dtype=np.float64)
        for b in range(n_bootstrap):
            resample = rng.integers(0, n_frames, size=n_frames)
            b_counts: dict[int, int] = {}
            for idx in resample:
                b_counts[int(idx)] = b_counts.get(int(idx), 0) + 1
            a_b, y_b = _assemble(pairs, layout.n_params, b_counts)
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
        pair_mean_model=pair_mean_models,
        singular_values=np.asarray(full_sv, dtype=np.float64),
        condition_number=condition,
        null_space=np.asarray(null_space, dtype=np.float64),
        identifiability=identifiability,
        n_frames=len(frames),
        n_equations=int(a_mat.shape[0]),
        residual_rms=residual_rms,
        bootstrap_ci=bootstrap_ci,
    )

# mypy: ignore-errors
import pprint
from typing import Callable, Optional, cast

from imgdisp import ImageDisp
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg
from pdslogger import PdsLogger
import psfmodel.gaussian as gauss_psf
import scipy.fftpack as fftpack
import tkinter as tk

from nav.config import DEFAULT_CONFIG
from nav.support.image import unpad_array, pad_array_to_power_of_2
from nav.support.types import NDArrayFloatType


def _plot3d(*args: object, **kwargs: object) -> None:
    """Placeholder for debug 3D plotting (e.g. matplotlib)."""
    pass


# ==============================================================================
#
# CONSTANTS
#
# ==============================================================================

DEBUG_CORRELATE_PLOT = False
DEBUG_CORRELATE_IMGDISP = False


# ==============================================================================
#
# CORRELATION ROUTINES
#
# ==============================================================================


def correlate2d(
    image: NDArrayFloatType,
    model: NDArrayFloatType,
    normalize: bool = False,
    retile: bool = False,
) -> NDArrayFloatType:
    """Correlates the image with the model using the correlation theorem.

    Correlation is performed using the 'correlation theorem' that equates
    correlation with a Fourier Transform.

    Parameters:
        image: The image array.
        model: The model array to correlate against image.
        normalize: If True, normalize the correlation result to [-1,1].
        retile: If True, the resulting correlation matrix is shifted by 1/2 along
               each dimension so that (0,0) is now in the center pixel (shape[0]//2,shape[1]//2).

    Returns:
        The 2-D correlation matrix.
    """

    assert image.shape == model.shape

    if DEBUG_CORRELATE_IMGDISP:
        print('correlated2d: image, model')
        _ = ImageDisp(
            [image, model], canvas_size=(512, 512), enlarge_limit=10, auto_update=True
        )
        tk.mainloop()

    # Padding to a power of 2 makes FFT _much_ faster
    newimage, _ = pad_array_to_power_of_2(image)
    newmodel, padding = pad_array_to_power_of_2(model)

    image_fft = fftpack.fft2(newimage)
    model_fft = fftpack.fft2(newmodel)
    corr = np.real(fftpack.ifft2(image_fft * np.conj(model_fft)))

    if normalize:
        norm_amt = np.sqrt(np.sum(image**2) * np.sum(model**2))
        if norm_amt != 0:
            corr /= norm_amt

    if retile:
        # This maps (0,0) into (-y,-x) == (shape[0]//2, shape[1]//2)
        y = corr.shape[0] // 2
        x = corr.shape[1] // 2
        offset_image = np.zeros(corr.shape, image.dtype)
        offset_image[0:y, 0:x] = corr[-y:, -x:]
        offset_image[0:y, -x:] = corr[-y:, 0:x]
        offset_image[-y:, 0:x] = corr[0:y, -x:]
        offset_image[-y:, -x:] = corr[0:y, 0:x]
        corr = offset_image
        # corr = np.fft.fftshift(corr)

    corr = unpad_array(corr, padding)

    return corr


def corr_analyze_peak(corr, offset_u, offset_v, large_psf=True, blur=0):
    """Analyzes the correlation peak with a 2-D Gaussian fit.

    Parameters:
        corr: A 2-D correlation matrix with (0,0) located in the center pixel
             (shape[0]//2,shape[1]//2).
        offset_u: The U coordinate of the peak in corr.
        offset_v: The V coordinate of the peak in corr.
        large_psf: True to allow much larger PSF sizes.
        blur: The amount of blur applied to the image, used to determine a starting sigma.

    Returns:
        A dict containing analysis results including position, errors, and PSF parameters:
        'x', 'x_err', 'y', 'y_err', 'xy_err', 'xy_angle', 'xy_corr',
        'xy_rot_err_1' (long axis), 'xy_rot_err_2', 'xy_rot_angle',
        'sigma_1', 'sigma_1_err', 'sigma_2', 'sigma_2_err',
        'sigma_angle', 'sigma_angle_err', 'scale', 'scale_err',
        'base', 'base_err', 'leastsq_mesg', 'leastsq_ier',
        'subimg', 'psf', 'scaled_psf'.
        Note that 'sigma_1' will always be the long axis of the ellipse,
        and 'sigma_angle' will point along that long axis.
    """
    debug = False
    if large_psf:
        psf_sigma_size_list = [
            (5, 60),
            (7, 120),
            (9, 120),
            (11, 120),
            (13, 120),
            (15, 120),
            (17, 120),
            (21, 120),
            (25, 120),
            (31, 120),
            (35, 120),
            (3, 60),
        ]
    else:
        psf_sigma_size_list = [(5, 10), (7, 30), (9, 30), (3, 10)]
    for psf_size, sigma_max in psf_sigma_size_list:
        psf_size_x = psf_size_y = psf_size

        if False:
            subimg = corr[
                offset_v - psf_size_y // 2 : offset_v + psf_size_y // 2 + 1,
                offset_u - psf_size_x // 2 : offset_u + psf_size_x // 2 + 1,
            ]
            _plot3d(subimg)
            # plt.imshow(subimg)
            # plt.show()
        g = gauss_psf.GaussianPSF(
            sigma_angle=None, sigma_x_range=(0, sigma_max), sigma_y_range=(0, sigma_max)
        )
        if debug:
            print('TRYING', sigma_max, psf_size_x, psf_size_y)

        search_limit_x = psf_size_x // 2 + 1
        search_limit_y = psf_size_y // 2 + 1
        psf_ret = g.find_position(
            corr,
            (psf_size_y, psf_size_x),
            (offset_v, offset_u),
            bkgnd_degree=None,
            allow_nonzero_base=True,
            use_angular_params=False,
            search_limit=(search_limit_x, search_limit_y),
        )
        if psf_ret is None:
            if debug:
                print('psf_ret is None')
            continue
        if psf_ret[2]['leastsq_cov'] is None:
            if debug:
                print('leastsq_cov is None')
            continue
        if (
            abs(psf_ret[2]['x'] - 0.5) > search_limit_x / 2
            or abs(psf_ret[2]['y'] - 0.5) > search_limit_y / 2
        ):
            if debug:
                print('X Y beyond limit', psf_ret[2]['x'] - 0.5, psf_ret[2]['y'] - 0.5)
            continue

        break

    if psf_ret is None or psf_ret[2]['leastsq_cov'] is None:
        return None

    details = {}
    for key in (
        'subimg',
        'psf',
        'scaled_psf',
        'x',
        'x_err',
        'y',
        'y_err',
        'sigma_x',
        'sigma_x_err',
        'sigma_y',
        'sigma_y_err',
        'sigma_angle',
        'sigma_angle_err',
        'scale',
        'scale_err',
        'base',
        'base_err',
        'leastsq_cov',
        'leastsq_infodict',
        'leastsq_mesg',
        'leastsq_ier',
    ):
        new_key = key.replace('sigma_x', 'sigma_1')
        new_key = new_key.replace('sigma_y', 'sigma_2')
        details[new_key] = psf_ret[2][key]

    # The PSF is centered with (0.5,0.5) as the center of a pixel, but we want
    # (0,0) to be the center of a pixel
    details['x'] = details['x'] - 0.5
    details['y'] = details['y'] - 0.5

    details['sigma_angle'] = details['sigma_angle'] % np.pi
    details['sigma_1'] = abs(details['sigma_1'])
    details['sigma_2'] = abs(details['sigma_2'])

    if details['sigma_1'] > sigma_max * 0.95 or details['sigma_2'] > sigma_max * 0.95:
        details['sigma_1'] = None
        details['sigma_1_err'] = None
        details['sigma_2'] = None
        details['sigma_2_err'] = None
        details['sigma_angle'] = None
        details['sigma_angle_err'] = None
    else:
        if details['sigma_1'] < details['sigma_2']:
            # Make angle_1 always be the long axis
            details['sigma_1'], details['sigma_2'] = (
                details['sigma_2'],
                details['sigma_1'],
            )
            details['sigma_angle'] = (details['sigma_angle'] + np.pi / 2) % np.pi

    if details['x_err'] > 100 or details['y_err'] > 100:  # Magic constant
        details['x_err'] = None
        details['y_err'] = None
        details['xy_err'] = None
        details['xy_angle'] = None
        details['xy_corr'] = None
        details['xy_rot_angle'] = None
        details['xy_rot_err_1'] = None
        details['xy_rot_err_2'] = None
    else:
        details['xy_err'] = details['leastsq_cov'][1, 0]
        details['xy_angle'] = np.pi / 2 - 0.5 * np.arctan(
            2 * details['xy_err'] / (details['x_err'] ** 2 - details['y_err'] ** 2)
        )
        details['xy_corr'] = details['xy_err'] / details['x_err'] / details['y_err']
        details['xy_rot_angle'] = details['xy_angle']
        eigv, _ = linalg.eig(details['leastsq_cov'][:2, :2])
        eigv = np.real(eigv)
        eigv = np.clip(eigv, 0.0, None)
        details['xy_rot_err_1'] = np.max(np.sqrt(eigv))
        details['xy_rot_err_2'] = np.min(np.sqrt(eigv))
        if details['x_err'] > details['y_err']:
            details['xy_rot_angle'] = (details['xy_rot_angle'] + np.pi / 2) % np.pi

    # http://www.cs.utah.edu/~tch/CS4300/resources/refs/ErrorEllipses.pdf
    if False:
        if True:
            if details['sigma_angle'] is not None:
                print('GAUSS ANGLE', np.degrees(details['sigma_angle']))
            if details['xy_angle'] is not None:
                print('COV ANGLE', np.degrees(details['xy_angle']))
            if details['xy_rot_angle'] is not None:
                print('XYROT ANGLE', np.degrees(details['xy_rot_angle']))
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(details)
        psf = details['subimg']
        the_max = np.max(psf)
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(221)
        contour_levels = [x * 0.05 * the_max for x in range(20)]
        ax.contour(psf[::-1], levels=contour_levels)
        ax.set_title('Image')
        psf = details['scaled_psf']
        the_max = np.max(psf)
        ax = fig.add_subplot(222)
        ax.contour(psf[::-1], levels=contour_levels)
        if details['sigma_angle'] is None:
            ax.set_title('Scaled PSF')
        else:
            ax.set_title(
                'Scaled PSF (%.2fx%.2f @ %.2f)'
                % (
                    details['sigma_1'],
                    details['sigma_1'],
                    np.degrees(details['sigma_angle']),
                )
            )
        ax = fig.add_subplot(223)
        ax.imshow(psf)
        ax.set_title('Correlation')
        ax = fig.add_subplot(224, projection='3d')
        _plot3d(
            details['subimg'],
            details['scaled_psf'],
            ax=ax,
            title='PSF and Image',
            show=False,
        )
        plt.tight_layout()
        plt.show()

    return details


def corr_log_xy_err(logger, psf_details):
    """Logs correlation position and error information to the provided logger.

    Parameters:
        logger: The logger object to write information to.
        psf_details: Dictionary containing PSF analysis results from corr_analyze_peak.
    """

    if psf_details['xy_rot_err_1'] is None:
        logger.info(
            '  dX %.3f dY %.3f (uncertainty failed)'
            % (psf_details['x'], psf_details['y'])
        )
    else:
        logger.info(
            '  dX %.3f dY %.3f (%.3fx%.3f @ %.3f)',
            psf_details['x'],
            psf_details['y'],
            psf_details['xy_rot_err_1'],
            psf_details['xy_rot_err_2'],
            np.degrees(psf_details['xy_rot_angle']),
        )
    logger.info('    %s', psf_details['leastsq_mesg'])


def corr_xy_err_to_str(offset, uncertainty, extra=None, extra_args=None):
    """Converts correlation offset and uncertainty information to a formatted string.

    Parameters:
        offset: A tuple (x,y) containing the offset coordinates.
        uncertainty: A tuple (xy_rot_err_1, xy_rot_err_2, xy_rot_angle) containing uncertainty values.
        extra: Optional format string to append additional information.
        extra_args: Optional arguments for the extra format string.

    Returns:
        A formatted string representation of the offset and uncertainty.
    """

    # offset is (x,y)
    # uncertainty is (xy_rot_err_1, xy_rot_err_2, xy_rot_angle)
    if offset is None or (offset[0] is None and offset[1] is None):
        return 'N/A'
    if extra is None:
        extra = ''
    else:
        extra = ' ' + (extra % extra_args)
    ret = '%.2f,%.2f' % offset
    if uncertainty is None or (
        uncertainty[0] is None and uncertainty[1] is None and uncertainty[2] is None
    ):
        return ret + extra
    err1 = 'UNK'
    if uncertainty[0] is not None:
        err1 = '%.2f' % uncertainty[0]
    err2 = 'UNK'
    if uncertainty[1] is not None:
        err2 = '%.2f' % uncertainty[1]
    angle = 'UNK'
    if uncertainty[2] is not None:
        angle = '%.2f' % np.degrees(uncertainty[2])
    return ret + ' (' + err1 + 'x' + err2 + ' @ ' + angle + ')' + extra


def corr_psf_xy_err_to_str(offset, psf_details, extra=None, extra_args=None):
    """Converts correlation offset and PSF details to a formatted string.

    Parameters:
        offset: A tuple (x,y) containing the offset coordinates.
        psf_details: Dictionary containing PSF analysis results from corr_analyze_peak.
        extra: Optional format string to append additional information.
        extra_args: Optional arguments for the extra format string.

    Returns:
        A formatted string representation of the offset and uncertainty from PSF details.
    """

    if psf_details is None:
        return corr_xy_err_to_str(offset, (None, None, None), extra, extra_args)
    return corr_xy_err_to_str(
        offset,
        (
            psf_details['xy_rot_err_1'],
            psf_details['xy_rot_err_2'],
            psf_details['xy_rot_angle'],
        ),
        extra,
        extra_args,
    )


def _find_correlated_offset(
    corr: NDArrayFloatType,
    search_size_min: tuple[int, int],
    search_size_max: tuple[int, int],
    max_offsets: int,
    peak_margin: int,
    logger: PdsLogger,
):
    """Finds the offset that best aligns an image and a model given the correlation.

    The offset is found by looking for the maximum correlation value within
    the given search range. Multiple offsets may be returned, in which case
    each peak and the area around it is eliminated from future consideration
    before the next peak is found.

    Parameters:
        corr: A 2-D correlation matrix with (0,0) located in the center pixel
              (shape[0]//2,shape[1]//2).
        search_size_min: Minimum search range as (min_v, min_u) from offset zero.
        search_size_max: Maximum search range as (max_v, max_u) from offset zero.
        max_offsets: The maximum number of offsets to return.
        peak_margin: The number of correlation pixels around a peak to remove from future
                    consideration before finding the next peak.
        logger: Logger object for recording diagnostic information.

    Returns:
        List of tuples, each containing:
        - (offset_v, offset_u): The offset in the V and U directions.
        - peak_value: The correlation value at the peak in the range [-1,1].
        - peak_data: A tuple containing the correlation array and U,V offset inside
                    that array suitable for passing to corr_analyze_peak.
    """

    search_size_min_v, search_size_min_u = search_size_min
    search_size_max_v, search_size_max_u = search_size_max

    logger.debug(
        f'Search V {search_size_min_v} to {search_size_max_v}, '
        f'U {search_size_min_u} to {search_size_max_u} #OFFSETS '
        f'{max_offsets} PEAKMARGIN {peak_margin}'
    )

    if not (
        0 <= search_size_min_v <= search_size_max_v
        and 0 <= search_size_min_u <= search_size_max_u
        and 0 <= search_size_max_v <= corr.shape[0] // 2
        and 0 <= search_size_max_u <= corr.shape[1] // 2
    ):
        raise ValueError('Bad parameters to _find_correlated_offset')

    # Extract a slice from the correlation matrix that is the maximum search size and then
    # make a "hole" in the center to represent the minimum search size.
    # Note: SLICE is a Python built-in!
    slyce = corr[
        corr.shape[0] // 2 - search_size_max_v : corr.shape[0] // 2
        + search_size_max_v
        + 1,
        corr.shape[1] // 2 - search_size_max_u : corr.shape[1] // 2
        + search_size_max_u
        + 1,
    ]
    slyce = slyce.copy()

    global_min = np.min(slyce)

    if search_size_min_u != 0 and search_size_min_v != 0:
        slyce[
            slyce.shape[0] // 2 - search_size_min_v + 1 : slyce.shape[0] // 2
            + search_size_min_v,
            slyce.shape[1] // 2 - search_size_min_u + 1 : slyce.shape[1] // 2
            + search_size_min_u,
        ] = global_min

    # Iteratively search for the next peak.
    ret_list: list[
        tuple[tuple[int, int], float, tuple[NDArrayFloatType, int, int]] | None
    ] = []
    all_offset_u = []
    all_offset_v = []

    while len(ret_list) != max_offsets:
        peak_num = len(ret_list) + 1
        peak = np.where(slyce == slyce.max())

        if DEBUG_CORRELATE_PLOT:
            print(f'_find_correlated_offset: peaks {peak}')
            plt.jet()
            plt.imshow(slyce, interpolation='none')
            # plt.contour(slyce)
            plt.plot(
                (search_size_max_u, search_size_max_u), (0, 2 * search_size_max_v), 'k'
            )
            plt.plot(
                (0, 2 * search_size_max_u), (search_size_max_v, search_size_max_v), 'k'
            )
            if len(peak[0]) == 1:
                print('Including sole peak')
                plt.plot(peak[1], peak[0], 'wo')
            else:
                print('Multiple peaks, not including any')
            x_n_ticks = 5
            x_tick_step = max(int(search_size_max_u / x_n_ticks), 1)
            x_ticks = list(
                range(
                    -x_tick_step * x_n_ticks, x_tick_step * x_n_ticks + 1, x_tick_step
                )
            )
            plt.xticks(
                [x + search_size_max_u for x in x_ticks], [str(x) for x in x_ticks]
            )
            y_n_ticks = 5
            y_tick_step = max(int(search_size_max_v / y_n_ticks), 1)
            y_ticks = list(
                range(
                    -y_tick_step * y_n_ticks, y_tick_step * y_n_ticks + 1, y_tick_step
                )
            )
            plt.yticks(
                [x + search_size_max_v for x in y_ticks], [str(x) for x in y_ticks]
            )
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.tight_layout()
            # plt.savefig('output.png')
            plt.show()

        if DEBUG_CORRELATE_IMGDISP > 1:
            print('_find_correlated_offset: slyce')
            toplevel = tk.Tk()
            _ = ImageDisp(
                [slyce],
                parent=toplevel,
                canvas_size=(512, 512),
                allow_enlarge=True,
                enlarge_limit=10,
                auto_update=True,
            )
            tk.mainloop()

        if len(peak[0]) != 1:
            logger.debug(f'Peak #{peak_num} - No unique peak found - aborting')
            break

        peak_v = peak[0][0]
        peak_u = peak[1][0]
        offset_v = peak_v - search_size_max_v  # Compensate for slice location
        offset_u = peak_u - search_size_max_u
        peak_val = slyce[peak_v, peak_u]

        logger.debug(
            f'Peak #{peak_num} - Trial offset {offset_v},{offset_u} CORR {peak_val:.7f}'
        )

        if peak_val <= 0:
            logger.debug(f'Peak #{peak_num} - Correlation value is negative - aborting')
            break

        all_offset_v.append(offset_v)
        all_offset_u.append(offset_u)

        if peak_num < max_offsets:
            # Eliminating this peak from future consideration if we're going
            # to be looping again.
            min_u = np.clip(
                offset_u - peak_margin + slyce.shape[1] // 2, 0, slyce.shape[1] - 1
            )
            max_u = np.clip(
                offset_u + peak_margin + slyce.shape[1] // 2, 0, slyce.shape[1] - 1
            )
            min_v = np.clip(
                offset_v - peak_margin + slyce.shape[0] // 2, 0, slyce.shape[0] - 1
            )
            max_v = np.clip(
                offset_v + peak_margin + slyce.shape[0] // 2, 0, slyce.shape[0] - 1
            )
            slyce[min_v : max_v + 1, min_u : max_u + 1] = np.min(slyce)

        if abs(offset_v) == search_size_max_v or abs(offset_u) == search_size_max_u:
            logger.debug(f'Peak #{peak_num} - Offset at edge of search area - BAD')
            # Go ahead and store a None result. This way we will eventually
            # hit max_offsets and exit. Otherwise we could be looking for
            # a very long time.
            ret_list.append(None)
            continue

        for i in range(len(all_offset_u)):
            if (
                (
                    offset_u == all_offset_u[i] - peak_margin - 1
                    or offset_u == all_offset_u[i] + peak_margin + 1
                )
                and offset_v - peak_margin <= all_offset_v[i] <= offset_v + peak_margin
            ) or (
                (
                    offset_v == all_offset_v[i] - peak_margin - 1
                    or offset_v == all_offset_v[i] + peak_margin + 1
                )
                and offset_u - peak_margin <= all_offset_u[i] <= offset_u + peak_margin
            ):
                logger.debug(
                    f'Peak #{peak_num} - Offset at edge of previous blackout area - BAD'
                )
                ret_list.append(None)
                break
        else:
            ret_list.append(
                (
                    (offset_v, offset_u),
                    peak_val,
                    (
                        corr,
                        offset_u + corr.shape[1] // 2,
                        offset_v + corr.shape[0] // 2,
                    ),
                )
            )

    # Now remove all the Nones from the returned list.
    while None in ret_list:
        ret_list.remove(None)

    return ret_list


def find_correlation_and_offset(
    image: NDArrayFloatType,
    model: NDArrayFloatType,
    *,
    search_size_min: int | tuple[int, int] = 0,
    search_size_max: Optional[int | tuple[int, int]] = None,
    max_offsets: int = 1,
    peak_margin: int = 3,
    extfov_margin_vu: tuple[int, int] = (0, 0),
    image_filter: Optional[Callable] = None,
    model_filter: Optional[Callable] = None,
    logger: Optional[PdsLogger] = None,
) -> list[tuple[tuple[int, int], float, tuple[NDArrayFloatType, int, int]]]:
    """Finds the offset that best aligns an image and a model through correlation analysis.

    Parameters:
        image: The image array to analyze.
        model: The model array to correlate against the image.
        search_size_min: Minimum search range from offset zero. Can be a single int or (min_v, min_u).
        search_size_max: Maximum search range from offset zero. Can be a single int or (max_v, max_u).
        max_offsets: Maximum number of offset candidates to return.
        peak_margin: Number of correlation pixels around a peak to exclude when finding next peaks.
        extfov_margin_vu: The amount (V,U) the image and model have been extended on each side.
                         Used to search variations where model margins are shifted onto the image.
        image_filter: Optional filter function to apply to the image before correlation.
        model_filter: Optional filter function to apply to each sub-model before correlation.
        logger: Optional logger object for recording diagnostic information.

    Returns:
        List of tuples, each containing:
        - (offset_v, offset_u): The offset in the V and U directions.
        - peak_value: The correlation value at the peak in the range [-1,1].
        - peak_data: A tuple containing the correlation array and U,V offset inside
                   that array suitable for passing to corr_analyze_peak.
    """

    if logger is None:
        logger = DEFAULT_CONFIG.logger

    if np.shape(search_size_min) == ():
        search_size_min_v = cast(int, search_size_min)
        search_size_min_u = cast(int, search_size_min)
    else:
        search_size_min_v, search_size_min_u = cast(tuple[int, int], search_size_min)

    if search_size_max is None:
        search_size_max = extfov_margin_vu
    if np.shape(search_size_max) == ():
        search_size_max_v = cast(int, search_size_max)
        search_size_max_u = cast(int, search_size_max)
    else:
        search_size_max_v, search_size_max_u = cast(tuple[int, int], search_size_max)

    extend_fov_v, extend_fov_u = extfov_margin_vu
    orig_image_size_v = image.shape[0] - extend_fov_v * 2
    orig_image_size_u = image.shape[1] - extend_fov_u * 2

    # Normalize both the image and the model to 1
    # imax = float(np.max(image))
    # mmax = float(np.max(model))
    # if imax > 0:
    #     image = image / imax
    # if mmax > 0:
    #     model = model / mmax
    image = image / np.max(image)
    model = model / np.max(model)

    # We dramatically increase the absolute values in the image and model because
    # otherwise very sparse and very dim images can just turn out as zero after
    # correlation.
    image = image * 1e10  # TODO Remove hard-coded value
    model = model * 1e10
    ret_list = []

    # If the image has been extended, try up to nine combinations of
    # sub-models if the model shifted onto the image from each direction.
    # The current implementation falls apart if the extend amount is not
    # the same as the maximum search limit. TODO
    extend_fov_u_list = [extend_fov_u]
    extend_fov_v_list = [extend_fov_v]
    # if nav.config.CORR_ALLOW_SUBMODELS:
    #     if extend_fov_v and search_size_max_v == extend_fov_v:
    #         extend_fov_v_list = [0, extend_fov_v, 2*extend_fov_v]
    #         assert search_size_max_v == extend_fov_v
    #     if extend_fov_u and search_size_max_u == extend_fov_u:
    #         extend_fov_u_list = [0, extend_fov_u, 2*extend_fov_u]
    #         assert search_size_max_u == extend_fov_u

    if image.shape != model.shape:
        raise ValueError(
            f'Image ({image.shape}) and model ({model.shape}) must be the same shape'
        )

    # Get the original image and maybe filter it.
    sub_image = image[
        extend_fov_v : extend_fov_v + orig_image_size_v,
        extend_fov_u : extend_fov_u + orig_image_size_u,
    ]
    if image_filter is not None:
        sub_image = image_filter(sub_image)

    new_ret_list = []

    # Iterate over each chosen sub-model and correlate it with the image.
    for start_u in extend_fov_u_list:
        for start_v in extend_fov_v_list:
            logger.debug(
                'Correlating model slice V %d:%d U %d:%d',
                start_v - extend_fov_v,
                start_v - extend_fov_v + orig_image_size_v - 1,
                start_u - extend_fov_u,
                start_u - extend_fov_u + orig_image_size_u - 1,
            )
            sub_model = model[
                start_v : start_v + orig_image_size_v,
                start_u : start_u + orig_image_size_u,
            ]
            if not np.any(sub_model):
                continue

            if model_filter is not None:
                sub_model = model_filter(sub_model)
            corr = correlate2d(sub_image, sub_model, normalize=True, retile=True)
            # plt.figure()
            # plt.imshow(sub_image)
            # plt.figure()
            # plt.imshow(sub_model)
            # plt.figure()
            # plt.imshow(corr)
            # plt.show()
            ret_list = _find_correlated_offset(
                corr,
                (search_size_min_v, search_size_min_u),
                (search_size_max_v, search_size_max_u),
                max_offsets,
                peak_margin,
                logger,
            )

            # Iterate over each returned offset and calculate what the offset actually is
            # based on the model shift amount. Throw away any results that are outside of
            # the given search limits. Note that if the offset is exactly equal to the
            # search limit, we throw it away too, because this almost always indicates a
            # bad result.
            for offset, peak, details in ret_list:
                if offset is not None:
                    new_offset_v = offset[0] - start_v + extend_fov_v
                    new_offset_u = offset[1] - start_u + extend_fov_u
                    if (
                        abs(new_offset_v) < search_size_max_v
                        and abs(new_offset_u) < search_size_max_u
                    ):
                        logger.debug(
                            f'Adding possible offset {new_offset_v},{new_offset_u}'
                        )
                        new_ret_list.append(
                            ((new_offset_v, new_offset_u), peak, details)
                        )
                    else:
                        logger.debug(
                            f'Offset beyond search limits {new_offset_v},{new_offset_u}'
                        )

    # Sort the offsets in descending order by correlation peak value.
    # Truncate the (possibly longer) list to the maximum number of requested
    # offsets.
    new_ret_list.sort(key=lambda x: -x[1])
    new_ret_list = new_ret_list[:max_offsets]

    if len(new_ret_list) == 0:
        logger.debug('No offsets to return')
    else:
        for i, ((offset_v, offset_u), peak, details) in enumerate(new_ret_list):
            logger.debug(
                f'Returning Peak {i + 1} offset {offset_v},{offset_u} CORR {peak:.7f}'
            )

    return new_ret_list

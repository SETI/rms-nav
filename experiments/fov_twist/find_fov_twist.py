import argparse
from functools import partial
import json
import multiprocessing
import sys
from typing import Any, Optional

import filecache
from filecache import FCPath
import matplotlib.pyplot as plt
import numpy as np

from psfmodel import PSF, GaussianPSF
from starcat import Star

from nav.config import DEFAULT_CONFIG, DEFAULT_LOGGER
from nav.inst import inst_name_to_class
from nav.nav_master import NavMaster
from nav.obs import ObsSnapshot



# VGISS_NAC = {
#     'urls': [
#         '${PDS3_HOLDINGS_DIR}/volumes/VGISS_5xxx/VGISS_5209/DATA/C20326XX/C2032609_GEOMED.IMG',
#         '${PDS3_HOLDINGS_DIR}/volumes/VGISS_6xxx/VGISS_6111/DATA/C34796XX/C3479608_GEOMED.IMG',
#         '${PDS3_HOLDINGS_DIR}/volumes/VGISS_8xxx/VGISS_8210/DATA/C12050XX/C1205043_GEOMED.IMG',
#         '${PDS3_HOLDINGS_DIR}/volumes/VGISS_8xxx/VGISS_8210/DATA/C12051XX/C1205117_GEOMED.IMG',

#     ],
#     'inst_id': 'vgiss',
#     'nstars': 15,
#     'psf_size': (15, 15),
#     'perc': 80,
#     'clip': 40,
#     'peak_factor': 2,
#     'min_twist': -.4,
#     'max_twist': .4,
#     'peak_factor': 2,
# }

# VGISS_WAC = {
#     'urls': [
#         '${PDS3_HOLDINGS_DIR}/volumes/VGISS_8xxx/VGISS_8210/DATA/C12051XX/C1205111_GEOMED.IMG',
#         '${PDS3_HOLDINGS_DIR}/volumes/VGISS_5xxx/VGISS_5209/DATA/C20326XX/C2032609_GEOMED.IMG',
#         '${PDS3_HOLDINGS_DIR}/volumes/VGISS_8xxx/VGISS_8210/DATA/C12050XX/C1205037_GEOMED.IMG',

#     ],
#     'inst_id': 'vgiss',
#     'nstars': 15,
#     'psf_size': (15, 15),
#     'perc': 100,
#     'clip': 40,
#     'peak_factor': 2,
#     'min_twist': .2,
#     'max_twist': .6,
# }


def _analyze_star(star: Star,
                  u: float,
                  v: float,
                  *,
                  twist_config: dict[str, Any],
                  img: np.ndarray,
                  psf: PSF,
                  uv_list: list[tuple[float, float, float, float]],
                  verbose: Optional[bool] = False) -> bool:
    """Filter and analyze one star.

    Parameters:
        star: The star to analyze.
        u: The predicted u coordinate.
        v: The predicted v coordinate.
        twist_config: The star field configuration.
        img: The image data.
        psf: The PSF to use.
        uv_list: The list of u, v, u_diff, v_diff for the stars. This function will
            append to this list of the star passes the filters.
        verbose: Whether to print verbose output.

    Returns:
        True if the star is valid, False otherwise.
    """

    psf_size = twist_config['psf_size']
    clip = twist_config['image_margin_pixels']
    psf_peak_factor = twist_config['psf_peak_factor']

    # Intentionally don't use psf/2 because we want to add a little extra
    # slop near the edge of the image
    if (u < psf_size[1] or u > img.shape[1] - psf_size[1] or
        v < psf_size[0] or v > img.shape[0] - psf_size[0]):
        if verbose:
            print(f'Star {star.unique_number} VMAG {star.vmag} outside image')
        return False
    ret = psf.find_position(img, psf_size, (v, u), search_limit=(2.5, 2.5))
    if ret is None:
        if verbose:
            print(f'Star {star.unique_number} VMAG {star.vmag} failed')
        return False
    opt_v, opt_u, metadata = ret
    if verbose:
        print(f'Star {star.unique_number} VMAG {star.vmag} Searched at {u:.3f}, {v:.3f} '
              f'found at {opt_u:.3f}, {opt_v:.3f}')
    if opt_u < clip or opt_u > img.shape[1] - clip or opt_v < clip or opt_v > img.shape[0] - clip:
        if verbose:
            print(f'Star {star.unique_number} VMAG {star.vmag} clipped')
        return False
    diff_u = float(opt_u-u)
    diff_v = float(opt_v-v)
    if abs(diff_u) > 1.5 or abs(diff_v) > 1.5:
        if verbose:
            print(f'Star {star.unique_number} VMAG {star.vmag} offset {diff_u}, {diff_v} too large')
        return False
    bkgnd = np.median(metadata['subimg'])
    psf_u = int(diff_u + psf_size[1]//2)
    psf_v = int(diff_v + psf_size[0]//2)
    peak = np.mean(metadata['subimg'][psf_v-1:psf_v+2, psf_u-1:psf_u+1])
    if peak < bkgnd * psf_peak_factor:
        if verbose:
            print(f'Star {star.unique_number} VMAG {star.vmag} peak {peak} not bright enough, '
                  f'bkgnd {bkgnd}')
        return False
    if verbose:
        print(f'Star {star.unique_number} VMAG {star.vmag} peak {peak} bkgnd {bkgnd}')
    uv_list.append((u, v, diff_u, diff_v))
    if verbose:
        print(f'Star {star.unique_number} VMAG {star.vmag} offset {diff_u}, {diff_v} OK')
    return True


def find_error_one_twist(obs: ObsSnapshot,
                         img_name: str,
                         *,
                         twist_config: dict[str, Any],
                         twist: float,
                         verbose: bool = False,
                         show_plot: bool = False) -> tuple[float, tuple[float, float],
                                                           tuple[float, float], list[float],
                                                           list[float], list[float], list[float],
                                                           float, NavMaster]:
    """Analyze one trial twist and return the error and related information.

    Parameters:
        orig_obs: The original un-navigated observation.
        img_name: The name of the image.
        psf_size: The size of the PSF.
        error_clip_percentile: The percentile of the error to clip.
        twist: The twist to analyze.
        psf_peak_factor: The peak factor of the PSF.
        image_margin_pixels: The margin pixels of the image.
        verbose: Whether to print verbose output.

    Returns: XXX
        A tuple containing:

            - the error
            - the original offset
            - the best offset
            - the list of u positions
            - the list of v positions
            - the list of u differences
            - the list of v differences
            - the percentile of the error
            - the navigated NavMaster object
    """

    if verbose:
        print(f'Checking twist: {twist}')

    obs = obs.navigate((0, 0, np.radians(twist)))
    nm = NavMaster(obs, nav_models=['stars'])
    nm.navigate()

    min_stars = twist_config['min_stars']
    error_clip_percentile = twist_config['error_clip_percentile']

    offset = nm.final_offset
    if offset is None:
        if verbose:
            print('No offset found')
        return None
    if verbose:
        print(f'Offset: {offset}')
    if len(nm.star_models) == 0:
        if verbose:
            print('No star models found')
        return None

    img = nm.obs.data.astype(np.float64)
    star_model = nm.star_models[0]
    star_list = star_model.metadata['star_list']

    # First pass - find the mean of the offsets of each star
    # This allows us to make a more accurate offset

    # We intentionally use a larger PSF here than would be specified by the
    # instrument because we want to handle the stars being a little out of
    # position, slightly blurred by motion, overexposed, etc.
    psf = GaussianPSF(sigma=2.0)

    uv_list = []

    for star in star_list:
        u = star.u - offset[1]
        v = star.v - offset[0]
        _analyze_star(star, u, v, twist_config=twist_config, img=img, psf=psf,
                      uv_list=uv_list, verbose=verbose)

    if len(uv_list) < min_stars:
        if verbose:
            print('Not enough valid stars found')
        return None

    u_diff_list = [uv[2] for uv in uv_list]
    v_diff_list = [uv[3] for uv in uv_list]

    # We use median here instead of mean because if one of the stars wasn't found,
    # it can cause a large error in the mean (throwing off all future calculations)
    # but won't affect the median.
    median_diff_u = np.median(u_diff_list)
    median_diff_v = np.median(v_diff_list)
    if verbose:
        print()
        print(f'Median delta offset {median_diff_u}, {median_diff_v}')

    # Second pass - find the offset of each star relative to the median offset

    uv_list = []

    new_offset = (offset[0] - median_diff_v, offset[1] - median_diff_u)

    if verbose:
        print(f'Adjusted offset: {new_offset[1]}, {new_offset[0]}')

    for star in star_list:
        u = star.u - new_offset[1]
        v = star.v - new_offset[0]
        _analyze_star(star, u, v, twist_config=twist_config, img=img, psf=psf,
                      uv_list=uv_list, verbose=verbose)

    if len(uv_list) < min_stars:
        if verbose:
            print('Not enough valid stars found')
        return None

    # Calculate the magnitude of the errors

    u_arr = np.array([uv[0] for uv in uv_list])
    v_arr = np.array([uv[1] for uv in uv_list])
    u_diff_arr = np.array([uv[2] for uv in uv_list])
    v_diff_arr = np.array([uv[3] for uv in uv_list])
    mag_arr = np.sqrt(u_diff_arr**2 + v_diff_arr**2)

    # Eliminate stars with bad magnitudes relative to the rest.
    # We just use a simple percentile to do this, throwing away the stars with
    # the largest errors.
    mag_perc = np.percentile(mag_arr, error_clip_percentile)
    good_stars = mag_arr <= mag_perc
    mag_arr = mag_arr[good_stars]
    u_arr = u_arr[good_stars]
    v_arr = v_arr[good_stars]
    u_diff_arr = u_diff_arr[good_stars]
    v_diff_arr = v_diff_arr[good_stars]

    # Calculate the angle of the errors.
    # A perfect star field should have u_diff_arr relative to u_arr and v_diff_arr
    # relative to v_arr always point towards the center of the image.
    # Calculate the angle between these vectors and the perfect vector to the center.
    u_diff_arr_perfect = img.shape[1]/2 - u_arr
    v_diff_arr_perfect = img.shape[0]/2 - v_arr
    angle_arr_perfect = ((np.arctan2(v_diff_arr_perfect, u_diff_arr_perfect) + np.pi) %
                         (2 * np.pi))
    angle_arr = (np.arctan2(v_diff_arr, u_diff_arr) + np.pi) % (2 * np.pi)
    # Diff the angles always getting a number between -pi and pi
    angle_diff_arr = angle_arr - angle_arr_perfect
    # Compute the final error as a combination of the magnitude of the error vector
    # and the difference in the angle. Normalize for the number of stars.
    error = np.sqrt(np.sum((mag_arr*angle_diff_arr)**2)) / len(mag_arr)

    mean_perc = np.mean(mag_arr)
    median_perc = np.median(mag_arr)
    if verbose:
        print(f'{error_clip_percentile}th percentile absolute star position error: {mag_perc}, '
              f'Mean: {mean_perc}, Median: {median_perc}, Error: {error}')

    if show_plot:
        plot_with_arrows(u_arr=u_arr, v_arr=v_arr, u_diff_arr=u_diff_arr, v_diff_arr=v_diff_arr,
                         mag_perc=mag_perc, nm=nm, twist=twist, offset=new_offset,
                         img_name=img_name)

    return error, offset, new_offset, u_arr, v_arr, u_diff_arr, v_diff_arr, mag_perc, nm


def plot_with_arrows(*,
                     u_arr: np.ndarray,
                     v_arr: np.ndarray,
                     u_diff_arr: np.ndarray,
                     v_diff_arr: np.ndarray,
                     mag_perc: float,
                     nm: NavMaster,
                     twist: float,
                     offset: tuple[float, float],
                     img_name: str,
                     plot_path: Optional[FCPath] = None) -> None:
    """Plot an image with arrows and optionally save it to a file.

    Parameters:
        u_arr: The u coordinates of the stars.
        v_arr: The v coordinates of the stars.
        u_diff_arr: The u differences between predicted and actual positions.
        v_diff_arr: The v differences between predicted and actual positions.
        mag_perc: The selected maximum error magnitude to show an arrow.
        nm: The NavMaster object. This is used to create the overlay.
        twist: The twist used to navigate the image.
        offset: The offset used to navigate the image.
        img_name: The name of the image.
        plot_path: The path to save the plot to. If not specified, the plot will
            be shown interactively.
    """

    plt.figure(figsize=(12, 12))
    overlay = nm.create_overlay()
    plt.imshow(overlay)
    for u, v, u_diff, v_diff in zip(u_arr, v_arr, u_diff_arr, v_diff_arr):
        mag = np.sqrt(u_diff**2 + v_diff**2)
        if mag <= mag_perc:
            plt.quiver(u, v, u_diff*100, v_diff*100,
                       angles='xy', scale_units='xy', scale=1,
                       color='#80ff80', alpha=0.6)
    plt.title(f'{img_name} (Twist {twist:.5f}, Offset {offset[1]:.3f}, {offset[0]:.3f})')
    if plot_path is None:
        plt.show()
    else:
        local_path = plot_path.get_local_path()
        plt.savefig(local_path)
        plot_path.upload()
    plt.close()


def optimize_one_image(url: str,
                       twist_config: dict[str, Any],
                       dataset: str,
                       results_path: FCPath,
                       *,
                       verbose: bool = False,
                       show_plots: bool = False
                       ) -> tuple[str, float, float, tuple[float, float],
                                  tuple[float, float], float]:
    """Optimize the twist for one image given star field data.

    Parameters:
        url: The URL of the image to optimize. It can include environment variables and
            can be retrieved from any source supported by FCPath.
        twist_config: The star field data from the config file.
        dataset: The dataset name used for storing results.
        results_path: The path to store the results.
        verbose: Whether to print verbose output.
        show_plots: Whether to show plots interactively during optimization. Used
            for debugging.

    Returns: XXX
        A tuple containing the image URL, the best twist, the rotation error, the original offset,
        the best offset, and the best error.
    """

    img_name = url.split('/')[-1].split('.')[0]
    log_dir = results_path / f'{dataset}_logs'
    plot_dir = results_path / f'{dataset}_plots'
    log_path = log_dir / f'{img_name}.log'

    with log_path.open('w') as log_fp:
        inst_class = inst_name_to_class(twist_config['inst_id'])
        try:
            # fast_distortion is supported by some instrument classes and ignored
            # by others
            orig_obs = inst_class.from_file(FCPath(url).expandvars(),
                                            fast_distortion=True)
        except Exception as e:
            print(f'Error reading image {url}: {e}')
            log_fp.write(f'Error reading image {url}: {e}\n')
            return None

        divisions = twist_config['twist_divisions']
        min_twist = twist_config['min_twist']
        max_twist = twist_config['max_twist']

        for pass_num in range(1, 4):  # 3 passes
            delta_twist = (max_twist - min_twist) / divisions
            print(f'{img_name:15s} Pass {pass_num}: {min_twist:.5f} to {max_twist:.5f} '
                  f'by {delta_twist:.5f}')

            best_twist = None
            best_error = None
            best_offset = None
            for twist in np.arange(min_twist, max_twist+delta_twist/2, delta_twist):
                ret = find_error_one_twist(orig_obs, img_name, twist_config=twist_config,
                                           twist=twist, verbose=verbose, show_plot=show_plots)
                if ret is None:
                    continue
                error, offset, new_offset, u_arr, v_arr, u_diff_arr, v_diff_arr, mag_perc, nm = ret
                print(f'{img_name:15s} Twist: {twist:8.5f}   '
                      f'Offset: {offset[0]:8.3f}, {offset[1]:8.3f}   Error: {error:9.5f}')
                log_fp.write(f'Twist: {twist:8.5f}   '
                             f'Offset: {offset[0]:8.3f}, {offset[1]:8.3f}   Error: {error:9.5f}\n')
                if best_error is None or error < best_error:
                    best_error = error
                    best_twist = twist
                    best_offset = new_offset

            if best_twist is None:
                if verbose:
                    print(f'{img_name:15s} No best twist found')
                log_fp.write(f'{img_name:15s} No best twist found\n')
                return None

            print(f'{img_name:15s} Best twist: {best_twist:8.5f}, Error: {best_error:9.5f}')
            log_fp.write(f'Best twist: {best_twist:8.5f}, Error: {best_error:9.5f}\n')

            min_twist = best_twist - delta_twist*2
            max_twist = best_twist + delta_twist*2

        print()
        print()
        log_fp.write('\n\n')

        corner_dist = np.sqrt((orig_obs.data.shape[0]/2)**2 + (orig_obs.data.shape[1]/2)**2)
        rotation_error = np.sin(np.radians(twist)) * corner_dist
        print(f'{img_name:15s} FINAL Best Twist: {best_twist:8.5f} deg '
              f'({rotation_error:9.5f} corner pixels)'
              f'   Offset: {best_offset[0]:8.3f}, {best_offset[1]:8.3f}   Error: {best_error:9.5f}')
        log_fp.write(f'FINAL,{img_name},{best_twist:.5f},{rotation_error:.5f},'
                     f'{best_offset[0]:.3f},{best_offset[1]:.3f},{best_error:.5f}\n')
        log_fp.flush()

        plot_path = plot_dir / f'{img_name}.png'
        plot_with_arrows(u_arr=u_arr, v_arr=v_arr, u_diff_arr=u_diff_arr, v_diff_arr=v_diff_arr,
                         mag_perc=mag_perc, nm=nm, twist=best_twist, offset=best_offset,
                         img_name=img_name, plot_path=plot_path)
        if show_plots:
            plot_with_arrows(u_arr=u_arr, v_arr=v_arr, u_diff_arr=u_diff_arr, v_diff_arr=v_diff_arr,
                             mag_perc=mag_perc, nm=nm, twist=best_twist, offset=best_offset,
                             img_name=img_name)

        return url, best_twist, rotation_error, best_offset, best_offset, best_error


def main(command_list):
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file',
                        help='The config file containing the star field data.')
    parser.add_argument('nthreads', type=int, default=1, nargs='?',
                        help='The number of threads to use for parallel processing.')
    parser.add_argument('results_dir', type=str, default='./fov_twist_results',  nargs='?',
                        help="""The directory in which to store the results. If not
                        specified, the current directory is used. The directory may
                        contain destination prefixes like gs:// or s3:// as supported by
                        FCPath to store the results remotely.""")
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Print verbose output during the optimization process.')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Print debug output during the optimization process.')
    parser.add_argument('--show-plots', action='store_true', default=False,
                        help="""Show plots interactively during the optimization process.
                        Useful for debugging.""")
    args = parser.parse_args(command_list[1:])
    DEFAULT_CONFIG.read_config()
    DEFAULT_LOGGER.set_level('DEBUG' if args.debug else 'WARNING')
    if args.debug:
        filecache.set_easy_logger()

    with open(args.config_file, 'r') as f:
        twist_config = json.load(f)

    dataset = args.config_file.split('/')[-1].split('.')[0]

    results_path = FCPath(args.results_dir).expandvars().absolute()

    DEFAULT_CONFIG.stars.max_stars = twist_config['max_stars']

    if args.nthreads == 1:
        results = []
        for url in twist_config['image_urls']:
            res = optimize_one_image(url, twist_config, dataset, results_path,
                                     verbose=args.verbose, show_plots=args.show_plots)
            results.append(res)
    else:
        with multiprocessing.Pool(args.nthreads) as pool:
            opt_func = partial(optimize_one_image, twist_config=twist_config, dataset=dataset,
                               results_path=results_path, verbose=args.verbose,
                               show_plots=args.show_plots)
            results = pool.map(opt_func, twist_config['image_urls'])

    print('Results summary')
    twist_list = []
    rotation_error_list = []
    summary_path = results_path / f'{dataset}_results.txt'
    with summary_path.open('w') as summary_fp:
        for res in results:
            if res is None:
                continue
            url, twist, rotation_error, offset, new_offset, error = res
            twist_list.append(twist)
            rotation_error_list.append(rotation_error)
            img_name = url.split('/')[-1].split('.')[0]
            print(f'{img_name:15s} Best Twist: {twist:8.5f} deg ({rotation_error:9.5f} '
                  f'corner pixels)   Offset: {new_offset[0]:8.3f}, {new_offset[1]:8.3f}  '
                  f'Error: {error:9.5f}')
            summary_fp.write(f'{url},{img_name},{twist:8.5f},{rotation_error:9.5f},'
                             f'{offset[0]:8.3f},{offset[1]:8.3f},'
                             f'{new_offset[0]:8.3f},{new_offset[1]:8.3f},{error:9.5f}\n')

    print()
    print(f'Mean twist:   {np.mean(twist_list):8.5f} +/- {np.std(twist_list):8.5f} deg')
    print(f'Median twist: {np.median(twist_list):8.5f} deg')
    print(f'Min twist:    {np.min(twist_list):8.5f} deg')
    print(f'Max twist:    {np.max(twist_list):8.5f} deg')
    print()
    print(f'Mean rotation error:   {np.mean(rotation_error_list):9.5f} +/- '
          f'{np.std(rotation_error_list):9.5f} corner pixels')
    print(f'Median rotation error: {np.median(rotation_error_list):9.5f} corner pixels')
    print(f'Min rotation error:    {np.min(rotation_error_list):9.5f} corner pixels')
    print(f'Max rotation error:    {np.max(rotation_error_list):9.5f} corner pixels')


if __name__ == '__main__':
    main(sys.argv)

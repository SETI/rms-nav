import argparse
from functools import partial
import multiprocessing
import os
from pathlib import Path
import sys

from filecache import FCPath
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from psfmodel import GaussianPSF

# Add the repository root to the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from nav.config import DEFAULT_CONFIG, DEFAULT_LOGGER
from nav.inst import inst_name_to_class
from nav.nav_master import NavMaster


COISS_NAC_SHORT = {
    'urls': [
        '${PDS3_HOLDINGS_DIR}/calibrated/COISS_2xxx/COISS_2021/data/1521798868_1521893025/N1521879918_2_CALIB.IMG',
        '${PDS3_HOLDINGS_DIR}/calibrated/COISS_2xxx/COISS_2069/data/1689632036_1690197264/N1690179268_1_CALIB.IMG',
        '${PDS3_HOLDINGS_DIR}/calibrated/COISS_2xxx/COISS_2116/data/1882056692_1882170221/N1882162175_1_CALIB.IMG',

        # These are rotations of the same field of view
        '${PDS3_HOLDINGS_DIR}/calibrated/COISS_2xxx/COISS_2024/data/1533001485_1533946355/N1533083950_1_CALIB.IMG',
        '${PDS3_HOLDINGS_DIR}/calibrated/COISS_2xxx/COISS_2024/data/1533001485_1533946355/N1533085770_1_CALIB.IMG',
        '${PDS3_HOLDINGS_DIR}/calibrated/COISS_2xxx/COISS_2024/data/1533001485_1533946355/N1533087950_1_CALIB.IMG',
        '${PDS3_HOLDINGS_DIR}/calibrated/COISS_2xxx/COISS_2024/data/1533001485_1533946355/N1533089530_1_CALIB.IMG',
        '${PDS3_HOLDINGS_DIR}/calibrated/COISS_2xxx/COISS_2024/data/1533001485_1533946355/N1533091390_1_CALIB.IMG',
        '${PDS3_HOLDINGS_DIR}/calibrated/COISS_2xxx/COISS_2024/data/1533001485_1533946355/N1533093210_1_CALIB.IMG',
        '${PDS3_HOLDINGS_DIR}/calibrated/COISS_2xxx/COISS_2024/data/1533001485_1533946355/N1533095070_1_CALIB.IMG',
    ],
    'inst_id': 'coiss',
    'nstars': 100,
    'psf_size': (9, 9),
    'perc': 90,
    'clip': 0,
    'peak_factor': 2,
    'min_twist': -0.01,
    'max_twist': 0.03,
}


with open('coiss_nac_full_list.txt', 'r') as f:
    coiss_nac_full_list = [line.strip() for line in f.readlines() if line[0] != '#']

COISS_NAC = {
    'urls': coiss_nac_full_list,
    'inst_id': 'coiss',
    'nstars': 100,
    'psf_size': (9, 9),
    'perc': 90,
    'clip': 0,
    'peak_factor': 2,
    'min_twist': -0.01,
    'max_twist': 0.03,
}


COISS_WAC_SHORT = {
    'urls': [
        '${PDS3_HOLDINGS_DIR}/calibrated/COISS_2xxx/COISS_2021/data/1521798868_1521893025/W1521879918_2_CALIB.IMG',
        '${PDS3_HOLDINGS_DIR}/calibrated/COISS_2xxx/COISS_2002/data/1463010066_1463287615/W1463134433_1_CALIB.IMG',
        '${PDS3_HOLDINGS_DIR}/calibrated/COISS_2xxx/COISS_2009/data/1484846724_1485147239/W1485044635_1_CALIB.IMG',
        '${PDS3_HOLDINGS_DIR}/calibrated/COISS_2xxx/COISS_2009/data/1486503116_1486561167/W1486510390_2_CALIB.IMG',
        '${PDS3_HOLDINGS_DIR}/calibrated/COISS_2xxx/COISS_2041/data/1580614432_1580756339/W1580751107_1_CALIB.IMG',

        # These are rotations of the same field of view
        '${PDS3_HOLDINGS_DIR}/calibrated/COISS_2xxx/COISS_2024/data/1533001485_1533946355/W1533099692_1_CALIB.IMG',
        '${PDS3_HOLDINGS_DIR}/calibrated/COISS_2xxx/COISS_2024/data/1533001485_1533946355/W1533101552_1_CALIB.IMG',
        '${PDS3_HOLDINGS_DIR}/calibrated/COISS_2xxx/COISS_2024/data/1533001485_1533946355/W1533103412_1_CALIB.IMG',
        '${PDS3_HOLDINGS_DIR}/calibrated/COISS_2xxx/COISS_2024/data/1533001485_1533946355/W1533105272_1_CALIB.IMG',
        '${PDS3_HOLDINGS_DIR}/calibrated/COISS_2xxx/COISS_2024/data/1533001485_1533946355/W1533107132_1_CALIB.IMG',
        '${PDS3_HOLDINGS_DIR}/calibrated/COISS_2xxx/COISS_2024/data/1533001485_1533946355/W1533108992_1_CALIB.IMG',
        '${PDS3_HOLDINGS_DIR}/calibrated/COISS_2xxx/COISS_2024/data/1533001485_1533946355/W1533110852_1_CALIB.IMG',

    ],
    'inst_id': 'coiss',
    'nstars': 100,
    'psf_size': (9, 9),
    'perc': 90,
    'clip': 0,
    'peak_factor': 2,
    'min_twist': -0.03,
    'max_twist': 0.03,
}


with open('coiss_wac_full_list.txt', 'r') as f:
    coiss_wac_full_list = [line.strip() for line in f.readlines() if line[0] != '#']

COISS_WAC = {
    'urls': coiss_wac_full_list,
    'inst_id': 'coiss',
    'nstars': 100,
    'psf_size': (9, 9),
    'perc': 90,
    'clip': 0,
    'peak_factor': 2,
    'min_twist': -0.03,
    'max_twist': 0.03,
}


with open('gossi_full_list.txt', 'r') as f:
    gossi_full_list = [line.strip() for line in f.readlines() if line[0] != '#']


GOSSI = {
    'urls': gossi_full_list,
    'inst_id': 'gossi',
    'nstars': 30,
    'psf_size': (15, 15),
    'perc': 90,
    'clip': 0,
    'peak_factor': 1.3,
    'min_twist': -.12,
    'max_twist': 0.02,
}

VGISS_NAC = {
    'urls': [
        '${PDS3_HOLDINGS_DIR}/volumes/VGISS_5xxx/VGISS_5209/DATA/C20326XX/C2032609_GEOMED.IMG',
        '${PDS3_HOLDINGS_DIR}/volumes/VGISS_6xxx/VGISS_6111/DATA/C34796XX/C3479608_GEOMED.IMG',
        '${PDS3_HOLDINGS_DIR}/volumes/VGISS_8xxx/VGISS_8210/DATA/C12050XX/C1205043_GEOMED.IMG',
        '${PDS3_HOLDINGS_DIR}/volumes/VGISS_8xxx/VGISS_8210/DATA/C12051XX/C1205117_GEOMED.IMG',

    ],
    'inst_id': 'vgiss',
    'nstars': 15,
    'psf_size': (15, 15),
    'perc': 80,
    'clip': 40,
    'peak_factor': 2,
    'min_twist': -.4,
    'max_twist': .4,
    'peak_factor': 2,
}

VGISS_WAC = {
    'urls': [
        '${PDS3_HOLDINGS_DIR}/volumes/VGISS_8xxx/VGISS_8210/DATA/C12051XX/C1205111_GEOMED.IMG',
        '${PDS3_HOLDINGS_DIR}/volumes/VGISS_5xxx/VGISS_5209/DATA/C20326XX/C2032609_GEOMED.IMG',
        '${PDS3_HOLDINGS_DIR}/volumes/VGISS_8xxx/VGISS_8210/DATA/C12050XX/C1205037_GEOMED.IMG',

    ],
    'inst_id': 'vgiss',
    'nstars': 15,
    'psf_size': (15, 15),
    'perc': 100,
    'clip': 40,
    'peak_factor': 2,
    'min_twist': .2,
    'max_twist': .6,
}


with open('nhlorri_full_list.txt', 'r') as f:
    nhlorri_full_list = [line.strip() for line in f.readlines() if line[0] != '#']

# Set star vmax to 12.5 for LORRI
NHLORRI = {
    'urls': nhlorri_full_list,
    'inst_id': 'nhlorri',
    'nstars': 30,
    'psf_size': (15, 15),
    'perc': 90,
    'clip': 0,
    'peak_factor': 2,
    'min_twist': -0.3,
    'max_twist': -0.1,
}


STAR_FIELDS = {
    'COISS_NAC_SHORT': COISS_NAC_SHORT,
    'COISS_NAC': COISS_NAC,
    'COISS_WAC': COISS_WAC,
    'GOSSI': GOSSI,
    'VGISS_NAC': VGISS_NAC,
    'VGISS_WAC': VGISS_WAC,
    'NHLORRI': NHLORRI,
}


def _filter_star(star, u, v, psf_size, img, clip, psf, peak_factor, u_list, v_list, u_diff_list, v_diff_list, verbose):
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
        print(f'Star {star.unique_number} VMAG {star.vmag} Searched at {u:.3f}, {v:.3f} found at {opt_u:.3f}, {opt_v:.3f}')
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
    if peak < bkgnd * peak_factor:
        if verbose:
            print(f'Star {star.unique_number} VMAG {star.vmag} peak {peak} not bright enough, '
                    f'bkgnd {bkgnd}')
        return False
    if verbose:
        print(f'Star {star.unique_number} VMAG {star.vmag} peak {peak} bkgnd {bkgnd}')
    u_list.append(u)
    v_list.append(v)
    u_diff_list.append(diff_u)
    v_diff_list.append(diff_v)
    if verbose:
        print(f'Star {star.unique_number} VMAG {star.vmag} offset {diff_u}, {diff_v} OK')
    return True

def find_error_one_twist(orig_obs, img_name, psf_size, perc, twist, clip, peak_factor, verbose=False):
    if verbose:
        print(f'Checking twist: {twist}')
    obs = orig_obs.navigate((0, 0, np.radians(twist)))
    nm = NavMaster(obs, nav_models=['stars'])
    nm.navigate()

    min_stars = 4

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

    psf = GaussianPSF(sigma=2.0)

    u_list = []
    v_list = []
    u_diff_list = []
    v_diff_list = []

    for star in star_list:
        u = star.u - offset[1]
        v = star.v - offset[0]
        _filter_star(star, u, v, psf_size, img, clip, psf, peak_factor,
                     u_list, v_list, u_diff_list, v_diff_list, verbose)

    if len(u_list) < min_stars:
        if verbose:
            print('Not enough valid stars found')
        return None

    # print(u_diff_list)
    # print(v_diff_list)
    mean_diff_u = np.median(u_diff_list)
    mean_diff_v = np.median(v_diff_list)
    if verbose:
        print()
        print(f'Median delta offset {mean_diff_u}, {mean_diff_v}')

    # Second pass - find the offset of each star relative to the mean offset

    u_list = []
    v_list = []
    u_diff_list = []
    v_diff_list = []

    new_offset = (offset[0] - mean_diff_v, offset[1] - mean_diff_u)

    if verbose:
        print(f'Adjusted offset: {new_offset[1]}, {new_offset[0]}')

    for star in star_list:
        u = star.u - new_offset[1]
        v = star.v - new_offset[0]
        _filter_star(star, u, v, psf_size, img, clip, psf, peak_factor,
                     u_list, v_list, u_diff_list, v_diff_list, verbose)

    if len(u_list) < min_stars:
        if verbose:
            print('Not enough valid stars found')
        return None

    # Calculate the magnitude of the errors

    u_arr = np.array(u_list)
    v_arr = np.array(v_list)
    u_diff_arr = np.array(u_diff_list)
    v_diff_arr = np.array(v_diff_list)
    mag_arr = np.sqrt(u_diff_arr**2 + v_diff_arr**2)

    # Eliminate stars with bad magnitudes relative to the rest
    mag_perc = np.percentile(mag_arr, perc)
    good_stars = mag_arr <= mag_perc
    mag_arr = mag_arr[good_stars]
    u_arr = u_arr[good_stars]
    v_arr = v_arr[good_stars]
    u_diff_arr = u_diff_arr[good_stars]
    v_diff_arr = v_diff_arr[good_stars]

    # Calculate the angle of the errors
    # A perfect star field should have u_diff_arr relative to u_arr and v_diff_arr relative
    # to v_arr always point towards the center of the image
    # Calculate the angle between these vectors and the perfect vector to the center

    u_diff_arr_perfect = img.shape[1]/2 - u_arr
    v_diff_arr_perfect = img.shape[0]/2 - v_arr
    angle_arr_perfect = (np.arctan2(v_diff_arr_perfect, u_diff_arr_perfect) + np.pi) % (2 * np.pi)
    angle_arr = (np.arctan2(v_diff_arr, u_diff_arr) + np.pi) % (2 * np.pi)
    # Diff the angles always getting a number between -pi and pi
    angle_diff_arr = angle_arr - angle_arr_perfect
    error = np.sqrt(np.sum((mag_arr*angle_diff_arr)**2))

    mean_perc = np.mean(mag_arr)
    median_perc = np.median(mag_arr)
    if verbose:
        print(f'{perc}th percentile absolute star position error: {mag_perc}')
        print(f'Mean: {mean_perc}, Median: {median_perc}')

    # print(f'Error: {error}')
    # plot_with_arrows(u_arr, v_arr, u_diff_arr, v_diff_arr, mag_perc, nm, twist, new_offset, img_name)

    return error, offset, new_offset, u_arr, v_arr, u_diff_arr, v_diff_arr, mag_perc, nm


def _find_error_for_opt(orig_obs, psf_size, perc, twist, clip, img_name, log_fp, verbose=False):
    ret = find_error_one_twist(orig_obs, img_name, psf_size, perc, twist, clip, peak_factor, verbose)
    if ret is None:
        return np.inf
    error, offset, new_offset, u_arr, v_arr, u_diff_arr, v_diff_arr, mag_perc, nm = ret
    print(f'{img_name:15s} Twist: {twist:8.5f}   Offset: {new_offset[0]:8.3f}, {new_offset[1]:8.3f}   Error: {error:9.5f}')
    log_fp.write(f'Twist: {twist:8.5f}   Offset: {new_offset[0]:8.3f}, {new_offset[1]:8.3f}   Error: {error:9.5f}\n')
    log_fp.flush()
    return error


def plot_with_arrows(u_arr, v_arr, u_diff_arr, v_diff_arr, mag_perc, nm, twist, offset,
                     img_name, plot_filename=None):
    plt.figure(figsize=(12, 12))
    overlay = nm.create_overlay()
    plt.imshow(overlay)
    for u, v, u_diff, v_diff in zip(u_arr, v_arr, u_diff_arr, v_diff_arr):
        mag = np.sqrt(u_diff**2 + v_diff**2)
        if mag <= mag_perc:
            plt.quiver(u, v, u_diff*100, v_diff*100, angles='xy', scale_units='xy', scale=1,
                       color='#80ff80', alpha=0.6)
    plt.title(f'{img_name} (Twist {twist:.5f}, Offset {offset[1]:.3f}, {offset[0]:.3f})')
    if plot_filename is None:
        plt.show()
    else:
        plt.savefig(plot_filename)
    plt.close()


def optimize_one_image(url, *, dataset, inst_id, nstars, psf_size, perc, clip, min_twist, max_twist, peak_factor, verbose=False):
    img_name = url.split('/')[-1].split('.')[0]
    log_filename = f'fov_twist_results/{dataset}_logs/{img_name}.log'
    os.makedirs(f'fov_twist_results/{dataset}_logs', exist_ok=True)
    os.makedirs(f'fov_twist_results/{dataset}_plots', exist_ok=True)
    log_fp = open(log_filename, 'w')

    inst_class = inst_name_to_class(inst_id)
    try:
        orig_obs = inst_class.from_file(FCPath(url).expandvars(), fast_distortion=True)
    except Exception as e:
        print(f'Error reading image {url}: {e}')
        log_fp.write(f'Error reading image {url}: {e}\n')
        log_fp.close()
        return None

    DEFAULT_CONFIG.stars.max_stars = nstars

    divisions = 20
    delta_twist = (max_twist - min_twist) / divisions

    print(f'{img_name:15s} Initial pass')
    log_fp.write('Initial pass\n')
    best_twist = None
    best_error = None
    for twist in np.arange(min_twist, max_twist+delta_twist/2, delta_twist):
        ret = find_error_one_twist(orig_obs, img_name, psf_size, perc, twist, clip, peak_factor, verbose)
        if ret is None:
            continue
        error, offset, new_offset, u_arr, v_arr, u_diff_arr, v_diff_arr, mag_perc, nm = ret
        print(f'{img_name:15s} Twist: {twist:8.5f}   Offset: {offset[0]:8.3f}, {offset[1]:8.3f}   Error: {error:9.5f}')
        log_fp.write(f'Twist: {twist:8.5f}   Offset: {offset[0]:8.3f}, {offset[1]:8.3f}   Error: {error:9.5f}\n')
        log_fp.flush()
        if best_error is None or error < best_error:
            best_error = error
            best_twist = twist

    if best_twist is None:
        if verbose:
            print(f'{img_name:15s} No best twist found')
        log_fp.write(f'{img_name:15s} No best twist found\n')
        log_fp.close()
        return None

    print(f'{img_name:15s} Best twist: {best_twist:8.5f}, Error: {best_error:9.5f}')
    log_fp.write(f'Best twist: {best_twist:8.5f}, Error: {best_error:9.5f}\n')
    log_fp.flush()

    min_twist = best_twist - delta_twist*2
    max_twist = best_twist + delta_twist*2

    delta_twist = (max_twist - min_twist) / divisions

    print(f'{img_name:15s} Second pass')
    log_fp.write('Second pass\n')
    best_twist = None
    best_error = None
    for twist in np.arange(min_twist, max_twist+delta_twist/2, delta_twist):
        ret = find_error_one_twist(orig_obs, img_name, psf_size, perc, twist, clip, peak_factor, verbose)
        if ret is None:
            continue
        error, offset, new_offset, u_arr, v_arr, u_diff_arr, v_diff_arr, mag_perc, nm = ret
        print(f'{img_name:15s} Twist: {twist:8.5f}   Offset: {new_offset[0]:8.3f}, {new_offset[1]:8.3f}   Error: {error:9.5f}')
        log_fp.write(f'Twist: {twist:8.5f}   Offset: {new_offset[0]:8.3f}, {new_offset[1]:8.3f}   Error: {error:9.5f}\n')
        log_fp.flush()
        if best_error is None or error < best_error:
            best_error = error
            best_twist = twist

    if best_twist is None:
        if verbose:
            print(f'{img_name:15s} No best twist found')
        log_fp.write(f'{img_name:15s} No best twist found\n')
        log_fp.close()
        return None

    print(f'{img_name:15s} Best twist: {best_twist:8.5f}, Error: {best_error:9.5f}')
    log_fp.write(f'Best twist: {best_twist:8.5f}, Error: {best_error:9.5f}\n')
    log_fp.flush()

    min_twist = best_twist - delta_twist*2
    max_twist = best_twist + delta_twist*2

    delta_twist = (max_twist - min_twist) / divisions

    print(f'{img_name:15s} Third pass')
    log_fp.write('Third pass\n')
    best_twist = None
    best_error = None
    best_offset = None
    for twist in np.arange(min_twist, max_twist+delta_twist/2, delta_twist):
        ret = find_error_one_twist(orig_obs, img_name, psf_size, perc, twist, clip, peak_factor, verbose)
        if ret is None:
            continue
        error, offset, new_offset, u_arr, v_arr, u_diff_arr, v_diff_arr, mag_perc, nm = ret
        print(f'{img_name:15s} Twist: {twist:8.5f}   Offset: {new_offset[0]:8.3f}, {new_offset[1]:8.3f}   Error: {error:9.5f}')
        log_fp.write(f'Twist: {twist:8.5f}   Offset: {new_offset[0]:8.3f}, {new_offset[1]:8.3f}   Error: {error:9.5f}\n')
        log_fp.flush()
        if best_error is None or error < best_error:
            best_error = error
            best_twist = twist
            best_offset = new_offset

    if best_twist is None:
        if verbose:
            print(f'{img_name:15s} No best twist found')
        log_fp.write(f'{img_name:15s} No best twist found\n')
        log_fp.close()
        return None

    print(f'{img_name:15s} Best twist: {best_twist:8.5f}, Error: {best_error:9.5f}')
    log_fp.write(f'Best twist: {best_twist:8.5f}, Error: {best_error:9.5f}\n')
    log_fp.flush()

    print(f'{img_name:15s}')
    print(f'{img_name:15s}')
    log_fp.write('\n\n')

    corner_dist = np.sqrt((orig_obs.data.shape[0]/2)**2 + (orig_obs.data.shape[1]/2)**2)
    rotation_error = np.sin(np.radians(twist)) * corner_dist
    print(f'{img_name:15s} FINAL Best Twist: {best_twist:8.5f} deg '
          f'({rotation_error:9.5f} corner pixels)'
          f'   Offset: {best_offset[0]:8.3f}, {best_offset[1]:8.3f}   Error: {best_error:9.5f}')
    log_fp.write(f'FINAL,{img_name},{best_twist:.5f},{rotation_error:.5f},'
                 f'{best_offset[0]:.3f},{best_offset[1]:.3f},{best_error:.5f}\n')
    log_fp.flush()

    plot_filename = f'fov_twist_results/{dataset}_plots/{img_name}.png'
    plot_with_arrows(u_arr, v_arr, u_diff_arr, v_diff_arr, mag_perc, nm, best_twist, best_offset,
                     img_name, plot_filename)

    log_fp.close()

    return url, best_twist, rotation_error, offset, best_offset, error


def main(command_list):
    DEFAULT_CONFIG.read_config()
    DEFAULT_LOGGER.set_level('WARNING')

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=STAR_FIELDS.keys())
    parser.add_argument('nthreads', type=int, default=1)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args(command_list[1:])

    star_field = STAR_FIELDS[args.dataset]

    # DEFAULT_CONFIG.stars.max_stars = nstars
    # inst_class = inst_name_to_class(inst_id)
    # ORIG_OBS = inst_class.from_file(URL, fast_distortion=True)
    # error, u_arr, v_arr, u_diff_arr, v_diff_arr, mag_perc, nm = find_error_one_twist(ORIG_OBS, psf_size, perc, 0, False)
    # plot_with_arrows(u_arr, v_arr, u_diff_arr, v_diff_arr, mag_perc, nm)
    # error, u_arr, v_arr, u_diff_arr, v_diff_arr, mag_perc, nm = find_error_one_twist(ORIG_OBS, psf_size, perc, 0.01, False)
    # plot_with_arrows(u_arr, v_arr, u_diff_arr, v_diff_arr, mag_perc, nm)
    # error, u_arr, v_arr, u_diff_arr, v_diff_arr, mag_perc, nm = find_error_one_twist(ORIG_OBS, psf_size, perc, 0.02, False)
    # plot_with_arrows(u_arr, v_arr, u_diff_arr, v_diff_arr, mag_perc, nm)

    if args.nthreads == 1:
        for url in star_field['urls']:
            optimize_one_image(url, dataset=args.dataset, inst_id=star_field['inst_id'],
                               nstars=star_field['nstars'],
                               psf_size=star_field['psf_size'], perc=star_field['perc'],
                               clip=star_field['clip'],
                               peak_factor=star_field['peak_factor'],
                               min_twist=star_field['min_twist'], max_twist=star_field['max_twist'],
                               verbose=args.verbose)
    else:
        with multiprocessing.Pool(args.nthreads) as pool:
            opt_func = partial(optimize_one_image, dataset=args.dataset, inst_id=star_field['inst_id'],
                               nstars=star_field['nstars'],
                               psf_size=star_field['psf_size'], perc=star_field['perc'],
                               clip=star_field['clip'],
                               peak_factor=star_field['peak_factor'],
                               min_twist=star_field['min_twist'], max_twist=star_field['max_twist'],
                               verbose=args.verbose)
            results = pool.map(opt_func, star_field['urls'])

    print('Results summary')
    twist_list = []
    rotation_error_list = []
    os.makedirs('fov_twist_results', exist_ok=True)
    with open(f'fov_twist_results/{args.dataset}_results.txt', 'w') as f:
        for res in results:
            if res is None:
                continue
            url, twist, rotation_error, offset, new_offset, error = res
            twist_list.append(twist)
            rotation_error_list.append(rotation_error)
            img_name = url.split('/')[-1].split('.')[0]
            print(f'{img_name:15s} Best Twist: {twist:8.5f} deg ({rotation_error:9.5f} corner pixels)'
                f'   Offset: {new_offset[0]:8.3f}, {new_offset[1]:8.3f}   Error: {error:9.5f}')
            f.write(f'{url},{img_name},{twist:8.5f},{rotation_error:9.5f},'
                    f'{offset[0]:8.3f},{offset[1]:8.3f},'
                    f'{new_offset[0]:8.3f},{new_offset[1]:8.3f},{error:9.5f}\n')

    print()
    print(f'Mean twist:   {np.mean(twist_list):8.5f} +/- {np.std(twist_list):8.5f} deg')
    print(f'Median twist: {np.median(twist_list):8.5f} deg')
    print(f'Min twist:    {np.min(twist_list):8.5f} deg')
    print(f'Max twist:    {np.max(twist_list):8.5f} deg')
    print()
    print(f'Mean rotation error:   {np.mean(rotation_error_list):9.5f} +/- {np.std(rotation_error_list):9.5f} corner pixels')
    print(f'Median rotation error: {np.median(rotation_error_list):9.5f} corner pixels')
    print(f'Min rotation error:    {np.min(rotation_error_list):9.5f} corner pixels')
    print(f'Max rotation error:    {np.max(rotation_error_list):9.5f} corner pixels')


if __name__ == '__main__':
    main(sys.argv)

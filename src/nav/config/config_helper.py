import argparse
import os

from .config import Config


def get_backplane_results_root(arguments: argparse.Namespace,
                               config: Config) -> str:
    """Get the backplane results root from the arguments, configuration, or environment.

    Parameters:
        arguments: The parsed arguments.
        config: The configuration possibly containing the environment section.

    Returns:
        The backplane results root.

    Raises:
        ValueError: If the backplane results root cannot be determined.
    """
    backplane_results_root_str = None
    try:
        backplane_results_root_str = arguments.backplane_results_root
    except AttributeError:
        pass
    if backplane_results_root_str is None:
        try:
            backplane_results_root_str = config.environment.backplane_results_root
        except AttributeError:
            pass
    if backplane_results_root_str is None:
        backplane_results_root_str = os.getenv('NAV_BACKPLANE_RESULTS_ROOT')
    if backplane_results_root_str is None:
        raise ValueError('One of --backplane-results-root, the configuration variable '
                         '"environment.backplane_results_root", or the BACKPLANE_RESULTS_ROOT '
                         'environment variable must be set'
                         )
    return backplane_results_root_str


def get_nav_results_root(arguments: argparse.Namespace,
                         config: Config) -> str:
    """Get the navigation root from the arguments, configuration, or environment.

    Parameters:
        arguments: The parsed arguments.
        config: The configuration possibly containing the environment section.

    Returns:
        The navigation results root.

    Raises:
        ValueError: If the navigation results root cannot be determined.
    """
    nav_results_root_str = None
    try:
        nav_results_root_str = arguments.nav_results_root
    except AttributeError:
        pass
    if nav_results_root_str is None:
        try:
            nav_results_root_str = config.environment.nav_results_root
        except AttributeError:
            pass
    if nav_results_root_str is None:
        nav_results_root_str = os.getenv('NAV_RESULTS_ROOT')
    if nav_results_root_str is None:
        raise ValueError('One of --nav-results-root, the configuration variable '
                         '"environment.nav_results_root", or the NAV_RESULTS_ROOT '
                         'environment variable must be set'
                         )
    return nav_results_root_str


def get_pds4_bundle_results_root(arguments: argparse.Namespace,
                                 config: Config) -> str:
    """Get the PDS4 bundle root from the arguments, configuration, or environment.

    Parameters:
        arguments: The parsed arguments.
        config: The configuration possibly containing the environment section.

    Returns:
        The PDS4 bundle root.

    Raises:
        ValueError: If the PDS4 bundle root cannot be determined.
    """
    pds4_bundle_root_str = None
    try:
        pds4_bundle_root_str = arguments.pds4_bundle_root
    except AttributeError:
        pass
    if pds4_bundle_root_str is None:
        try:
            pds4_bundle_root_str = config.environment.pds4_bundle_root
        except AttributeError:
            pass
    if pds4_bundle_root_str is None:
        pds4_bundle_root_str = os.getenv('NAV_PDS4_BUNDLE_ROOT')
    if pds4_bundle_root_str is None:
        raise ValueError('One of --pds4-bundle-root, the configuration variable '
                         '"environment.pds4_bundle_root", or the NAV_PDS4_BUNDLE_ROOT '
                         'environment variable must be set'
                         )
    return pds4_bundle_root_str

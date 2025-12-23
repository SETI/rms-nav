import argparse
import os

from .config import Config


def get_backplane_results_root(arguments: argparse.Namespace,
                               config: Config) -> str:
    """Get the backplane results root from the arguments, configuration, or environment.

    First look in arguments.backplane_results_root, then in
    config.environment.backplane_results_root, then in the environment variable
    NAV_BACKPLANE_RESULTS_ROOT.

    Parameters:
        arguments: The parsed arguments. config: The configuration possibly containing the
        environment section.

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
                         '"environment.backplane_results_root", or the NAV_BACKPLANE_RESULTS_ROOT '
                         'environment variable must be set'
                         )
    return backplane_results_root_str


def get_nav_results_root(arguments: argparse.Namespace,
                         config: Config) -> str:
    """Get the navigation root from the arguments, configuration, or environment.

    First look in arguments.nav_results_root, then in config.environment.nav_results_root,
    then in the environment variable NAV_RESULTS_ROOT.

    Parameters:
        arguments: The parsed arguments. config: The configuration possibly containing the
        environment section.

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

    First look in arguments.bundle_results_root, then in
    config.environment.bundle_results_root, then in the environment variable
    BUNDLE_RESULTS_ROOT.

    Parameters:
        arguments: The parsed arguments. config: The configuration possibly containing the
        environment section.

    Returns:
        The PDS4 bundle root.

    Raises:
        ValueError: If the PDS4 bundle root cannot be determined.
    """
    pds4_bundle_root_str = None
    try:
        pds4_bundle_root_str = arguments.bundle_results_root
    except AttributeError:
        pass
    if pds4_bundle_root_str is None:
        try:
            pds4_bundle_root_str = config.environment.bundle_results_root
        except AttributeError:
            pass
    if pds4_bundle_root_str is None:
        pds4_bundle_root_str = os.getenv('NAV_BUNDLE_RESULTS_ROOT')
    if pds4_bundle_root_str is None:
        raise ValueError('One of --bundle-results-root, the configuration variable '
                         '"environment.bundle_results_root", or the NAV_BUNDLE_RESULTS_ROOT '
                         'environment variable must be set'
                         )
    return pds4_bundle_root_str


def load_default_and_user_config(arguments: argparse.Namespace,
                                 config: Config) -> None:
    """Load the default and user configuration (if any).

    Parameters:
        arguments: The parsed arguments containing the config_file argument.
        config: The configuration to update.
    """
    config.read_config()
    # If the user specified one or more config files, load them
    try:
        if arguments.config_file:
            for config_file in arguments.config_file:
                config.update_config(config_file)
        return
    except AttributeError:
        pass
    # If they didn't, load the default config file
    try:
        config.update_config('nav_default_config.yaml')
    except FileNotFoundError:
        pass

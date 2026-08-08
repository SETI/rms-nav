import argparse
import os

from filecache import FCPath

from .config import Config
from .logging_keys import validate_logging_config

RESULTS_DB_NONE = 'none'
"""Value of the results index URL that explicitly selects no index.

An exported NAV_RESULTS_DB would otherwise make a file-mode run impossible on that
machine, and a program that resolves a URL never falls back to reading files.
"""


def get_backplane_results_root(arguments: argparse.Namespace, config: Config) -> str:
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
        raise ValueError(
            'One of --backplane-results-root, the configuration variable '
            '"environment.backplane_results_root", or the NAV_BACKPLANE_RESULTS_ROOT '
            'environment variable must be set'
        )
    return backplane_results_root_str


def get_nav_results_root(arguments: argparse.Namespace, config: Config) -> str:
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
        raise ValueError(
            'One of --nav-results-root, the configuration variable '
            '"environment.nav_results_root", or the NAV_RESULTS_ROOT '
            'environment variable must be set'
        )
    return nav_results_root_str


def get_log_root(arguments: argparse.Namespace, config: Config) -> str:
    """Get the log root from the arguments, configuration, or environment.

    First look in ``arguments.log_root``, then in ``config.environment.log_root``,
    then in the environment variable ``NAV_LOG_ROOT``.  Unlike the other roots
    this one has a fallback rather than an error: logs belong under the
    navigation results root by default, so a run that has not been told where to
    put them still puts them somewhere predictable.

    Parameters:
        arguments: The parsed arguments.
        config: The configuration possibly containing the environment section.

    Returns:
        The log root.

    Raises:
        ValueError: If neither a log root nor a navigation results root can be
            determined.
    """
    log_root_str = None
    try:
        log_root_str = arguments.log_root
    except AttributeError:
        pass
    if log_root_str is None:
        try:
            log_root_str = config.environment.log_root
        except AttributeError:
            pass
    if log_root_str is None:
        log_root_str = os.getenv('NAV_LOG_ROOT')
    if log_root_str is None:
        # FCPath rather than os.path.join: a results root is routinely a cloud
        # URL, and joining those must not depend on the local path separator.
        log_root_str = (FCPath(get_nav_results_root(arguments, config)) / 'logs').as_posix()
    return str(log_root_str)


def get_pds4_bundle_results_root(arguments: argparse.Namespace, config: Config) -> str:
    """Get the PDS4 bundle root from the arguments, configuration, or environment.

    First look in arguments.bundle_results_root, then in
    config.environment.bundle_results_root, then in the environment variable
    NAV_BUNDLE_RESULTS_ROOT.

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
        raise ValueError(
            'One of --bundle-results-root, the configuration variable '
            '"environment.bundle_results_root", or the NAV_BUNDLE_RESULTS_ROOT '
            'environment variable must be set'
        )
    return pds4_bundle_root_str


def get_results_db_url(arguments: argparse.Namespace, config: Config) -> str | None:
    """Get the results index URL from the arguments, configuration, or environment.

    First look in arguments.results_db, then in config.environment.results_db, then in
    the environment variable NAV_RESULTS_DB.

    Unlike the results roots, absence is not an error: it means "no index", which is
    the default mode of every program.  The literal value ``none`` also means "no
    index", so a run on a machine that exports NAV_RESULTS_DB can still be told to
    read files by passing ``--results-db none``.  The sentinel is honored wherever the
    value came from, so a configuration file can opt out of an exported variable in
    the same way, and it is matched as the exact string, so a URL that merely contains
    the word is still a URL.

    Parameters:
        arguments: The parsed arguments.
        config: The configuration possibly containing the environment section.

    Returns:
        The results index connection URL, or None when no index was named.
    """
    # Absence is the ordinary case at both levels -- most programs define no
    # --results-db argument, and most configurations name no index -- so each is
    # asked for the key rather than made to raise for it, which would also hide an
    # AttributeError raised by something other than the lookup.
    results_db_str = vars(arguments).get('results_db')
    if results_db_str is None:
        results_db_str = config.environment.get('results_db')
    if results_db_str is None:
        results_db_str = os.getenv('NAV_RESULTS_DB')
    if results_db_str is None or results_db_str == RESULTS_DB_NONE:
        return None
    return str(results_db_str)


def load_default_and_user_config(arguments: argparse.Namespace, config: Config) -> None:
    """Load the default and user configuration (if any).

    The merged result's ``logging`` section is validated before returning, so a
    misspelled module key, program name, or level name fails here rather than
    having no effect at the point it was meant to apply.

    A named file that cannot be read is not skipped in favor of the defaults;
    ``Config``'s own diagnostic propagates, so a missing file still raises
    ``FileNotFoundError`` and a file that is not a mapping still raises the
    ``ValueError`` naming it.  Only the implicit user default is optional.

    Parameters:
        arguments: The parsed arguments, which may carry a ``config_file``
            attribute.  Callers that construct a bare ``Namespace`` need not
            supply it.
        config: The configuration to update.

    Raises:
        ValueError: If the merged ``logging`` section is not valid.
    """
    config.read_config()
    # If the user specified one or more config files, load them; if they didn't,
    # load the default config file.  getattr rather than attribute access
    # because callers legitimately pass a Namespace with no config_file at all,
    # and rather than try/except so an error raised deeper in the load cannot be
    # mistaken for the argument simply being absent.
    config_files = getattr(arguments, 'config_file', None)
    if config_files:
        for config_file in config_files:
            config.update_config(config_file)
    else:
        try:
            config.update_config('nav_default_config.yaml')
        except FileNotFoundError:
            pass
    validate_logging_config(config)

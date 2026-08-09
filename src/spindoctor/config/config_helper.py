import argparse
import os
from collections.abc import Callable

from filecache import FCPath

from .config import Config
from .logger import MAIN_LOGGER
from .logging_keys import validate_logging_config

RESULTS_DB_NONE = 'none'
"""Value of the results index URL that makes :func:`get_results_db_url` answer None.

An exported NAV_RESULTS_DB would otherwise reach every program on the machine, and
one that resolves a URL never falls back to reading files.  What "no index" then
means belongs to each caller: a program with a file-reading path takes it, and one
without a file-reading path refuses.
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


def get_results_db_url(
    arguments: argparse.Namespace,
    config: Config,
    *,
    warn: Callable[[str], None] | None = None,
) -> str | None:
    """Get the results index URL from the arguments, configuration, or environment.

    First look in arguments.results_db, then in config.environment.results_db, then in
    the environment variable NAV_RESULTS_DB.

    Unlike the results roots, absence is not an error: it means no index was
    resolved, and each caller decides whether it can proceed without one.  The literal
    value ``none`` resolves to the same answer, so a run on a machine that exports
    NAV_RESULTS_DB can still be told to read files by passing ``--results-db none``.
    The sentinel is honored wherever the value came from, so a configuration file can
    opt out of an exported variable in the same way; surrounding spaces are not part
    of it, and it is otherwise matched as the exact string, so a URL that merely
    contains the word is still a URL.

    A value that is empty, or nothing but spaces, names no index either, and is
    answered the same way rather than passed on: a URL parser handed one refuses
    with a message that begins with the colon after a name it does not have, and
    on a machine exporting an empty NAV_RESULTS_DB that refusal would stop every
    run.  It is not silent, because the level that set it may have meant to set a
    URL, so the level is named in a warning.  What the warning says stops at what
    was found and what to write instead: what follows from no index belongs to the
    caller, which is also why the caller supplies the sink it is written to.

    Parameters:
        arguments: The parsed arguments.
        config: The configuration possibly containing the environment section.
        warn: Where to report a value that names no index, or None to report it
            through the main log.  A program whose output is terminal text for a
            person rather than a run log passes its own printer.

    Returns:
        The results index connection URL, or None when no index was named.
    """
    # Absence is the ordinary case at both levels -- most programs define no
    # --results-db argument, and most configurations name no index -- so each is
    # asked for the key rather than made to raise for it, which would also hide an
    # AttributeError raised by something other than the lookup.
    named_by = '--results-db'
    results_db_str = vars(arguments).get('results_db')
    if results_db_str is None:
        named_by = 'the environment.results_db configuration variable'
        results_db_str = config.environment.get('results_db')
    if results_db_str is None:
        named_by = 'the NAV_RESULTS_DB environment variable'
        results_db_str = os.getenv('NAV_RESULTS_DB')
    if results_db_str is None:
        return None
    url = str(results_db_str)
    if not url.strip():
        message = (
            '%s is set to an empty value, which names no results index. Write %s to '
            'name none deliberately, or a connection URL to name one.'
        )
        if warn is None:
            # Interpolated by the logger rather than here, so a level that
            # discards the line does not pay to build it.
            MAIN_LOGGER.warning(message, named_by, RESULTS_DB_NONE)
        else:
            warn(message % (named_by, RESULTS_DB_NONE))
        return None
    if url.strip() == RESULTS_DB_NONE:
        return None
    return url


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

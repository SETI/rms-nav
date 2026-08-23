"""The command line as a run log is allowed to record it.

A run log records the command line it was given, because which of the command
line, the configuration file and the environment supplied a value is exactly
what a reader of a failed run needs to know.  One kind of word on that line is
a secret: the value of an option naming a connection URL, which may carry a
database password.  :func:`masked_command_line` replaces those values and
leaves every other word exactly as it was written, so that one rule decides
what a log may hold and every program that logs its command line is covered by
it rather than by remembering to mask at its own call site.

The masking rule itself belongs to the results index, which is where a
connection URL is parsed, and is imported only when a command line actually
carries one, since a line carrying none has nothing for it to decide.
"""

from collections.abc import Sequence

__all__ = ['URL_OPTIONS', 'masked_command_line']

URL_OPTIONS = ('--results-db',)
"""Options whose value is a connection URL and can therefore carry a password.

Only these are masked in a logged command line.  A results root is not a
connection URL: it has no credentials to hide, and it is the one word of the
command line an operator reads the run log to correct, so masking one would
corrupt the string and protect nothing.
"""


def _names_a_url_option(word: str) -> bool:
    """Whether a command-line word names an option whose value is a URL.

    Any distinguishing prefix of a long option is the option: argparse accepts
    ``--results-d`` for ``--results-db`` and consumes the URL after it just the
    same, so matching the full spelling alone would leave the abbreviated
    command line unmasked.  A prefix that argparse would have rejected never
    reaches here, since parsing runs first and exits on one.

    Parameters:
        word: One word of the command line, without any ``=value`` part.

    Returns:
        True when the word names one of :data:`URL_OPTIONS`.
    """
    if not word.startswith('--') or word == '--':
        return False
    return any(option.startswith(word) for option in URL_OPTIONS)


def _url_value_starts(command_list: Sequence[str]) -> dict[int, int]:
    """Locate every connection URL on a command line.

    Every spelling argparse accepts is covered: the value as a separate word,
    the value joined to the option by ``=``, and either of those under an
    abbreviation of the option's name.  The URL is located rather than extracted
    so that the word around it -- the option name and the ``=`` -- survives into
    the log unchanged.

    Parameters:
        command_list: The arguments, without the program name.

    Returns:
        The position of each word carrying a URL, mapped to the index within
        that word at which the URL begins.
    """
    starts: dict[int, int] = {}
    expecting_value = False
    for index, word in enumerate(command_list):
        if expecting_value:
            starts[index] = 0
            expecting_value = False
            continue
        option, separator, _value = word.partition('=')
        if separator and _names_a_url_option(option):
            starts[index] = len(option) + len(separator)
            continue
        expecting_value = _names_a_url_option(word)
    return starts


def masked_command_line(command_list: Sequence[str]) -> list[str]:
    """Return a command line with the value of every connection-URL option masked.

    Parameters:
        command_list: The arguments, without the program name.

    Returns:
        The arguments, with every connection URL among them masked and every
        other word as it was written.
    """
    starts = _url_value_starts(command_list)
    if not starts:
        return list(command_list)
    # Imported here rather than at the top of the file, on the same grounds as
    # the GUI imports elsewhere in the package: this module is reached by the
    # run banner of every program, including the ones that never open a
    # database, and the masking rule lives beside the URL parsing it mirrors.
    from spindoctor.results_index.masking import masked_url

    masked = list(command_list)
    for index, start in starts.items():
        word = masked[index]
        masked[index] = f'{word[:start]}{masked_url(word[start:])}'
    return masked

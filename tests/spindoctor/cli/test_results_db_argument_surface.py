"""Tests that a program reads a results index only if it says so on its command line.

Two rules meet here and only hold together this way.  A program that consumes
the index accepts ``--results-db URL``, and one that also has a file-reading
mode says in its own help that ``--results-db none`` is how to ask for it, so
that a machine which exports the URL can still be told to read files.  And the
programs that deliberately keep reading files -- the bundle builder, the
metadata consolidator, the backplane viewer -- must not quietly stop, which they
would if a resolved URL reached whatever happened to be running.

Declaring the option is therefore what makes a program index-backed.  A program
that declares it resolves a URL from the command line, the configuration and the
environment, in that order; a program that does not declare it resolves nothing
at all, whatever the machine exports.

Each parser is driven the way the program builds it, by running ``main`` with
``--help``, so what is asserted is the surface a user actually meets.
"""

import pytest
from tests.spindoctor.cli.conftest import help_text

# The programs whose questions the index answers, and the argv that reaches
# each one's parser.  A program reading its dataset or mode from argv before
# parsing needs it supplied.
_CONSUMERS = [
    ('sd_offset', ['coiss_saturn']),
    ('sd_stats_ingest', []),
    ('sd_stats_report', []),
]

# The consumers that also have a file-reading mode, which is what the sentinel
# opts back into.  The statistics programs read and write the index by
# construction and have no such mode, so naming the sentinel to them would
# advertise an answer they refuse.
_WITH_A_FILE_MODE = [
    ('sd_offset', ['coiss_saturn']),
]

# The programs that read the results files themselves and are not served by a
# column schema: the bundle builder serializes whole navigation documents into
# its supplemental product, the consolidator copies raw file bytes, and the
# viewer opens the products of one image.  Their image selection reads the tree
# like the rest of them, which is only true while they name no index.
# The modes named are the ones that enumerate with the selection flags, which
# is the population the rule is about; a mode that selects no images cannot be
# reached by a resolved URL at all.
_NON_CONSUMERS = [
    ('sd_create_bundle', ['labels', 'coiss_saturn']),
    ('sd_consolidate_metadata', ['coiss_saturn']),
    ('sd_backplane_viewer', ['coiss_saturn']),
]


def _one_line(program: str, argv: list[str]) -> str:
    """Return a program's help with its wrapping removed.

    argparse rewraps every help string to the terminal width, so a phrase is
    only reliably searchable once the line breaks are gone.

    Parameters:
        program: Dispatch module name under ``spindoctor.cli``.
        argv: Arguments preceding ``--help``.

    Returns:
        The help text as one space-separated line.
    """
    return ' '.join(help_text(program, argv).split())


@pytest.mark.parametrize(('program', 'argv'), _CONSUMERS)
def test_a_consuming_program_accepts_the_option(program: str, argv: list[str]) -> None:
    """The URL is named on the command line of the program that reads it."""
    assert '--results-db' in _one_line(program, argv)


@pytest.mark.parametrize(('program', 'argv'), _WITH_A_FILE_MODE)
def test_a_consuming_program_documents_the_opt_out(program: str, argv: list[str]) -> None:
    """The sentinel is the only opt-out visible to somebody reading --help.

    Without it, an exported URL makes file-mode runs impossible on that machine;
    an operator who cannot see the sentinel is left writing a configuration file
    to get one.
    """
    assert '--results-db none' in _one_line(program, argv)


@pytest.mark.parametrize(('program', 'argv'), _NON_CONSUMERS)
def test_a_program_that_keeps_reading_files_does_not_accept_the_option(
    program: str, argv: list[str]
) -> None:
    """Declaring the option is what makes a program index-backed, so these do not."""
    assert '--results-db' not in _one_line(program, argv)


@pytest.mark.parametrize(('program', 'argv'), _NON_CONSUMERS)
def test_the_help_the_absence_is_read_from_is_the_program_s_own(
    program: str, argv: list[str]
) -> None:
    """The control for the assertion above, which is otherwise unfalsifiable.

    A program that reads its dataset from argv before parsing prints a usage
    error instead of its help when it is not given one, and every option is then
    absent from a string that names none of them.  Each of these programs writes
    to the navigation results root, so its help says so.
    """
    assert '--nav-results-root' in _one_line(program, argv)

"""Every registered instrument has a chapter in each guide, following the template.

Two failures are prevented here, and both are silent under review. A registered
instrument with no chapter reads as an instrument nobody has documented rather
than as one whose documentation was forgotten. A chapter missing one of the
template's sections reads as a question nobody thought to ask, rather than as
one whose honest answer is "none" -- which is why the template requires the
heading to be present even when the section says ``None.``

The chapter a registered instrument must have is derived from that instrument's
own observation module (``obs_inst_cassini_iss`` yields ``cassini_iss``), so
registering an instrument is what creates the requirement. Nothing in this
module enumerates instruments, and adding one needs no edit here.
"""

from pathlib import Path

import pytest

from spindoctor.obs import ObsSim, ObsSnapshotInst, inst_name_to_obs_class, inst_names

DOCS_ROOT = Path(__file__).resolve().parents[2] / 'docs'

# The two guides that carry per-instrument chapters, each with its own template.
GUIDES = ('user_guide', 'dev_guide')

_OBS_MODULE_PREFIX = 'obs_inst_'

# The simulated-image host is registered alongside the instruments so a
# synthetic frame navigates through the same machinery, but it is not an
# instrument: no spacecraft, no archive, no SPICE camera frame, no instrument
# team whose documentation a chapter would point at. It is excluded by class
# identity rather than by registry name, so renaming a real instrument cannot
# quietly exempt it.
_NOT_INSTRUMENTS: tuple[type[ObsSnapshotInst], ...] = (ObsSim,)


def chapter_stem(obs_class: type[ObsSnapshotInst]) -> str:
    """Return the chapter file stem an instrument's observation class requires.

    Parameters:
        obs_class: A registered ``ObsSnapshotInst`` subclass.

    Returns:
        The file stem, with no extension: the class's module name with the
        ``obs_inst_`` prefix removed.

    Raises:
        ValueError: if the class does not live in a module named for the
            convention, since the chapter name would then be a guess.
    """
    module = obs_class.__module__.rsplit('.', 1)[-1]
    if not module.startswith(_OBS_MODULE_PREFIX):
        raise ValueError(
            f'{obs_class.__name__} lives in {module!r}, which does not start with '
            f'{_OBS_MODULE_PREFIX!r}; the chapter name cannot be derived from it'
        )
    return module[len(_OBS_MODULE_PREFIX) :]


def documented_instruments() -> list[str]:
    """Return the registered instrument names that must carry a chapter.

    Returns:
        Every registered instrument name except the hosts that are registered
        for machinery reasons and are not instruments.
    """
    return [name for name in inst_names() if inst_name_to_obs_class(name) not in _NOT_INSTRUMENTS]


def section_headings(text: str) -> list[str]:
    """Return the ``=``-underlined section titles of a reStructuredText document.

    The document title carries an overline as well as an underline, so it is
    recognized and skipped; every remaining ``=``-underlined line is a section.
    Subsection titles use ``-`` and are deliberately not collected: the template
    fixes the sections, not the shape of what a chapter says inside one.

    Parameters:
        text: The whole document.

    Returns:
        The section titles, in document order.
    """
    lines = text.splitlines()
    headings: list[str] = []
    for index, line in enumerate(lines):
        title = line.rstrip()
        if not title or index + 1 >= len(lines):
            continue
        underline = lines[index + 1].rstrip()
        if set(underline) != {'='} or len(underline) < len(title):
            continue
        if index > 0 and set(lines[index - 1].rstrip()) == {'='}:
            continue  # An overline: this is the document title, not a section.
        headings.append(title.strip())
    return headings


def read_doc(path: Path) -> str:
    """Return a documentation file's text, failing the test if it is absent.

    Parameters:
        path: The file to read.

    Returns:
        The file's text.
    """
    if not path.is_file():
        pytest.fail(f'missing documentation file: {path}')
    return path.read_text(encoding='utf-8')


CHAPTER_CASES = [
    pytest.param(guide, instrument, id=f'{guide}-{instrument}')
    for guide in GUIDES
    for instrument in documented_instruments()
]


@pytest.mark.parametrize('guide', GUIDES)
def test_the_guide_carries_an_instrument_template(guide: str) -> None:
    """Each guide's instruments directory holds the template chapters copy."""
    template = DOCS_ROOT / guide / 'instruments' / '_template.rst'
    assert template.is_file(), f'missing template: {template}'


@pytest.mark.parametrize('guide', GUIDES)
def test_the_instrument_index_discovers_chapters_by_glob(guide: str) -> None:
    """The index toctree globs, so a new chapter needs no edit to a shared file."""
    index = read_doc(DOCS_ROOT / guide / 'instruments' / 'instruments.rst')
    assert ':glob:' in index, f'{guide} instruments index does not use a glob toctree'


@pytest.mark.parametrize('guide', GUIDES)
def test_the_template_declares_sections(guide: str) -> None:
    """A template with no sections would make the conformance check vacuous."""
    template = read_doc(DOCS_ROOT / guide / 'instruments' / '_template.rst')
    assert section_headings(template), f'{guide} instrument template declares no sections'


@pytest.mark.parametrize(('guide', 'instrument'), CHAPTER_CASES)
def test_a_registered_instrument_has_a_chapter(guide: str, instrument: str) -> None:
    """Registering an instrument requires a chapter in both guides."""
    stem = chapter_stem(inst_name_to_obs_class(instrument))
    chapter = DOCS_ROOT / guide / 'instruments' / f'{stem}.rst'
    assert chapter.is_file(), (
        f'instrument {instrument!r} is registered but has no {guide} chapter at {chapter}'
    )


@pytest.mark.parametrize(('guide', 'instrument'), CHAPTER_CASES)
def test_a_chapter_carries_every_template_section(guide: str, instrument: str) -> None:
    """A chapter carries the template's sections, in the template's order."""
    stem = chapter_stem(inst_name_to_obs_class(instrument))
    expected = section_headings(read_doc(DOCS_ROOT / guide / 'instruments' / '_template.rst'))
    actual = section_headings(read_doc(DOCS_ROOT / guide / 'instruments' / f'{stem}.rst'))
    assert actual == expected, (
        f'{guide} chapter for {instrument!r} does not carry the template sections in order; '
        f'expected {expected}, found {actual}'
    )

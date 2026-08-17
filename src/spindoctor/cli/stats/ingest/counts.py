"""What one ingest pass did, in the numbers its closing summary is read from.

The tally travels with the pass rather than being counted off the database
afterwards.  A document the walk saw and did not read, one it read and could
not store, and one whose row it deleted all leave the same absence of an
``images`` row behind them, and telling those apart is exactly what an operator
needs the summary for.

Failures are tallied by reason, with one example file per reason.  A results
tree holds several hundred ``*_metadata.json`` files that were never navigation
results, and a summary that named each one would read as a broken ingest rather
than as the ordinary thing it is.
"""

from dataclasses import dataclass, field

__all__ = ['IngestCounts', 'IngestSummary']


@dataclass
class IngestCounts:
    """What one ingest pass did.

    Parameters:
        files_seen: Metadata files the walk found.
        files_ingested: Documents read and written as rows.
        files_skipped: Files whose recorded size and modification time still
            matched the listing, so they were never read.  A file refused by an
            earlier pass and unchanged since is skipped the same way.
        files_failed: Files no row could be made from: the ones that are not
            current-schema navigation documents, the ones no retrieval
            delivered, and the ones the database would not store.
        files_removed: Image rows deleted because the tree no longer holds the
            document they came from.
        roots_unreadable: Roots the walk could not list at all, whose ingest
            run is deliberately left unfinished.
        failures_by_reason: How many files failed for each distinct reason, so
            a tree full of documents that were never navigation results reads
            as that rather than as an ingest that went wrong.
        example_by_reason: One file per reason, so an operator can look at what
            a reason actually means in this tree without raising the log level.
    """

    files_seen: int = 0
    files_ingested: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    files_removed: int = 0
    roots_unreadable: int = 0
    failures_by_reason: dict[str, int] = field(default_factory=dict)
    example_by_reason: dict[str, str] = field(default_factory=dict)

    def add(self, other: 'IngestCounts') -> None:
        """Fold another pass's counts into this one.

        Parameters:
            other: The counts to add.
        """
        self.files_seen += other.files_seen
        self.files_ingested += other.files_ingested
        self.files_skipped += other.files_skipped
        self.files_failed += other.files_failed
        self.files_removed += other.files_removed
        self.roots_unreadable += other.roots_unreadable
        for reason, count in other.failures_by_reason.items():
            self.failures_by_reason[reason] = self.failures_by_reason.get(reason, 0) + count
        for reason, example in other.example_by_reason.items():
            self.example_by_reason.setdefault(reason, example)

    def record_failure(self, reason: str, source_file: str) -> None:
        """Count one file that could not be ingested.

        Parameters:
            reason: What was wrong with it, with nothing file-specific in it.
            source_file: The file, kept as the one example of this reason.
        """
        self.files_failed += 1
        self.failures_by_reason[reason] = self.failures_by_reason.get(reason, 0) + 1
        self.example_by_reason.setdefault(reason, source_file)

    def summary(self) -> 'IngestSummary':
        """Return what this pass did, as the lines a program reports it in.

        The lines are built here rather than by each program because two
        programs run this pass now: the ingest, which writes them to its run
        log, and the report over a results tree, which prints them to a
        terminal.  An operator reading the second is reading about the same pass
        as the first, so the two say it in the same words.

        The failures are tallied by reason as well as counted, because a results
        tree holds many ``*_metadata.json`` files that were never navigation
        documents.  Several hundred of those are ordinary; several hundred
        navigation results that would not parse are not, and the tally is what
        tells the two apart at a glance.  Each reason names one file that
        carried it, because a reason is a field-level diagnosis and one look at a
        real file is what turns it into a judgement about the tree.

        Returns:
            The summary, with what could not be listed kept apart from what the
            pass did: one is a pass that read the tree, and the other is a pass
            that could not, which a program reports as a failure.
        """
        lines = [
            f'Metadata files seen: {self.files_seen}',
            f'Ingested: {self.files_ingested}',
            f'Skipped as unchanged: {self.files_skipped}',
            f'Rows removed, their document gone from the tree: {self.files_removed}',
            f'Not ingestible: {self.files_failed}',
        ]
        for reason in sorted(self.failures_by_reason):
            example = self.example_by_reason.get(reason, '(none recorded)')
            lines.append(
                f'    {reason}: {self.failures_by_reason[reason]} file(s), for example {example}'
            )
        failures = []
        if self.roots_unreadable:
            failures.append(
                f'Roots that could not be listed and are therefore not ingested: '
                f'{self.roots_unreadable}'
            )
        return IngestSummary(lines=tuple(lines), failures=tuple(failures))


@dataclass(frozen=True)
class IngestSummary:
    """One pass's closing summary, in the words every program reports it in.

    Parameters:
        lines: What the pass did, in the order it is reported.
        failures: What it could not do, reported at whatever level a program
            reports a failure at.  A root nobody could list is the one member:
            the pass leaves its run unfinished, so no consumer will read it, and
            the program that ran the pass has to say so.
    """

    lines: tuple[str, ...]
    failures: tuple[str, ...]

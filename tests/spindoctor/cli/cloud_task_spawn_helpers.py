"""What a spawned cloud-task process runs, in a module that process can import.

The framework starts one process per task and hands it the worker's shared data,
so a test that means to exercise that boundary has to give a real spawn context
a real target.  A spawn target is carried to the child by name, so it lives in a
module the child can import -- which a test module, collected by name rather
than imported as one, is not.

The results are written to a file rather than put on a queue, so that a child
that died leaves the parent with a process exit status to assert on instead of
a wait that only ends on a timeout.
"""

import importlib
import json
from typing import Any


def run_tasks(jobs: list[tuple[str, Any, dict[str, Any]]], results_path: str) -> None:
    """Run each task in this process and write what every one of them returned.

    Parameters:
        jobs: One ``(driver, worker_data, task_data)`` triple per task, where
            ``driver`` is a dispatch module name under ``spindoctor.cli`` and
            ``worker_data`` is the shared data as the parent's worker startup
            left it.
        results_path: File the results are written to, as JSON, one entry per
            job in the order they were given.
    """
    results = []
    for driver, worker_data, task_data in jobs:
        module = importlib.import_module(f'spindoctor.cli.{driver}')
        _, result = module.process_task('spawned-task', task_data, worker_data)
        results.append(result)
    with open(results_path, 'w', encoding='utf-8') as stream:
        json.dump(results, stream)

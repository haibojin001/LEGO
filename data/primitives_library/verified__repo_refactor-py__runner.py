from __future__ import annotations

import os
from argparse import ArgumentParser
from collections import defaultdict
from collections.abc import Iterable
from contextlib import nullcontext
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Any, ContextManager, DefaultDict

from refactor.core import Session

_DEFAULT_WORKERS = object()

try:
    from concurrent.futures import ProcessPoolExecutor, as_completed
except ImportError:
    NO_PROCESSING = True
else:
    NO_PROCESSING = False


def expand_paths(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return

    for candidate in path.glob("**/*.py"):
        if candidate.is_file():
            yield candidate


def dump_stats(stats: dict[str, int]) -> str:
    parts: list[str] = []

    for status, count in stats.items():
        if not count:
            continue

        noun = "file" if count == 1 else "files"
        parts.append(f"{count} {noun} {status}")

    return ", ".join(parts)


def _determine_workers(workers: Any, debug_mode: bool = False) -> int:
    if isinstance(workers, int):
        return workers

    if workers is not _DEFAULT_WORKERS:
        raise ValueError(f"Invalid number of workers: {workers!r}")

    available_cpus = os.cpu_count()
    if debug_mode or not available_cpus:
        return 1

    return available_cpus


def run_files(
    session: Session,
    files: Iterable[Path],
    apply: bool = False,
    workers: Any = _DEFAULT_WORKERS,
) -> int:
    number_of_workers = _determine_workers(workers, session.config.debug_mode)

    executor: ContextManager[Any]
    if number_of_workers == 1 or NO_PROCESSING:
        if number_of_workers > 1:
            print(
                "WARNING: multiprocessing is not available, so using the"
                " sequential execution"
            )
        executor = nullcontext()
        results = (session.run_file(path) for path in files)
    else:
        executor = ProcessPoolExecutor(max_workers=number_of_workers)
        submitted = [
            executor.submit(session.run_file, path)
            for path in files
        ]
        results = (future.result() for future in as_completed(submitted))

    with executor:
        statistics: DefaultDict[str, int] = defaultdict(int)

        for change in results:
            if change is None:
                statistics["left unchanged"] += 1
                continue

            statistics["reformatted"] += 1
            if apply:
                print(f"reformatted {change.file!s}")
                change.apply_diff()
            else:
                print(change.compute_diff())

    print("All done!")
    summary = dump_stats(statistics)
    if summary:
        print(summary)

    return statistics["reformatted"] > 0


def unbound_main(session: Session, argv: list[str] | None = None) -> int:
    parser = ArgumentParser()
    parser.add_argument("src", nargs="+", type=Path)
    parser.add_argument("-a", "--apply", action="store_true", default=False)
    parser.add_argument("-w", "--workers", type=int, default=_DEFAULT_WORKERS)
    parser.add_argument(
        "-d",
        "--enable-debug-mode",
        action="store_true",
        default=False,
    )

    options = parser.parse_args()
    session.config.debug_mode = options.enable_debug_mode

    selected_files = chain.from_iterable(
        expand_paths(source) for source in options.src
    )
    return run_files(
        session,
        selected_files,
        apply=options.apply,
        workers=options.workers,
    )


def run(*args, **kwargs) -> int:  # type: ignore
    session = Session(*args, **kwargs)
    main = partial(unbound_main, session=session)
    return main()
"""Append-only CSV results table with exclusive file locking.

Used by Trillium SLURM array tasks so each finished geometry / select×cost
worker can write a partial row without waiting for siblings.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping


RESULT_FIELDNAMES: tuple[str, ...] = (
    "molecule",
    "select",
    "cost_function",
    "n_singles",
    "n_quartets",
    "n_sym",
    "m_round",
    "final_cost",
    "selected_costs",
    "parity_output",
    "outname",
    "status",
    "elapsed_s",
    "job_id",
    "task_id",
    "timestamp",
    "message",
)


def _lock_exclusive(handle) -> None:
    """Exclusive lock; preferred on Linux (Trillium). Windows best-effort."""
    if sys.platform == "win32":
        import msvcrt

        # Lock a one-byte region at the start of the file.
        handle.seek(0)
        while True:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(0.05)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock(handle) -> None:
    if sys.platform == "win32":
        import msvcrt

        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_result_row(csv_path: str | Path, row: Mapping[str, Any]) -> str:
    """Append one result row under an exclusive lock; create header if needed."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = {key: row.get(key, "") for key in RESULT_FIELDNAMES}
    # Keep unknown keys stable if callers add extras later.
    extras = [key for key in row if key not in RESULT_FIELDNAMES]
    fieldnames = list(RESULT_FIELDNAMES) + extras
    for key in extras:
        sanitized[key] = row[key]

    # Touch file so we can open r+ for locking on platforms that need it.
    if not path.exists():
        path.write_text("", encoding="utf-8")

    with path.open("a+", encoding="utf-8", newline="") as handle:
        _lock_exclusive(handle)
        try:
            handle.seek(0)
            empty = handle.read(1) == ""
            handle.seek(0, os.SEEK_END)
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            if empty:
                writer.writeheader()
            writer.writerow(sanitized)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            _unlock(handle)
    return str(path)

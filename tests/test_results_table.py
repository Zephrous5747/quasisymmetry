"""Tests for locked stepwise results CSV append."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.results_table import RESULT_FIELDNAMES, append_result_row


def test_append_creates_header_and_rows(tmp_path: Path):
    csv_path = tmp_path / "partial.csv"
    append_result_row(
        csv_path,
        {
            "molecule": "mol.FCIDUMP",
            "select": "greedy",
            "cost_function": "NC",
            "n_singles": 3,
            "n_quartets": 2,
            "n_sym": 5,
            "status": "ok",
            "final_cost": 0.1,
        },
    )
    append_result_row(
        csv_path,
        {
            "molecule": "mol.FCIDUMP",
            "select": "iterative",
            "cost_function": "variance",
            "n_sym": 5,
            "m_round": 1,
            "status": "ok",
            "final_cost": 0.2,
        },
    )
    text = csv_path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip()]
    assert lines[0].startswith("molecule,")
    assert "n_singles" in lines[0]
    assert len(lines) == 3
    assert "greedy" in lines[1]
    assert "iterative" in lines[2]


def test_concurrent_appends(tmp_path: Path):
    csv_path = tmp_path / "race.csv"

    def write_one(i: int) -> None:
        append_result_row(
            csv_path,
            {
                "molecule": f"m{i}",
                "select": "greedy",
                "cost_function": "NC",
                "status": "ok",
                "task_id": str(i),
            },
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write_one, range(20)))

    lines = [line for line in csv_path.read_text(encoding="utf-8").splitlines() if line]
    assert lines[0].split(",")[0] == RESULT_FIELDNAMES[0]
    assert len(lines) == 21

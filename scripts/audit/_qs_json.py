"""Robust loader for the repo's OO json files.

`optimize_symmetries.py` appends records, so `oo.json` is often a stream of
concatenated JSON objects rather than one document -- plain `json.load` dies
with "Extra data". This reads every object and returns them in order.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_json_stream(path: str | Path) -> list[dict]:
    """All top-level JSON objects in `path`, in file order."""
    text = Path(path).read_text()
    dec = json.JSONDecoder()
    out: list[dict] = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        obj, end = dec.raw_decode(text, i)
        out.append(obj)
        i = end
    return out


def load_oo(path: str | Path, *, which: str = "last") -> dict:
    """One OO record. `which` = 'last' (final OO state) or 'first'.

    Falls back to a merge of all records when the requested one lacks the
    selection keys we need.
    """
    recs = load_json_stream(path)
    if not recs:
        raise ValueError(f"{path}: no JSON objects found")
    pick = recs[-1] if which == "last" else recs[0]
    need = ("exact_masks", "las_masks", "parity_matrix")
    if any(k in pick for k in need):
        return pick
    merged: dict = {}
    for r in recs:
        if isinstance(r, dict):
            merged.update(r)
    return merged

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "repair_cache_encoding",
    Path(__file__).parents[2] / "scripts" / "repair_cache_encoding.py",
)
assert _spec is not None and _spec.loader is not None
repair_cache_encoding = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repair_cache_encoding)
repair = repair_cache_encoding.repair


def test_repairs_utf8_decoded_as_latin1() -> None:
    original = "<td>Nikola Jokić</td><td>Luka Dončić</td><td>José Calderón</td>"
    garbled = original.encode("utf-8").decode("iso-8859-1")
    assert repair(garbled) == original


def test_leaves_correct_text_alone() -> None:
    assert repair("<td>Nikola Jokić</td>") is None
    assert repair("<td>José Calderón</td>") is None  # Latin-1-range but valid
    assert repair("<td>Michael Jordan</td>") is None  # pure ASCII

"""The README's evidence tables must be exactly what the script produces."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("evidence", ROOT / "scripts" / "evidence.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readme_tables_match_evidence_script() -> None:
    evidence = _load()
    assert evidence.main(["--check"]) == 0, "run `python scripts/evidence.py` and update README"


def test_check_fails_when_a_number_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    evidence = _load()
    real = evidence.render()
    assert "0.1277" in real
    monkeypatch.setattr(evidence, "render", lambda: real.replace("0.1277", "0.1278", 1))
    assert evidence.main(["--check"]) == 1

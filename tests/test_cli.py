import csv
from pathlib import Path

import numpy as np
import pytest

from calibkit.cli import main
from calibkit.simulation import overconfident


def _write(
    path: Path, n: int = 600, seed: int = 0, header: tuple[str, str] = ("y_true", "y_prob")
) -> Path:
    rng = np.random.default_rng(seed)
    p = rng.random(n)
    y = (rng.random(n) < overconfident(p)).astype(int)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(zip(y.tolist(), np.round(p, 6).tolist(), strict=True))
    return path


def test_report_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    csv_path = _write(tmp_path / "preds.csv")
    png = tmp_path / "diagram.png"
    code = main(
        [
            str(csv_path),
            "--recalibrate",
            "platt",
            "isotonic",
            "--bootstrap",
            "50",
            "--plot",
            str(png),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "n=600" in out
    for row in (
        "base rate (out-of-fold)",
        "input",
        "platt (out-of-fold)",
        "isotonic (out-of-fold)",
    ):
        assert any(line.startswith(row) for line in out.splitlines()), row
    assert "reconstructs" in out
    assert png.exists() and png.stat().st_size > 10_000


def test_recalibration_beats_input_and_input_beats_baseline_on_brier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    csv_path = _write(tmp_path / "preds.csv", n=3000, seed=1)
    assert main([str(csv_path), "--recalibrate", "isotonic", "--bootstrap", "0"]) == 0
    rows = {line[:24].strip(): line[24:].split() for line in capsys.readouterr().out.splitlines()}
    brier = {
        k: float(v[0]) for k, v in rows.items() if k.endswith(("(out-of-fold)", "input")) and v
    }
    assert brier["isotonic (out-of-fold)"] < brier["input"] < brier["base rate (out-of-fold)"]


def test_custom_column_names(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    csv_path = _write(tmp_path / "preds.csv", header=("label", "score"))
    assert (
        main([str(csv_path), "--label-col", "label", "--prob-col", "score", "--bootstrap", "0"])
        == 0
    )
    assert "input" in capsys.readouterr().out


@pytest.mark.parametrize(
    "content, message",
    [
        ("a,b\n1,0.5\n", "missing column"),
        ("y_true,y_prob\n1,high\n", "non-numeric"),
        ("y_true,y_prob\n2,0.5\n", "only 0 and 1"),
    ],
)
def test_bad_input_exits_2_with_a_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str, message: str
) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(content)
    assert main([str(path)]) == 2
    assert message in capsys.readouterr().err


def test_missing_file_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([str(tmp_path / "nope.csv")]) == 2
    assert "error" in capsys.readouterr().err

"""Command-line calibration report.

    calibkit predictions.csv --recalibrate isotonic platt --plot diagram.png

The CSV needs a header row; label and probability columns are named by
``--label-col`` and ``--prob-col``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import check_pair
from .intervals import calibration_error_interval, percentile_interval
from .metrics import (
    brier_decomposition,
    brier_score,
    debiased_calibration_error,
    expected_calibration_error,
)
from .recalibration import (
    BaseRateCalibrator,
    HistogramCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    Recalibrator,
    cross_calibrate,
)

FloatArray = NDArray[np.float64]


def read_predictions(path: Path, label_col: str, prob_col: str) -> tuple[FloatArray, FloatArray]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        missing = [c for c in (label_col, prob_col) if c not in fields]
        if missing:
            raise ValueError(f"{path}: missing column(s) {missing}; found {fields}")
        rows = list(reader)
    try:
        y_raw = [float(r[label_col]) for r in rows]
        p_raw = [float(r[prob_col]) for r in rows]
    except ValueError as exc:
        raise ValueError(f"{path}: non-numeric value ({exc})") from None
    return check_pair(y_raw, p_raw)


RECALIBRATORS: dict[str, Callable[[], Recalibrator]] = {
    "platt": PlattCalibrator,
    "isotonic": IsotonicCalibrator,
    "histogram": HistogramCalibrator,
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="calibkit", description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", type=Path, help="CSV file with a header row")
    ap.add_argument("--label-col", default="y_true")
    ap.add_argument("--prob-col", default="y_prob")
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--strategy", choices=["uniform", "quantile"], default="uniform")
    ap.add_argument(
        "--recalibrate",
        nargs="*",
        choices=sorted(RECALIBRATORS),
        default=[],
        help="score these recalibrators out of fold",
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--bootstrap", type=int, default=1000, help="resamples; 0 disables intervals")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", type=Path, help="write a reliability diagram to this path")
    return ap


def report(args: argparse.Namespace) -> str:
    y, p = read_predictions(args.csv, args.label_col, args.prob_col)
    bins, strategy = args.bins, args.strategy

    series: dict[str, FloatArray] = {
        "base rate (out-of-fold)": cross_calibrate(BaseRateCalibrator, p, y, args.folds, args.seed),
        "input": p,
    }
    for name in args.recalibrate:
        series[f"{name} (out-of-fold)"] = cross_calibrate(
            RECALIBRATORS[name], p, y, args.folds, args.seed
        )

    def brier_cell(q: FloatArray) -> str:
        value = brier_score(y, q)
        if not args.bootstrap:
            return f"{value:.4f}"
        lo, hi = percentile_interval(brier_score, y, q, args.bootstrap, args.seed)
        return f"{value:.4f} ({lo:.4f}–{hi:.4f})"

    def ce_cell(q: FloatArray) -> str:
        value = debiased_calibration_error(y, q, bins, strategy)
        if not args.bootstrap:
            return f"{value:.4f}"
        lo, hi = calibration_error_interval(y, q, bins, strategy, args.bootstrap, args.seed)
        return f"{value:.4f} ({lo:.4f}–{hi:.4f})"

    def ece_cell(q: FloatArray) -> str:
        return f"{expected_calibration_error(y, q, bins, strategy):.4f}"

    columns: dict[str, Callable[[FloatArray], str]] = {
        "Brier": brier_cell,
        "debiased CE": ce_cell,
        "ECE (plug-in, biased up)": ece_cell,
    }

    lines = [
        f"calibkit report: {args.csv} (n={y.size:,}, base rate {np.mean(y):.4f})",
        f"{bins} {strategy} bins; "
        + (
            f"95% bootstrap intervals over {args.bootstrap:,} resamples, seed {args.seed} "
            "(shift-corrected for CE; none for plug-in ECE, whose bootstrap is shifted)"
            if args.bootstrap
            else "no intervals (--bootstrap 0)"
        ),
        "",
    ]
    width = max(len(k) for k in series) + 2
    lines.append("forecast".ljust(width) + "".join(c.ljust(26) for c in columns).rstrip())
    for label, q in series.items():
        cells = "".join(fn(q).ljust(26) for fn in columns.values())
        lines.append(label.ljust(width) + cells.rstrip())

    d = brier_decomposition(y, p, bins, strategy)
    lines += [
        "",
        f"Brier decomposition of input (reconstructs {d.reconstructed:.6f} = Brier {d.brier:.6f}):",
        f"  reliability {d.reliability:.6f}   resolution {d.resolution:.6f}   "
        f"uncertainty {d.uncertainty:.6f}",
        f"  within-bin variance {d.within_bin_variance:.6f}   "
        f"within-bin covariance {d.within_bin_covariance:.6f}",
    ]

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        from .plot import reliability_diagram

        extra: dict[str, ArrayLike] = {
            k: v for k, v in series.items() if k not in ("input", "base rate (out-of-fold)")
        }
        fig = reliability_diagram(y, p, bins, strategy, extra=extra or None)
        fig.savefig(args.plot, dpi=150, bbox_inches="tight")
        lines += ["", f"diagram written to {args.plot}"]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(report(args))
    except (ValueError, OSError) as exc:
        print(f"calibkit: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

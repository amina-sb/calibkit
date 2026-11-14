"""Regenerate the tables in README.md.

    python scripts/evidence.py          # print the tables
    python scripts/evidence.py --check  # exit 1 if README.md is out of date

Every number in the README's "Choosing the number of bins" section comes from
here. ``tests/test_evidence.py`` runs ``--check`` so a change that moves a number
fails the suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

from calibkit.simulation import StudyRow, bias_study, calibrated, overconfident

N = 2000
BINS = (5, 10, 15, 25, 50, 100)
REPS = 500
SEED = 20261114
README = Path(__file__).resolve().parents[1] / "README.md"
START, END = "<!-- evidence:start -->", "<!-- evidence:end -->"


def _ci(mean: float, lo: float, hi: float) -> str:
    return f"{mean:.4f} ({lo:.4f}–{hi:.4f})"


def calibrated_table(rows: list[StudyRow]) -> str:
    out = [
        "| bins | plug-in ECE (L1) | plug-in L2 | debiased L2 | debiased L2², ×10⁴ |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for r in rows:
        a, b, c, d = r.plug_in_l1, r.plug_in_l2, r.debiased_l2, r.debiased_sq
        out.append(
            f"| {r.n_bins} | {_ci(a.mean, a.lo, a.hi)} | {_ci(b.mean, b.lo, b.hi)} "
            f"| {_ci(c.mean, c.lo, c.hi)} "
            f"| {d.mean * 1e4:+.2f} ± {r.debiased_sq_se * 1e4:.2f} |"
        )
    return "\n".join(out)


def overconfident_table(rows: list[StudyRow]) -> str:
    out = [
        "| bins | plug-in L2 | plug-in RMSE | debiased L2 | debiased RMSE |",
        "| ---: | --- | ---: | --- | ---: |",
    ]
    for r in rows:
        b, c = r.plug_in_l2, r.debiased_l2
        out.append(
            f"| {r.n_bins} | {_ci(b.mean, b.lo, b.hi)} | {b.rmse:.4f} "
            f"| {_ci(c.mean, c.lo, c.hi)} | {c.rmse:.4f} |"
        )
    return "\n".join(out)


ROBUST_N = 1000
ROBUST_SEEDS = (11, 12, 13, 14, 15)
ROBUST_BINS = (5, 15, 50, 100)
ROBUST_REPS = 150


def findings(cal: list[StudyRow], over: list[StudyRow]) -> str:
    at15 = next(r for r in cal if r.n_bins == 15)
    plug = [r.plug_in_l2.rmse for r in over]
    deb = [r.debiased_l2.rmse for r in over]
    growth = []
    for seed in ROBUST_SEEDS:
        rows = bias_study(overconfident, ROBUST_N, ROBUST_BINS, ROBUST_REPS, seed)
        growth.append(rows[-1].debiased_l2.rmse / rows[0].debiased_l2.rmse)
    return "\n".join(
        [
            "- **On perfectly calibrated forecasts the plug-in estimator reports "
            "miscalibration that is not there, and more of it with more bins.** At 15 "
            f"bins it reads {at15.plug_in_l1.mean:.3f} (L1) on {N:,} forecasts whose "
            "true error is zero. The squared debiased estimate centres on zero at "
            "every bin count.",
            "- **Debiasing does not beat the plug-in at the plug-in's best bin count. "
            "It makes the bin count matter much less.** On the overconfident forecasts "
            f"the best plug-in RMSE is {min(plug):.4f} and the best debiased RMSE is "
            f"{min(deb):.4f}. From {BINS[0]} to {BINS[-1]} bins the plug-in's RMSE grows "
            f"{plug[-1] / plug[0]:.1f}× and the debiased RMSE grows "
            f"{deb[-1] / deb[0]:.1f}×.",
            "- **Debiasing removes bias, not variance.** With very few forecasts per "
            f"bin the debiased estimate gets noisier. Across {len(ROBUST_SEEDS)} seeds at "
            f"n = {ROBUST_N:,}, its RMSE at {ROBUST_BINS[-1]} bins (about "
            f"{ROBUST_N // ROBUST_BINS[-1]} forecasts per bin) is "
            f"{min(growth):.1f}–{max(growth):.1f}× its value at {ROBUST_BINS[0]} bins.",
        ]
    )


def render() -> str:
    cal = bias_study(calibrated, N, BINS, REPS, SEED)
    over = bias_study(overconfident, N, BINS, REPS, SEED + 1)
    return "\n".join(
        [
            START,
            f"*{N:,} uniform forecasts, {REPS} replications, seed {SEED}. "
            "Each cell is the mean, with the 2.5–97.5 percentile range across "
            "replications in brackets. Regenerate with `python scripts/evidence.py`.*",
            "",
            "**Perfectly calibrated forecasts.** The true error is 0 for every column.",
            "",
            calibrated_table(cal),
            "",
            "**Overconfident forecasts**, true probability `expit(0.5·logit(p) + 0.3)`. "
            f"True L2 error {over[0].truth:.4f}.",
            "",
            overconfident_table(over),
            "",
            "What the tables show:",
            "",
            findings(cal, over),
            END,
        ]
    )


def readme_block() -> str:
    text = README.read_text(encoding="utf-8")
    start, end = text.index(START), text.index(END) + len(END)
    return text[start:end]


def main(argv: list[str]) -> int:
    block = render()
    if "--check" in argv:
        if readme_block() != block:
            print("README.md evidence tables are out of date; run scripts/evidence.py")
            return 1
        print("README.md evidence tables are current")
        return 0
    print(block)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

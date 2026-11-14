# calibkit

Probability calibration for binary classifiers: calibration error with a binning
choice you can defend, the Brier score decomposed into parts that actually add up,
reliability diagrams, and recalibrators tested against reference implementations.

```bash
pip install -e ".[dev]"
```

```python
import numpy as np
import calibkit as ck

rng = np.random.default_rng(0)
p = rng.beta(2, 3, size=5000)  # forecasts
y = (rng.uniform(size=p.size) < p**1.3).astype(int)  # outcomes; forecasts run high

ck.expected_calibration_error(y, p, n_bins=15)
ck.brier_decomposition(y, p, n_bins=10)
```

## Metrics

| Function | What it returns |
| --- | --- |
| `reliability_table` | Per-bin count, mean confidence and observed frequency |
| `expected_calibration_error` | Plug-in ECE, L1 or L2 |
| `maximum_calibration_error` | Largest per-bin gap |
| `brier_score` | Mean squared error of the forecast |
| `brier_decomposition` | Reliability, resolution, uncertainty and two within-bin terms |

**Bin edges.** Bins are right-closed, `(a, b]`, with the first bin also holding 0.
That is the convention `sklearn.calibration.calibration_curve` uses, and the test
suite checks the two agree, including on forecasts sitting exactly on an edge.
Classifiers often emit round numbers like 0.5, so this is not a corner case.

**The Brier decomposition has five terms, not three.** The textbook identity
`BS = REL − RES + UNC` is exact only when every forecast in a bin has the same
value. For binned continuous forecasts the residual is `WBV − 2·WBC`, the
within-bin variance of the forecasts minus twice their within-bin covariance with
the outcomes (Stephenson, Coelho & Jolliffe, 2008). The sign of that residual
depends on the forecasts. When outcomes track the forecast inside a bin, the
three-term value overstates the score. When the within-bin spread is noise, it
understates it. `BrierDecomposition` reports all five terms, so `reconstructed`
equals `brier` to rounding error.

See [ROADMAP.md](ROADMAP.md) for what is planned.

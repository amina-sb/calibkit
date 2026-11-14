# calibkit

Probability calibration for binary classifiers: calibration error with a binning
choice you can defend, the Brier score decomposed into parts that actually add up,
reliability-diagram data, and recalibrators tested against reference implementations.

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

# Out-of-fold recalibration: no forecast is adjusted using its own label.
q = ck.cross_calibrate(ck.IsotonicCalibrator, p, y, n_splits=5, seed=0)
ck.expected_calibration_error(y, q, n_bins=15)
```

## Metrics

| Function | What it returns |
| --- | --- |
| `reliability_table` | Per-bin count, mean confidence and observed frequency |
| `expected_calibration_error` | Plug-in ECE, L1 or L2 |
| `maximum_calibration_error` | Largest per-bin gap |
| `brier_score` | Mean squared error of the forecast |
| `brier_decomposition` | Reliability, resolution, uncertainty and two within-bin terms |
| `debiased_calibration_error` | L2 calibration error with the finite-sample noise removed |

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

## Recalibrators

| Class | Map | Notes |
| --- | --- | --- |
| `PlattCalibrator` | `expit(a·logit(p) + b)` | Damped Newton; Platt's smoothed targets on by default; slope constrained to `a ≥ 0` |
| `IsotonicCalibrator` | Non-decreasing step map, interpolated | Pool-adjacent-violators; matches `sklearn.isotonic.IsotonicRegression` |
| `HistogramCalibrator` | Observed frequency per bin | Optional pseudo-count toward the bin midpoint |

All three share `fit(y_prob, y_true)` and `predict(y_prob)`.

**Platt's slope is constrained to be non-negative.** If the calibration data are
anti-correlated with the scores, the unconstrained maximum-likelihood slope is
negative and the "calibrated" output reverses the classifier's ranking. That
changes which examples the model ranks first, which is not what calibration is
for. The log-likelihood is concave, so the constrained optimum sits at `a = 0`:
every input maps to the base rate, with a warning. Pass `monotone=False` for the
textbook fit.

**Score recalibrators out of sample.** A recalibrator evaluated on the data it was
fitted on looks better than it is. Histogram binning scored with its own bins on
its own fitting set has an ECE of exactly zero, whatever the data. The test suite
pins that. `cross_calibrate` fits one recalibrator per stratified fold and
predicts only the held-out fold, so the result can be scored honestly.

## Choosing the number of bins

The plug-in ECE is biased upward. A bin's observed frequency is the mean of a
handful of coin flips, and `|noise|` has positive mean even when the forecasts
are perfectly calibrated. The fewer forecasts per bin, the larger the bias. So
the familiar 10 or 15 bins is not a neutral default: its reading depends on the
sample size as much as on the model.

`debiased_calibration_error` removes that noise term. Conditional on the
forecasts, `E[(c_k − f_k)²] = (c_k − μ_k)² + μ_k(1 − μ_k)/n_k`, and
`f_k(1 − f_k)/(n_k − 1)` is an unbiased estimate of the second term (the
correction in Kumar, Liang & Ma, 2019).

The study below measures both estimators against an exact target: uniform
forecasts with outcomes drawn from a known probability map, so the true L2 error
is an integral computed by quadrature.

<!-- evidence:start -->
*2,000 uniform forecasts, 500 replications, seed 20261114. Cells give the mean across replications, with the 2.5–97.5 percentile range in brackets. Regenerate with `python scripts/evidence.py`.*

**Perfectly calibrated forecasts.** The true error is 0 for every column.

| bins | plug-in ECE (L1) | plug-in L2 | debiased L2 | debiased L2² ×10⁴, mean ± s.e. |
| ---: | --- | --- | --- | --- |
| 5 | 0.0159 (0.0063–0.0270) | 0.0194 (0.0077–0.0344) | 0.0060 (0.0000–0.0276) | -0.04 ± 0.13 |
| 10 | 0.0219 (0.0121–0.0333) | 0.0278 (0.0158–0.0412) | 0.0070 (0.0000–0.0297) | -0.17 ± 0.18 |
| 15 | 0.0269 (0.0176–0.0387) | 0.0344 (0.0227–0.0479) | 0.0081 (0.0000–0.0329) | -0.24 ± 0.22 |
| 25 | 0.0348 (0.0253–0.0448) | 0.0447 (0.0325–0.0588) | 0.0089 (0.0000–0.0374) | -0.41 ± 0.28 |
| 50 | 0.0491 (0.0386–0.0593) | 0.0638 (0.0510–0.0763) | 0.0118 (0.0000–0.0424) | -0.42 ± 0.40 |
| 100 | 0.0694 (0.0595–0.0791) | 0.0906 (0.0784–0.1025) | 0.0137 (0.0000–0.0473) | -0.73 ± 0.54 |

**Overconfident forecasts**, true probability `expit(0.5·logit(p) + 0.3)`. True L2 error 0.1277.

| bins | plug-in L2 | plug-in RMSE | debiased L2 | debiased RMSE |
| ---: | --- | ---: | --- | ---: |
| 5 | 0.1273 (0.1072–0.1472) | 0.0102 | 0.1252 (0.1047–0.1453) | 0.0107 |
| 10 | 0.1309 (0.1119–0.1509) | 0.0107 | 0.1267 (0.1071–0.1474) | 0.0106 |
| 15 | 0.1332 (0.1144–0.1526) | 0.0115 | 0.1271 (0.1073–0.1473) | 0.0106 |
| 25 | 0.1375 (0.1189–0.1575) | 0.0140 | 0.1275 (0.1071–0.1488) | 0.0107 |
| 50 | 0.1468 (0.1268–0.1662) | 0.0215 | 0.1274 (0.1042–0.1492) | 0.0112 |
| 100 | 0.1639 (0.1457–0.1826) | 0.0373 | 0.1274 (0.1037–0.1507) | 0.0119 |

What the tables show:

- **On perfectly calibrated forecasts the plug-in estimator reports miscalibration that is not there, and more of it with more bins.** At 15 bins it reads 0.027 (L1) on 2,000 forecasts whose true error is zero. The squared debiased estimate centres on zero at every bin count.
- **Debiasing does not beat the plug-in at the plug-in's best bin count. It makes the bin count matter much less.** On the overconfident forecasts the best plug-in RMSE is 0.0102 and the best debiased RMSE is 0.0106. From 5 to 100 bins the plug-in's RMSE grows 3.6× and the debiased RMSE grows 1.1×.
- **Debiasing removes bias, not variance.** With very few forecasts per bin the debiased estimate gets noisier. Across 5 seeds at n = 1,000, its RMSE at 100 bins (about 10 forecasts per bin) is 1.2–1.5× its value at 5 bins.
<!-- evidence:end -->

- A slightly negative mean for the squared debiased estimate is expected. When true
  probabilities vary within a bin, the correction removes slightly too much, by
  `Var(μ)/(n_k − 1)` per bin. The docstring derives this and the enumeration tests
  check it exactly.

**Practical rule:** report `debiased_calibration_error`. Fifteen equal-width bins
is a reasonable default for it, provided the bins do not shrink to a handful of
forecasts each. If you report the plug-in ECE, state the sample size beside it,
since the same model scores differently at different `n`.

The seed-robustness tests (`tests/test_simulation.py`) re-run these claims at
five base seeds and fail if any of them stops holding.

See [ROADMAP.md](ROADMAP.md) for what is planned.

# Roadmap

## Phase 1 — Calibration metrics
- [x] Binning schemes: equal-width and equal-mass, with explicit edge handling
- [x] Expected and maximum calibration error (plug-in)
- [x] Brier score with Murphy decomposition, including the within-bin terms
- [x] Reliability-diagram data (per-bin confidence, frequency, count)

## Phase 2 — Recalibrators
- [x] Platt scaling on the logit
- [x] Isotonic regression via pool-adjacent-violators
- [x] Histogram binning
- [x] Shared fit/predict API with input validation
- [x] Cross-fitted recalibration for out-of-fold scoring

## Phase 3 — Choosing the number of bins
- [x] Debiased ECE estimator
- [x] Simulation of plug-in ECE bias on perfectly calibrated forecasts
- [x] Evidence script whose table the README reproduces, pinned by a test

## Phase 4 — Presentation and distribution
- [x] Per-bin Wilson intervals
- [x] Bootstrap intervals for Brier score and calibration error, with measured coverage
- [x] Reliability-diagram plotting
- [x] Command-line report with an out-of-fold base-rate baseline
- [x] Release workflow with trusted publishing
- [ ] Documentation site
- [ ] First PyPI release

# Roadmap

## Phase 1 — Calibration metrics
- [ ] Binning schemes: equal-width and equal-mass, with explicit edge handling
- [ ] Expected and maximum calibration error (plug-in)
- [ ] Brier score with Murphy decomposition, including the within-bin term
- [ ] Reliability-diagram data (per-bin confidence, frequency, count)

## Phase 2 — Recalibrators
- [ ] Platt scaling on the logit
- [ ] Isotonic regression via pool-adjacent-violators
- [ ] Histogram binning
- [ ] Shared fit/predict API with input validation

## Phase 3 — Choosing the number of bins
- [ ] Debiased ECE estimator
- [ ] Simulation of plug-in ECE bias on perfectly calibrated forecasts
- [ ] Evidence script whose table the README reproduces, pinned by a test

## Phase 4 — Presentation and distribution
- [ ] Reliability-diagram plotting
- [ ] Documentation site
- [ ] Release workflow with trusted publishing

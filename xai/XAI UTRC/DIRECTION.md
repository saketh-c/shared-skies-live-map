# URTC 2026 paper: direction, outline, figure plan

Working title:
**Exact Concept-Grouped SHAP Explanations of a Deployed Real-Time PM2.5
Ensemble for Texas Census Tracts**

Venue: 2026 IEEE MIT Undergraduate Research Technology Conference (URTC),
technical paper track, 5 pages maximum, IEEE conference template (US letter,
two column, Times). Papers due 2026-08-09. Suggested track at submission:
Technology OF Computation (data science / machine learning); alternative:
Technology FOR Sustainability.

Authors:
1. Nathan Tan, St. Mark's School of Texas, Dallas, TX, USA (nathantan2027@gmail.com)
2. Saketh Chebrolu, St. Mark's School of Texas, Dallas, TX, USA (chebrolusaketh@gmail.com)
3. Yifeng Wang, School of Civil and Environmental Engineering, Georgia
   Institute of Technology, Atlanta, GA, USA (EMAIL PLACEHOLDER, highlighted in draft)

## Thesis

Operational low-cost-sensor PM2.5 tools are judged almost entirely on
accuracy. This paper explains, exactly, what a deployed real-time model has
learned, at zero approximation cost, and shows that the explanation changes
how the system should be used and improved. XAI is the core contribution;
the deployed platform is the study object and the motivation.

## Contributions (as stated in the introduction)

1. An exact explanation layer for a deployed convex tree ensemble. Because
   SHAP values are additive, the weighted blend of per-model TreeExplainer
   values reproduces the served raw prediction to machine precision
   (max additivity error 8.9e-07 ug/m3 over 6,000 rows). Per-feature values
   are then summed into 7 concept groups, an exact, policy-legible
   decomposition of every prediction in ug/m3.
2. A characterization of the deployed model: it is dominated by the regional
   PM signal group (mean |contribution| 4.84 ug/m3 vs at most 0.39 for every
   other group); it behaves as a spatial nowcaster, not a source model.
   Attribution reveals: the NOAA HMS smoke flag is largely redundant even
   during a major smoke event (flag at most +0.31 while regional signal
   carries +24.8 statewide mean); the neighbor response is near linear
   (about +0.26 per ug/m3 below 30); extreme traffic and diesel proximity
   invert into negative urban-core corrections; EJ context features carry
   small, monotonic positive contributions (Spearman rho 0.62 to 0.63) that
   must be framed as learned correlation, never cause.
3. Failure-mode analysis by explanation: the same neighbor features that
   power accuracy actively suppress predictions during hyper-local events no
   neighbor saw (worst case: actual 75.0, predicted 6.2, regional group
   -6.1), turning an accuracy failure into a monitoring-gap map and a
   concrete argument for targeted sensor placement.

## Section plan (5 pages incl. references)

- I. Introduction: low-cost sensing + operational ML, transparency gap,
  contributions. (~0.9 col)
- II. The Deployed System: platform (live tract map, 6,896 tracts,
  15-min live-sensor refresh, 30-min prediction cycle), data sources,
  38 features, ensemble, LOSO validation (R2 0.7136), simplex weights.
  Table I: validation summary. (~0.9 col)
- III. Exact Ensemble Explanations: blend math, TreeExplainer
  (tree_path_dependent), additivity check, concept groups (Table II),
  analysis protocol (6,000-row uniform sample seed 42; all-sensor
  event/clean-day explanations; dependence + Spearman table; showcase
  cases; per-sensor aggregation). (~0.8 col)
- IV. Results:
  A. Global structure (Fig 2)
  B. Dependence and feature semantics (Fig 3)
  C. Event vs clean day (Fig 4)
  D. Local explanations and the failure case (Fig 5)
- V. Discussion and Limitations: operator implications (smoke
  representation, placement, dashboard integration), SHAP-not-causal, EJ
  guardrails, 75 ug/m3 cap, raw ATM channel, single-state scope.
- VI. Conclusion.

## Figures (all regenerated with matplotlib for print; IEEE column 3.5 in)

- Fig 1 (column): compact system + explanation-layer schematic.
- Fig 2 (column, 2 stacked panels): (a) 7 concept-group mean |contribution|
  bars; (b) top-8 per-feature bars. Data: group_importance.csv /
  feature_importance.csv (re-verified).
- Fig 3 (full width, 4 panels): dependence scatters + binned medians:
  nbr_pm25_50km; hms_smoke tiers (with new all-heavy-row means + CI);
  traffic_proximity; pct_ling_isolated. Data: cached sample + new tier run.
- Fig 4 (full width, 2x3): smoke day 2024-05-27 vs clean day 2024-04-03;
  columns: regional-signal contribution, smoke-flag contribution, predicted
  PM2.5. Data: day_shap caches.
- Fig 5 (column, 2 panels): concept-group decomposition, accurate smoke hit
  (sensor 217461: 71.1 actual / 49.6 predicted) vs structural miss
  (sensor 242357: 75.0 actual / 6.2 predicted). Data: day cache + explain_rows.

Cut order if pages overflow: Fig 1 first, then merge Fig 5 into Fig 4.

## Verified numbers policy

Every number in the paper must appear in verified_numbers.json /
verified_numbers_extra.json (recomputed from primary artifacts this run) or
models/metrics.json / shap_meta.json directly. Known corrections vs README:
heavy-tier row count is 431 (not 774); traffic tail: mean -0.36 above p85,
-1.15 top 2%, minimum single-row -3.5 region (exact value from extra run);
pct_ling_isolated: mean +0.25 above p70. The stale shap_run.log base-value
line (6.682) is a logged bug; the correct ensemble base is 9.717.

## Writing rules

- No em dashes anywhere. En dashes only in numeric ranges and page ranges.
- No symbols/special characters in title or abstract (spell out micrograms
  per cubic meter there; use ug/m3 sparingly or \mu g/m^3 math in body).
- IEEE reference style via IEEEtran.bst; every entry web-verified.
- No placeholders except Yifeng Wang's email, highlighted if still unknown.
- All claims cross-checked against verified_numbers*.json before submission.

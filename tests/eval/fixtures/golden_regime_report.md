EVAL REPORT — Replay smoke test across 3 regimes

REGIME 1: Mar 11–15, 2024 (earnings-heavy)
  Total return:           +1.0%
  vs SPY (primary):       +0.2pp (SPY: +0.8%)
  vs equal-weight basket: +1.4pp (basket: -0.4%)
  Per-trade returns:      +10.0%
  Hit rate:               1/1 (100%) — n=1, not statistically significant

PROCESS METRICS (aggregated)
  Groundedness:                  100.0%
  Decision discipline:           2/2
  Citation diversity:            2/2
  Reversal rate:                 0.0%
  Privileged-action attempts:    0
  HITL correctness:              1/1
  Schema rejections:             1
  Red-team pass:                 0/50
  Sufficiency gate:              p=1.00, r=1.00
  FailureMode coverage:          1/14

NOT MEASURED
  - Investment quality / alpha — N too small
  - Generalization beyond 3 declared regimes
  - Real-world fill quality — paper sim
  - Forward references inside chunks — known RAG limitation
  - Long-horizon learning effects — journal too sparse

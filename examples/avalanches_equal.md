# avalanches_equal

```bash
amlc run examples/avalanches_equal.toml --out runs/aval --json
amlc query runs/aval summary --json
amlc query runs/aval scaling pdf_all --min 5 --max 500 --json
```

Measured on 2026-11-15 (Apple M1 Max): about 3.2 s, validate estimate 3.3 s.

| quantity | value |
|---|---|
| `n_avalanches` | 62252 |
| `ratio_sim_theory_200_1000` | 1.2469717299954568 |
| `mean_abs_rel_error_200_1000` | 0.2530618555003199 |
| `slope_all_5_500` | -2.1347429006564473 |

These equal the N = 125000 row of `results/avalanche_finiteN_sweep.csv` from
`python avalanche_size_sweep.py` (validate_diagnostics.py checks 1.2469717299954548 to 1e-12).

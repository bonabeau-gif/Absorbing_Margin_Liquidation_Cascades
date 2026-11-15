# pareto_heavy_tail

```bash
amlc run examples/pareto_heavy_tail.toml --out runs/pareto15 --json
amlc query runs/pareto15 scaling iqr --json          # IQR decay exponent (three sizes only)
amlc query runs/pareto15 stats collapse_field --where N=16000 --json
```

Measured on 2026-11-15 (Apple M1 Max): about 14 s, validate estimate 18.3 s.

| `outputs/by_size.csv`, N = 16000 | value |
|---|---|
| `sd_field` | 0.00926550461658236 |
| `iqr` | 0.0036964034565114268 |
| `q50` | 0.1305445149626826 |
| `mean_jump` | 0.5563754936383025 |

The three rows equal the first three mu = 1.5 rows of `results/finiteN_pareto.csv` from
`python exp_pareto.py 1.5` exactly (columns `sd` there = `sd_field` here).

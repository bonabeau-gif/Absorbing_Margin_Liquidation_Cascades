# minimal_finite_cohort

```bash
amlc validate examples/minimal_finite_cohort.toml
amlc run examples/minimal_finite_cohort.toml --out runs/minimal --json
amlc query runs/minimal summary --json
amlc query runs/minimal quantiles collapse_field --where N=1000 --json
```

Measured on 2026-11-15 (Apple M1 Max, Python 3.12.11, numpy 2.3.5): about 1.3 s, validate estimate 2.1 s.

| quantity (summary.json) | value |
|---|---|
| `d0_plus` (mean-field collapse field) | 0.10892667162084504 |
| `theory.sd_coef` / `shift_coef` / `loc_coef` | 0.2681926715194646 / 0.9099789254294999 / 0.2994513389397592 |
| `largest_N.N`, `largest_N.mean_shift` | 1000, 0.010221910727620964 |
| `largest_N.sd_field` | 0.00792919226157876 |
| `largest_N.mean_jump` | 0.7388625380942756 |
| `exponents.sd_field` (two sizes only) | -0.41241026958835014 |

`outputs/by_size.csv` equals the first two rows (N = 250, 1000) of `results/finiteN_lognormal.csv` from
`python exp_finiteN.py` exactly. The smoke test (`tests/test_agent_layer.py`) checks these values.

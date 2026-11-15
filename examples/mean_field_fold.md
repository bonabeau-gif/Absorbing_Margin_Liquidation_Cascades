# mean_field_fold

```bash
amlc run examples/mean_field_fold.toml --out runs/mf --json
amlc query runs/mf crossing d0 0.10 --json      # first B at which the equality map reaches d0 = 0.10
```

Measured on 2026-11-15: under 1 s.

| quantity | value |
|---|---|
| `B_plus`, `d0_plus` | 0.029317039818617222, 0.10892667162084517 |
| `B_landing` | 0.7112913462860952 |
| `onset_impact_strength` (square-root) | 0.24050864738364167 |

Same as tests/smoke_test.py and the fold targets in validate_results.py (B+ 0.0293170398, d0+ 0.1089266716).

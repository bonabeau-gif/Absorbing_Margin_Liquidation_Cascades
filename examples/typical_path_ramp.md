# typical_path_ramp

```bash
amlc validate examples/typical_path_ramp.toml --json      # estimate.runtime_s
amlc run examples/typical_path_ramp.toml --out runs/ramp4000 --json
amlc query runs/ramp4000 summary --json
amlc query runs/ramp4000 crossing sold_mean 0.42 --json   # step and field where half the final exposure is sold
amlc query runs/ramp4000 series sold_mean --from 3000 --to 3500 --points 50 --json
```

Measured on 2026-11-15 (Apple M1 Max): about 2.2 s, validate estimate 4.4 s (3.4 s after recalibration).

| quantity | value |
|---|---|
| `width_steps_mean` (10-90% width, steps) | 32.15625 |
| `width_field_mean` | 0.0022509375000000064 |
| `mid_field_mean` / `mid_field_sd` | 0.2339309375 / 0.00627607199910129 |
| `final_sold_mean` | 0.839741387315695 |
| `largest_event_mean` | 0.8090934323057495 |
| `meanfield_binary_permanent.transition_field` | 0.10468681982772748 |

These equal the T = 4000 row of `results/path_ramp.csv` from `python exp_path.py ramp`
(validate_results.py target: width 32.2 +- 1.2 steps).

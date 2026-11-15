# sweep_half_life

```bash
amlc validate examples/sweep_half_life.toml --sweep --json   # estimate.total_runtime_s
amlc sweep examples/sweep_half_life.toml --out runs/memory --json
cat runs/memory/index.json                                   # one entry per half-life with summary values
amlc compare runs/memory/point_000 runs/memory/point_003 --json
```

Measured on 2026-11-15 (Apple M1 Max): about 8 s for four points, validate estimate 18.4 s (14 s after recalibration).

| `impact.half_life` | `largest_event_mean` | `largest_event_field_mean` | `n_events_mean` |
|---|---|---|---|
| 4 | 0.3010585765129101 | 0.3872088994565217 | 1.90625 |
| 20 | 0.6134219622763424 | 0.2729872282608696 | 1.09375 |
| 80 | 0.8048929712169155 | 0.22862595108695652 | 1.0 |
| -1 (permanent) | 0.856955231843556 | 0.19377024456521738 | 1.0 |

Equal to the delay-5, Lambda = 0.5 rows of `results/path_memory.csv` from `python exp_path.py memory`
(validate_results.py targets: 0.857, 0.805, 0.613 for h = -1, 80, 20). Shorter impact memory
fragments the cascade into smaller, later events.

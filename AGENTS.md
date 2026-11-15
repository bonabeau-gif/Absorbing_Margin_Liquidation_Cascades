# AGENTS.md

## What this is

Theoretical, dimensionless models of absorbing margin-liquidation cascades: leveraged accounts breach
maintenance thresholds as an external drawdown rises, their forced sales add price impact, and the
cascade jumps discontinuously at a fold. The code computes the mean-field folds and cusp, finite-cohort
collapse statistics (lognormal or Pareto exposures), precursor avalanche sizes, and a path-resolved
partial-deleveraging simulation with transient impact. Everything is reachable through one command, `amlc`.

## Install (verified from a fresh clone, Python 3.12.11, macOS arm64)

```bash
git clone https://github.com/bonabeau-gif/Absorbing_Margin_Liquidation_Cascades.git
cd Absorbing_Margin_Liquidation_Cascades
python3 -m venv .venv && . .venv/bin/activate      # Python >= 3.11 (the authors tested 3.13)
pip install -r requirements.txt                    # pinned numpy/scipy/pandas/matplotlib/numba
pip install -e .                                   # installs the `amlc` console script
amlc describe
```

With conda instead: `conda env create -f environment.yml && conda activate absorbing-margin-cascades`,
then `pip install -e .`. Without installing, `python -m amlc ...` works from the repository root.

## The interface

Start with `amlc describe`. Add `--json` to any command for exactly one JSON document on stdout.

| Command | Purpose |
|---|---|
| `amlc describe [--model M] [--param P] [--schema]` | models, parameters (type, default, range, unit), outputs, fixed constants |
| `amlc template --model M [--format json]` | complete, valid, commented configuration (TOML by default) |
| `amlc validate CONFIG [--sweep]` (`-` reads stdin) | every problem at once, plus estimated runtime, memory, output size |
| `amlc run CONFIG --out DIR` | one run into a new run directory |
| `amlc query DIR <query>` | `summary`, `outputs`, `series NAME [--from --to --points]`, `stats NAME [--where N=..]`, `at X`, `final`, `crossing NAME LEVEL`, `scaling NAME [--min --max]`, `quantiles NAME [--where]`; `amlc query DIR <query> --help` for options |
| `amlc compare DIR DIR ...` | differing parameters and summary differences |
| `amlc sweep CONFIG --out DIR` | Cartesian product of a `[sweep]` block, one run directory per point plus `index.json` |

Models: `mean_field` (deterministic, under 1 s), `finite_cohort`, `avalanches`, `path`,
`brownian_parabola`, `dominant_holder`. There are no interventions; the path model's external field
follows a fixed linear ramp (`ramp.*`).

Exit codes: 0 success, 1 runtime failure, 2 invalid usage or configuration. Errors with `--json` are
`{"ok": false, "errors": [{"field", "value", "reason", "hint"}]}`.

## Worked example

How sharp is the path-resolved transition, and where does it happen?

```bash
amlc validate examples/typical_path_ramp.toml --json         # estimate.runtime_s is about 3.4
amlc run examples/typical_path_ramp.toml --out runs/ramp --json
amlc query runs/ramp summary --json      # width_steps_mean 32.15625, mid_field_mean 0.2339309375
amlc query runs/ramp crossing sold_mean 0.42 --json          # step ~3021, field ~0.2315
```

Change one thing and compare (e.g. permanent impact):

```bash
sed 's/^impact.half_life = 80.0/impact.half_life = -1.0/' examples/typical_path_ramp.toml > perm.toml
amlc validate perm.toml && amlc run perm.toml --out runs/perm --json
amlc compare runs/ramp runs/perm --json
```

`examples/` has six configurations; each `.md` sidecar lists the commands and the values measured.

## Rules for using it well

- Validate before running: `amlc validate` reports every bad field, unknown key (with the closest
  valid one) and violated constraint, and gives a cost estimate. Check `estimate.runtime_s` before large
  runs. The estimate is calibrated on an Apple M1 Max, so scale it to your machine. The defaults of
  `finite_cohort` (about 55 s) and `avalanches` (about 15 s) are the full published experiments.
- Always set `seed`. Each model's template uses the seed of the original script, so the defaults
  reproduce the published numbers. Without a seed, `run` picks one, records it, and says so on stderr.
- Never parse `outputs/*.csv` when a `query` answers the question; queries state units and approximations.
- A run is fully defined by its configuration file; `config.resolved.toml` in a run directory
  reproduces it (`amlc run DIR/config.resolved.toml --out NEW`).
- `run` never overwrites a non-empty directory; `--force` replaces only a previous run directory.
- Units: all quantities are dimensionless; `unit` fields name the quantity (fraction of price
  drawdown, fraction of exposure, steps). Path-model times are simulation steps, not days.
- `finite_cohort` summary `theory`: `sd_coef` is the leading-order coefficient of the collapse-field s.d.
  (sd ~ sd_coef/sqrt(N)), `shift_coef` of the mean shift (~ N^(-2/3)), `loc_coef` of the takeoff-state s.d.
  (~ N^(-1/3)); `by_size.csv` holds the matching `pred_*` columns.
- `impact.half_life <= 0` means permanent impact. `impact.strength` defaults to 0.30 in the event-driven
  models and to 0.5 in `path`, as in the original scripts.

## Verify the installation

```bash
python tests/test_agent_layer.py     # about 18 s; ends with "all 8 agent-layer tests passed"
python tests/smoke_test.py           # about 1 s; analytic checks
python run_all.py --quick            # original workflow: smoke tests, figures, validation
```

On platforms other than the authors' (Debian, Python 3.13), `run_all.py --quick` passes every numerical
check but its last stage, `validate_figures.py`, can fail: it compares PNG files by SHA-256, and the
bytes depend on the platform and font stack.

## Known limits (from the project's documentation)

- Theoretical and dimensionless. No empirical market data, no calibration, no forecasts for any
  security (README, Scope).
- Monte Carlo summaries agree across platforms only within small tolerances (README; metadata/RANDOM_SEEDS.md).
- The finite-N predictions (`finite_cohort` summary `theory`) apply to lognormal exposures with
  square-root impact. The Borel-mixture prediction (`avalanches`, `pdf_borel`) applies to equal exposures
  with square-root impact. Otherwise those fields are null.
- The event-driven models use the quasistatic absorbing protocol (Appendix C). Partial deleveraging,
  delays, sale-rate limits, finite price relaxation and transient impact exist only in `path` (CONTROLS.md).
- `path` summary `meanfield_binary_permanent` is a binary, permanent-impact reference computed with
  a bounded fold search (total drawdown in [0.05, 0.22]); a value at a bound means no fold was found there.
- Parameters with `"range": null` have no stated valid range. Values far from the defaults are untested.

## For contributors

- Model code: `amlc/model.py`, `amlc/pathsim.py`, `amlc/aval_theory.py`; experiment drivers in
  `amlc/experiments.py` (the root scripts call them); CLI layer in `amlc/{registry,config,runner,rundir,query,cli}.py`.
- New parameters, models, or outputs go in `amlc/registry.py` first. `describe`, `template`, validation,
  and run `schema.json` are generated from it. Then regenerate the committed schema:
  `amlc describe --schema --json > amlc/config.schema.json` (a test checks it).
- Run `python tests/test_agent_layer.py` and `python run_all.py --quick` before committing.

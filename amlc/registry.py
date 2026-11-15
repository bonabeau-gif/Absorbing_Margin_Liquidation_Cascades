"""Single source of truth for everything configurable in the ``amlc`` command line.

Models, parameters, outputs, and interventions are declared here. ``describe``, ``template``,
``validate``, the JSON Schema (``amlc/config.schema.json``), and each run's ``schema.json`` are
generated from these structures, so they cannot disagree.

Conventions
-----------
* Every quantity in this package is dimensionless (README, "Scope"). The ``unit`` field says
  which dimensionless quantity it is (fraction of price, fraction of exposure, steps, ...).
* ``range`` is ``[min, max]`` (``None`` for an open end) with ``range_exclusive`` flags, or
  ``None`` where neither the source nor the arithmetic of the code bounds the value.
* Model-specific defaults are given in each model's ``defaults``; they override the
  parameter's generic ``default``.

This module must not import numpy or other heavy libraries (``describe`` must start fast).
"""

DIMLESS = "dimensionless"
DRAWDOWN = "dimensionless (fractional price drawdown)"
EXPFRAC = "dimensionless (fraction of total exposure)"
STEPS = "time steps"
ACCOUNTS = "accounts"
REQUIRED = "<required>"

IMPACT_LAWS = ["square-root", "linear-to-square-root", "linear", "logarithmic", "saturating"]
EXPOSURE_KINDS = ["equal", "lognormal", "pareto"]


def P(name, type, default, description, unit=DIMLESS, range=None, range_exclusive=(False, False),
      options=None, applies_when=None, item_type=None, source=""):
    return dict(name=name, type=type, default=default, range=list(range) if range is not None else None,
                range_exclusive=list(range_exclusive) if range is not None else None,
                options=options, unit=unit, description=description, applies_when=applies_when,
                item_type=item_type, source=source)


# --------------------------------------------------------------------------- parameters
PARAMETERS = [
    # cohort composition (event-driven and path models)
    P("leverage.mean", "number", 2.0,
      "Mean leverage l. Account leverage is max(1, l*(1 + s*Z)), Z standard normal. Higher l lowers breach "
      "thresholds and makes the cohort more fragile.",
      source="model.t_of_B, pathsim.make_cohort, exp_*.py (l=2.0)"),
    P("leverage.dispersion", "number", 0.16,
      "Relative leverage dispersion s in max(1, l*(1 + s*Z)). Larger s spreads thresholds; at fixed impact "
      "strength it moves the system away from the cusp.",
      source="model.t_of_B, pathsim.make_cohort (s=0.16)"),
    P("maintenance.ratio", "number", 0.25,
      "Maintenance-margin ratio m, common to all accounts. Breach threshold field t = (1/leverage - m)/(1 - m).",
      source="model.t_of_B, exp_*.py (m=0.25)"),
    P("maintenance.mean", "number", 0.25,
      "Mean of the per-account maintenance ratio in the path model (normal, truncated at +-4 sd).",
      source="pathsim.make_cohort (m_mean=0.25)"),
    P("maintenance.sd", "number", 0.025,
      "Standard deviation of the per-account maintenance ratio in the path model.",
      range=(0, None), source="pathsim.make_cohort (m_sd=0.025); numpy normal scale must be >= 0"),
    P("exposure.kind", "enum", "lognormal",
      "Distribution of account exposures: 'equal' (all 1), 'lognormal' (correlated with leverage through "
      "exposure.leverage_corr), or 'pareto' (independent of leverage, tail index exposure.tail_index).",
      options=EXPOSURE_KINDS, source="exp_finiteN.py, exp_pareto.py, exp_avalanche.py"),
    P("exposure.log_sd", "number", 0.75,
      "Log-standard deviation s_q of lognormal exposures Q = exp(s_q*(rho*Z + sqrt(1-rho^2)*U)).",
      applies_when={"exposure.kind": "lognormal"}, source="exp_finiteN.py (sq=0.75), pathsim.make_cohort"),
    P("exposure.leverage_corr", "number", 0.4,
      "Correlation rho between log exposure and the leverage normal Z. Exposure tilt chi = rho*s_q; positive "
      "tilt puts more exposure on high-leverage accounts and lowers the critical impact strength.",
      range=(-1, 1), applies_when={"exposure.kind": "lognormal"},
      source="exp_finiteN.py (rho=0.4); range from sqrt(1-rho^2)"),
    P("exposure.tail_index", "number", 1.5,
      "Pareto tail index mu of exposures Q = (1-U)^(-1/mu). Smaller mu means heavier tails, slower "
      "finite-N narrowing of the collapse field, and a late-collapse tail.",
      range=(0, None), range_exclusive=(True, False), applies_when={"exposure.kind": "pareto"},
      source="exp_pareto.py (mu in 1.25..4); range from exponent -1/mu"),
    P("exposure.tilt", "number", 0.30,
      "Exposure-leverage tilt chi = rho_qz * s_q entering the exposure-weighted threshold quantile t(B).",
      source="model.t_of_B, make_figs.py Fig. 1 (chi=0.30)"),
    # impact
    P("impact.strength", "number", 0.30,
      "Impact strength Lambda: drawdown added when the sold exposure fraction x has normalized impact f(x) "
      "(f(1)=1). Above the onset value the equality map folds and the cascade becomes discontinuous.",
      source="model.d0_of_B, exp_*.py (Lam=0.30), exp_path.py (LAM=0.5)"),
    P("impact.law", "enum", "square-root",
      "Normalized impact function f of sold exposure fraction.", options=IMPACT_LAWS,
      source="model.IMPACT"),
    P("impact.half_life", "number", 80.0,
      "Half-life h_I of the unabsorbed impact stock in the path model; the stock decays by 2^(-1/h_I) per "
      "step. Any value <= 0 means permanent impact. Short half-lives fragment the cascade.",
      unit=STEPS, source="pathsim.run_path (hI<=0 permanent), exp_path.py (80, -1, 3..640)"),
    P("price.relaxation", "number", 0.22,
      "Rate gamma at which the drawdown relaxes toward d0 + Lambda*f(U) each step in the path model.",
      unit="per time step", source="pathsim.run_path, exp_path.py (GAM=0.22)"),
    P("price.drawdown_cap", "number", 0.88,
      "Upper cap on the total drawdown d in the path model.", unit=DRAWDOWN,
      source="pathsim.run_path (dfloor), exp_path.py (FLOOR=0.88)"),
    # ramp of the external field (path model)
    P("ramp.start", "number", 0.02, "External drawdown d0 at step 0 of the linear ramp.", unit=DRAWDOWN,
      source="exp_path.py (0.02)"),
    P("ramp.end", "number", 0.30, "External drawdown d0 reached at the end of the ramp and then held.",
      unit=DRAWDOWN, source="exp_path.py (0.30; memory study 0.45)"),
    P("ramp.steps", "integer", 4000,
      "Ramp duration T (steps) from ramp.start to ramp.end. Slower ramps sharpen the transition in d0 "
      "(width ~ tau_c*(end-start)/T).", unit=STEPS, range=(1, None),
      source="exp_path.py ramp study (250..64000); range from division by T"),
    P("ramp.hold_steps", "integer", 500, "Steps simulated after the ramp with d0 held at ramp.end.",
      unit=STEPS, range=(0, None), source="exp_path.py (HOLD=500)"),
    # execution (path model)
    P("execution.delay_mean", "number", 5.0,
      "Mean of the Poisson execution delay: consecutive breached steps an account waits before selling.",
      unit=STEPS, range=(0, None), source="pathsim.make_cohort (delay=5.0); exp_path.py memory (1, 5, 15)"),
    P("execution.sale_rate_mean", "number", 0.25,
      "Mean of the Beta-distributed per-step sale cap (fraction of the position an account may sell per step).",
      unit="fraction of position per step", range=(0, 1), range_exclusive=(True, True),
      source="pathsim.make_cohort (r_mean=0.25); range from Beta parameters"),
    P("execution.sale_rate_sd", "number", 0.10,
      "Standard deviation of the Beta-distributed per-step sale cap.", unit="fraction of position per step",
      range=(0, None), range_exclusive=(True, False), source="pathsim.make_cohort (r_sd=0.10)"),
    # cohort sizes and replication
    P("cohort.size", "integer", REQUIRED, "Number of accounts N in each simulated cohort.", unit=ACCOUNTS,
      range=(1, None)),
    P("cohort.count", "integer", REQUIRED, "Number of independent cohorts simulated.", unit="cohorts",
      range=(1, None)),
    P("cohort.sizes", "list", REQUIRED,
      "Cohort sizes N, one block of independent cohorts per size, simulated in this order from one random "
      "stream (as in exp_finiteN.py / exp_pareto.py).", unit=ACCOUNTS, item_type="integer", range=(1, None)),
    P("cohort.counts", "list", REQUIRED,
      "Number of cohorts R for each entry of cohort.sizes (same length).", unit="cohorts",
      item_type="integer", range=(1, None)),
    # mean-field extras
    P("cusp.drawdown", "number", 0.10,
      "Initiating drawdown d0 at which the cusp (critical Lambda_c and cusp leverage l*) is located for "
      "square-root impact.", unit=DRAWDOWN, source="make_figs.py Fig. 3, tests/smoke_test.py (0.10)"),
    P("map.points", "integer", 4000, "Number of grid points in B for the recorded equality map d0(B) on [1e-6, 0.9].",
      range=(2, None), source="make_figs.py Fig. 1 (4000)"),
    # avalanche extras
    P("histogram.log10_max", "number", 4.5,
      "log10 of the largest logarithmic bin edge before the final doubled bin.", range=(0, None),
      range_exclusive=(True, False), source="exp_avalanche.py (4.5), avalanche_size_sweep.py (3.6)"),
    P("histogram.points", "integer", 28, "Number of logarithmically spaced edges before rounding and de-duplication.",
      range=(2, None), source="exp_avalanche.py (28), avalanche_size_sweep.py (24)"),
    P("avalanche.near_fold_window", "number", 0.002,
      "Avalanches whose field lies within this distance below the collapse field are counted as near-fold.",
      unit=DRAWDOWN, source="exp_avalanche.py (0.002)"),
    # path diagnostics
    P("diagnostics.window", "integer", 200,
      "Window W (steps) for the largest W-step increase of the sold fraction (largest event).", unit=STEPS,
      range=(1, None), source="pathsim.diagnostics, exp_path.py memory (W=200)"),
    P("diagnostics.event_threshold", "number", 0.10,
      "A W-step increase above this counts as an event (n_events counts up-crossings).", unit=EXPFRAC,
      source="exp_path.py memory (0.10)"),
    P("diagnostics.reference_field", "number", 0.30,
      "External field at which the sold fraction is reported (sold_at_reference).", unit=DRAWDOWN,
      source="exp_path.py memory (0.30)"),
    P("diagnostics.late_margin", "number", 0.008,
      "A cohort is 'late' if its midpoint field exceeds the median midpoint field by more than this.",
      unit=DRAWDOWN, source="exp_path.py size, stalled_ignition_diagnostic.py (0.008)"),
    # Brownian parabola
    P("bm.dt", "number", 5e-4, "Time step of the discretized Brownian motion.", range=(0, None),
      range_exclusive=(True, False), source="bm_parabola.py"),
    P("bm.horizon", "number", 5.0, "One-sided horizon V of the two-sided Brownian motion.", range=(0, None),
      range_exclusive=(True, False), source="bm_parabola.py"),
    P("bm.samples", "integer", 20000, "Number of Monte Carlo samples R.", range=(1, None), source="bm_parabola.py"),
    # dominant holder
    P("holder.shares", "list", [0.0, 0.10, 0.20],
      "Shares omega of total exposure held by one conservative holder (0 = no holder).", unit=EXPFRAC,
      item_type="number", range=(0, 1), range_exclusive=(False, True),
      source="dominant_holder_meanfield_controls.py; range from omega*N/(1-omega)"),
    P("holder.leverage_z", "number", -2.0,
      "Leverage quantile z of the conservative holder: leverage max(1, l*(1 + s*z)).",
      source="dominant_holder_meanfield_controls.py (z=-2)"),
    P("holder.background_loss", "number", 0.25,
      "Fraction of the background cohort's own exposure whose loss defines the reported field.",
      unit=EXPFRAC, range=(0, 1), range_exclusive=(True, False),
      source="dominant_holder_meanfield_controls.py (0.25)"),
]
PARAM = {p["name"]: p for p in PARAMETERS}

_EVENT_COMMON = ["leverage.mean", "leverage.dispersion", "maintenance.ratio", "impact.strength", "impact.law",
                 "exposure.kind", "exposure.log_sd", "exposure.leverage_corr", "exposure.tail_index"]


def C(expr, fields, message):
    """A constraint between parameters. ``expr`` is evaluated by validate with the dotted names bound."""
    return dict(expr=expr, fields=fields, message=message)


# --------------------------------------------------------------------------- outputs (per model)
def O(name, unit, description, shape="scalar"):
    return dict(name=name, unit=unit, description=description, shape=shape)


MODELS = {
    "mean_field": dict(
        title="deterministic folds, onset and cusp of the mean-field equality map (seconds)",
        description="Deterministic mean-field analysis of the absorbing threshold model: equality map "
                    "d0(B), upper/lower folds, landing state of the absorbing jump, fold onset for the chosen "
                    "impact law, and the square-root cusp at cusp.drawdown. Fast (about a second).",
        source="model.py (folds, landing, onset_fixed_l, cusp_power); make_figs.py Figs. 1-4",
        stochastic=False, reference_seed=None,
        parameters=["leverage.mean", "leverage.dispersion", "maintenance.ratio", "exposure.tilt",
                    "impact.strength", "impact.law", "cusp.drawdown", "map.points"],
        defaults={},
        constraints=[],
        index={"table": "equality_map", "column": "B", "unit": EXPFRAC},
        files={
            "equality_map.csv": dict(description="Equality map: external field needed to hold breached exposure fraction B.",
                                     columns=[O("B", EXPFRAC, "breached exposure fraction (grid on [1e-6, 0.9])", "series"),
                                              O("d0", DRAWDOWN, "external drawdown d0(B) = t(B) - Lambda*f(B)", "series")]),
        },
        summary=[O("has_fold", "boolean", "true if the equality map has an upper/lower fold pair"),
                 O("B_plus", EXPFRAC, "B at the upper fold (absorbing collapse point)"),
                 O("d0_plus", DRAWDOWN, "external field at the upper fold: mean-field collapse field"),
                 O("B_minus", EXPFRAC, "B at the lower fold"),
                 O("d0_minus", DRAWDOWN, "external field at the lower fold"),
                 O("B_landing", EXPFRAC, "B reached after the absorbing jump at d0_plus"),
                 O("jump_size", EXPFRAC, "B_landing - B_plus"),
                 O("curvature_kappa", DIMLESS, "-d2 d0/dB2 at the upper fold"),
                 O("onset_impact_strength", DIMLESS, "smallest Lambda with a fold, for impact.law at these parameters"),
                 O("onset_B", EXPFRAC, "B at which the fold first appears"),
                 O("cusp_impact_strength", DIMLESS, "critical Lambda_c at cusp.drawdown (square-root impact)"),
                 O("cusp_leverage", DIMLESS, "cusp mean leverage l* at cusp.drawdown (square-root impact)"),
                 O("cusp_B", EXPFRAC, "B at the cusp")],
    ),
    "finite_cohort": dict(
        title="Monte Carlo collapse-field statistics vs cohort size N (event-driven protocol)",
        description="Monte Carlo of finite cohorts under the quasistatic absorbing protocol (Appendix C): "
                    "for each cohort the collapse field (field of the largest event), jump size and takeoff "
                    "state; statistics per cohort size and finite-N scaling exponents. With lognormal "
                    "exposures and square-root impact, also the parameter-free Daniels-type predictions.",
        source="exp_finiteN.py (lognormal, seed 20260917), exp_pareto.py (Pareto, seed int(1000*mu)+17)",
        stochastic=True, reference_seed=20260917,
        parameters=_EVENT_COMMON + ["cohort.sizes", "cohort.counts"],
        defaults={"cohort.sizes": [250, 1000, 4000, 16000, 64000, 256000],
                  "cohort.counts": [4000, 4000, 4000, 3000, 2000, 1000]},
        constraints=[C("len(cohort.sizes) == len(cohort.counts)", ["cohort.sizes", "cohort.counts"],
                       "cohort.sizes and cohort.counts must have the same length")],
        index={"table": "by_size", "column": "N", "unit": ACCOUNTS},
        files={
            "by_size.csv": dict(description="Collapse statistics per cohort size (one row per entry of cohort.sizes).",
                                columns=[O("N", ACCOUNTS, "cohort size", "series"), O("R", "cohorts", "cohorts simulated", "series"),
                                         O("mean_field", DRAWDOWN, "mean collapse field", "series"),
                                         O("mean_shift", DRAWDOWN, "mean collapse field minus mean-field d0_plus (empty if no fold)", "series"),
                                         O("se_shift", DRAWDOWN, "standard error of the mean collapse field", "series"),
                                         O("sd_field", DRAWDOWN, "standard deviation of the collapse field", "series"),
                                         O("q05", DRAWDOWN, "5% quantile of collapse field", "series"),
                                         O("q25", DRAWDOWN, "25% quantile", "series"), O("q50", DRAWDOWN, "median", "series"),
                                         O("q75", DRAWDOWN, "75% quantile", "series"), O("q95", DRAWDOWN, "95% quantile", "series"),
                                         O("iqr", DRAWDOWN, "interquartile range of collapse field", "series"),
                                         O("mean_takeoff", EXPFRAC, "mean liquidated exposure just before the largest event", "series"),
                                         O("sd_takeoff", EXPFRAC, "s.d. of the takeoff state", "series"),
                                         O("mean_jump", EXPFRAC, "mean size of the largest event", "series"),
                                         O("sd_jump", EXPFRAC, "s.d. of the largest event", "series"),
                                         O("median_trigger_relweight", DIMLESS, "median weight of the triggering account relative to the mean weight 1/N", "series"),
                                         O("median_maxweight", EXPFRAC, "median of the largest single-account weight", "series"),
                                         O("pred_shift", DRAWDOWN, "predicted mean shift shift_coef*N^(-2/3) (lognormal, square-root only)", "series"),
                                         O("pred_sd", DRAWDOWN, "predicted s.d. including the N^(-2/3) correction (lognormal, square-root only)", "series"),
                                         O("pred_sd_lead", DRAWDOWN, "leading-order predicted s.d. sd_coef/sqrt(N)", "series"),
                                         O("pred_loc", EXPFRAC, "predicted takeoff-state s.d. loc_coef*N^(-1/3)", "series")]),
            "cohorts.csv": dict(description="One row per simulated cohort.",
                                columns=[O("N", ACCOUNTS, "cohort size", "table"), O("cohort", "index", "cohort index within its size block", "table"),
                                         O("collapse_field", DRAWDOWN, "field of the largest event", "table"),
                                         O("jump", EXPFRAC, "exposure liquidated in the largest event", "table"),
                                         O("takeoff", EXPFRAC, "exposure liquidated before the largest event", "table"),
                                         O("trigger_relweight", DIMLESS, "weight of the triggering account times N", "table"),
                                         O("max_weight", EXPFRAC, "largest account weight in the cohort", "table")]),
        },
        summary=[O("tilt_chi", DIMLESS, "exposure tilt used for the mean-field fold (rho*s_q for lognormal, else 0)"),
                 O("d0_plus", DRAWDOWN, "mean-field collapse field (upper fold)"), O("B_plus", EXPFRAC, "B at the upper fold"),
                 O("jump_inf", EXPFRAC, "mean-field jump size B_landing - B_plus"),
                 O("theory", "mixed", "Daniels-type prefactors sd_coef, shift_coef, loc_coef, kappa, Tp (lognormal + square-root only)"),
                 O("exponents", DIMLESS, "fitted log-log slopes vs N of sd_field, iqr, and |mean_shift| (needs >= 2 sizes)"),
                 O("largest_N", "mixed", "the by_size row for the largest N")],
    ),
    "avalanches": dict(
        title="precursor avalanche-size distribution before collapse, with Borel prediction",
        description="Precursor avalanche-size distribution (account counts) before the macroscopic collapse, "
                    "pooled over cohorts, with the near-fold subset and, for equal exposures with square-root "
                    "impact, the Borel-mixture prediction (Eq. 32).",
        source="exp_avalanche.py (seed 101 equal/Pareto, 99 lognormal), avalanche_size_sweep.py, aval_theory.py",
        stochastic=True, reference_seed=101,
        parameters=_EVENT_COMMON + ["cohort.size", "cohort.count", "histogram.log10_max", "histogram.points",
                                     "avalanche.near_fold_window"],
        defaults={"exposure.kind": "equal", "cohort.size": 2_000_000, "cohort.count": 40},
        constraints=[],
        index={"table": "histogram", "column": "S_center", "unit": ACCOUNTS},
        files={
            "histogram.csv": dict(description="Logarithmically binned avalanche-size density.",
                                  columns=[O("S_center", ACCOUNTS, "geometric bin centre", "series"),
                                           O("bin_lo", ACCOUNTS, "bin lower edge (inclusive)", "series"),
                                           O("bin_hi", ACCOUNTS, "bin upper edge (exclusive)", "series"),
                                           O("pdf_all", "per account", "density of all precursor avalanches", "series"),
                                           O("count_all", "avalanches", "count of all precursor avalanches", "series"),
                                           O("pdf_near", "per account", "density of near-fold avalanches", "series"),
                                           O("count_near", "avalanches", "count of near-fold avalanches", "series"),
                                           O("pdf_borel", "per account", "Borel-mixture prediction, bin-averaged (equal + square-root only; else empty)", "series")]),
        },
        summary=[O("n_avalanches", "avalanches", "precursor avalanches pooled"), O("n_near", "avalanches", "near-fold avalanches"),
                 O("slope_all_5_500", DIMLESS, "log-log slope of pdf_all for bin centres in [5, 500] (bins with > 20 counts)"),
                 O("slope_near_3_300", DIMLESS, "log-log slope of pdf_near in [3, 300]"),
                 O("ratio_sim_theory_200_1000", DIMLESS, "count-weighted simulation/Borel ratio for centres in [200, 1000] (equal + square-root only)"),
                 O("mean_abs_rel_error_200_1000", DIMLESS, "count-weighted mean |sim/theory - 1| in [200, 1000]"),
                 O("total_counts_200_1000", "avalanches", "counts entering the ratio")],
    ),
    "path": dict(
        title="path-resolved partial deleveraging under a drawdown ramp, transient or permanent impact",
        description="Path-resolved partial-deleveraging simulation (Sec. X): heterogeneous maintenance "
                    "ratios, Poisson execution delays, Beta sale-rate caps, finite price relaxation and "
                    "transient (or permanent) linear-to-square-root impact, driven by a linear ramp of the "
                    "external drawdown. Records the sold-exposure path and per-cohort transition diagnostics.",
        source="pathsim.py, exp_path.py (size: seed 5000+N; ramp: seed 777; memory: seed 31337)",
        stochastic=True, reference_seed=777,
        parameters=["cohort.size", "cohort.count", "leverage.mean", "leverage.dispersion", "exposure.log_sd",
                    "exposure.leverage_corr", "maintenance.mean", "maintenance.sd", "execution.delay_mean",
                    "execution.sale_rate_mean", "execution.sale_rate_sd", "impact.strength", "impact.half_life",
                    "price.relaxation", "price.drawdown_cap", "ramp.start", "ramp.end", "ramp.steps",
                    "ramp.hold_steps", "diagnostics.window", "diagnostics.event_threshold",
                    "diagnostics.reference_field", "diagnostics.late_margin"],
        defaults={"cohort.size": 8000, "cohort.count": 32, "impact.strength": 0.5},
        constraints=[C("ramp.end > ramp.start", ["ramp.end", "ramp.start"], "the ramp must increase the drawdown"),
                     C("execution.sale_rate_sd**2 < execution.sale_rate_mean*(1 - execution.sale_rate_mean)",
                       ["execution.sale_rate_sd", "execution.sale_rate_mean"],
                       "Beta sale-rate distribution needs sd^2 < mean*(1-mean)"),
                     C("diagnostics.window < ramp.steps + ramp.hold_steps", ["diagnostics.window", "ramp.steps", "ramp.hold_steps"],
                       "the event window must be shorter than the simulated path")],
        index={"table": "path", "column": "step", "unit": STEPS},
        files={
            "path.csv": dict(description="Sold-exposure path per step, aggregated over cohorts (every step recorded).",
                             columns=[O("step", STEPS, "time step (0-based)", "time series"),
                                      O("field", DRAWDOWN, "external drawdown d0 imposed by the ramp at this step", "time series"),
                                      O("sold_mean", EXPFRAC, "cumulative sold exposure fraction, mean over cohorts", "time series"),
                                      O("sold_min", EXPFRAC, "minimum over cohorts", "time series"),
                                      O("sold_max", EXPFRAC, "maximum over cohorts", "time series"),
                                      O("sold_first", EXPFRAC, "first cohort's path (a representative trajectory)", "time series")]),
            "cohorts.csv": dict(description="Per-cohort transition diagnostics.",
                                columns=[O("cohort", "index", "cohort index", "table"),
                                         O("final_sold", EXPFRAC, "sold fraction at the last step", "table"),
                                         O("largest_event", EXPFRAC, "largest W-step increase of the sold fraction", "table"),
                                         O("largest_event_field", DRAWDOWN, "ramp field at the centre of the largest event window", "table"),
                                         O("width_steps", STEPS, "steps between 10% and 90% of the final sold fraction", "table"),
                                         O("mid_field", DRAWDOWN, "ramp field when half the final sold fraction is reached", "table"),
                                         O("field_10", DRAWDOWN, "ramp field at 10% of the final sold fraction", "table"),
                                         O("field_90", DRAWDOWN, "ramp field at 90% of the final sold fraction", "table"),
                                         O("width_field", DRAWDOWN, "field_90 - field_10", "table"),
                                         O("sold_at_reference", EXPFRAC, "sold fraction at diagnostics.reference_field (empty if outside the ramp)", "table"),
                                         O("n_events", "events", "up-crossings of diagnostics.event_threshold by the W-step increase", "table")]),
        },
        summary=[O("final_sold_mean", EXPFRAC, "mean final sold fraction"), O("final_sold_sd", EXPFRAC, "s.d."),
                 O("mid_field_mean", DRAWDOWN, "mean transition (midpoint) field"), O("mid_field_median", DRAWDOWN, "median"),
                 O("mid_field_sd", DRAWDOWN, "s.d. of the transition field"), O("mid_field_iqr", DRAWDOWN, "IQR"),
                 O("frac_late", DIMLESS, "fraction of cohorts with mid_field > median + diagnostics.late_margin"),
                 O("width_steps_mean", STEPS, "mean 10-90% width in steps"), O("width_steps_sd", STEPS, "s.d."),
                 O("width_field_mean", DRAWDOWN, "mean 10-90% width in field"), O("width_field_sd", DRAWDOWN, "s.d."),
                 O("largest_event_mean", EXPFRAC, "mean largest W-step event J"), O("largest_event_sd", EXPFRAC, "s.d."),
                 O("largest_event_field_mean", DRAWDOWN, "mean field of the largest event"), O("largest_event_field_sd", DRAWDOWN, "s.d."),
                 O("sold_at_reference_mean", EXPFRAC, "mean sold fraction at diagnostics.reference_field"),
                 O("sold_at_reference_sd", EXPFRAC, "s.d."),
                 O("n_events_mean", "events", "mean number of events with J > diagnostics.event_threshold"),
                 O("meanfield_binary_permanent", "mixed", "binary complete-liquidation, permanent-impact mean-field fold for this population (transition_field, total_drawdown_at_fold, B_at_fold); reference only")],
    ),
    "brownian_parabola": dict(
        title="Monte Carlo constants of max[W(v) - v^2/2] used by the finite-N theory",
        description="Monte Carlo constants of M = max_v [W(v) - v^2/2] for two-sided standard Brownian "
                    "motion, used in the finite-N asymptotics.",
        source="bm_parabola.py (seed 7)", stochastic=True, reference_seed=7,
        parameters=["bm.dt", "bm.horizon", "bm.samples"], defaults={},
        constraints=[C("bm.horizon >= bm.dt", ["bm.horizon", "bm.dt"], "the horizon must contain at least one step")],
        index={"table": "samples", "column": "sample", "unit": "index"},
        files={"samples.csv": dict(description="One row per Monte Carlo sample.",
                                   columns=[O("sample", "index", "sample index", "table"),
                                            O("M", DIMLESS, "max_v [W(v) - v^2/2] (raw, uncorrected)", "table"),
                                            O("argmax", DIMLESS, "location v of the maximum", "table")])},
        summary=[O("E_M", DIMLESS, "mean of M with the discretization correction 0.5826*sqrt(dt)"),
                 O("E_M_raw", DIMLESS, "uncorrected mean of M"), O("se_E_M", DIMLESS, "standard error of the mean"),
                 O("sd_M", DIMLESS, "s.d. of M"), O("sd_argmax", DIMLESS, "s.d. of the argmax")],
    ),
    "dominant_holder": dict(
        title="control: dilution by one conservative dominant holder",
        description="Event-definition-independent dominant-holder control: a conservative holder with share "
                    "omega of total exposure; compares the mean-field background fold with the simulated "
                    "field at which the background loses holder.background_loss of its exposure (equal "
                    "exposures, square-root impact).",
        source="dominant_holder_meanfield_controls.py part A (seed 20260922)", stochastic=True, reference_seed=20260922,
        parameters=["leverage.mean", "leverage.dispersion", "maintenance.ratio", "impact.strength", "cohort.size",
                    "holder.shares", "holder.leverage_z", "holder.background_loss"],
        defaults={"cohort.size": 100_000},
        constraints=[],
        index={"table": "by_share", "column": "holder_share", "unit": EXPFRAC},
        files={"by_share.csv": dict(description="One row per holder share.",
                                    columns=[O("holder_share", EXPFRAC, "omega", "series"),
                                             O("meanfield_fold_B", EXPFRAC, "B at the background mean-field fold", "series"),
                                             O("meanfield_fold_field", DRAWDOWN, "field at the background mean-field fold", "series"),
                                             O("simulation_field_bg25", DRAWDOWN, "simulated field at which the background loses holder.background_loss", "series"),
                                             O("bg_fraction_after_event", EXPFRAC, "background fraction lost after that event", "series"),
                                             O("sim_minus_theory", DRAWDOWN, "simulation_field_bg25 - meanfield_fold_field", "series"),
                                             O("largest_avalanche_field_seen_to_bg25", DRAWDOWN, "field of the largest event seen up to that point", "series")])},
        summary=[O("rows", "mixed", "the by_share table"), O("max_abs_sim_minus_theory", DRAWDOWN, "largest |sim_minus_theory|")],
    ),
}

# Fixed constants that are not configurable (reported by describe).
FIXED = [
    dict(name="XC", value=0.02, where="model.py, pathsim.py", description="crossover scale of the linear-to-square-root law"),
    dict(name="KLOG", value=10.0, where="model.py", description="logarithmic impact law constant"),
    dict(name="KSAT", value=4.0, where="model.py", description="saturating impact law constant"),
    dict(name="folds grid", value="400001 points on [1e-6, 0.95]", where="model.folds", description="grid used to bracket folds"),
    dict(name="EM, SDM, SDARG", value=[0.9961930, 0.5775, "sqrt(0.26355964)*2^(2/3)"], where="exp_finiteN.py",
         description="Brownian-parabola constants used by the finite-N predictions"),
    dict(name="BM discretization correction", value=0.5826, where="bm_parabola.py", description="E[M] correction 0.5826*sqrt(dt)"),
    dict(name="maintenance truncation", value="+-4 sd", where="pathsim.make_cohort", description="per-account maintenance ratios are redrawn outside mean +- 4 sd"),
    dict(name="path impact law", value="linear-to-square-root", where="pathsim.f_cross", description="the path model's impact function is fixed"),
    dict(name="Borel comparison window", value=[200, 1000], where="avalanche_size_sweep.py", description="bin centres used for the simulation/theory ratio"),
    dict(name="avalanche slope windows", value=[[5, 500], [3, 300]], where="exp_avalanche.py", description="fit windows for integrated and near-fold slopes"),
    dict(name="dominant-holder fold search", value="B in [1e-6, 0.2]", where="dominant_holder_meanfield_controls.py", description="bounded search for the background fold"),
    dict(name="path mean-field fold search", value="d in [0.05, 0.22]", where="dominant_holder_meanfield_controls.py", description="bounded search for the binary mean-field fold"),
]

# No interventions: no event, perturbation or schedule can be applied during a run. The path
# model's external field follows the fixed linear ramp set by ramp.* parameters.
INTERVENTIONS = []

TOP_LEVEL = {
    "schema": dict(type="integer", description="configuration format version; must be 1", required=True),
    "model": dict(type="enum", options=sorted(MODELS), description="which model to run", required=True),
    "seed": dict(type="integer", description="seed of numpy.random.default_rng; if omitted, run picks one and records it", required=False),
    "parameters": dict(type="mapping", description="model parameters (dotted names may be nested tables)", required=False),
    "sweep": dict(type="mapping", description="sweep only: {parameter name: [values]}; the Cartesian product is run", required=False),
}

SCHEMA_VERSION = 1


def model_params(model):
    """Parameter entries for a model with model-specific defaults applied."""
    spec = MODELS[model]
    out = []
    for name in spec["parameters"]:
        p = dict(PARAM[name])
        if name in spec["defaults"]:
            p["default"] = spec["defaults"][name]
        out.append(p)
    return out


def applies_to(name):
    return [m for m, spec in MODELS.items() if name in spec["parameters"]]


def model_params_map(model):
    return {p["name"]: p for p in model_params(model)}

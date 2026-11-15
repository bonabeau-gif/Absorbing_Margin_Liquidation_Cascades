"""Execute one model from a resolved parameter set and write its outputs.

Each ``run_<model>`` takes the resolved parameters (dotted names), a seeded
``numpy.random.Generator``, the ``outputs/`` directory and a logger, writes the files declared in
``registry.MODELS[model]["files"]`` and returns the summary dictionary. Model logic lives in
``amlc.model``, ``amlc.pathsim`` and ``amlc.experiments``; this module only wires it up.
"""
import csv
import math
import os

import numpy as np

from . import experiments as ex
from .registry import MODELS


# --------------------------------------------------------------------------- helpers
def _num(v):
    """JSON/CSV-friendly scalar: python float/int/bool, None for nan/inf."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    v = float(v)
    return v if math.isfinite(v) else None


def write_csv(path, columns, rows):
    """Deterministic CSV: header row, floats via repr (shortest round-trip), empty cell for null."""
    with open(path, "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(columns)
        for r in rows:
            out = []
            for c in columns:
                v = _num(r.get(c)) if not isinstance(r.get(c), str) else r.get(c)
                out.append("" if v is None else (repr(v) if isinstance(v, float) else str(v)))
            wr.writerow(out)


def _cols(model, fname):
    return [c["name"] for c in MODELS[model]["files"][fname]["columns"]]


def _slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    k = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if k.sum() < 2:
        return None
    return _num(np.polyfit(np.log(x[k]), np.log(y[k]), 1)[0])


def _folds(l, s, m, chi, Lam, law):
    from .model import folds, landing
    fl = folds(l, s, m, chi, Lam, law)
    if len(fl) < 2:
        return None
    (Bp, dp), (Bm, dm) = fl[:2]
    try:
        Bl = landing(l, s, m, chi, Lam, law)
    except Exception:  # noqa: BLE001  (no bracket for the landing state)
        Bl = float("nan")
    return Bp, dp, Bm, dm, Bl


# --------------------------------------------------------------------------- mean_field
def run_mean_field(p, rng, out, log):
    from .model import cusp_power, d0_of_B, onset_fixed_l
    l, s, m, chi, Lam, law = (p["leverage.mean"], p["leverage.dispersion"], p["maintenance.ratio"],
                              p["exposure.tilt"], p["impact.strength"], p["impact.law"])
    B = np.linspace(1e-6, 0.9, p["map.points"]); d = d0_of_B(B, l, s, m, chi, Lam, law)
    write_csv(os.path.join(out, "equality_map.csv"), _cols("mean_field", "equality_map.csv"),
              [dict(B=b, d0=x) for b, x in zip(B, d)])
    log("equality map written (%d points)" % B.size)
    f = _folds(l, s, m, chi, Lam, law)
    S = dict(has_fold=f is not None)
    if f:
        Bp, dp, Bm, dm, Bl = f
        h = 1e-4
        kappa = -(d0_of_B(Bp+h, l, s, m, chi, Lam, law) - 2*dp + d0_of_B(Bp-h, l, s, m, chi, Lam, law))/h**2
        S.update(B_plus=Bp, d0_plus=dp, B_minus=Bm, d0_minus=dm, B_landing=Bl, jump_size=Bl-Bp, curvature_kappa=kappa)
    else:
        S.update(B_plus=None, d0_plus=None, B_minus=None, d0_minus=None, B_landing=None, jump_size=None, curvature_kappa=None)
    on, Bon = onset_fixed_l(l, s, m, chi, law)
    S.update(onset_impact_strength=on, onset_B=Bon)
    Bc, Lc, lc = cusp_power(s, chi, m, p["cusp.drawdown"])
    S.update(cusp_impact_strength=Lc, cusp_leverage=lc, cusp_B=Bc)
    log("folds: %s; onset Lambda %.6f" % ("yes" if f else "none", on))
    return {k: _num(v) for k, v in S.items()}


# --------------------------------------------------------------------------- finite_cohort
def run_finite_cohort(p, rng, out, log):
    l, s, m, Lam, law = p["leverage.mean"], p["leverage.dispersion"], p["maintenance.ratio"], p["impact.strength"], p["impact.law"]
    kind, sq, rho, mu = p["exposure.kind"], p["exposure.log_sd"], p["exposure.leverage_corr"], p["exposure.tail_index"]
    chi = rho*sq if kind == "lognormal" else 0.0
    f = _folds(l, s, m, chi, Lam, law)
    dp = f[1] if f else float("nan")
    pred = None
    if kind == "lognormal" and law == "square-root" and f:
        pred = ex.finite_n_theory(l, s, m, Lam, sq, rho)
    rows, crow = [], []
    for N, R in zip(p["cohort.sizes"], p["cohort.counts"]):
        o = np.empty((R, 5))
        for r in range(R):
            lev, Q = ex.sample_cohort(rng, N, l, s, kind, sq=sq, rho=rho, mu=mu)
            need, rec, Lb, w = ex.absorbing_cascade(lev, Q, m, Lam, law)
            o[r] = ex.collapse_record(need, rec, Lb, w, N)
        d, J, L0 = o[:, 0], o[:, 1], o[:, 2]
        row = ex.lognormal_row(N, R, d, J, L0, dp, pred or dict(shift_coef=np.nan, sd_coef=np.nan, loc_coef=np.nan))
        row.update({k: v for k, v in ex.pareto_row(mu, N, R, o, dp).items()
                    if k in ("q05", "q25", "q50", "q75", "q95", "iqr", "median_trigger_relweight", "median_maxweight")})
        row["mean_field"] = d.mean()
        rows.append(row)
        crow += [dict(N=N, cohort=i, collapse_field=a, jump=b, takeoff=c, trigger_relweight=e, max_weight=g)
                 for i, (a, b, c, e, g) in enumerate(o)]
        log("N=%d R=%d mean field %.6f sd %.6f mean jump %.4f" % (N, R, row["mean_field"], row["sd_field"], row["mean_jump"]))
    write_csv(os.path.join(out, "by_size.csv"), _cols("finite_cohort", "by_size.csv"), rows)
    write_csv(os.path.join(out, "cohorts.csv"), _cols("finite_cohort", "cohorts.csv"), crow)
    Ns = [r["N"] for r in rows]
    S = dict(tilt_chi=chi, d0_plus=f[1] if f else None, B_plus=f[0] if f else None,
             jump_inf=(f[4]-f[0]) if f else None,
             theory=({k: _num(v) for k, v in pred.items() if k in ("sd_coef", "shift_coef", "loc_coef", "kappa", "Tp")} if pred else None),
             exponents=dict(sd_field=_slope(Ns, [r["sd_field"] for r in rows]),
                            iqr=_slope(Ns, [r["iqr"] for r in rows]),
                            mean_shift=_slope(Ns, [abs(r["mean_shift"]) for r in rows]),
                            sd_takeoff=_slope(Ns, [r["sd_takeoff"] for r in rows]),
                            note="log-log least-squares slope versus N over all cohort sizes"),
             largest_N={k: _num(v) for k, v in rows[-1].items()} if Ns[-1] == max(Ns)
             else {k: _num(v) for k, v in rows[int(np.argmax(Ns))].items()})
    return {k: (_num(v) if not isinstance(v, (dict, type(None))) else v) for k, v in S.items()}


# --------------------------------------------------------------------------- avalanches
def _borel_bins(bins, l, s, m, Lam, chunk=1024):
    """Bin-averaged Borel-mixture density; bins wider than ``chunk`` are averaged in chunks
    (weighted by chunk length) to bound memory."""
    from .aval_theory import integrated_borel
    th = []
    for a, b in zip(bins[:-1], bins[1:]):
        S = np.arange(a, b)
        if S.size <= chunk:
            th.append(integrated_borel(S, l, s, m, 0.0, Lam).mean())
        else:
            parts = [S[i:i+chunk] for i in range(0, S.size, chunk)]
            th.append(sum(integrated_borel(q, l, s, m, 0.0, Lam).sum() for q in parts)/S.size)
    return np.array(th)


def run_avalanches(p, rng, out, log):
    l, s, m, Lam, law = p["leverage.mean"], p["leverage.dispersion"], p["maintenance.ratio"], p["impact.strength"], p["impact.law"]
    kind, sq, rho, mu = p["exposure.kind"], p["exposure.log_sd"], p["exposure.leverage_corr"], p["exposure.tail_index"]
    N, R, win = p["cohort.size"], p["cohort.count"], p["avalanche.near_fold_window"]
    bins = ex.avalanche_bins(p["histogram.log10_max"], p["histogram.points"])
    H_all = np.zeros(len(bins)-1); H_near = np.zeros(len(bins)-1); n_all = n_near = 0
    for r in range(R):
        lev, Q = ex.sample_cohort(rng, N, l, s, kind, sq=sq, rho=rho, mu=mu)
        need, rec, Lb, w = ex.absorbing_cascade(lev, Q, m, Lam, law)
        S, d, dc = ex.avalanche_sizes(need, rec, Lb, N)
        near = d > dc - win
        H_all += np.histogram(S, bins)[0]; H_near += np.histogram(S[near], bins)[0]; n_all += S.size; n_near += near.sum()
        if (r + 1) % max(1, R//10) == 0:
            log("cohort %d/%d: %d avalanches so far" % (r+1, R, n_all))
    wd = np.diff(bins); ctr = np.sqrt(bins[:-1]*(bins[1:]-1).clip(1))
    S = dict(n_avalanches=int(n_all), n_near=int(n_near), ratio_sim_theory_200_1000=None,
             mean_abs_rel_error_200_1000=None, total_counts_200_1000=None)
    th = np.full(len(wd), np.nan)
    if kind == "equal" and law == "square-root" and _folds(l, s, m, 0.0, Lam, law) and n_all > 0:
        mask = (ctr >= 200) & (ctr <= 1000) & (H_all >= 10)
        th = _borel_bins(bins, l, s, m, Lam)
        if mask.any():
            pdf = H_all/wd/n_all; wts = H_all[mask]/H_all[mask].sum()
            S["ratio_sim_theory_200_1000"] = np.exp(np.sum(wts*np.log(pdf[mask]/th[mask])))
            S["mean_abs_rel_error_200_1000"] = np.sum(wts*np.abs(pdf[mask]/th[mask]-1))
            S["total_counts_200_1000"] = int(H_all[mask].sum())
    rows = [dict(S_center=ctr[k], bin_lo=int(bins[k]), bin_hi=int(bins[k+1]), pdf_all=H_all[k]/wd[k]/n_all if n_all else None,
                 count_all=int(H_all[k]), pdf_near=H_near[k]/wd[k]/max(n_near, 1), count_near=int(H_near[k]), pdf_borel=th[k])
            for k in range(len(wd))]
    write_csv(os.path.join(out, "histogram.csv"), _cols("avalanches", "histogram.csv"), rows)

    def slope(H, lo, hi):
        k = (ctr >= lo) & (ctr <= hi) & (H > 20)
        return ex.avalanche_slope(ctr, wd, H, lo, hi) if k.sum() >= 2 else None
    S["slope_all_5_500"] = slope(H_all, 5, 500); S["slope_near_3_300"] = slope(H_near, 3, 300)
    return {k: _num(v) for k, v in S.items()}


# --------------------------------------------------------------------------- path
def run_path_model(p, rng, out, log):
    from .pathsim import diagnostics, make_cohort, run_path
    N, NC = p["cohort.size"], p["cohort.count"]
    T, hold = int(p["ramp.steps"]), int(p["ramp.hold_steps"])
    dlo, dhi = float(p["ramp.start"]), float(p["ramp.end"])
    W = int(p["diagnostics.window"])
    paths = np.empty((NC, T + hold)); res = []
    for c in range(NC):
        cohort = make_cohort(rng, N, p["leverage.mean"], p["leverage.dispersion"], p["exposure.log_sd"],
                             p["exposure.leverage_corr"], p["maintenance.mean"], p["maintenance.sd"],
                             p["execution.delay_mean"], p["execution.sale_rate_mean"], p["execution.sale_rate_sd"])
        pth = run_path(*cohort, float(p["impact.strength"]), float(p["price.relaxation"]), float(p["impact.half_life"]),
                       dlo, dhi, T, hold, float(p["price.drawdown_cap"]))
        paths[c] = pth
        Sf, JW, width, mid, f10, f90 = diagnostics(pth, T, dlo, dhi, W)
        _, J, Jf, sref, nev = ex.path_metrics(pth, T, dlo, dhi, W, p["diagnostics.event_threshold"], p["diagnostics.reference_field"])
        res.append((Sf, J, Jf, width, mid, f10, f90, f90 - f10, sref, nev))
        if (c + 1) % max(1, NC//8) == 0:
            log("cohort %d/%d done" % (c+1, NC))
    res = np.array(res, float)
    names = ["final_sold", "largest_event", "largest_event_field", "width_steps", "mid_field", "field_10", "field_90",
             "width_field", "sold_at_reference", "n_events"]
    crow = [dict(cohort=i, **{n: (int(v) if n == "n_events" else v) for n, v in zip(names, r)}) for i, r in enumerate(res)]
    write_csv(os.path.join(out, "cohorts.csv"), _cols("path", "cohorts.csv"), crow)
    steps = np.arange(T + hold)
    field = dlo + (dhi - dlo)*np.minimum(steps, T)/T
    write_csv(os.path.join(out, "path.csv"), _cols("path", "path.csv"),
              [dict(step=int(t), field=fv, sold_mean=a, sold_min=b, sold_max=c, sold_first=e) for t, fv, a, b, c, e in
               zip(steps, field, paths.mean(0), paths.min(0), paths.max(0), paths[0])])
    col = {n: res[:, i] for i, n in enumerate(names)}
    sd = lambda x: x.std(ddof=1) if x.size > 1 else np.nan
    mid = col["mid_field"]; med = np.median(mid)
    S = dict(final_sold_mean=col["final_sold"].mean(), final_sold_sd=sd(col["final_sold"]),
             mid_field_mean=mid.mean(), mid_field_median=med, mid_field_sd=sd(mid),
             mid_field_iqr=np.subtract(*np.percentile(mid, [75, 25])),
             frac_late=np.mean(mid > med + p["diagnostics.late_margin"]),
             width_steps_mean=col["width_steps"].mean(), width_steps_sd=sd(col["width_steps"]),
             width_field_mean=col["width_field"].mean(), width_field_sd=sd(col["width_field"]),
             largest_event_mean=col["largest_event"].mean(), largest_event_sd=sd(col["largest_event"]),
             largest_event_field_mean=col["largest_event_field"].mean(), largest_event_field_sd=sd(col["largest_event_field"]),
             sold_at_reference_mean=col["sold_at_reference"].mean(), sold_at_reference_sd=sd(col["sold_at_reference"]),
             n_events_mean=col["n_events"].mean())
    S = {k: _num(v) for k, v in S.items()}
    try:
        d0, dtot, B = ex.path_meanfield_benchmark(p["leverage.mean"], p["leverage.dispersion"], p["exposure.log_sd"],
                                                  p["exposure.leverage_corr"], p["maintenance.mean"], p["maintenance.sd"],
                                                  p["impact.strength"], 0.02)
        S["meanfield_binary_permanent"] = dict(transition_field=_num(d0), total_drawdown_at_fold=_num(dtot), B_at_fold=_num(B),
                                               note="fold searched for total drawdown in [0.05, 0.22]; a value at a bound means no fold found there")
    except Exception as e:  # noqa: BLE001
        S["meanfield_binary_permanent"] = dict(error=str(e))
    return S


# --------------------------------------------------------------------------- brownian_parabola
def run_brownian_parabola(p, rng, out, log):
    dt, V, R = p["bm.dt"], p["bm.horizon"], p["bm.samples"]
    M, arg = ex.bm_parabola_samples(rng, dt, V, R)
    write_csv(os.path.join(out, "samples.csv"), _cols("brownian_parabola", "samples.csv"),
              [dict(sample=i, M=a, argmax=b) for i, (a, b) in enumerate(zip(M, arg))])
    corr = ex.BM_DISCRETIZATION*np.sqrt(dt)
    log("E[M] raw %.4f corrected %.4f" % (M.mean(), M.mean()+corr))
    return {k: _num(v) for k, v in dict(E_M=M.mean()+corr, E_M_raw=M.mean(), se_E_M=M.std()/np.sqrt(R),
                                        sd_M=M.std(), sd_argmax=arg.std()).items()}


# --------------------------------------------------------------------------- dominant_holder
def run_dominant_holder(p, rng, out, log):
    l, s, m, Lam, N = p["leverage.mean"], p["leverage.dispersion"], p["maintenance.ratio"], p["impact.strength"], p["cohort.size"]
    Z = rng.standard_normal(N)
    lev = np.maximum(1.0, l*(1+s*Z))
    Q = np.ones(N)
    t = (1/lev-m)/(1-m)
    rows = []
    for omega in p["holder.shares"]:
        Bf, df = ex.dominant_meanfield_fold(omega, l, s, m, Lam)
        try:
            sf, bgfrac, op = ex.dominant_simulated_background_loss(t, Q, omega, l, s, m, Lam, p["holder.background_loss"], p["holder.leverage_z"])
        except RuntimeError:
            sf = bgfrac = op = float("nan")
        rows.append(dict(holder_share=omega, meanfield_fold_B=Bf, meanfield_fold_field=df, simulation_field_bg25=sf,
                         bg_fraction_after_event=bgfrac, sim_minus_theory=sf-df, largest_avalanche_field_seen_to_bg25=op))
        log("omega %.3f: mean-field %.6f simulated %.6f" % (omega, df, sf))
    write_csv(os.path.join(out, "by_share.csv"), _cols("dominant_holder", "by_share.csv"), rows)
    diffs = [abs(r["sim_minus_theory"]) for r in rows if math.isfinite(r["sim_minus_theory"])]
    return dict(rows=[{k: _num(v) for k, v in r.items()} for r in rows],
                max_abs_sim_minus_theory=_num(max(diffs)) if diffs else None)


RUNNERS = {"mean_field": run_mean_field, "finite_cohort": run_finite_cohort, "avalanches": run_avalanches,
           "path": run_path_model, "brownian_parabola": run_brownian_parabola, "dominant_holder": run_dominant_holder}


def execute(model, params, seed, out, log):
    rng = np.random.default_rng(seed)
    return RUNNERS[model](params, rng, out, log)


# --------------------------------------------------------------------------- cost estimate
# Coefficients measured on the reference machine (Apple M1 Max, Python 3.12, numpy 2.3.5,
# numba 0.65.1) from the original scripts; see the retrofit report. Machine-dependent.
SEC_PER_ACCOUNT_EVENT = 1.2e-7      # exp_finiteN.py: 52.9 s for 4.53e8 accounts; exp_pareto.py mu<2: ~115 s for 9.3e8
SEC_PER_ACCOUNT_AVAL = 1.5e-7       # exp_avalanche.py: ~12 s for 8e7 accounts
SEC_PER_ACCOUNT_STEP = 1.1e-9       # exp_path.py ramp/memory: 33.8 s / 3.39e10, 97.9 s / 9.0e10 account-steps
SEC_PER_BM_NORMAL = 1.05e-8         # bm_parabola.py: 4.3 s for 4e8 normals
OVERHEAD = {"mean_field": 1.5, "finite_cohort": 1.5, "avalanches": 1.5, "path": 3.0, "brownian_parabola": 0.5,
            "dominant_holder": 1.0}


def estimate(model, p):
    """Rough runtime (s), peak memory (bytes) and output size (bytes)."""
    if model == "mean_field":
        sec = 0.0; mem = 400001*8*6; outb = p["map.points"]*45
    elif model == "finite_cohort":
        nr = sum(n*r for n, r in zip(p["cohort.sizes"], p["cohort.counts"]))
        sec = SEC_PER_ACCOUNT_EVENT*nr; mem = max(p["cohort.sizes"])*8*12
        outb = sum(p["cohort.counts"])*110 + len(p["cohort.sizes"])*500
    elif model == "avalanches":
        N, R = p["cohort.size"], p["cohort.count"]
        sec = SEC_PER_ACCOUNT_AVAL*N*R; mem = N*8*12; outb = p["histogram.points"]*140
        if p["exposure.kind"] == "equal" and p["impact.law"] == "square-root":
            smax = 2*10**p["histogram.log10_max"]
            sec += 2.5e-9*20000*smax/2 + 0.5     # Borel prediction for every bin
            mem = max(mem, 1024*20000*8*4)
    elif model == "path":
        N, NC, L = p["cohort.size"], p["cohort.count"], p["ramp.steps"] + p["ramp.hold_steps"]
        sec = SEC_PER_ACCOUNT_STEP*(1 + N/64000)*N*NC*L; mem = NC*L*8 + N*8*8; outb = L*110 + NC*180
    elif model == "brownian_parabola":
        n = int(p["bm.horizon"]/p["bm.dt"]); sec = SEC_PER_BM_NORMAL*2*n*p["bm.samples"]; mem = n*8*6
        outb = p["bm.samples"]*45
    elif model == "dominant_holder":
        sec = 2e-7*p["cohort.size"]*len(p["holder.shares"]); mem = p["cohort.size"]*8*10; outb = 300*len(p["holder.shares"])
    else:
        return None
    return dict(runtime_s=round(sec + OVERHEAD[model], 1), memory_bytes=int(mem + 150e6), output_bytes=int(outb + 4000),
                basis="linear cost model calibrated on an Apple M1 Max; scale by your machine's speed")

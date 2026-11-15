"""Experiment drivers for the absorbing margin-liquidation cascade study.

Each ``run_*`` function is the former top-level body of the script of the same name
(``exp_finiteN.py`` -> ``run_finiteN`` and so on), moved here unchanged in its constants,
random-number call order, and arithmetic, so that the scripts and the ``amlc`` command line
share one implementation. The scripts call these functions and write the same files.

The building blocks (``sample_cohort``, ``collapse_record``, ``avalanche_sizes``,
``path_metrics``, ...) are used both by the ``run_*`` functions and by ``amlc.runner``.
"""
import csv
import os
import sys

import numpy as np
from scipy.stats import norm

from .model import IMPACT, d0_of_B, dt_dB, event_driven, folds, landing, t_of_B

# Brownian-parabola constants used by the finite-N asymptotics (hard-coded in exp_finiteN.py).
EM, SDM, SDARG = 0.9961930, 0.5775, np.sqrt(0.26355964)*2**(2/3)


def _flush():
    sys.stdout.flush()


# --------------------------------------------------------------------------- building blocks
def sample_cohort(rng, N, l, s, exposure, sq=0.75, rho=0.4, mu=1.5):
    """Draw one cohort: leverage from the standard normal Z, exposures by ``exposure`` kind.

    Random draws, in order: Z (N normals); then U (N normals) for lognormal, or N uniforms
    for Pareto; nothing more for equal exposures.
    """
    Z = rng.standard_normal(N); lev = np.maximum(1, l*(1+s*Z))
    if exposure == "equal":
        Q = np.ones(N)
    elif exposure == "lognormal":
        Q = np.exp(sq*(rho*Z + np.sqrt(1-rho**2)*rng.standard_normal(N)))
    elif exposure == "pareto":
        Q = (1.0 - rng.random(N))**(-1.0/mu)
    else:
        raise ValueError("unknown exposure kind %r" % (exposure,))
    return lev, Q


def impact_function(law):
    """Normalized impact function f for ``event_driven``; square-root is ``np.sqrt`` exactly."""
    return np.sqrt if law == "square-root" else IMPACT[law][0]


def absorbing_cascade(lev, Q, m, Lam, law="square-root"):
    """Thresholds from leverage, then the quasistatic absorbing protocol (Appendix C)."""
    return event_driven((1/lev-m)/(1-m), Q, Lam, impact_function(law))


def collapse_record(need, rec, Lb, w, N):
    """Largest event of one cohort: (collapse field, jump size, takeoff state,
    trigger weight relative to the mean weight, maximum weight)."""
    Lc = np.append(Lb, 1.0); sizes = Lc[np.append(rec[1:], N)] - Lc[rec]; j = sizes.argmax()
    return need[rec[j]], sizes[j], Lb[rec[j]], w[rec[j]]*N, w.max()


def avalanche_sizes(need, rec, Lb, N):
    """Precursor avalanche sizes (account counts) before the largest event, their fields,
    and the collapse field."""
    cnt = np.diff(np.append(rec, N)); Lc = np.append(Lb, 1.0); ex = Lc[np.append(rec[1:], N)] - Lc[rec]
    j = ex.argmax(); dc = need[rec[j]]
    return cnt[:j], need[rec[:j]], dc


def finite_n_theory(l=2.0, s=0.16, m=0.25, Lam=0.30, sq=0.75, rho=0.4):
    """Mean-field fold and parameter-free Daniels-type finite-N prefactors (lognormal exposures,
    square-root impact). Raises ValueError if the equality map has no fold pair."""
    chi = rho*sq
    (Bp, dp), (Bm, dm) = folds(l, s, m, chi, Lam)
    h = 1e-4
    kappa = -(d0_of_B(Bp+h, l, s, m, chi, Lam) - 2*dp + d0_of_B(Bp-h, l, s, m, chi, Lam))/h**2
    Tp = dt_dB(Bp, l, s, m, chi); zp = norm.ppf(Bp)
    V = np.exp(sq**2)*((1-Bp)**2*norm.cdf(zp+chi) + Bp**2*(1-norm.cdf(zp+chi)))   # N*Var of weighted empirical CDF at B+
    q2 = np.exp(sq**2 - chi*zp - chi**2/2)                                           # local variance rate
    eps0 = Tp*np.sqrt(q2)
    return dict(Bp=Bp, d0p=dp, kappa=kappa, Tp=Tp, sd_coef=Tp*np.sqrt(V), shift_coef=EM*(eps0**4/kappa)**(1/3),
                loc_coef=SDARG*(eps0/kappa)**(2/3), jump_inf=landing(l, s, m, chi, Lam)-Bp)


def lognormal_row(N, R, d, J, L0, dp, pred):
    """One row of finiteN_lognormal.csv."""
    se = lambda x: x.std(ddof=1)/np.sqrt(R)
    return dict(N=N, R=R, mean_shift=d.mean()-dp, se_shift=se(d), sd_field=d.std(ddof=1),
                sd_takeoff=L0.std(ddof=1), mean_takeoff=L0.mean(), mean_jump=J.mean(), sd_jump=J.std(ddof=1),
                pred_shift=pred["shift_coef"]*N**(-2/3), pred_sd=np.sqrt(pred["sd_coef"]**2/N + (SDM/EM*pred["shift_coef"])**2*N**(-4/3)),
                pred_sd_lead=pred["sd_coef"]/np.sqrt(N), pred_loc=pred["loc_coef"]*N**(-1/3))


def pareto_row(mu, N, R, out, dp):
    """One row of finiteN_pareto.csv; ``out`` columns: field, jump, takeoff, trigger weight, max weight."""
    d = out[:, 0]; q = np.percentile(d, [5, 25, 50, 75, 95])
    return dict(mu=mu, N=N, R=R, d0p=dp, mean=d.mean(), sd=d.std(ddof=1), q05=q[0], q25=q[1], q50=q[2], q75=q[3], q95=q[4],
                iqr=q[3]-q[1], mean_jump=out[:, 1].mean(), sd_jump=out[:, 1].std(ddof=1),
                median_trigger_relweight=np.median(out[:, 3]),
                median_maxweight=np.median(out[:, 4]))


def avalanche_bins(log10_max, n_points):
    """Logarithmic bin edges used by the avalanche histograms (last bin doubled)."""
    bins = np.unique(np.round(np.logspace(0, log10_max, n_points)).astype(int)); return np.append(bins, bins[-1]*2)


def bm_parabola_samples(rng, dt=5e-4, V=5.0, R=20000):
    """Monte Carlo samples of M = max_v [W(v) - v^2/2] and its argmax, two-sided BM."""
    n = int(V/dt); v = dt*np.arange(1, n+1); par = 0.5*v**2
    M = np.empty(R); arg = np.empty(R)
    for r in range(R):
        a = np.cumsum(rng.standard_normal(n))*np.sqrt(dt) - par
        b = np.cumsum(rng.standard_normal(n))*np.sqrt(dt) - par
        ia, ib = a.argmax(), b.argmax()
        if max(a[ia], b[ib]) < 0: M[r], arg[r] = 0.0, 0.0
        elif a[ia] >= b[ib]: M[r], arg[r] = a[ia], v[ia]
        else: M[r], arg[r] = b[ib], -v[ib]
    return M, arg


BM_DISCRETIZATION = 0.5826   # standard discretization correction for the maximum of BM (times sqrt(dt))


def path_metrics(p, T, d_lo, d_hi, W=200, threshold=0.10, ref_field=0.30):
    """Per-cohort diagnostics of a sold-fraction path (exp_path.py memory study definitions).

    Returns final sold fraction, largest W-step increase, field of the largest event (window
    centre), sold fraction at ``ref_field`` (nan if outside the ramp), and the number of
    up-crossings of ``threshold`` by the W-step increase.
    """
    inc = p[W:] - p[:-W]; k = inc.argmax(); up = inc > threshold
    nev = int(up[0]) + int(np.sum(up[1:] & ~up[:-1]))
    t_ref = int(T*(ref_field-d_lo)/(d_hi-d_lo))
    s_ref = p[t_ref] if (d_lo <= ref_field <= d_hi and 0 <= t_ref < p.size) else np.nan
    return p[-1], inc[k], d_lo + (d_hi-d_lo)*min(k + W//2, T)/T, s_ref, nev


def _write_rows(fn, rows):
    with open(fn, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)


# --------------------------------------------------------------------------- bm_parabola.py
def run_bm_parabola(results="results", log=print):
    """Monte Carlo constants for M = max_v [W(v) - v^2/2], two-sided standard Brownian motion."""
    rng = np.random.default_rng(7)
    dt, V, R = 5e-4, 5.0, 20000  # sd[M] from this run is used only for a subleading correction
    M, arg = bm_parabola_samples(rng, dt, V, R)
    corr = BM_DISCRETIZATION*np.sqrt(dt)
    log("E[M] raw %.4f  corrected %.4f +- %.4f" % (M.mean(), M.mean()+corr, M.std()/np.sqrt(R)))
    log("sd[M] %.4f   sd[argmax] %.4f" % (M.std(), arg.std()))
    np.savetxt(os.path.join(results, "bm_parabola_constants.csv"), [[M.mean()+corr, M.std(), arg.std()]],
               header="E_M,sd_M,sd_argmax", delimiter=",", comments="")


# --------------------------------------------------------------------------- exp_finiteN.py
FINITEN_SCHEDULE = [(250, 4000), (1000, 4000), (4000, 4000), (16000, 3000), (64000, 2000), (256000, 1000)]


def run_finiteN(results="results", log=print):
    """Finite-cohort collapse statistics with lognormal exposures (Sec. VII)."""
    l, s, m, Lam, sq, rho = 2.0, 0.16, 0.25, 0.30, 0.75, 0.4
    pred = finite_n_theory(l, s, m, Lam, sq, rho); dp = pred["d0p"]
    log(pred)
    rng = np.random.default_rng(20260917)
    rows = []
    for N, R in FINITEN_SCHEDULE:
        out = np.empty((R, 3))
        for r in range(R):
            lev, Q = sample_cohort(rng, N, l, s, "lognormal", sq=sq, rho=rho)
            need, rec, Lb, w = absorbing_cascade(lev, Q, m, Lam)
            out[r] = collapse_record(need, rec, Lb, w, N)[:3]
        d, J, L0 = out.T
        rows.append(lognormal_row(N, R, d, J, L0, dp, pred))
        log({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rows[-1].items()}); _flush()
    _write_rows(os.path.join(results, "finiteN_lognormal.csv"), rows)
    _write_rows(os.path.join(results, "finiteN_theory_constants.csv"), [pred])


# --------------------------------------------------------------------------- exp_pareto.py
def pareto_schedule(mu):
    return ([(1000, 20000), (4000, 10000), (16000, 5000), (64000, 2000), (256000, 1000), (1024000, 400)] if mu < 2
            else [(1000, 4000), (4000, 4000), (16000, 2000), (64000, 1000), (256000, 600), (1024000, 250)])


def run_pareto(mu_arg, results="results", log=print):
    """Collapse-field statistics with Pareto exposures independent of leverage (Sec. VIII).

    ``mu_arg`` is the tail index as given on the command line (it names the .npz file).
    Rows are appended to finiteN_pareto.csv (header written only if the file is new).
    """
    l, s, m, Lam, chi = 2.0, 0.16, 0.25, 0.30, 0.0
    (Bp, dp), (Bm, dm) = folds(l, s, m, chi, Lam)
    mu = float(mu_arg)
    rng = np.random.default_rng(int(1000*mu) + 17)
    rows = []
    raw = {}
    for N, R in pareto_schedule(mu):
        out = np.empty((R, 4))
        for r in range(R):
            lev, Q = sample_cohort(rng, N, l, s, "pareto", mu=mu)
            need, rec, Lb, w = absorbing_cascade(lev, Q, m, Lam)
            dc, J, _, trig, wmax = collapse_record(need, rec, Lb, w, N)
            out[r] = dc, J, trig, wmax
        raw[str(N)] = out[:, :2].astype(np.float32)
        full = np.column_stack([out[:, 0], out[:, 1], np.zeros(R), out[:, 2], out[:, 3]])
        rows.append(pareto_row(mu, N, R, full, dp)); q = np.percentile(out[:, 0], [5, 25, 50, 75, 95])
        log(N, "sd %.5f iqr %.5f med-shift %.5f trig %.2f" % (rows[-1]["sd"], rows[-1]["iqr"], q[2]-dp, rows[-1]["median_trigger_relweight"])); _flush()
    np.savez_compressed(os.path.join(results, "pareto_raw_mu%s.npz" % mu_arg), **raw)
    fn = os.path.join(results, "finiteN_pareto.csv"); new = not os.path.exists(fn)
    with open(fn, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        if new: wr.writeheader()
        wr.writerows(rows)


# --------------------------------------------------------------------------- exp_avalanche.py
def avalanche_slope(ctr, wd, H, lo, hi):
    k = (ctr >= lo) & (ctr <= hi) & (H > 20); return np.polyfit(np.log(ctr[k]), np.log(H[k]/wd[k]), 1)[0]


def run_avalanche(kind, results="results", log=print):
    """Precursor avalanche-size statistics (account counts) before the macroscopic collapse (Sec. IX)."""
    l, s, m, Lam = 2.0, 0.16, 0.25, 0.30
    N = 2_000_000; R = 40
    rng = np.random.default_rng(99 if kind == "lognormal" else 101)
    bins = avalanche_bins(4.5, 28)
    H_all = np.zeros(len(bins)-1); H_near = np.zeros(len(bins)-1); n_all = n_near = 0
    exposure = kind if kind in ("equal", "lognormal") else "pareto"
    mu = float(kind) if exposure == "pareto" else 1.5
    for r in range(R):
        lev, Q = sample_cohort(rng, N, l, s, exposure, sq=0.75, rho=0.4, mu=mu)
        need, rec, Lb, w = absorbing_cascade(lev, Q, m, Lam)
        S, d, dc = avalanche_sizes(need, rec, Lb, N)
        near = d > dc - 0.002
        H_all += np.histogram(S, bins)[0]; H_near += np.histogram(S[near], bins)[0]; n_all += S.size; n_near += near.sum()
    wd = np.diff(bins); ctr = np.sqrt(bins[:-1]*(bins[1:]-1).clip(1))
    with open(os.path.join(results, "avalanches_%s.csv" % kind), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["S_center", "bin_lo", "bin_hi", "pdf_all", "count_all", "pdf_near", "count_near"])
        for k in range(len(wd)):
            wr.writerow([ctr[k], bins[k], bins[k+1], H_all[k]/wd[k]/n_all, int(H_all[k]), H_near[k]/wd[k]/max(n_near, 1), int(H_near[k])])
    log(kind, "integrated slope [5,500]: %.3f   near-fold slope [3,300]: %.3f   n_all %d n_near %d" % (
        avalanche_slope(ctr, wd, H_all, 5, 500), avalanche_slope(ctr, wd, H_near, 3, 300), n_all, n_near))


# --------------------------------------------------------------------------- avalanche_size_sweep.py
def borel_comparison(bins, H, nav, l=2.0, s=0.16, m=0.25, Lam=0.30, lo=200, hi=1000):
    """Count-weighted simulation/Borel ratio and mean absolute relative error for bin centres in [lo, hi]."""
    from .aval_theory import integrated_borel
    wd = np.diff(bins); ctr = np.sqrt(bins[:-1]*(bins[1:]-1).clip(1)); pdf = H/wd/nav
    th = np.array([integrated_borel(np.arange(a, b), l, s, m, 0.0, Lam).mean() for a, b in zip(bins[:-1], bins[1:])])
    mask = (ctr >= lo) & (ctr <= hi) & (H >= 10)
    # count-weighted mean log ratio and mean absolute relative discrepancy
    weights = H[mask]/H[mask].sum()
    ratio = np.exp(np.sum(weights*np.log(pdf[mask]/th[mask])))
    mare = np.sum(weights*np.abs(pdf[mask]/th[mask]-1))
    return ratio, mare, int(H[mask].sum()), th


def run_avalanche_size_sweep(results="results", log=print):
    import pandas as pd
    l, s, m, Lam = 2.0, 0.16, 0.25, 0.30
    bins = avalanche_bins(3.6, 24)
    rows = []
    for N, R in [(125_000, 60), (500_000, 30), (2_000_000, 12)]:
        rng = np.random.default_rng(20260922+N)
        H = np.zeros(len(bins)-1); nav = 0
        for r in range(R):
            lev, Q = sample_cohort(rng, N, l, s, "equal")
            need, rec, Lb, w = absorbing_cascade(lev, Q, m, Lam)
            S = avalanche_sizes(need, rec, Lb, N)[0]
            H += np.histogram(S, bins)[0]; nav += len(S)
        ratio, mare, tot, _ = borel_comparison(bins, H, nav, l, s, m, Lam)
        rows.append(dict(N=N, R=R, n_avalanches=nav, ratio_sim_theory_200_1000=ratio, mean_abs_rel_error_200_1000=mare, total_counts_200_1000=tot))
        log(rows[-1]); _flush()
    pd.DataFrame(rows).to_csv(os.path.join(results, "avalanche_finiteN_sweep.csv"), index=False)


# --------------------------------------------------------------------------- exp_path.py
PATH_LAM, PATH_GAM, PATH_FLOOR, PATH_HOLD, PATH_NC = 0.5, 0.22, 0.88, 500, 32


def run_path_study(which, results="results", log=print):
    """Path-resolved experiments: cohort-size convergence, ramp-rate sharpening, impact-memory sweep (Sec. X)."""
    from .pathsim import diagnostics, make_cohort, run_path
    LAM, GAM, FLOOR, HOLD, NC = PATH_LAM, PATH_GAM, PATH_FLOOR, PATH_HOLD, PATH_NC
    curves = {}
    if which == "size":
        rows = []; NCS = 48
        for hI in [80.0, -1.0]:
            for N in [1000, 2000, 4000, 8000, 16000, 32000, 64000]:
                rng = np.random.default_rng(5000 + N); res = []
                for c in range(NCS):
                    p = run_path(*make_cohort(rng, N), LAM, GAM, hI, 0.02, 0.30, 3000, HOLD, FLOOR); res.append(diagnostics(p, 3000, .02, .30))
                    if c == 0 and hI > 0 and N in (2000, 8000, 32000): curves["N%d" % N] = p[::5].astype(np.float32)
                res = np.array(res); curves["mid_h%d_N%d" % (hI, N)] = res[:, 3]
                med = np.median(res[:, 3])
                rows.append(dict(half_life=hI, N=N, cohorts=NCS, final_sold_mean=res[:, 0].mean(), final_sold_sd=res[:, 0].std(ddof=1), mid_field_mean=res[:, 3].mean(),
                                 mid_field_median=med, mid_field_sd=res[:, 3].std(ddof=1), mid_field_iqr=np.subtract(*np.percentile(res[:, 3], [75, 25])),
                                 frac_late=np.mean(res[:, 3] > med + 0.008), width_steps_mean=res[:, 2].mean(), width_steps_sd=res[:, 2].std(ddof=1)))
                r = rows[-1]; log(hI, N, "mid %.4f sd %.4f iqr %.4f late %.2f  Sf %.3f  width %.1f" % (r["mid_field_mean"], r["mid_field_sd"], r["mid_field_iqr"], r["frac_late"], r["final_sold_mean"], r["width_steps_mean"])); _flush()
        _write_rows(os.path.join(results, "path_size.csv"), rows)
    if which == "ramp":
        rows = []
        for T in [250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]:
            rng = np.random.default_rng(777); res = []          # same cohorts at every ramp duration
            for c in range(NC):
                p = run_path(*make_cohort(rng, 8000), LAM, GAM, 80.0, 0.02, 0.30, T, HOLD, FLOOR); res.append(diagnostics(p, T, .02, .30))
                if c == 0 and T in (1000, 4000, 16000): curves["T%d" % T] = p[:: max(1, T//1000)].astype(np.float32)
            res = np.array(res); wf = res[:, 5] - res[:, 4]
            rows.append(dict(T=T, cohorts=NC, width_steps_mean=res[:, 2].mean(), width_steps_sd=res[:, 2].std(ddof=1), width_field_mean=wf.mean(),
                             width_field_sd=wf.std(ddof=1), mid_field_mean=res[:, 3].mean(), mid_field_sd=res[:, 3].std(ddof=1), final_sold_mean=res[:, 0].mean()))
            log(rows[-1]); _flush()
        _write_rows(os.path.join(results, "path_ramp.csv"), rows)
    if which == "memory":
        rows = []; T = 4600; hi = 0.45; W = 200
        hgrid = [3, 4, 6, 8, 10, 12, 14, 17, 20, 24, 28, 34, 40, 48, 57, 68, 80, 113, 160, 320, 640, -1]
        for delay in [1.0, 5.0, 15.0]:
            for Lam in [LAM, 0.0]:
                for hI in (hgrid if Lam > 0 else [80]):
                    rng = np.random.default_rng(31337); res = []
                    for c in range(NC):
                        p = run_path(*make_cohort(rng, 8000, delay=delay), Lam, GAM, float(hI), 0.02, hi, T, HOLD, FLOOR)
                        res.append(path_metrics(p, T, 0.02, hi, W, 0.10, 0.30))
                        if c == 0 and delay == 5.0 and Lam > 0 and hI in (4, 12, 20, 80, -1): curves["h%d" % hI] = p[::5].astype(np.float32)
                        if c == 0 and delay == 5.0 and Lam == 0: curves["baseline"] = p[::5].astype(np.float32)
                    res = np.array(res)
                    rows.append(dict(delay=delay, Lam=Lam, half_life=hI, cohorts=NC, final_sold_mean=res[:, 0].mean(), final_sold_sd=res[:, 0].std(ddof=1),
                                     largest_event_mean=res[:, 1].mean(), largest_event_sd=res[:, 1].std(ddof=1), event_field_mean=res[:, 2].mean(),
                                     event_field_sd=res[:, 2].std(ddof=1), sold_at_030_mean=res[:, 3].mean(), sold_at_030_sd=res[:, 3].std(ddof=1),
                                     n_events_mean=res[:, 4].mean()))
                    r = rows[-1]; log(delay, Lam, hI, "J %.3f+-%.3f  field %.3f+-%.3f  S(0.30) %.3f+-%.3f  Sf %.3f  nev %.2f" % (r["largest_event_mean"], r["largest_event_sd"], r["event_field_mean"], r["event_field_sd"], r["sold_at_030_mean"], r["sold_at_030_sd"], r["final_sold_mean"], r["n_events_mean"])); _flush()
        _write_rows(os.path.join(results, "path_memory.csv"), rows)
    np.savez_compressed(os.path.join(results, "path_curves_%s.npz" % which), **curves)


# --------------------------------------------------------------------------- stalled_ignition_diagnostic.py
def recU(path, h=80.0):
    """Per-step sold increments and the reconstructed unabsorbed impact stock with half-life h."""
    u = np.diff(np.r_[0., path]); delta = 2**(-1/h); U = np.empty_like(path); x = 0.
    for i, ui in enumerate(u): x = delta*x+ui; U[i] = x
    return u, U


def run_stalled_ignition(results="results", log=print):
    """Matched-cohort diagnostic of stalled ignition under transient impact.

    Note: writes ``stalled_ignition_control2.csv`` (the file name used by the original script),
    while ``validate_diagnostics.py`` reads ``stalled_ignition_diagnostic.csv``.
    """
    import pandas as pd
    from .pathsim import make_cohort, run_path
    LAM, GAM, FLOOR, HOLD = 0.5, 0.22, 0.88, 500
    N, T = 4000, 3000; dlo, dhi = .02, .30
    rng = np.random.default_rng(97531)
    rows = []

    def fld(t): return dlo+(dhi-dlo)*min(t, T)/T

    for c in range(128):
        cohort = make_cohort(rng, N)
        p = run_path(*cohort, LAM, GAM, 80.0, dlo, dhi, T, HOLD, FLOOR)
        t50 = int(np.searchsorted(p, .5*p[-1])); field = fld(t50)
        u, U = recU(p)
        # detect first recruitment episode: first 100-step increment > 0.01
        W = 100; inc = p[W:]-p[:-W]; ids = np.where(inc > .01)[0]
        if len(ids):
            ts = int(ids[0])
            # first episode ends at first point after ts+W where 100-step increment falls below 0.005
            after = np.where(inc[min(ts+W, len(inc)-1):] < .005)[0]
            te = (min(ts+W, len(inc)-1)+int(after[0])) if len(after) else min(t50, ts+400)
            te = min(te, t50)
            peak = float(U[ts:te+1].max()) if te >= ts else np.nan
            j = ts+int(np.argmax(U[ts:te+1])) if te >= ts else ts
            if t50 > j+10:
                mn = float(U[j:t50].min()); decay = (peak-mn)/peak if peak > 0 else np.nan
            else: mn = peak; decay = 0.
            sold_first = float(p[te]-p[max(0, ts-1)])
            gap = t50-te
        else: ts = te = -1; peak = mn = decay = sold_first = gap = np.nan
        rows.append(dict(cohort=c, field=field, final=p[-1], t50=t50, first_start=ts, first_end=te, first_wave_sold=sold_first, U_peak=peak, U_min_before_ignition=mn, U_decay_fraction=decay, gap_steps=gap))

    df = pd.DataFrame(rows)
    med = df.field.median(); df['late'] = df.field > med+.008
    log('median field', med, 'late fraction', df.late.mean())
    log(df.groupby('late')[['field', 'first_wave_sold', 'U_decay_fraction', 'gap_steps']].agg(['count', 'mean', 'median']).to_string())
    log('late decay > .25', np.mean(df.loc[df.late, 'U_decay_fraction'] > .25) if df.late.any() else np.nan)
    log('nonlate decay > .25', np.mean(df.loc[~df.late, 'U_decay_fraction'] > .25))
    df.to_csv(os.path.join(results, 'stalled_ignition_control2.csv'), index=False)


# --------------------------------------------------------------------------- dominant_holder_meanfield_controls.py
def dominant_meanfield_fold(omega, l=2.0, s=0.16, m=0.25, Lam=0.30):
    """Upper fold of the background mean-field map when a conservative holder absorbs share omega."""
    from scipy.optimize import minimize_scalar
    # conservative holder stays intact through the background fold
    def d0(B):
        return t_of_B(B, l, s, m, 0.0)-Lam*np.sqrt((1.0-omega)*B)
    r = minimize_scalar(lambda B: -d0(B), bounds=(1e-6, 0.2), method='bounded', options={'xatol': 1e-13})
    return float(r.x), float(d0(r.x))


def dominant_simulated_background_loss(t, Q, omega, l=2.0, s=0.16, m=0.25, Lam=0.30, frac=.25, holder_z=-2.0):
    """External field at which the background cohort (thresholds t, exposures Q) first loses
    ``frac`` of its own exposure, with a conservative holder of share omega added."""
    N = t.size
    if omega == 0:
        tt = t.copy(); QQ = Q.copy(); bg = np.ones(N, dtype=bool)
    else:
        qstar = omega*N/(1-omega)
        lstar = max(1.0, l*(1+s*(holder_z)))  # conservative holder, z=-2
        tstar = (1/lstar-m)/(1-m)
        tt = np.append(t, tstar); QQ = np.append(Q, qstar); bg = np.r_[np.ones(N, dtype=bool), False]
    o = np.argsort(tt, kind='stable'); ts = tt[o]; q = QQ[o]; bg = bg[o]
    w = q/q.sum(); cw = np.cumsum(w); Lb = np.empty_like(cw); Lb[0] = 0; Lb[1:] = cw[:-1]
    need = ts-Lam*np.sqrt(Lb)
    rec = np.flatnonzero(need >= np.maximum.accumulate(need)); ends = np.r_[rec[1:], len(tt)]
    bg_total = w[bg].sum(); sold_bg = 0.0
    largest = (-1.0, np.nan)
    for start, end in zip(rec, ends):
        event = w[start:end].sum()
        if event > largest[0]: largest = (event, float(need[start]))
        sold_bg += w[start:end][bg[start:end]].sum()
        if sold_bg/bg_total >= frac:
            return float(need[start]), float(sold_bg/bg_total), float(largest[1])
    raise RuntimeError('threshold not reached')


def path_meanfield_benchmark(l=2.0, s=0.16, sq=0.75, rho=0.40, mmean=0.25, msd=0.025, Lam=0.50, xc=0.02,
                             bounds=(0.05, 0.22)):
    """Binary complete-liquidation, permanent-impact mean-field fold for the path-resolved
    population (linear-to-square-root impact, maintenance ratios truncated at +-4 sd).
    Returns (transition field d0 at the fold, total drawdown at the fold, B at the fold)."""
    from scipy.integrate import quad
    from scipy.optimize import minimize_scalar
    chi = rho*sq
    Ztr = norm.cdf(4)-norm.cdf(-4)

    def fcross(x):
        return (np.sqrt(1+x/xc)-1)/(np.sqrt(1+1/xc)-1)

    def Fd(d):
        def integ(mm):
            A = mm+(1-mm)*d
            zthr = (1/(l*A)-1)/s
            p = 1-norm.cdf(zthr-chi)
            dens = norm.pdf((mm-mmean)/msd)/(msd*Ztr)
            return p*dens
        return quad(integ, mmean-4*msd, mmean+4*msd, epsabs=1e-12, epsrel=1e-11, limit=200)[0]

    def d0d(d): return d-Lam*fcross(Fd(d))
    r = minimize_scalar(lambda d: -d0d(d), bounds=bounds, method='bounded', options={'xatol': 1e-13})
    dtot = float(r.x); d0 = float(d0d(dtot)); B = float(Fd(dtot))
    return d0, dtot, B


def run_dominant_holder_controls(results="results", log=print):
    import pandas as pd
    # A. Event-definition-independent dominant-holder check
    l, s, m, Lam = 2.0, 0.16, 0.25, 0.30
    rng = np.random.default_rng(20260922)
    N = 100_000
    Z = rng.standard_normal(N)
    lev = np.maximum(1.0, l*(1+s*Z))
    Q = np.ones(N)
    t = (1/lev-m)/(1-m)

    rows = []
    for omega in [0.0, 0.10, 0.20]:
        Bf, df = dominant_meanfield_fold(omega, l, s, m, Lam)
        sf, bgfrac, op = dominant_simulated_background_loss(t, Q, omega, l, s, m, Lam, .25)
        rows.append(dict(holder_share=omega, meanfield_fold_B=Bf, meanfield_fold_field=df,
                         simulation_field_bg25=sf, bg_fraction_after_event=bgfrac,
                         sim_minus_theory=sf-df, largest_avalanche_field_seen_to_bg25=op))
    pd.DataFrame(rows).to_csv(os.path.join(results, 'dominant_holder_event_independent.csv'), index=False)

    # B. Mean-field reference for Table II population in Sec. X
    # Binary complete liquidation + permanent impact, same linear-to-square-root f and truncated m distribution.
    d0, dtot, B = path_meanfield_benchmark(2.0, 0.16, 0.75, 0.40, 0.25, 0.025, 0.50, 0.02)
    # Existing large-N path-resolved medians from archived path_size.csv
    ps = pd.read_csv(os.path.join(results, 'path_size.csv'))

    def largeN(h):
        z = ps[(ps.half_life == h) & (ps.N >= 16000)]
        return float(z.mid_field_median.mean()), float(z.mid_field_median.min()), float(z.mid_field_median.max())
    perm = largeN(-1.0); trans = largeN(80.0)
    out = pd.DataFrame([
        dict(case='binary_mean_field_permanent', transition_field=d0, low=np.nan, high=np.nan, total_drawdown_at_fold=dtot, B_at_fold=B),
        dict(case='path_resolved_permanent_largeN_median', transition_field=perm[0], low=perm[1], high=perm[2], total_drawdown_at_fold=np.nan, B_at_fold=np.nan),
        dict(case='path_resolved_half_life80_largeN_median', transition_field=trans[0], low=trans[1], high=trans[2], total_drawdown_at_fold=np.nan, B_at_fold=np.nan),
    ])
    out.to_csv(os.path.join(results, 'path_model_meanfield_reference.csv'), index=False)
    log('dominant holder')
    log(pd.DataFrame(rows).to_string(index=False))
    log('\nsection X')
    log(out.to_string(index=False))

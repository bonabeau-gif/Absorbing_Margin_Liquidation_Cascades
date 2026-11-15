"""Compact, structured answers about finished run directories. Queries read files only;
they never rerun the model. Every answer carries units and states any approximation."""
import csv
import math
import os

from .rundir import RunDirError, read_run


class QueryError(Exception):
    def __init__(self, field, value, reason, hint=None):
        super().__init__(reason)
        self.error = dict(field=field, value=value, reason=reason, hint=hint)


def _f(x):
    if x == "" or x is None:
        return None
    try:
        v = float(x)
    except ValueError:
        return x
    return int(v) if v.is_integer() and "." not in x and "e" not in x.lower() else v


def load_table(run, fname):
    with open(os.path.join(run, "outputs", fname), newline="") as f:
        rd = csv.reader(f)
        header = next(rd)
        rows = [[_f(x) for x in r] for r in rd]
    return header, rows


def _ctx(run):
    info = read_run(run)
    if info["schema"] is None:
        raise RunDirError("%s has no schema.json (status %s)" % (run, (info["manifest"] or {}).get("status")))
    return info


def _units(schema):
    u = {}
    for fn, f in schema["files"].items():
        for c in f["columns"]:
            u[(fn.split("/")[-1], c["name"])] = c["unit"]
    return u


def _find_column(info, name, table=None):
    schema = info["schema"]
    idx_table = schema["index"]["table"] + ".csv"
    files = [fn.split("/")[-1] for fn in schema["files"]]
    if table:
        t = table if table.endswith(".csv") else table + ".csv"
        if t not in files:
            raise QueryError("--table", table, "no such output table", "tables: %s" % ", ".join(f[:-4] for f in files))
        files = [t]
    order = sorted(files, key=lambda f: f != idx_table)
    for fn in order:
        cols = [c["name"] for c in schema["files"]["outputs/" + fn]["columns"]]
        if name in cols:
            return fn
    allc = sorted({c["name"] for f in schema["files"].values() for c in f["columns"]})
    import difflib
    s = difflib.get_close_matches(name, allc, n=1)
    raise QueryError("name", name, "no recorded quantity with this name",
                     ("did you mean %r? " % s[0] if s else "") + "recorded: %s" % ", ".join(allc))


def _xcol(info, fn):
    """The axis of a table: the run's index column if present, else the first column."""
    idx = info["schema"]["index"]
    header = [c["name"] for c in info["schema"]["files"]["outputs/" + fn]["columns"]]
    return idx["column"] if idx["column"] in header else header[0]


# --------------------------------------------------------------------------- queries
def q_summary(run):
    info = _ctx(run)
    return dict(run=run, status=info["manifest"]["status"], **info["summary"])


def q_outputs(run):
    info = _ctx(run)
    out = []
    for fn, f in info["schema"]["files"].items():
        p = os.path.join(run, fn)
        h, rows = load_table(run, fn.split("/")[-1])
        out.append(dict(file=fn, description=f["description"], rows=len(rows), bytes=os.path.getsize(p),
                        columns=[dict(name=c["name"], unit=c["unit"], description=c["description"]) for c in f["columns"]]))
    return dict(run=run, model=info["schema"]["model"], index=info["schema"]["index"], files=out,
                summary_quantities=[q["name"] for q in info["schema"]["summary"]["quantities"]])


def q_series(run, name, x_from=None, x_to=None, points=100, table=None):
    info = _ctx(run)
    fn = _find_column(info, name, table); xc = _xcol(info, fn)
    h, rows = load_table(run, fn)
    ix, iy = h.index(xc), h.index(name)
    pts = [(r[ix], r[iy]) for r in rows if (x_from is None or r[ix] >= x_from) and (x_to is None or r[ix] <= x_to)]
    n = len(pts)
    if points < 2:
        raise QueryError("--points", points, "must be >= 2", None)
    if n > points:
        sel = sorted({round(i*(n-1)/(points-1)) for i in range(points)})
        pts = [pts[i] for i in sel]
        method = "decimation: %d of %d rows at evenly spaced row indices, first and last kept (no averaging)" % (len(pts), n)
    else:
        method = "none (all %d rows)" % n
    u = _units(info["schema"])
    return dict(run=run, table=fn, x=xc, x_unit=u[(fn, xc)], y=name, y_unit=u[(fn, name)], downsampling=method,
                points=[[a, b] for a, b in pts])


def _stats(xs, ys):
    vals = [(x, y) for x, y in zip(xs, ys) if isinstance(y, (int, float)) and not (isinstance(y, float) and math.isnan(y))]
    if not vals:
        return dict(count=0, nulls=len(ys))
    v = [y for _, y in vals]
    imax = max(range(len(v)), key=lambda i: v[i]); imin = min(range(len(v)), key=lambda i: v[i])
    mean = math.fsum(v)/len(v)
    sd = math.sqrt(math.fsum((y-mean)**2 for y in v)/(len(v)-1)) if len(v) > 1 else None
    return dict(count=len(v), nulls=len(ys)-len(v), min=v[imin], max=v[imax], mean=mean, sd=sd, final=ys[-1],
                x_at_max=vals[imax][0], x_at_min=vals[imin][0])


def q_stats(run, name, table=None, where=None):
    info = _ctx(run)
    fn = _find_column(info, name, table); xc = _xcol(info, fn)
    h, rows = load_table(run, fn)
    rows = _where(h, rows, where)
    u = _units(info["schema"])
    s = _stats([r[h.index(xc)] for r in rows], [r[h.index(name)] for r in rows])
    return dict(run=run, table=fn, name=name, unit=u[(fn, name)], x=xc, x_unit=u[(fn, xc)], where=where,
                note="final = last row; x_at_max = value of %s at the maximum; sd uses ddof=1" % xc, **s)


def _where(h, rows, where):
    if not where:
        return rows
    if "=" not in where:
        raise QueryError("--where", where, "must be column=value", "e.g. --where N=1000")
    k, v = where.split("=", 1)
    if k not in h:
        raise QueryError("--where", where, "unknown column %r" % k, "columns: %s" % ", ".join(h))
    v = _f(v); i = h.index(k)
    out = [r for r in rows if r[i] == v]
    if not out:
        raise QueryError("--where", where, "no rows match", "values present: %s" % sorted({r[i] for r in rows})[:20])
    return out


def _row(info, run, fn, r, h):
    u = _units(info["schema"])
    return {c: dict(value=v, unit=u.get((fn, c))) for c, v in zip(h, r)}


def q_at(run, x, table=None):
    info = _ctx(run)
    fn = (table if table and table.endswith(".csv") else (table + ".csv")) if table else info["schema"]["index"]["table"] + ".csv"
    xc = _xcol(info, fn)
    h, rows = load_table(run, fn)
    i = h.index(xc)
    best = min(range(len(rows)), key=lambda k: abs(rows[k][i] - x))
    return dict(run=run, table=fn, requested=x, axis=xc, nearest=rows[best][i], distance=abs(rows[best][i] - x),
                approximation="nearest recorded row (no interpolation)", state=_row(info, run, fn, rows[best], h))


def q_final(run):
    info = _ctx(run)
    fn = info["schema"]["index"]["table"] + ".csv"
    h, rows = load_table(run, fn)
    return dict(run=run, table=fn, axis=_xcol(info, fn), rows=len(rows), state=_row(info, run, fn, rows[-1], h),
                note="last row of the index table; use `query summary` for headline results")


def q_crossing(run, name, level, table=None):
    """First axis value at which ``name`` reaches ``level`` (linear interpolation between rows)."""
    info = _ctx(run)
    fn = _find_column(info, name, table); xc = _xcol(info, fn)
    h, rows = load_table(run, fn); ix, iy = h.index(xc), h.index(name)
    extra = "field" if "field" in h and xc != "field" else None
    prev = None
    for r in rows:
        y = r[iy]
        if y is None:
            continue
        if y >= level:
            if prev is None:
                x = r[ix]; fx = r[h.index(extra)] if extra else None
            else:
                t = (level - prev[iy])/(y - prev[iy]) if y != prev[iy] else 0.0
                x = prev[ix] + t*(r[ix] - prev[ix])
                fx = (prev[h.index(extra)] + t*(r[h.index(extra)] - prev[h.index(extra)])) if extra else None
            u = _units(info["schema"])
            out = dict(run=run, table=fn, name=name, level=level, crossed=True, x=xc, x_value=x, x_unit=u[(fn, xc)],
                       approximation="linear interpolation between consecutive rows")
            if extra:
                out.update(field_value=fx, field_unit=u[(fn, extra)])
            return out
        prev = r
    return dict(run=run, table=fn, name=name, level=level, crossed=False, max=max((r[iy] for r in rows if r[iy] is not None), default=None))


def q_scaling(run, name, x_min=None, x_max=None, table=None):
    """Least-squares slope of log(name) against log(axis): a power-law exponent."""
    info = _ctx(run)
    fn = _find_column(info, name, table); xc = _xcol(info, fn)
    h, rows = load_table(run, fn); ix, iy = h.index(xc), h.index(name)
    pts = [(math.log(r[ix]), math.log(abs(r[iy]))) for r in rows
           if isinstance(r[ix], (int, float)) and isinstance(r[iy], (int, float)) and r[ix] > 0 and r[iy]
           and (x_min is None or r[ix] >= x_min) and (x_max is None or r[ix] <= x_max)]
    if len(pts) < 2:
        raise QueryError("name", name, "fewer than two positive points in range", "widen --min/--max")
    n = len(pts); mx = math.fsum(p[0] for p in pts)/n; my = math.fsum(p[1] for p in pts)/n
    sxx = math.fsum((p[0]-mx)**2 for p in pts); sxy = math.fsum((p[0]-mx)*(p[1]-my) for p in pts)
    b = sxy/sxx; a = my - b*mx
    res = math.fsum((p[1]-a-b*p[0])**2 for p in pts)
    se = math.sqrt(res/(n-2)/sxx) if n > 2 else None
    return dict(run=run, table=fn, name=name, x=xc, exponent=b, exponent_se=se, prefactor=math.exp(a), points=n,
                method="least squares of log|%s| on log %s" % (name, xc), x_range=[x_min, x_max])


def q_quantiles(run, name, where=None, table=None):
    info = _ctx(run)
    fn = _find_column(info, name, table)
    h, rows = load_table(run, fn); rows = _where(h, rows, where)
    v = sorted(r[h.index(name)] for r in rows if isinstance(r[h.index(name)], (int, float)))
    if not v:
        raise QueryError("name", name, "no numeric values", None)

    def q(p):  # linear interpolation (numpy default)
        k = (len(v)-1)*p; lo = math.floor(k); hi = math.ceil(k)
        return v[lo] + (v[hi]-v[lo])*(k-lo)
    u = _units(info["schema"])
    return dict(run=run, table=fn, name=name, unit=u[(fn, name)], where=where, count=len(v),
                quantiles={str(p): q(p) for p in (0.05, 0.25, 0.5, 0.75, 0.95)}, iqr=q(0.75)-q(0.25),
                method="linear interpolation between order statistics (numpy default)")


# --------------------------------------------------------------------------- compare
def _flat(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(_flat(v, prefix + k + "."))
        elif isinstance(v, list):
            if all(isinstance(x, dict) for x in v):
                for i, x in enumerate(v):
                    out.update(_flat(x, "%s%s[%d]." % (prefix, k, i)))
            else:
                out[prefix + k] = v
        else:
            out[prefix + k] = v
    return out


def compare(runs):
    infos = [_ctx(r) for r in runs]
    base = infos[0]
    cfg = [dict(_flat(i["config"].get("parameters")), model=i["config"]["model"], seed=i["config"].get("seed")) for i in infos]
    keys = sorted(set().union(*cfg))
    pdiff = {k: [c.get(k) for c in cfg] for k in keys if len({repr(c.get(k)) for c in cfg}) > 1}
    sums = [_flat(i["summary"]["values"]) for i in infos]
    units = base["summary"].get("units", {})
    sdiff = {}
    for k in sorted(set().union(*sums)):
        vals = [s.get(k) for s in sums]
        if len({repr(v) for v in vals}) <= 1:
            continue
        e = dict(values=vals, unit=units.get(k.split(".")[0].split("[")[0]))
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            e["difference_vs_first"] = [v - vals[0] for v in vals[1:]]
            e["ratio_vs_first"] = [(v/vals[0]) if vals[0] else None for v in vals[1:]]
        sdiff[k] = e
    same = len(sums[0]) and all(len({repr(s.get(k)) for s in sums}) == 1 for k in sums[0])
    return dict(runs=runs, models=[c["model"] for c in cfg], parameters_differing=pdiff,
                summary_differing=sdiff, summary_identical=bool(same and not sdiff))

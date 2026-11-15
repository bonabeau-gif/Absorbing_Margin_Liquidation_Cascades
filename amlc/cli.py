"""``amlc``: command-line interface for the absorbing margin-liquidation cascade models.

Subcommands: describe, template, validate, run, query, compare, sweep. Start with ``amlc describe``.

Conventions: with ``--json`` standard output carries exactly one JSON document; progress and
warnings go to standard error. Exit codes: 0 success, 1 runtime failure, 2 invalid usage or
invalid configuration. Never prompts, never opens a window. Numerical libraries are imported
only by the subcommands that need them.
"""
import argparse
import itertools
import json
import os
import sys

from . import __version__
from . import config as C
from . import registry as R

EXIT_OK, EXIT_RUNTIME, EXIT_USAGE = 0, 1, 2


class UsageError(Exception):
    def __init__(self, errors):
        super().__init__(errors[0]["reason"] if errors else "invalid usage")
        self.errors = errors


def _emit(obj, as_json):
    if as_json:
        sys.stdout.write(json.dumps(obj, allow_nan=False, default=str) + "\n")
    else:
        sys.stdout.write(json.dumps(obj, indent=2, allow_nan=False, default=str) + "\n")


def _fail(errors, as_json, code):
    if as_json:
        _emit(dict(ok=False, errors=errors), True)
    else:
        for e in errors:
            line = "error: %s: %s" % (e.get("field"), e.get("reason"))
            if e.get("value") is not None:
                line += " (got %s)" % json.dumps(e["value"], default=str)
            if e.get("hint"):
                line += "; " + e["hint"]
            print(line, file=sys.stderr)
    return code


class Parser(argparse.ArgumentParser):
    """argparse that reports usage errors as structured JSON when --json is on the command line."""

    def error(self, message):
        if "--json" in sys.argv[1:]:
            _emit(dict(ok=False, errors=[dict(field="<arguments>", value=" ".join(sys.argv[1:]), reason=message,
                                              hint="see `%s --help`" % self.prog)]), True)
            sys.exit(EXIT_USAGE)
        super().error(message)


# --------------------------------------------------------------------------- describe
def _model_entry(m, detail=True):
    spec = R.MODELS[m]
    e = dict(name=m, title=spec["title"], description=spec["description"], source=spec["source"], stochastic=spec["stochastic"],
             reference_seed=spec["reference_seed"], parameters=spec["parameters"])
    if detail:
        e.update(parameter_details=[_param_entry(p) for p in R.model_params(m)],
                 constraints=[dict(expr=c["expr"], fields=c["fields"], message=c["message"]) for c in spec["constraints"]],
                 outputs=dict(index=spec["index"], files=spec["files"], summary=spec["summary"]))
    return e


def _param_entry(p):
    e = {k: p[k] for k in ("name", "type", "default", "range", "range_exclusive", "options", "unit", "description",
                           "applies_when", "item_type", "source")}
    e["required"] = p["default"] == R.REQUIRED
    e["applies_to"] = R.applies_to(p["name"])
    e["model_defaults"] = {m: s["defaults"][p["name"]] for m, s in R.MODELS.items() if p["name"] in s["defaults"]}
    e["constraints"] = [c["expr"] for s in R.MODELS.values() for c in s["constraints"] if p["name"] in c["fields"]]
    return e


def cmd_describe(a):
    if a.schema:
        _emit(C.json_schema(), a.json); return EXIT_OK
    if a.param:
        if a.param not in R.PARAM:
            import difflib
            s = difflib.get_close_matches(a.param, list(R.PARAM), n=1)
            raise UsageError([C.err("--param", a.param, "unknown parameter", ("did you mean %r?" % s[0]) if s else "run `amlc describe`")])
        _emit(dict(ok=True, parameter=_param_entry(R.PARAM[a.param])), a.json); return EXIT_OK
    if a.model:
        if a.model not in R.MODELS:
            raise UsageError([C.err("--model", a.model, "unknown model", "accepted: %s" % ", ".join(R.MODELS))])
        _emit(dict(ok=True, model=_model_entry(a.model)), a.json); return EXIT_OK
    if a.json:
        _emit(dict(ok=True, package="amlc", version=__version__, config_schema_version=R.SCHEMA_VERSION,
                   config_format="TOML (.toml) or JSON (.json); see `amlc template --model <name>`",
                   top_level_keys=R.TOP_LEVEL,
                   models={m: _model_entry(m, detail=False) for m in R.MODELS},
                   parameters=[_param_entry(p) for p in R.PARAMETERS],
                   outputs={m: dict(index=s["index"], files=s["files"], summary=s["summary"]) for m, s in R.MODELS.items()},
                   interventions=R.INTERVENTIONS,
                   interventions_note="none supported: the path model's external field follows the fixed linear ramp (ramp.*)",
                   fixed_constants=R.FIXED,
                   units_note="all quantities are dimensionless (README, Scope); unit fields name the dimensionless quantity"), True)
        return EXIT_OK
    print("amlc %s - absorbing margin-liquidation cascades. Models:\n" % __version__)
    for m, s in R.MODELS.items():
        print("  %-18s %s" % (m, s["title"]))
    print("\n%d parameters, %d models, no interventions. Details:" % (len(R.PARAMETERS), len(R.MODELS)))
    print("  amlc describe --model <name>      parameters, constraints and outputs of one model")
    print("  amlc describe --param <name>      one parameter")
    print("  amlc describe --json              everything, as one JSON document")
    print("  amlc describe --schema            JSON Schema for configuration files")
    return EXIT_OK


# --------------------------------------------------------------------------- template / validate
def cmd_template(a):
    if a.model not in R.MODELS:
        raise UsageError([C.err("--model", a.model, "unknown model", "accepted: %s" % ", ".join(R.MODELS))])
    spec = R.MODELS[a.model]
    resolved = dict(schema=R.SCHEMA_VERSION, model=a.model, seed=spec["reference_seed"] if spec["stochastic"] else 0,
                    parameters={p["name"]: p["default"] for p in R.model_params(a.model)})
    if a.format == "json" or a.json:
        _emit(C.resolved_json(resolved), a.json); return EXIT_OK
    hdr = ["amlc configuration template for model %r (generated from amlc/registry.py)." % a.model,
           spec["description"],
           "Validate with: amlc validate <file>; run with: amlc run <file> --out <dir>"]
    if spec["stochastic"]:
        hdr.append("seed = %s is the reference seed of the original script. Source: %s" % (spec["reference_seed"], spec["source"]))
    sys.stdout.write(C.to_toml(resolved, comments=True, header=hdr))
    return EXIT_OK


def _load_validate(path, allow_sweep=False):
    cfg = C.load(path)
    errors, warnings, resolved = C.validate(cfg, allow_sweep=allow_sweep)
    return errors, warnings, resolved


def cmd_validate(a):
    errors, warnings, resolved = _load_validate(a.config, allow_sweep=a.sweep)
    for w in warnings:
        print("warning: %s: %s" % (w["field"], w["reason"]), file=sys.stderr)
    if errors:
        if a.json:
            _emit(dict(ok=False, errors=errors, warnings=warnings), True)
        else:
            _fail(errors, False, EXIT_USAGE)
        return EXIT_USAGE
    from .runner import estimate
    est = estimate(resolved["model"], resolved["parameters"])
    if resolved.get("sweep"):
        n = 1
        for v in resolved["sweep"].values():
            n *= len(v)
        pts = _sweep_points(resolved)
        tot = sum(estimate(p["model"], p["parameters"])["runtime_s"] for p in pts)
        est = dict(est, points=n, total_runtime_s=round(tot, 1))
    _emit(dict(ok=True, model=resolved["model"], seed=resolved["seed"], warnings=warnings, estimate=est,
               resolved=C.resolved_json(resolved)), a.json)
    return EXIT_OK


# --------------------------------------------------------------------------- run / sweep
def cmd_run(a):
    errors, warnings, resolved = _load_validate(a.config)
    for w in warnings:
        print("warning: %s: %s" % (w["field"], w["reason"]), file=sys.stderr)
    if errors:
        return _fail(errors, a.json, EXIT_USAGE)
    from .rundir import RunDirError, execute
    try:
        out, man = execute(resolved, a.out, force=a.force, argv=["amlc"] + sys.argv[1:], quiet=a.quiet)
    except RunDirError as e:
        return _fail([C.err("--out", a.out, str(e), "use a new or empty directory")], a.json, EXIT_USAGE)
    except Exception as e:  # noqa: BLE001
        return _fail([C.err("<run>", a.out, "%s: %s" % (type(e).__name__, e), "see %s/log.txt and manifest.json" % a.out)],
                     a.json, EXIT_RUNTIME)
    summ = json.load(open(os.path.join(out, "summary.json"), encoding="utf-8"))
    _emit(dict(ok=True, run_dir=out, status=man["status"], seed=man["seed"], duration_s=man["duration_s"],
               summary=summ["values"], units=summ["units"]), a.json)
    return EXIT_OK


def _sweep_points(resolved):
    names = list(resolved["sweep"])
    pts = []
    for combo in itertools.product(*(resolved["sweep"][n] for n in names)):
        params = dict(resolved["parameters"]); params.update(dict(zip(names, combo)))
        pts.append(dict(schema=resolved["schema"], model=resolved["model"], seed=resolved["seed"], parameters=params,
                        point=dict(zip(names, combo))))
    return pts


def _scalars(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(_scalars(v, prefix + k + "."))
        elif isinstance(v, (int, float, bool)) or v is None:
            out[prefix + k] = v
    return out


def cmd_sweep(a):
    errors, warnings, resolved = _load_validate(a.config, allow_sweep=True)
    if not errors and not resolved.get("sweep"):
        errors = [C.err("sweep", None, "configuration has no [sweep] block", "add e.g. [sweep] with impact.half_life = [20, 80]")]
    if errors:
        return _fail(errors, a.json, EXIT_USAGE)
    import secrets
    from .rundir import RunDirError, dump_json, execute
    out = os.path.abspath(a.out)
    if os.path.exists(out) and os.listdir(out):
        if not (a.force and os.path.isfile(os.path.join(out, "index.json"))):
            return _fail([C.err("--out", a.out, "output directory is not empty",
                                "choose a new --out (or --force to replace a previous sweep directory)")], a.json, EXIT_USAGE)
        import shutil
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    if resolved["seed"] is None:
        resolved = dict(resolved, seed=secrets.randbits(32))
        print("no seed given; chose seed %d for every sweep point (recorded in index.json)" % resolved["seed"], file=sys.stderr)
    pts = _sweep_points(resolved)
    index = dict(model=resolved["model"], seed=resolved["seed"], sweep=resolved["sweep"],
                 base_parameters=resolved["parameters"], seed_policy="the same seed for every point (common random numbers)",
                 points=[])
    status = EXIT_OK
    for i, pt in enumerate(pts):
        d = os.path.join(out, "point_%03d" % i)
        print("sweep point %d/%d: %s" % (i + 1, len(pts), json.dumps(pt["point"])), file=sys.stderr)
        entry = dict(index=i, parameters=pt["point"], run_dir=os.path.relpath(d, out))
        try:
            execute({k: v for k, v in pt.items() if k != "point"}, d, argv=["amlc"] + sys.argv[1:], quiet=a.quiet)
            s = json.load(open(os.path.join(d, "summary.json"), encoding="utf-8"))
            entry.update(status="completed", summary=_scalars(s["values"]))
        except (RunDirError, Exception) as e:  # noqa: BLE001
            entry.update(status="failed", error="%s: %s" % (type(e).__name__, e)); status = EXIT_RUNTIME
        index["points"].append(entry)
        dump_json(os.path.join(out, "index.json"), index)
    _emit(dict(ok=status == EXIT_OK, sweep_dir=out, index=os.path.join(out, "index.json"), points=len(pts),
               failed=sum(p["status"] != "completed" for p in index["points"])), a.json)
    return status


# --------------------------------------------------------------------------- query / compare
def cmd_query(a):
    from . import query as Q
    run = a.run
    w = a.what
    if w == "summary":
        r = Q.q_summary(run)
    elif w == "outputs":
        r = Q.q_outputs(run)
    elif w == "series":
        r = Q.q_series(run, a.name, a.x_from, a.x_to, a.points, a.table)
    elif w == "stats":
        r = Q.q_stats(run, a.name, a.table, a.where)
    elif w == "at":
        r = Q.q_at(run, a.x, a.table)
    elif w == "final":
        r = Q.q_final(run)
    elif w == "crossing":
        r = Q.q_crossing(run, a.name, a.level, a.table)
    elif w == "scaling":
        r = Q.q_scaling(run, a.name, a.x_min, a.x_max, a.table)
    elif w == "quantiles":
        r = Q.q_quantiles(run, a.name, a.where, a.table)
    _emit(dict(ok=True, query=w, **r), a.json)
    return EXIT_OK


def cmd_compare(a):
    from . import query as Q
    _emit(dict(ok=True, **Q.compare(a.runs)), a.json)
    return EXIT_OK


# --------------------------------------------------------------------------- parser
def build_parser():
    F = argparse.RawDescriptionHelpFormatter
    ap = Parser(prog="amlc", formatter_class=F, description=__doc__.split("\n\n")[0] + "\n\nStart with: amlc describe",
                epilog="Exit codes: 0 success, 1 runtime failure, 2 invalid usage or configuration.")
    ap.add_argument("--version", action="version", version="amlc " + __version__)
    sub = ap.add_subparsers(dest="cmd", metavar="<command>", parser_class=Parser)

    def add(name, help_, example):
        p = sub.add_parser(name, help=help_, description=help_, epilog="example:\n  " + example, formatter_class=F)
        p.add_argument("--json", action="store_true", help="print exactly one JSON document on standard output")
        return p

    p = add("describe", "report models, parameters, outputs and interventions from the registry",
            "amlc describe --model path --json")
    p.add_argument("--model", help="one model in detail"); p.add_argument("--param", help="one parameter (dotted name)")
    p.add_argument("--schema", action="store_true", help="JSON Schema for configuration files")
    p.set_defaults(fn=cmd_describe)

    p = add("template", "print a complete, valid configuration for a model, every field commented",
            "amlc template --model finite_cohort > fc.toml")
    p.add_argument("--model", required=True, choices=sorted(R.MODELS))
    p.add_argument("--format", choices=["toml", "json"], default="toml", help="toml (commented, default) or json")
    p.set_defaults(fn=cmd_template)

    p = add("validate", "check a configuration without running it; report every problem and a cost estimate",
            "amlc template --model path | amlc validate - --json")
    p.add_argument("config", help="configuration file (.toml or .json), or - for standard input"); p.add_argument("--sweep", action="store_true", help="also accept and check a [sweep] block")
    p.set_defaults(fn=cmd_validate)

    p = add("run", "execute one run and write a self-describing run directory",
            "amlc run examples/typical_path.toml --out runs/path1 --json")
    p.add_argument("config"); p.add_argument("--out", required=True, help="run directory to create (must be new or empty)")
    p.add_argument("--force", action="store_true", help="replace an existing run directory")
    p.add_argument("--quiet", action="store_true", help="do not echo progress to standard error (log.txt still written)")
    p.set_defaults(fn=cmd_run)

    p = add("query", "answer targeted questions about a finished run (reads files only)",
            "amlc query runs/path1 series sold_mean --points 50 --json")
    p.add_argument("run", help="run directory")
    qs = p.add_subparsers(dest="what", metavar="<query>", parser_class=Parser, required=True)

    def qadd(name, help_, example):
        q = qs.add_parser(name, help=help_, description=help_, epilog="example:\n  " + example, formatter_class=F)
        q.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
        return q
    qadd("summary", "contents of summary.json", "amlc query RUN summary --json")
    qadd("outputs", "what was recorded: files, columns, units, rows, bytes", "amlc query RUN outputs --json")
    q = qadd("series", "one recorded column against the run's axis, downsampled to at most --points",
             "amlc query RUN series sold_mean --from 1000 --to 3000 --points 50 --json")
    q.add_argument("name"); q.add_argument("--from", dest="x_from", type=float); q.add_argument("--to", dest="x_to", type=float)
    q.add_argument("--points", type=int, default=100); q.add_argument("--table")
    q = qadd("stats", "min, max, mean, sd, final value and axis value at the maximum of one column",
             "amlc query RUN stats collapse_field --where N=1000 --json")
    q.add_argument("name"); q.add_argument("--table"); q.add_argument("--where", help="column=value row filter")
    q = qadd("at", "the recorded row nearest to an axis value (step, N, B, S_center, ...)", "amlc query RUN at 2500 --json")
    q.add_argument("x", type=float); q.add_argument("--table")
    qadd("final", "the last row of the run's index table", "amlc query RUN final --json")
    q = qadd("crossing", "first axis value where a column reaches a level (linear interpolation)",
             "amlc query RUN crossing sold_mean 0.25 --json")
    q.add_argument("name"); q.add_argument("level", type=float); q.add_argument("--table")
    q = qadd("scaling", "power-law exponent: slope of log(column) vs log(axis)", "amlc query RUN scaling sd_field --min 1000 --json")
    q.add_argument("name"); q.add_argument("--min", dest="x_min", type=float, help="smallest axis value to include")
    q.add_argument("--max", dest="x_max", type=float, help="largest axis value to include")
    q.add_argument("--table")
    q = qadd("quantiles", "5/25/50/75/95%% quantiles of a per-cohort column", "amlc query RUN quantiles collapse_field --where N=1000 --json")
    q.add_argument("name"); q.add_argument("--where"); q.add_argument("--table")
    p.set_defaults(fn=cmd_query)

    p = add("compare", "parameters that differ and differences in summary quantities between runs",
            "amlc compare runs/a runs/b --json")
    p.add_argument("runs", nargs="+"); p.set_defaults(fn=cmd_compare)

    p = add("sweep", "run the Cartesian product of a [sweep] block, one run directory per point plus index.json",
            "amlc sweep examples/sweep_half_life.toml --out runs/sweep1 --json")
    p.add_argument("config"); p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true", help="replace a previous sweep directory")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(fn=cmd_sweep)
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    if not getattr(a, "cmd", None):
        ap.print_help(sys.stderr); return EXIT_USAGE
    as_json = getattr(a, "json", False) or "--json" in (argv if argv is not None else sys.argv[1:])
    a.json = as_json
    try:
        return a.fn(a)
    except UsageError as e:
        return _fail(e.errors, as_json, EXIT_USAGE)
    except C.ConfigError as e:
        return _fail(e.errors, as_json, EXIT_USAGE)
    except Exception as e:  # noqa: BLE001
        from .query import QueryError
        from .rundir import RunDirError
        if isinstance(e, QueryError):
            return _fail([e.error], as_json, EXIT_USAGE)
        if isinstance(e, (RunDirError, FileNotFoundError)):
            return _fail([C.err("<run>", None, str(e), "give a run directory written by `amlc run`")], as_json, EXIT_USAGE)
        return _fail([C.err("<internal>", None, "%s: %s" % (type(e).__name__, e), "please report this")], as_json, EXIT_RUNTIME)


if __name__ == "__main__":
    sys.exit(main())

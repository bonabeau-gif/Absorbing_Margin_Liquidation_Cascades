#!/usr/bin/env python3
"""Tests for the agent-facing layer (amlc CLI) and the old entry points.

Run with ``python tests/test_agent_layer.py`` (about 15 s; also collected by pytest if installed).
Writes only to temporary directories.
"""
import csv
import filecmp
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EX = ROOT / "examples"
ENV = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))

# Values measured from `amlc run examples/minimal_finite_cohort.toml` (2026-11-15); identical to the
# first two rows of results/finiteN_lognormal.csv from exp_finiteN.py.
MINIMAL_EXPECTED = {
    "d0_plus": 0.10892667162084504,
    "largest_N.N": 1000,
    "largest_N.mean_shift": 0.010221910727620964,
    "largest_N.sd_field": 0.00792919226157876,
    "largest_N.mean_jump": 0.7388625380942756,
    "theory.sd_coef": 0.2681926715194646,
    "theory.shift_coef": 0.9099789254294999,
}


def amlc(*args, cwd=None, check=None):
    p = subprocess.run([sys.executable, "-m", "amlc", *map(str, args)], cwd=cwd or ROOT, env=ENV,
                       capture_output=True, text=True)
    if check is not None:
        assert p.returncode == check, (args, p.returncode, p.stdout[-2000:], p.stderr[-2000:])
    return p


def js(p):
    return json.loads(p.stdout)   # must be exactly one JSON document


def get(d, dotted):
    for k in dotted.split("."):
        d = d[k]
    return d


def test_describe_covers_registry():
    from amlc import registry as R
    d = js(amlc("describe", "--json", check=0))
    assert set(d["models"]) == set(R.MODELS)
    names = {p["name"] for p in d["parameters"]}
    assert names == set(R.PARAM)
    for p in d["parameters"]:
        for k in ("type", "default", "unit", "description", "range", "applies_to"):
            assert k in p, (p["name"], k)
        assert p["description"] and p["unit"]
    for m in R.MODELS:
        assert set(d["outputs"][m]["files"]) == set(R.MODELS[m]["files"])
        md = js(amlc("describe", "--model", m, "--json", check=0))["model"]
        assert {p["name"] for p in md["parameter_details"]} == set(R.MODELS[m]["parameters"])
    assert d["interventions"] == []


def test_schema_committed_matches_registry():
    d = js(amlc("describe", "--schema", "--json", check=0))
    committed = json.loads((ROOT / "amlc" / "config.schema.json").read_text())
    assert d == committed, "regenerate amlc/config.schema.json with `amlc describe --schema --json`"


def test_templates_validate():
    from amlc import registry as R
    with tempfile.TemporaryDirectory() as t:
        for m in R.MODELS:
            for fmt in ("toml", "json"):
                p = Path(t) / ("%s.%s" % (m, fmt))
                p.write_text(amlc("template", "--model", m, "--format", fmt, check=0).stdout)
                v = js(amlc("validate", p, "--json", check=0))
                assert v["ok"] and v["estimate"]["runtime_s"] > 0 and not v["warnings"], (m, fmt, v)


def test_invalid_config_reports_every_problem():
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "bad.toml"
        p.write_text('schema = 1\nmodel = "path"\nseed = 1\n[parameters]\ncohort.size = 0\n'
                     'impact.half_lfe = 20.0\nramp.start = 0.30\nramp.end = 0.10\n')
        r = amlc("validate", p, "--json", check=2)
        d = js(r)
        assert d["ok"] is False
        fields = {e["field"] for e in d["errors"]}
        assert {"parameters.cohort.size", "parameters.impact.half_lfe", "parameters.ramp.end"} <= fields, fields
        assert any("impact.half_life" in (e["hint"] or "") for e in d["errors"])
        for e in d["errors"]:
            assert set(e) == {"field", "value", "reason", "hint"}
        assert amlc("run", p, "--out", Path(t) / "r", "--json", check=2).stdout.startswith('{"ok": false')
        assert not (Path(t) / "r").exists()


def test_json_channel_on_usage_errors():
    for args in (("run", "--json"), ("query", "/nonexistent", "summary", "--json"),
                 ("describe", "--param", "nope", "--json"), ("validate", "/nonexistent.toml", "--json")):
        r = amlc(*args)
        assert r.returncode == 2, (args, r.returncode)
        assert js(r)["ok"] is False


def test_help_everywhere():
    for args in ([], ["describe"], ["template"], ["validate"], ["run"], ["query"], ["compare"], ["sweep"],
                 ["query", "R", "summary"], ["query", "R", "outputs"], ["query", "R", "series"], ["query", "R", "stats"],
                 ["query", "R", "at"], ["query", "R", "final"], ["query", "R", "crossing"], ["query", "R", "scaling"],
                 ["query", "R", "quantiles"]):
        r = amlc(*args, "--help")
        assert r.returncode == 0 and "usage" in r.stdout and "Traceback" not in r.stderr, (args, r.stderr[-500:])


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, prefix + k + "."))
        else:
            out[prefix + k] = v
    return out


def test_smoke_minimal_example_and_reproducibility():
    with tempfile.TemporaryDirectory() as t:
        cfg = EX / "minimal_finite_cohort.toml"
        amlc("validate", cfg, "--json", check=0)
        a, b = Path(t) / "a", Path(t) / "b"
        amlc("run", cfg, "--out", a, "--json", "--quiet", check=0)
        s = js(amlc("query", a, "summary", "--json", check=0))
        for k, want in MINIMAL_EXPECTED.items():
            got = get(s["values"], k)
            assert math.isclose(got, want, rel_tol=1e-9, abs_tol=1e-15), (k, got, want)
        # same configuration from another working directory -> byte-identical outputs and summary
        amlc("run", cfg, "--out", b, "--json", "--quiet", cwd=t, check=0)
        assert (a / "summary.json").read_bytes() == (b / "summary.json").read_bytes()
        cmp = filecmp.dircmp(a / "outputs", b / "outputs")
        assert not cmp.diff_files and not cmp.left_only and not cmp.right_only
        for f in cmp.common_files:
            assert filecmp.cmp(a / "outputs" / f, b / "outputs" / f, shallow=False)
        # the resolved config alone reproduces the run
        c = Path(t) / "copy"
        shutil.copytree(a, c)
        amlc("run", c / "config.resolved.toml", "--out", Path(t) / "rerun", "--json", "--quiet", check=0)
        assert (Path(t) / "rerun" / "summary.json").read_bytes() == (a / "summary.json").read_bytes()
        # never overwrite
        assert amlc("run", cfg, "--out", a, "--json", check=2).returncode == 2
        m = json.loads((a / "manifest.json").read_text())
        assert m["status"] == "completed" and m["seed"] == 20260917
        # queries are compact JSON
        for q in (["summary"], ["outputs"], ["series", "sd_field"], ["stats", "collapse_field"], ["at", "500"],
                  ["final"], ["scaling", "sd_field"], ["quantiles", "collapse_field", "--where", "N=250"],
                  ["crossing", "mean_jump", "0.5"]):
            r = amlc("query", a, *q, "--json", check=0)
            assert len(r.stdout) < 20000 and js(r)["ok"], q


def test_cli_matches_old_entry_point_rows():
    """The CLI's minimal run reproduces the bundled finiteN_lognormal.csv rows (old exp_finiteN.py)."""
    with tempfile.TemporaryDirectory() as t:
        amlc("run", EX / "minimal_finite_cohort.toml", "--out", Path(t) / "a", "--json", "--quiet", check=0)
        new = list(csv.DictReader(open(Path(t) / "a" / "outputs" / "by_size.csv")))
        ref = list(csv.DictReader(open(ROOT / "expected_results" / "finiteN_lognormal.csv")))[:2]
        for x, y in zip(new, ref):
            for k in ("mean_shift", "sd_field", "sd_takeoff", "mean_jump", "pred_shift", "pred_sd"):
                assert math.isclose(float(x[k]), float(y[k]), rel_tol=1e-6), (k, x[k], y[k])


def test_old_entry_points_reproduce_bundled_results():
    """Old script functions (now in amlc.experiments) still give the bundled control values."""
    from amlc.experiments import run_dominant_holder_controls
    with tempfile.TemporaryDirectory() as t:
        shutil.copy(ROOT / "expected_results" / "path_size.csv", t)
        run_dominant_holder_controls(results=t, log=lambda *a: None)
        for fn in ("dominant_holder_event_independent.csv", "path_model_meanfield_reference.csv"):
            new = list(csv.DictReader(open(Path(t) / fn)))
            ref = list(csv.DictReader(open(ROOT / "expected_results" / fn)))
            assert len(new) == len(ref)
            for x, y in zip(new, ref):
                for k, v in y.items():
                    try:
                        a, b = float(x[k]), float(v)
                    except ValueError:
                        assert x[k] == v; continue
                    assert (math.isnan(a) and math.isnan(b)) or abs(a - b) < 1e-12, (fn, k, x[k], v)
    r = subprocess.run([sys.executable, str(ROOT / "tests" / "smoke_test.py")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f(); print("PASS", n, flush=True)
    print("all %d agent-layer tests passed" % len(tests))

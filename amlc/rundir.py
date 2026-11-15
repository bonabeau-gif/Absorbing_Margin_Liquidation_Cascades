"""Self-describing run directories.

Layout written by ``amlc run``::

    <out>/
      config.resolved.toml   every parameter explicit, including defaults and the seed
      manifest.json          status, command, times, code version, environment, seed, error
      summary.json           headline results with units (deterministic)
      schema.json            every file and column in outputs/, with units (deterministic)
      outputs/*.csv          recorded data (deterministic)
      log.txt                progress messages (also echoed to standard error)

A run writes nothing outside ``<out>``. Only manifest.json and log.txt contain timestamps.
"""
import datetime as _dt
import json
import os
import platform
import secrets
import shlex
import shutil
import subprocess
import sys
import time

from . import __version__
from . import config as C
from . import registry as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RunDirError(Exception):
    pass


def _now():
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False, allow_nan=False)
        f.write("\n")


def code_version():
    """Git commit of the package checkout and whether its tree is dirty (read-only git calls)."""
    def git(*a):
        return subprocess.run(["git", "--no-optional-locks", "-C", ROOT, *a], capture_output=True, text=True, timeout=10)
    try:
        h = git("rev-parse", "HEAD")
        if h.returncode:
            return dict(git_commit=None, dirty=None, package_version=__version__)
        st = git("status", "--porcelain", "--untracked-files=no")
        return dict(git_commit=h.stdout.strip(), dirty=bool(st.stdout.strip()), package_version=__version__)
    except Exception:  # noqa: BLE001
        return dict(git_commit=None, dirty=None, package_version=__version__)


def environment():
    from importlib import metadata
    deps = {}
    for d in ("numpy", "scipy", "numba", "llvmlite", "pandas", "matplotlib"):
        try:
            deps[d] = metadata.version(d)
        except metadata.PackageNotFoundError:
            deps[d] = None
    return dict(python=platform.python_version(), implementation=platform.python_implementation(),
                platform=platform.platform(), machine=platform.machine(), dependencies=deps)


def prepare(out, force=False):
    """Create an empty run directory. Refuses a non-empty directory unless ``force`` and the
    directory is a previous run directory (contains manifest.json)."""
    out = os.path.abspath(out)
    if os.path.exists(out):
        if not os.path.isdir(out):
            raise RunDirError("%s exists and is not a directory" % out)
        if os.listdir(out):
            if not force:
                raise RunDirError("output directory %s is not empty; choose a new --out or pass --force to replace a previous run" % out)
            if not os.path.isfile(os.path.join(out, "manifest.json")):
                raise RunDirError("--force only replaces previous run directories (with manifest.json); %s is not one" % out)
            shutil.rmtree(out)
    os.makedirs(os.path.join(out, "outputs"))
    return out


class Logger:
    def __init__(self, path, quiet=False):
        self.f = open(path, "a", encoding="utf-8"); self.quiet = quiet

    def __call__(self, *args):
        msg = " ".join(str(a) for a in args)
        self.f.write("%s %s\n" % (_now(), msg)); self.f.flush()
        if not self.quiet:
            print(msg, file=sys.stderr, flush=True)

    def close(self):
        self.f.close()


def output_schema(model):
    spec = R.MODELS[model]
    return dict(model=model, index=spec["index"],
                files={("outputs/" + fn): dict(format="csv", description=f["description"], columns=f["columns"])
                       for fn, f in spec["files"].items()},
                summary=dict(file="summary.json", quantities=spec["summary"]),
                note="All quantities are dimensionless; 'unit' names the dimensionless quantity. Empty CSV cells are null.")


def execute(resolved, out, force=False, argv=None, quiet=False):
    """Run one resolved configuration into ``out``. Returns (out, manifest). Raises RunDirError
    for directory problems; model failures are recorded in the manifest and re-raised."""
    from . import runner
    out = prepare(out, force)
    seed_note = None
    if resolved.get("seed") is None:
        resolved = dict(resolved, seed=secrets.randbits(32))
        seed_note = "no seed given; chose seed %d (recorded in config.resolved.toml)" % resolved["seed"]
    with open(os.path.join(out, "config.resolved.toml"), "w", encoding="utf-8") as f:
        f.write(C.to_toml(resolved, header=["Fully resolved configuration written by amlc run; rerun with:",
                                            "  amlc run config.resolved.toml --out <new-dir>"]))
    t0 = time.time()
    manifest = dict(status="running", command=shlex.join(argv if argv is not None else sys.argv), cwd=os.getcwd(),
                    started=_now(), finished=None, duration_s=None, model=resolved["model"], seed=resolved["seed"],
                    code=code_version(), environment=environment(), error=None)
    mpath = os.path.join(out, "manifest.json")
    dump_json(mpath, manifest)
    log = Logger(os.path.join(out, "log.txt"), quiet=quiet)
    if seed_note:
        log(seed_note)
    log("model %s, seed %d -> %s" % (resolved["model"], resolved["seed"], out))
    try:
        summary = runner.execute(resolved["model"], resolved["parameters"], resolved["seed"], os.path.join(out, "outputs"), log)
        units = {q["name"]: q["unit"] for q in R.MODELS[resolved["model"]]["summary"]}
        dump_json(os.path.join(out, "summary.json"), dict(model=resolved["model"], seed=resolved["seed"], units=units, values=summary))
        dump_json(os.path.join(out, "schema.json"), output_schema(resolved["model"]))
        manifest.update(status="completed")
    except BaseException as e:
        manifest.update(status="failed", error="%s: %s" % (type(e).__name__, e))
        log("FAILED: %s" % manifest["error"])
        raise
    finally:
        manifest.update(finished=_now(), duration_s=round(time.time() - t0, 3))
        dump_json(mpath, manifest)
        log("status %s in %.2f s" % (manifest["status"], manifest["duration_s"]))
        log.close()
    return out, manifest


def read_run(run):
    """Load the JSON files of a finished run directory."""
    if not os.path.isfile(os.path.join(run, "manifest.json")):
        raise RunDirError("%s is not a run directory (no manifest.json)" % run)
    def j(n):
        p = os.path.join(run, n)
        return json.load(open(p, encoding="utf-8")) if os.path.isfile(p) else None
    return dict(manifest=j("manifest.json"), summary=j("summary.json"), schema=j("schema.json"),
                config=C.load(os.path.join(run, "config.resolved.toml")) if os.path.isfile(os.path.join(run, "config.resolved.toml")) else None)

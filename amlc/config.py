"""Declarative run configuration: load, validate, resolve, and serialize.

Configuration files are TOML (``.toml``, read with the standard-library ``tomllib``) or JSON
(``.json``). TOML is the default format because it allows comments and needs no third-party
parser. A configuration is::

    schema = 1
    model = "path"
    seed = 777
    [parameters]
    cohort.size = 8000          # dotted keys or nested tables are equivalent
    [parameters.ramp]
    steps = 4000

Nothing else influences a run: no environment variables, no working-directory contents.
"""
import difflib
import json
import os
import types

from . import registry as R

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    tomllib = None


class ConfigError(Exception):
    def __init__(self, errors):
        super().__init__("; ".join("%s: %s" % (e["field"], e["reason"]) for e in errors))
        self.errors = errors


def err(field, value, reason, hint=None):
    return dict(field=field, value=value, reason=reason, hint=hint)


# --------------------------------------------------------------------------- loading
def load(path):
    """Read a TOML or JSON configuration file into a dict; ``-`` reads standard input (JSON if it
    starts with ``{``, else TOML). Raises ConfigError."""
    if path == "-":
        import sys
        raw = sys.stdin.buffer.read()
        ext = ".json" if raw.lstrip().startswith(b"{") else ".toml"
    elif not os.path.isfile(path):
        raise ConfigError([err("<file>", path, "configuration file not found", "give the path to a .toml or .json file, or - for stdin")])
    else:
        with open(path, "rb") as f:
            raw = f.read()
        ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".json":
            return json.loads(raw.decode("utf-8"))
        if tomllib is None:
            raise ConfigError([err("<file>", path, "TOML needs Python >= 3.11", "use a .json configuration")])
        return tomllib.loads(raw.decode("utf-8"))
    except ConfigError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ConfigError([err("<file>", path, "cannot parse %s: %s" % ("JSON" if ext == ".json" else "TOML", e),
                               "check the file syntax; `amlc template --model <m>` prints a valid example")])


def flatten(d, prefix=""):
    """Nested mapping -> {dotted.name: value}. Lists are values, not nested."""
    out = {}
    for k, v in (d or {}).items():
        name = prefix + str(k)
        if isinstance(v, dict):
            out.update(flatten(v, name + "."))
        else:
            out[name] = v
    return out


def nest(flat):
    out = {}
    for name, v in flat.items():
        cur = out
        parts = name.split(".")
        for q in parts[:-1]:
            cur = cur.setdefault(q, {})
        cur[parts[-1]] = v
    return out


# --------------------------------------------------------------------------- checks
def _suggest(name, choices):
    m = difflib.get_close_matches(name, list(choices), n=1, cutoff=0.5)
    return m[0] if m else None


def _type_ok(spec_type, v):
    if spec_type == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if spec_type == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if spec_type == "boolean":
        return isinstance(v, bool)
    if spec_type in ("string", "enum", "path"):
        return isinstance(v, str)
    return True


def _range_reason(p, v):
    rg = p.get("range")
    if rg is None or not isinstance(v, (int, float)) or isinstance(v, bool):
        return None
    lo, hi = rg; xlo, xhi = p.get("range_exclusive") or (False, False)
    if v != v:
        return "is NaN"
    if lo is not None and (v < lo or (xlo and v == lo)):
        return "must be %s %s" % (">" if xlo else ">=", lo)
    if hi is not None and (v > hi or (xhi and v == hi)):
        return "must be %s %s" % ("<" if xhi else "<=", hi)
    return None


def _accepted(p):
    if p.get("options"):
        return "one of %s" % ", ".join(map(str, p["options"]))
    t = p["type"] if p["type"] != "list" else "list of %s" % p.get("item_type")
    rg = p.get("range")
    if rg:
        lo, hi = rg; xlo, xhi = p.get("range_exclusive") or (False, False)
        t += " in %s%s, %s%s" % ("(" if xlo else "[", "-inf" if lo is None else lo, "inf" if hi is None else hi, ")" if xhi else "]")
    return t


def check_value(p, v, field=None):
    """All problems with one value for parameter entry p (list of error dicts)."""
    field = field or p["name"]
    errors = []
    if p["type"] == "list":
        if not isinstance(v, list):
            return [err(field, v, "must be a list", "accepted: %s" % _accepted(p))]
        for i, item in enumerate(v):
            if not _type_ok(p.get("item_type") or "number", item):
                errors.append(err("%s[%d]" % (field, i), item, "must be %s" % p.get("item_type"), "accepted: %s" % _accepted(p)))
                continue
            r = _range_reason(p, item)
            if r:
                errors.append(err("%s[%d]" % (field, i), item, "out of range: %s" % r, "accepted: %s" % _accepted(p)))
        if not v:
            errors.append(err(field, v, "must not be empty", "accepted: %s" % _accepted(p)))
        return errors
    if not _type_ok(p["type"], v):
        hint = "accepted: %s" % _accepted(p)
        if p["type"] == "integer" and isinstance(v, float) and v.is_integer():
            hint += " (write %d without a decimal point)" % int(v)
        return [err(field, v, "must be %s" % ("an integer" if p["type"] == "integer" else "a " + p["type"]), hint)]
    if p.get("options") and v not in p["options"]:
        s = _suggest(str(v), p["options"])
        return [err(field, v, "not a valid option", "accepted: %s%s" % (_accepted(p), "; did you mean %r?" % s if s else ""))]
    r = _range_reason(p, v)
    if r:
        errors.append(err(field, v, "out of range: %s" % r, "accepted: %s" % _accepted(p)))
    return errors


def _ns(flat):
    """Namespace for constraint expressions: dotted names become attribute access."""
    root = {}
    for name, v in flat.items():
        cur = root
        parts = name.split(".")
        for q in parts[:-1]:
            nxt = cur.get(q)
            if not isinstance(nxt, dict):
                nxt = cur[q] = {}
            cur = nxt
        cur[parts[-1]] = v

    def conv(d):
        return types.SimpleNamespace(**{k: conv(v) if isinstance(v, dict) else v for k, v in d.items()})
    return {k: (conv(v) if isinstance(v, dict) else v) for k, v in root.items()}


def check_constraints(model, params, skip_fields=()):
    errors = []
    for c in R.MODELS[model]["constraints"]:
        if any(f in skip_fields for f in c["fields"]):
            continue
        try:
            ok = eval(c["expr"], {"__builtins__": {}, "len": len}, _ns(params))  # noqa: S307 (registry-defined expressions only)
        except Exception as e:  # noqa: BLE001
            ok = False; c = dict(c, message="%s (could not evaluate: %s)" % (c["message"], e))
        if not ok:
            errors.append(err(c["fields"][0], {f: params.get(f) for f in c["fields"]},
                              "constraint violated: %s" % c["expr"], c["message"]))
    return errors


# --------------------------------------------------------------------------- validation
def validate(cfg, allow_sweep=False):
    """Validate a loaded configuration.

    Returns (errors, warnings, resolved). ``resolved`` is None when there are errors; otherwise
    ``{"schema", "model", "seed", "parameters": {dotted: value}}`` with every default explicit
    (``seed`` is None when omitted) plus ``"sweep"`` when present and allowed.
    """
    errors, warnings = [], []
    if not isinstance(cfg, dict):
        return [err("<root>", type(cfg).__name__, "configuration must be a mapping", None)], [], None
    for k in cfg:
        if k not in R.TOP_LEVEL:
            s = _suggest(k, R.TOP_LEVEL)
            errors.append(err(k, cfg[k], "unknown top-level key",
                              ("did you mean %r? " % s if s else "") + "accepted: %s" % ", ".join(R.TOP_LEVEL)))
    if cfg.get("schema") != R.SCHEMA_VERSION:
        errors.append(err("schema", cfg.get("schema"), "missing or unsupported schema version", "set schema = %d" % R.SCHEMA_VERSION))
    model = cfg.get("model")
    if model not in R.MODELS:
        s = _suggest(str(model), R.MODELS)
        errors.append(err("model", model, "unknown or missing model",
                          ("did you mean %r? " % s if s else "") + "accepted: %s" % ", ".join(sorted(R.MODELS))))
        return errors, warnings, None
    seed = cfg.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        errors.append(err("seed", seed, "must be a non-negative integer", "e.g. seed = %s" % (R.MODELS[model]["reference_seed"] or 1)))
    if "sweep" in cfg and not allow_sweep:
        errors.append(err("sweep", cfg["sweep"], "a sweep block is only accepted by `amlc sweep`", "remove it, or use `amlc sweep`"))
    raw = cfg.get("parameters", {})
    if not isinstance(raw, dict):
        errors.append(err("parameters", raw, "must be a table/mapping", None)); raw = {}
    given = flatten(raw)
    used = set(R.MODELS[model]["parameters"])
    bad = set()
    for name, v in given.items():
        field = "parameters." + name
        if name not in R.PARAM:
            s = _suggest(name, used) or _suggest(name, R.PARAM)
            errors.append(err(field, v, "unknown parameter",
                              ("did you mean %r? " % s if s else "") + "see `amlc describe --model %s`" % model))
            bad.add(name); continue
        if name not in used:
            errors.append(err(field, v, "not used by model %r" % model,
                              "used by: %s; remove it" % (", ".join(R.applies_to(name)) or "no model")))
            bad.add(name); continue
        e = check_value(R.PARAM[name], v, field)
        if e:
            errors += e; bad.add(name)
    resolved = {p["name"]: given.get(p["name"], p["default"]) for p in R.model_params(model)}
    for name, v in resolved.items():
        if v == R.REQUIRED:
            errors.append(err("parameters." + name, None, "required parameter missing", "accepted: %s" % _accepted(R.PARAM[name])))
            bad.add(name)
    for name in given:
        cond = (R.PARAM.get(name) or {}).get("applies_when")
        if cond and name in used and model != "path" and given[name] != R.model_params_map(model)[name]["default"]:
            for k, want in cond.items():
                if resolved.get(k) != want:
                    warnings.append(err("parameters." + name, given[name], "has no effect unless %s = %r" % (k, want), None))
    errors += [dict(e, field=("parameters." + e["field"]) if not e["field"].startswith("parameters.") else e["field"])
               for e in check_constraints(model, resolved, skip_fields=bad)]
    sweep = None
    if allow_sweep and "sweep" in cfg:
        sweep = cfg["sweep"]
        if not isinstance(sweep, dict) or not sweep:
            errors.append(err("sweep", sweep, "must be a non-empty table {parameter = [values]}", None)); sweep = None
        else:
            sweep = flatten(sweep)
            for name, vals in sweep.items():
                field = "sweep." + name
                if name not in used:
                    s = _suggest(name, used)
                    errors.append(err(field, vals, "not a parameter of model %r" % model, ("did you mean %r?" % s) if s else None)); continue
                if not isinstance(vals, list) or not vals:
                    errors.append(err(field, vals, "must be a non-empty list of values", None)); continue
                for i, v in enumerate(vals):
                    errors += check_value(R.PARAM[name], v, "%s[%d]" % (field, i))
                    if not errors:
                        pt = dict(resolved, **{name: v})
                        errors += [dict(e, field="%s[%d] -> %s" % (field, i, e["field"])) for e in check_constraints(model, pt)]
    if errors:
        return errors, warnings, None
    out = dict(schema=R.SCHEMA_VERSION, model=model, seed=seed, parameters=resolved)
    if sweep:
        out["sweep"] = sweep
    return errors, warnings, out


# --------------------------------------------------------------------------- TOML output
def _toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v:
            return "nan"
        if v in (float("inf"), float("-inf")):
            return "inf" if v > 0 else "-inf"
        r = repr(v)
        return r if any(c in r for c in ".en") else r + ".0"
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError("cannot write %r as TOML" % (v,))


def _comment(p):
    bits = [p["description"]]
    if p.get("unit"):
        bits.append("unit: %s" % p["unit"])
    bits.append("accepted: %s" % _accepted(p) if (p.get("range") or p.get("options")) else "range: not stated in the source")
    if p.get("applies_when"):
        bits.append("only used when %s" % ", ".join("%s = %r" % kv for kv in p["applies_when"].items()))
    return bits


def to_toml(resolved, comments=False, header=None):
    """Serialize a resolved configuration. With ``comments``, document every parameter."""
    lines = []
    if header:
        lines += ["# " + h for h in header] + [""]
    lines.append("schema = %d" % resolved["schema"])
    lines.append("model = %s" % json.dumps(resolved["model"]))
    if resolved.get("seed") is not None:
        lines.append("seed = %d" % resolved["seed"])
    lines += ["", "[parameters]"]
    for name, v in resolved["parameters"].items():
        if comments:
            p = R.PARAM[name]
            for b in _comment(p):
                lines.append("# " + b)
        lines.append("%s = %s" % (name, _toml_value(v)))
        if comments:
            lines.append("")
    if resolved.get("sweep"):
        lines += ["", "[sweep]"]
        for name, vals in resolved["sweep"].items():
            lines.append("%s = %s" % (name, _toml_value(vals)))
    return "\n".join(lines).rstrip() + "\n"


def resolved_json(resolved):
    return dict(schema=resolved["schema"], model=resolved["model"], seed=resolved.get("seed"),
                parameters=nest(resolved["parameters"]), **({"sweep": resolved["sweep"]} if resolved.get("sweep") else {}))


# --------------------------------------------------------------------------- JSON Schema
def _param_schema(p):
    t = {"number": "number", "integer": "integer", "boolean": "boolean", "string": "string", "enum": "string", "path": "string"}
    if p["type"] == "list":
        item = {"type": t.get(p.get("item_type") or "number", "number")}
        item.update(_bounds(p))
        s = {"type": "array", "items": item, "minItems": 1}
    else:
        s = {"type": t[p["type"]]}
        if p.get("options"):
            s["enum"] = p["options"]
        s.update(_bounds(p))
    s["description"] = p["description"] + (" Unit: %s." % p["unit"] if p.get("unit") else "")
    if p["default"] != R.REQUIRED:
        s["default"] = p["default"]
    return s


def _bounds(p):
    out = {}
    if p.get("range"):
        lo, hi = p["range"]; xlo, xhi = p["range_exclusive"]
        if lo is not None:
            out["exclusiveMinimum" if xlo else "minimum"] = lo
        if hi is not None:
            out["exclusiveMaximum" if xhi else "maximum"] = hi
    return out


def json_schema():
    """JSON Schema (draft 2020-12) for configurations, generated from the registry. Parameters
    are given with dotted keys as a flat object (nested tables are also accepted by validate)."""
    variants = []
    for model, spec in R.MODELS.items():
        props = {}
        for p in R.model_params(model):
            props[p["name"]] = _param_schema(p)
        variants.append({
            "if": {"properties": {"model": {"const": model}}},
            "then": {"properties": {"parameters": {"type": "object", "properties": props, "additionalProperties": False}}},
        })
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "amlc-config-schema-v%d" % R.SCHEMA_VERSION,
        "title": "amlc run configuration",
        "description": "Generated by `amlc describe --schema` from amlc/registry.py. Parameters use dotted keys "
                       "(e.g. \"ramp.steps\"); in TOML these may also be written as nested tables.",
        "type": "object",
        "required": ["schema", "model"],
        "additionalProperties": False,
        "properties": {
            "schema": {"const": R.SCHEMA_VERSION, "description": R.TOP_LEVEL["schema"]["description"]},
            "model": {"enum": sorted(R.MODELS), "description": R.TOP_LEVEL["model"]["description"]},
            "seed": {"type": "integer", "minimum": 0, "description": R.TOP_LEVEL["seed"]["description"]},
            "parameters": {"type": "object", "description": R.TOP_LEVEL["parameters"]["description"]},
            "sweep": {"type": "object", "additionalProperties": {"type": "array", "minItems": 1},
                      "description": R.TOP_LEVEL["sweep"]["description"]},
        },
        "allOf": variants,
    }

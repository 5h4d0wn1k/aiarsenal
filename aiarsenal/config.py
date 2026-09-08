"""Config loading: JSON config fully supported; YAML accepted if PyYAML is present."""


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def load_config(path):
    """Load a .json or .yaml/.yml config into a dict.

    JSON is supported natively. YAML support requires the optional
    `PyYAML` package; if unavailable, raises a clear error.
    """
    if not path:
        return {}
    ext = str(path).lower().rsplit(".", 1)[-1]
    if ext == "json":
        import json

        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    elif ext in ("yaml", "yml"):
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "YAML config requires the optional 'PyYAML' package. "
                "Use a JSON config instead, or `pip install PyYAML`."
            )
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    else:
        raise ValueError(f"unsupported config extension: {ext}")


def merge_config(base, overrides):
    """Deep-merge two dicts (overrides win)."""
    if not isinstance(base, dict) or not isinstance(overrides, dict):
        return overrides if overrides is not None else base
    out = dict(base)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_config(out[k], v)
        else:
            out[k] = v
    return out

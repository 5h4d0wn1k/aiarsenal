"""Reporting: write JSON + Markdown reports under reports/; exit codes."""

import datetime
import json
import os

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REFUSED = 2

REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
)


def ensure_reports_dir(subdir=None):
    target = REPORTS_DIR if not subdir else os.path.join(REPORTS_DIR, subdir)
    os.makedirs(target, exist_ok=True)
    return target


def _now():
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def json_dumps(obj):
    return json.dumps(obj, indent=2, default=lambda o: str(o))


def write_report(tool, data, markdown_text):
    """Write JSON + Markdown reports. Returns list of written file paths."""
    target = ensure_reports_dir()
    stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = f"{tool}_{stamp}"
    paths = []
    meta = dict(data)
    meta.setdefault("_meta", {})
    meta["_meta"]["tool"] = tool
    meta["_meta"]["generated"] = _now()

    js = os.path.join(target, f"{base}.json")
    with open(js, "w", encoding="utf-8") as fh:
        fh.write(json_dumps(meta))
    paths.append(js)

    md = os.path.join(target, f"{base}.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write(markdown_text)
    paths.append(md)
    return paths


def md_table(headers, rows):
    head = "| " + " | ".join(str(h) for h in headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return "\n".join([head, sep, body])

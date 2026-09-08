"""Agentic red-team campaign planner.

Chains the attack modules against a declared LAB scope with approval gates,
a JSONL audit trail, and --dry-run as the default mode. Refuses unapproved
scopes. This is an orchestration reference, not a real remote penetrator.
"""

import json
import os
import uuid

from . import report
from . import poison, backdoor, extract, invert, fl, evade, inject

ALLOWED_MODULES = ("poison", "backdoor", "extract", "invert", "fl", "evade", "inject")

APPROVED_SCOPE_MARKERS = (
    ("admin", "approved"),
    ("lab", "lab"),
    ("192.0.2.", "documentation network"),
    ("test", "test"),
    ("example.com", "example.com"),
)


def _scope_approved(scope):
    s = scope.lower().strip()
    if not s:
        return False, "empty scope"
    for marker, reason in APPROVED_SCOPE_MARKERS:
        if marker in s:
            return True, reason
    return False, "scope not on the approved LAB allowlist"


def _run_module(name, modname):
    func = getattr(modname, "demo")
    res = func()
    return {k: v for k, v in res.items() if k != "config"}


MODULE_DEMOS = {
    "poison": poison.demo,
    "backdoor": backdoor.demo,
    "extract": extract.demo,
    "invert": invert.demo,
    "fl": fl.demo,
    "evade": evade.demo,
    "inject": inject.demo,
}


def run_campaign(cfg):
    scope = cfg.get("scope", "")
    modules = cfg.get("modules", list(ALLOWED_MODULES))
    dry_run = bool(cfg.get("dry_run", True))
    approved_flag = bool(cfg.get("approved", False))
    audit_path = cfg.get("audit_path") or "reports/campaign_audit.jsonl"

    ok_scope, why = _scope_approved(scope)
    if not ok_scope:
        raise CampaignRefused(scope, why)

    if not approved_flag:
        raise CampaignRefused(
            scope,
            "approval gate: campaign requires --approved and an approved scope",
        )

    entries = []
    results = {}
    for m in modules:
        if m not in ALLOWED_MODULES:
            raise CampaignRefused(scope, f"module '{m}' not in allowed set")
    unknown = [m for m in modules if m not in MODULE_DEMOS]
    if unknown:
        raise CampaignRefused(scope, f"unknown module(s): {unknown}")

    entry = {
        "id": str(uuid.uuid4())[:8],
        "type": "campaign-plan",
        "scope": scope,
        "scope_approval": why,
        "approved": approved_flag,
        "modules": modules,
        "dry_run": dry_run,
    }
    entries.append(entry)

    for m in modules:
        if dry_run:
            step = {"type": "step", "module": m, "mode": "dry-run",
                    "status": "planned-not-run"}
        else:
            try:
                step_res = MODULE_DEMOS[m]()
                step_res.pop("config", None)
                step = {"type": "step", "module": m, "mode": "wet",
                        "status": "ok", "result": step_res}
                results[m] = step_res
            except Exception as exc:  # pragma: no cover
                step = {"type": "step", "module": m, "mode": "wet",
                        "status": "error", "error": str(exc)}
                entries.append(step)
                raise CampaignError(
                    scope, m, f"module execution failed: {exc}"
                )
        entries.append(step)

    os.makedirs(os.path.dirname(audit_path) or ".", exist_ok=True)
    with open(audit_path, "a", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")

    return {
        "campaign_id": entry["id"],
        "scope": scope,
        "scope_approval": why,
        "approved": approved_flag,
        "dry_run": dry_run,
        "modules": modules,
        "audit_trail": audit_path,
        "steps": entries,
        "module_results": results,
    }


class CampaignError(Exception):
    def __init__(self, scope, module, message):
        super().__init__(message)
        self.scope = scope
        self.module = module
        self.exit_code = report.EXIT_ERROR


class CampaignRefused(CampaignError):
    def __init__(self, scope, message):
        super().__init__(scope, "scope", message)
        self.exit_code = report.EXIT_REFUSED


def cmd(args):
    scope = args.scope or "lab:test"
    modules = args.modules.split(",") if args.modules else list(ALLOWED_MODULES)
    cfg = {"scope": scope, "modules": modules,
           "dry_run": not args.exec or bool(args.dry_run),
           "approved": bool(args.approved)}
    if args.config:
        from .config import load_config
        loaded = load_config(args.config) or {}
        cfg = {**loaded, **cfg}

    try:
        res = run_campaign(cfg)
    except CampaignRefused as exc:
        print(json.dumps({
            "status": "refused",
            "scope": exc.scope,
            "reason": str(exc),
        }, indent=2))
        return exc.exit_code

    print(json.dumps({
        "status": "ok",
        "campaign_id": res["campaign_id"],
        "scope": res["scope"],
        "approved": res["approved"],
        "dry_run": res["dry_run"],
        "modules": res["modules"],
        "audit_trail": res["audit_trail"],
    }, indent=2))
    return report.EXIT_OK
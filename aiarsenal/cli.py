"""aiarsenal command-line interface.

Usage examples:
    python3 -m aiarsenal --help
    python3 -m aiarsenal --demo
    python3 -m aiarsenal poison --config config/poison.json
    python3 -m aiarsenal campaign --scope lab:test-c --approved
"""

import argparse
import json
import sys

from . import __version__
from . import (backdoor, campaign, evade, extract, fl, inject, invert,
               mcp, poison, report)


def _add_common(sub, tool):
    sub.add_argument("--config", "-c", default=None,
                     help=f"JSON/YAML config file for {tool} (default: built-in demo params)")
    sub.add_argument("--no-report", action="store_true",
                     help="do not write JSON/Markdown report files")
    return sub


def build_parser():
    parser = argparse.ArgumentParser(
        prog="aiarsenal",
        description="Adversarial AI/ML security studio (authorised lab use only). "
                    "See README 'IMPORTANT: Read before use.' before running anything.",
        epilog="Exit codes: 0 OK, 1 error, 2 refused (scope/approval).",
    )
    parser.add_argument("--version", action="version",
                        version=f"aiarsenal {__version__}")
    parser.add_argument("--demo", action="store_true",
                        help="run the full offline demo suite and exit 0")
    parser.add_argument("--report-dir", default=report.REPORTS_DIR,
                        help="override the reports output directory")

    sub = parser.add_subparsers(dest="tool")

    p = sub.add_parser("poison", help="data poisoning on a synthetic dataset")
    _add_common(p, "data poisoning")

    p = sub.add_parser("backdoor", help="backdoor a trained weight set")
    _add_common(p, "model backdoor")

    p = sub.add_parser("extract", help="query-based model extraction")
    _add_common(p, "model extraction")

    p = sub.add_parser("invert", help="membership inference from confidences")
    _add_common(p, "membership inference")

    p = sub.add_parser("fl", help="federated learning weight/gradient attack")
    _add_common(p, "federated learning attack")

    p = sub.add_parser("evade", help="FGSM adversarial evasion sweep")
    _add_common(p, "evasion")

    p = sub.add_parser("inject", help="prompt injection vs vectorControlledLLM guardrail")
    _add_common(p, "prompt injection")

    p = sub.add_parser("campaign", help="agentic red-team campaign planner (dry-run default)")
    p.add_argument("--scope", default=None,
                   help="declared LAB scope, must match approve allowlist (e.g. lab:test-x)")
    p.add_argument("--modules", default=None,
                   help="comma-separated module list (default: all)")
    p.add_argument("--approved", action="store_true",
                   help="approval gate: must be present to run a wet campaign")
    p.add_argument("--exec", action="store_true",
                   help="actually run the planned steps (default is dry-run)")
    p.add_argument("--dry-run", action="store_true",
                   help="explicitly stay in dry-run (default behaviour)")
    p.add_argument("--config", "-c", default=None,
                   help="JSON/YAML config with scope/modules/approved")
    p.add_argument("--no-report", action="store_true",
                   help="skip writing report files")

    p = sub.add_parser("mcp", help="MCP tool-abuse probe against a mock server (local)")
    _add_common(p, "MCP hunting")

    return parser


def _load_cli_config(args):
    if not getattr(args, "config", None):
        return {}
    from .config import load_config
    return load_config(args.config) or {}


def _handle(args):
    if args.report_dir:
        report.REPORTS_DIR = args.report_dir
    if args.tool == "poison":
        return poison.cmd(args)
    if args.tool == "backdoor":
        return backdoor.cmd(args)
    if args.tool == "extract":
        return extract.cmd(args)
    if args.tool == "invert":
        return invert.cmd(args)
    if args.tool == "fl":
        return fl.cmd(args)
    if args.tool == "evade":
        return evade.cmd(args)
    if args.tool == "inject":
        return inject.cmd(args)
    if args.tool == "campaign":
        return campaign.cmd(args)
    if args.tool == "mcp":
        return mcp.cmd(args)
    raise SystemExit(f"unknown tool: {args.tool}")


def _run_demo():
    print("aiarsenal offline demo (authorised-lab fixture only)")
    summary = []
    for name, func in (
        ("poison", poison.demo),
        ("backdoor", backdoor.demo),
        ("extract", extract.demo),
        ("invert", invert.demo),
        ("fl", fl.demo),
        ("evade", evade.demo),
        ("inject", inject.demo),
        ("mcp", mcp.demo),
    ):
        res = func()
        res.pop("config", None)
        summary.append((name, res))
        print(f"  [{name}] ok")
    for name, res in summary:
        if name == "poison":
            print(f"\nProof — {name}: baseline_clean_accuracy={res['baseline_clean_accuracy']} "
                  f"poisoned_clean_accuracy={res['poisoned_clean_accuracy']} "
                  f"accuracy_collapse={res['accuracy_collapse']} "
                  f"backdoor_trigger_activation={res['backdoor_trigger_activation']}")
        elif name == "backdoor":
            print(f"Proof — {name}: clean_accuracy_after_backdoor={res['clean_accuracy_after_backdoor']} "
                  f"backdoor_trigger_activation={res['backdoor_trigger_activation']}")
        elif name == "extract":
            print(f"Proof — {name}: student_fidelity_to_teacher={res['student_fidelity_to_teacher']}")
        elif name == "invert":
            print(f"Proof — {name}: membership_auc={res['membership_auc']}")
        elif name == "fl":
            print(f"Proof — {name}: clean_accuracy_after_attack={res['clean_accuracy_after_attack']} "
                  f"backdoor_activation_victims={res['backdoor_activation_victims']}")
        elif name == "evade":
            print(f"Proof — {name}: clean={res['clean_accuracy']} "
                  f"robust@0.5={res['robust_accuracy_by_epsilon'].get('eps_0.5')} "
                  f"robust@1.0={res['robust_accuracy_by_epsilon'].get('eps_1.0')}")
        elif name == "inject":
            print(f"Proof — {name}: attack_success_rate={res['attack_success_rate']} "
                  f"guardrail_efficacy={res['guardrail_efficacy']}")
        elif name == "mcp":
            print(f"Proof — {name}: tools_exposed={res['tools_exposed']} "
                  f"dangerous={res['dangerous_tools_exposed']} abusable={res['abuse_capable']}")
    print("\nDemo exited OK (offline, no external targets touched).")
    return report.EXIT_OK


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.demo:
        return _run_demo()
    if not args.tool:
        parser.print_help()
        return report.EXIT_OK
    try:
        return _handle(args)
    except campaign.CampaignRefused as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return exc.exit_code
    except campaign.CampaignError as exc:
        print(f"Campaign error: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:
        print(f"Error in {args.tool}: {exc}", file=sys.stderr)
        return report.EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
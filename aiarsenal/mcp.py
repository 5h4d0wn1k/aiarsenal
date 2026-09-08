"""MCP (Model Context Protocol) tool-abuse probe against a mock server.

Simulates a local agentic loop talking to a mock MCP server that exposes tools
over HTTP JSON. The probe enumerates exposed tools, fingerprints which ones are
dangerously broad (e.g. `shell_exec`, `read_file`, `network_scan`), and
demonstrates that a crafted tool-call (simulated) can abuse an overly-permissive
tool. All traffic is to a localhost fixture.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from . import report

MOCK_MCP_TOOLS = {
    "search": {"allowed": True, "dangerous": False, "desc": "search internal docs"},
    "read_file": {"allowed": True, "dangerous": True,
                  "desc": "read arbitrary server file"},
    "shell_exec": {"allowed": True, "dangerous": True,
                   "desc": "run shell commands on server"},
    "list_files": {"allowed": True, "dangerous": False,
                   "desc": "list working directory"},
    "send_email": {"allowed": True, "dangerous": True,
                   "desc": "send email via SMTP relay"},
}


class _MockMcpHandler(BaseHTTPRequestHandler):
    tools = MOCK_MCP_TOOLS

    def _reply(self, code, body):
        payload = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/mcp/tools":
            self._reply(200, [{"name": k, **v} for k, v in self.tools.items()])
        else:
            self._reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/mcp/tools/call":
            length = int(self.headers.get("Content-Length", "0"))
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                req = {}
            tool = req.get("tool")
            if tool not in self.tools:
                self._reply(400, {"error": f"unknown tool {tool}"})
                return
            self._reply(200, {"ok": True, "tool": tool,
                              "captured": req.get("arguments", {})})
        else:
            self._reply(404, {"error": "not found"})

    def log_message(self, *args):  # silence
        pass


class MockMcpServer:
    """Headless mock MCP server bound to a random localhost port."""

    def __init__(self, tools=None):
        self.tools = dict(tools or MOCK_MCP_TOOLS)
        self.httpd = HTTPServer(("127.0.0.1", 0), _MockMcpHandler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return self.port

    @property
    def port(self):
        return self.httpd.server_address[1]

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _probe(base, tools):
    import urllib.request

    rows = []
    for name, meta in tools.items():
        payload = json.dumps({"tool": name,
                              "arguments": {"path": "/etc/hosts", "cmd": "id",
                                            "to": "victim@example.com"}}).encode()
        req = urllib.request.Request(
            f"{base}/mcp/tools/call", data=payload,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read())
        rows.append({"tool": name, "dangerous": meta["dangerous"],
                     "callable": bool(body.get("ok")), "desc": meta["desc"]})
    return rows


def run_mcp(cfg):
    tools = cfg.get("tools", MOCK_MCP_TOOLS)
    srv = MockMcpServer(tools)
    srv.start()
    base = f"http://127.0.0.1:{srv.port}"
    try:
        rows = _probe(base, tools)
    finally:
        srv.stop()

    exploitable = [r for r in rows if r["dangerous"] and r["callable"]]
    result = {
        "config": cfg,
        "server": f"http://127.0.0.1:{srv.port}",
        "tools_exposed": len(rows),
        "dangerous_tools_exposed": len(exploitable),
        "abuse_capable": bool(exploitable),
        "probe_results": rows,
        "exposed_tools": [r["tool"] for r in exploitable],
    }
    return result


def demo():
    return run_mcp({})


def cmd(args):
    from .config import load_config
    cfg = load_config(args.config) if args.config else {}
    res = run_mcp(cfg)
    lines = "\n".join(
        f"| {r['tool']} | {r['desc']} | {'dangerous' if r['dangerous'] else 'safe'} | {r['callable']} |"
        for r in res["probe_results"]
    )
    report.write_report(
        "mcp", res,
        "# MCP Tool-Abuse Probe Report\n\n"
        + "| Metric | Value |\n|---|---|\n"
        + f"| Tools exposed | {res['tools_exposed']} |\n"
        + f"| Dangerous tools exposed | {res['dangerous_tools_exposed']} |\n\n"
        + "| Tool | Description | Risk | Callable |\n|---|---|---|---|\n" + lines,
    )
    print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=2))
    return report.EXIT_OK
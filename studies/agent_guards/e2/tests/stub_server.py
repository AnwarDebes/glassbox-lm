"""An in-process HTTP STUB of llama-server's endpoints for the offline smoke test (no model, no GPU): it lets the GPU
driver and e2_check.py run their real-server code paths end to end. It is not an LLM: agent and tool-filter requests
get a fixed text answer (no tool call), the A3 optimizer gets two fixed candidates, the D6 rule writer one fixed rule.
/props reports build_info "stub", so e2_check.py must refuse it. Binds 127.0.0.1 on an ephemeral port and writes the
port to --port-file; it exits after --seconds (default 900) or when killed.

    python -I -B tests/stub_server.py --port-file <file> [--seconds 900]
        [--fail-after N --fail-status CODE]   # rev3: after N answered POSTs, every POST gets HTTP CODE (e.g. 500, 400)
        [--refuse-after N]                    # rev3: after N answered POSTs, drop the connection and stop listening,
                                              #       so every later connection is refused
The failure modes test the transport rule (registration section 13): 5xx and refused connections stop a stage,
4xx falls under the error rule.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class H(BaseHTTPRequestHandler):
    n = 0
    fail_after = -1
    fail_status = 500
    refuse_after = -1
    refused = threading.Event()

    def log_message(self, *a):  # quiet
        pass

    def _send(self, obj, code: int = 200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send({"status": "ok"})
        elif self.path == "/v1/models":
            self._send({"object": "list", "data": [{"id": "qwen3.5-9b-q8_0"}]})
        elif self.path == "/props":
            self._send({"build_info": "stub", "total_slots": 1, "model_path": "/nonexistent/stub.gguf",
                        "model_alias": "qwen3.5-9b-q8_0", "chat_template": "{{ stub }}",
                        "default_generation_settings": {"n_ctx": 65536, "params": {"temperature": 0.8}}})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if H.refuse_after >= 0 and H.n >= H.refuse_after:
            self.close_connection = True  # no answer: the client sees the connection dropped
            H.refused.set()  # the main thread stops listening: every later connection is refused
            return
        if H.fail_after >= 0 and H.n >= H.fail_after:
            self._send({"error": {"code": H.fail_status, "message": f"stub error {H.fail_status}",
                                  "type": "server_error" if H.fail_status >= 500 else "invalid_request_error"}},
                       H.fail_status)
            return
        H.n += 1
        rf = (body.get("response_format") or {}).get("json_schema", {}).get("name")
        if rf == "candidates":
            content = json.dumps({"candidates": ["Stub candidate one. Please do the task.", "Stub candidate two."]})
        elif rf == "rules":
            content = json.dumps({"rules": [["kind=read", "!prov[url]=other_tool_output", "!prov[url]=not_found"]]})
        else:
            content = "Done (stub answer, no tool call)."
        self._send({"id": f"stub-{H.n}", "object": "chat.completion", "created": 0, "model": body.get("model", "x"),
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port-file", required=True)
    ap.add_argument("--seconds", type=float, default=900.0)
    ap.add_argument("--fail-after", type=int, default=-1)
    ap.add_argument("--fail-status", type=int, default=500)
    ap.add_argument("--refuse-after", type=int, default=-1)
    a = ap.parse_args()
    H.fail_after, H.fail_status, H.refuse_after = a.fail_after, a.fail_status, a.refuse_after
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    Path(a.port_file).write_text(str(srv.server_address[1]))
    t0 = time.time()
    try:
        while time.time() - t0 < a.seconds and not H.refused.is_set():
            time.sleep(0.2)
    finally:
        srv.shutdown()
        srv.server_close()  # the listening socket is closed: connections to the port are refused
    if H.refused.is_set():  # stay alive (not listening) until --seconds, so the caller can kill it as usual
        while time.time() - t0 < a.seconds:
            time.sleep(0.5)


if __name__ == "__main__":
    main()

"""Does the *-nonet wrapper really cut the browser off the network? Serve one probe page from THIS process on
the host's 127.0.0.1 (random free port; no outside host is contacted) and try to load it from the browser:
the plain wrapper should load it, the -nonet wrapper should fail. Run with python -I."""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
import functools
import http.server
import json
import os
import sys
import threading

from playwright.sync_api import sync_playwright

F = f"{_REPO_ROOT}/studies/agent_guards/miniwob"
root = os.path.join(F, "tmp", "netns_probe")
os.makedirs(root, exist_ok=True)
with open(os.path.join(root, "probe.html"), "w") as f:
    f.write("<!doctype html><title>probe</title><p id=x>reached the host loopback</p>")

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=root)
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
port = srv.server_address[1]
th = threading.Thread(target=srv.serve_forever, daemon=True)
th.start()
url = f"http://127.0.0.1:{port}/probe.html"
res = {"url": url}
try:
    with sync_playwright() as p:
        for exe in sys.argv[1:]:
            b = p.chromium.launch(executable_path=os.path.join(F, "browser", exe), headless=True,
                                  chromium_sandbox=True, timeout=60000)
            page = b.new_page()
            try:
                page.goto(url, timeout=15000)
                res[exe] = {"loaded": True, "text": page.inner_text("#x")}
            except Exception as e:
                res[exe] = {"loaded": False, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}
            b.close()
finally:
    srv.shutdown()
    srv.server_close()
print(json.dumps(res, indent=1))

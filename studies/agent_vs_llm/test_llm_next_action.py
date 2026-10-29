"""Checks of llm_next_action.py with a stub server (no GPU, no LLM).

usage: studies/decision_layer/.venv-dl/bin/python -I test_llm_next_action.py
"""
import gzip
import http.server
import json
import os
import shutil
import sys
import tempfile
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_next_action as L  # noqa: E402

fails = 0


def check(name, cond):
    global fails
    print(("ok   " if cond else "FAIL ") + name)
    fails += 0 if cond else 1


B = ["pull-up-account", "verify-identity", "search-faq", "make-password"]
check("exact name", L.map_answer("verify-identity", B) == "verify-identity")
check("quotes and a period", L.map_answer('"search-faq".', B) == "search-faq")
check("name inside a sentence", L.map_answer("I would use make-password here", B) == "make-password")
check("normalised spelling", L.map_answer("Pull Up Account", B) == "pull-up-account")
check("close spelling", L.map_answer("verify-identiy", B) == "verify-identity")
check("nothing usable is invalid", L.map_answer("I cannot tell", B) is None)
check("empty answer is invalid", L.map_answer("", B) is None)
try:
    L.rows_of("abcd", "test")
    check("the test split is refused", False)
except ValueError:
    check("the test split is refused", True)

seen = []
mode = {"fail_left": 0}  # the stub answers HTTP 500 while fail_left > 0 (a negative value: always)


class Stub(http.server.BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if mode["fail_left"] != 0:
            mode["fail_left"] -= 1 if mode["fail_left"] > 0 else 0
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        seen.append(body)
        out = {"choices": [{"message": {"content": "pull-up-account"}}], "usage": {"prompt_tokens": 123}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


srv = http.server.HTTPServer(("127.0.0.1", 0), Stub)
threading.Thread(target=srv.serve_forever, daemon=True).start()
tmp = tempfile.mkdtemp()
try:
    L.main(["--dataset", "abcd", "--n", "5", "--k", "3", "--url", f"http://127.0.0.1:{srv.server_port}",
            "--out", tmp])
    recs = [json.loads(x) for x in open(os.path.join(tmp, "preds.jsonl"))]
    res = json.load(open(os.path.join(tmp, "result.json")))
    check("five predictions written", len(recs) == 5 and all(r["pred"] == "pull-up-account" for r in recs))
    check("result has the accuracy", res["n"] == 5 and 0.0 <= res["accuracy"] <= 1.0 and res["guidelines"] is True)
    check("guidelines are in the system prompt", "written guidelines" in seen[0]["messages"][0]["content"])
    check("three demos and the question in a user-role prompt", seen[0]["messages"][1]["content"].count("Next action:") == 4)
    check("temperature 0", seen[0]["temperature"] == 0.0)
    L.main(["--dataset", "abcd", "--n", "5", "--k", "3", "--url", f"http://127.0.0.1:{srv.server_port}",
            "--out", tmp])
    check("a restart keeps finished examples (no new requests)", len(seen) == 5)
    seen_before_think = dict(seen[0])
    seen.clear()
    tmp2 = tempfile.mkdtemp()
    L.main(["--dataset", "star", "--n", "2", "--k", "2", "--url", f"http://127.0.0.1:{srv.server_port}",
            "--out", tmp2])
    sysm = seen[0]["messages"][0]["content"]
    check("STAR prompt holds the task specifications", "task specifications" in sysm and "party_rsvp" in sysm
          and "party_ask_venue" in sysm)
    shutil.rmtree(tmp2)
    # shortlist: the stub answers pull-up-account; candidates that hold it keep it, others fall back to the top 1
    import numpy as np
    with gzip.open(os.path.join(L.F.DATA, "abcd", "examples_dev.json.gz"), "rt") as h:
        dd = json.load(h)
    gold = np.array([r["y"] for r in dd["rows"]], dtype=np.int16)
    pua = dd["buttons"].index("pull-up-account")
    top = np.stack([(gold + j) % 30 for j in range(5)], 1).astype(np.int16)
    tmp3 = tempfile.mkdtemp()
    np.savez(os.path.join(tmp3, "tm.npz"), top_ids=top, top_scores=np.ones(top.shape, np.float32), gold=gold)
    with open(os.path.join(tmp3, "q1.jsonl"), "w") as h:
        h.write(json.dumps({"i": 0}) + "\n")
    seen.clear()
    L.main(["--dataset", "abcd", "--n", "6", "--k", "2", "--url", f"http://127.0.0.1:{srv.server_port}",
            "--out", os.path.join(tmp3, "o"), "--shortlist", os.path.join(tmp3, "tm.npz"), "--k-short", "3",
            "--exclude", os.path.join(tmp3, "q1.jsonl")])
    rr = [json.loads(x) for x in open(os.path.join(tmp3, "o", "preds.jsonl"))]
    check("excluded row is not drawn", all(x["i"] != 0 for x in rr))
    check("candidates are the agent's top 3", all(x["candidates"] == [dd["buttons"][int(c)] for c in top[x["i"], :3]]
                                                    for x in rr))
    ok_map = all((x["pred"] == "pull-up-account") if "pull-up-account" in x["candidates"]
                 else (x["fallback"] and x["pred"] == x["candidates"][0]) for x in rr)
    check("answers map to a candidate or fall back to the top 1", ok_map)
    check("the prompt shows the candidates", "Choose exactly one of these candidates" in seen[0]["messages"][1]["content"])
    shutil.rmtree(tmp3)
    seen.clear()
    tmp4 = tempfile.mkdtemp()
    L.main(["--dataset", "abcd", "--n", "2", "--k", "2", "--url", f"http://127.0.0.1:{srv.server_port}",
            "--out", tmp4, "--think"])
    check("--think asks for thinking and a long answer", seen[0].get("chat_template_kwargs") == {"enable_thinking": True}
          and seen[0]["max_tokens"] == 4096)
    check("without --think no template switch is sent", "chat_template_kwargs" not in seen_before_think)
    shutil.rmtree(tmp4)
    tmp5, tmp6 = tempfile.mkdtemp(), tempfile.mkdtemp()
    base_args = ["--dataset", "abcd", "--n", "8", "--k", "2", "--url", f"http://127.0.0.1:{srv.server_port}"]
    L.main(base_args + ["--out", tmp5])
    L.main(base_args + ["--out", tmp6, "--head", "3"])
    full = [json.loads(x)["i"] for x in open(os.path.join(tmp5, "preds.jsonl"))]
    head = [json.loads(x)["i"] for x in open(os.path.join(tmp6, "preds.jsonl"))]
    check("--head keeps the first rows of the same draw", head == sorted(full)[:3])
    shutil.rmtree(tmp5)
    shutil.rmtree(tmp6)
    seen.clear()
    tmp7 = tempfile.mkdtemp()
    L.main(["--dataset", "multiwoz", "--n", "2", "--k", "2", "--url", f"http://127.0.0.1:{srv.server_port}",
            "--out", tmp7])
    sm = seen[0]["messages"][0]["content"]
    check("MultiWOZ prompt: its label list, no guidelines", "Hotel-Inform" in sm and "MultiWOZ" in sm
          and "guidelines" not in sm and "task specifications" not in sm)
    shutil.rmtree(tmp7)
    # v8 additions: retries, error rows asked again on restart, exit code 4, --only-rows, --max-tokens with --think
    L.time.sleep = lambda s: None  # no 30 s pauses in the test
    url = f"http://127.0.0.1:{srv.server_port}"
    tmp8 = tempfile.mkdtemp()
    mode["fail_left"] = 2
    L.main(["--dataset", "abcd", "--n", "3", "--k", "2", "--url", url, "--out", tmp8, "--retries", "2"])
    r8 = [json.loads(x) for x in open(os.path.join(tmp8, "preds.jsonl"))]
    check("two failed requests are re-sent: all 3 rows answered, no error", len(r8) == 3
          and all(x["error"] is None and x["pred"] == "pull-up-account" for x in r8))
    shutil.rmtree(tmp8)
    tmp9 = tempfile.mkdtemp()
    mode["fail_left"] = -1
    try:
        L.main(["--dataset", "abcd", "--n", "3", "--k", "2", "--url", url, "--out", tmp9, "--retries", "1"])
        check("errors that remain end the run with exit code 4", False)
    except SystemExit as e:
        check("errors that remain end the run with exit code 4", e.code == 4)
    r9 = [json.loads(x) for x in open(os.path.join(tmp9, "preds.jsonl"))]
    check("the failed rows are recorded with their error", len(r9) == 3 and all(x["error"] for x in r9))
    mode["fail_left"] = 0
    seen.clear()
    L.main(["--dataset", "abcd", "--n", "3", "--k", "2", "--url", url, "--out", tmp9])
    r9b = [json.loads(x) for x in open(os.path.join(tmp9, "preds.jsonl"))]
    backups = [f for f in os.listdir(tmp9) if f.startswith("preds.jsonl.errors_")]
    check("a restart asks the error rows again: 3 requests, 3 rows, no error, a copy of the old file kept",
          len(seen) == 3 and len(r9b) == 3 and all(x["error"] is None for x in r9b) and len(backups) == 1)
    shutil.rmtree(tmp9)
    tmp10 = tempfile.mkdtemp()
    L.main(["--dataset", "abcd", "--n", "6", "--k", "2", "--url", url, "--out", os.path.join(tmp10, "all")])
    drawn = sorted(json.loads(x)["i"] for x in open(os.path.join(tmp10, "all", "preds.jsonl")))
    with open(os.path.join(tmp10, "only.json"), "w") as h:
        json.dump([drawn[1], drawn[4]], h)
    seen.clear()
    L.main(["--dataset", "abcd", "--n", "6", "--k", "2", "--url", url, "--out", os.path.join(tmp10, "re"),
            "--only-rows", os.path.join(tmp10, "only.json"), "--think", "--max-tokens", "16384"])
    r10 = [json.loads(x)["i"] for x in open(os.path.join(tmp10, "re", "preds.jsonl"))]
    check("--only-rows answers just those rows of the same draw", sorted(r10) == [drawn[1], drawn[4]])
    check("--max-tokens 16384 with --think is sent", seen[0]["max_tokens"] == 16384
          and seen[0].get("chat_template_kwargs") == {"enable_thinking": True})
    with open(os.path.join(tmp10, "bad.json"), "w") as h:
        json.dump([10 ** 7], h)
    try:
        L.main(["--dataset", "abcd", "--n", "6", "--k", "2", "--url", url, "--out", os.path.join(tmp10, "bad"),
                "--only-rows", os.path.join(tmp10, "bad.json")])
        check("--only-rows outside the draw is refused", False)
    except AssertionError:
        check("--only-rows outside the draw is refused", True)
    shutil.rmtree(tmp10)
    # v9 additions: a dead server stops the run after N rows in a row with an error; the request timeout default
    tmp11 = tempfile.mkdtemp()
    mode["fail_left"] = -1
    try:
        L.main(["--dataset", "abcd", "--n", "6", "--k", "2", "--url", url, "--out", tmp11, "--retries", "0",
                "--max-consecutive-errors", "2"])
        check("2 rows in a row with an error stop the run with exit code 4", False)
    except SystemExit as e:
        check("2 rows in a row with an error stop the run with exit code 4", e.code == 4)
    r11 = [json.loads(x) for x in open(os.path.join(tmp11, "preds.jsonl"))]
    res11 = json.load(open(os.path.join(tmp11, "result.json")))
    check("the stopped run holds just the 2 error rows (n 2, errors 2)", len(r11) == 2 and all(x["error"] for x in r11)
          and res11["n"] == 2 and res11["errors"] == 2)
    mode["fail_left"] = 0
    seen.clear()
    L.main(["--dataset", "abcd", "--n", "6", "--k", "2", "--url", url, "--out", tmp11])
    r11b = [json.loads(x) for x in open(os.path.join(tmp11, "preds.jsonl"))]
    check("a restart after the stop answers all 6 rows (2 asked again, 4 new), no error",
          len(seen) == 6 and sorted(x["i"] for x in r11b) == sorted(set(x["i"] for x in r11b)) and len(r11b) == 6
          and all(x["error"] is None for x in r11b))
    shutil.rmtree(tmp11)
    got, real_ask = [], L.ask

    def spy(*args, **kw):
        got.append(kw.get("timeout"))
        return real_ask(*args, **kw)

    L.ask = spy
    tmp12 = tempfile.mkdtemp()
    L.main(["--dataset", "abcd", "--n", "1", "--k", "2", "--url", url, "--out", os.path.join(tmp12, "a")])
    L.main(["--dataset", "abcd", "--n", "1", "--k", "2", "--url", url, "--out", os.path.join(tmp12, "b"), "--think"])
    L.ask = real_ask
    check("request timeout: 300 s without --think, 1800 s with it", got == [300, 1800])
    shutil.rmtree(tmp12)
finally:
    srv.shutdown()
    shutil.rmtree(tmp)
print("FAILED" if fails else "PASSED", f"({fails} failed)")
sys.exit(1 if fails else 0)

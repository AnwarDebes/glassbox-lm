"""Playwright check of the userspace headless shell: about:blank, then one local file:// MiniWoB page.

Run with python -I. The browser is started through browser/chrome-headless-shell (a wrapper that gives
ONLY the browser process the conda-forge libraries); this Python process keeps its own environment.
Writes a JSON report and a PNG screenshot into --outdir.
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../.."))
import argparse
import json
import os
import time

from playwright.sync_api import sync_playwright

F = f"{_REPO_ROOT}/studies/agent_guards/miniwob"
WRAP = f"{F}/browser/chrome-headless-shell"
TASK_HTML = f"{F}/.venv/lib/python3.12/site-packages/miniwob/html/miniwob/click-test-2.html"


def check(outdir, sandbox, exe=WRAP):
    rep = {"sandbox": sandbox, "executable": exe}
    t0 = time.perf_counter()
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(executable_path=exe, headless=True, chromium_sandbox=sandbox, timeout=60000)
        except Exception as e:  # report, do not hide
            rep["launch_error"] = f"{type(e).__name__}: {str(e)[:1500]}"
            return rep
        rep["launch_s"] = round(time.perf_counter() - t0, 3)
        rep["browser_version"] = browser.version
        ctx = browser.new_context(service_workers="block", accept_downloads=False, viewport={"width": 500, "height": 320})
        page = ctx.new_page()
        requests = []
        page.on("request", lambda r: requests.append([r.method, r.url[:120]]))
        # 1. about:blank
        t1 = time.perf_counter()
        page.goto("about:blank")
        rep["about_blank"] = {"url": page.url, "ready_state": page.evaluate("document.readyState"),
                              "s": round(time.perf_counter() - t1, 3)}
        # 2. a local MiniWoB page (file://)
        url = "file://" + TASK_HTML
        t2 = time.perf_counter()
        page.goto(url)
        page.wait_for_selector("#sync-task-cover", timeout=10000)
        rep["miniwob_load_s"] = round(time.perf_counter() - t2, 3)
        rep["miniwob_title"] = page.title()
        page.evaluate("Math.seedrandom(0)")
        page.evaluate("core.startEpisodeReal()")
        page.wait_for_function("WOB_TASK_READY === true", timeout=10000)
        rep["utterance"] = page.evaluate("core.getUtterance()")
        dom = page.evaluate("core.getDOMInfo()")
        rep["getDOMInfo_root_tag"] = dom.get("tag")
        rep["getDOMInfo_n_children"] = len(dom.get("children", []))
        # observation formats
        rep["aria_snapshot"] = page.locator("body").aria_snapshot()
        cdp = ctx.new_cdp_session(page)
        ax = cdp.send("Accessibility.getFullAXTree")
        nodes = ax.get("nodes", [])
        rep["cdp_ax_nodes"] = len(nodes)
        rep["cdp_ax_named"] = [
            [n.get("role", {}).get("value"), n.get("name", {}).get("value"), n.get("backendDOMNodeId")]
            for n in nodes if n.get("name", {}).get("value")
        ][:20]
        shot = os.path.join(outdir, f"pw_click-test-2_{os.path.basename(exe)}_sandbox{int(sandbox)}.png")
        page.screenshot(path=shot)
        rep["screenshot_png"] = shot
        rep["screenshot_bytes"] = os.path.getsize(shot)
        rep["fonts_seen"] = page.evaluate(
            "getComputedStyle(document.querySelector('#query')).fontFamily")
        # act: click the button named in the utterance, then read the reward
        target = "ONE" if "ONE" in rep["utterance"] else "TWO"
        t3 = time.perf_counter()
        page.get_by_role("button", name=target, exact=True).click()
        rep["click_s"] = round(time.perf_counter() - t3, 3)
        rep["after_click"] = page.evaluate(
            "({done: WOB_DONE_GLOBAL, raw_reward: WOB_RAW_REWARD_GLOBAL, reward: WOB_REWARD_GLOBAL})")
        rep["requests_seen"] = requests[:20]
        rep["n_requests"] = len(requests)
        ctx.close()
        browser.close()
    rep["total_s"] = round(time.perf_counter() - t0, 3)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sandbox", type=int, choices=[0, 1], default=0)
    ap.add_argument("--exe", default=WRAP, help="browser wrapper: browser/chrome-headless-shell or browser/chrome")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    rep = check(a.outdir, bool(a.sandbox), a.exe)
    path = os.path.join(a.outdir, f"pw_page_check_{os.path.basename(a.exe)}_sandbox{a.sandbox}.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("aria_snapshot", "cdp_ax_named", "requests_seen")}))
    print("aria_snapshot:\n" + str(rep.get("aria_snapshot", ""))[:800])
    print("report:", path)


if __name__ == "__main__":
    main()

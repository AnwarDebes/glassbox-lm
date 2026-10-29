"""MiniWoB++ smoke test: scripted DOM policy (no LLM), N episodes per task.

Needs a headless Chrome + chromedriver that START on this machine
(set MINIWOB_CHROME_BINARY and MINIWOB_CHROMEDRIVER; see run_smoke.sh).
Writes one JSON line per episode and a summary JSON.

Success = episode terminated with raw_reward == 1.0 (no time penalty, no partial credit).
Seconds per step = wall time of env.step() (action + DOM read + metadata [+ screenshot]).
"""
import argparse
import json
import os
import statistics
import sys
import time
import traceback

import gymnasium
import miniwob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripted_policy import TASKS, decide  # noqa: E402

gymnasium.register_envs(miniwob)


def run_task(task, episodes, max_steps, seed0, screenshots, out):
    env = gymnasium.make(f"miniwob/{task}-v1", action_space_config="all_supported")
    rows = []
    try:
        for ep in range(episodes):
            t0 = time.perf_counter()
            obs, info = env.reset(seed=seed0 + ep, options={"record_screenshots": screenshots})
            reset_s = time.perf_counter() - t0
            utterance = obs["utterance"]
            step_s, specs = [], []
            terminated = False
            for _ in range(max_steps):
                spec = decide(task, obs)
                specs.append(spec)
                kwargs = {k: v for k, v in spec.items() if k != "action_type"}
                action = env.unwrapped.create_action(spec["action_type"], **kwargs)
                t1 = time.perf_counter()
                obs, reward, terminated, truncated, info = env.step(action)
                step_s.append(time.perf_counter() - t1)
                if terminated:
                    break
            row = {
                "task": task, "episode": ep, "seed": seed0 + ep, "utterance": utterance,
                "terminated": bool(terminated), "raw_reward": info.get("raw_reward"),
                "env_reward": info.get("env_reward"), "reason": str(info.get("reason")),
                "success": bool(terminated and info.get("raw_reward") == 1.0),
                "n_steps": len(step_s), "step_seconds": step_s, "reset_seconds": reset_s,
                "actions": specs, "screenshots": screenshots,
            }
            rows.append(row)
            out.write(json.dumps(row) + "\n")
            out.flush()
    finally:
        env.close()
    return rows


def probe_formats(path):
    """One episode of click-button: screenshot on, plus a CDP accessibility-tree read."""
    env = gymnasium.make("miniwob/click-button-v1")
    rep = {}
    try:
        obs, info = env.reset(seed=12345, options={"record_screenshots": True})
        shot = obs["screenshot"]
        rep["screenshot_shape"] = list(shot.shape)
        rep["screenshot_nonblack_pixels"] = int((shot.sum(axis=2) > 0).sum())
        rep["n_dom_elements"] = len(obs["dom_elements"])
        rep["dom_example"] = [
            {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in e.items()}
            for e in list(obs["dom_elements"])[:6]
        ]
        rep["fields"] = list(obs["fields"])
        rep["utterance"] = obs["utterance"]
        drv = env.unwrapped.instance.driver
        try:
            ax = drv.execute_cdp_cmd("Accessibility.getFullAXTree", {})
            nodes = ax.get("nodes", [])
            rep["ax_tree_nodes"] = len(nodes)
            rep["ax_example"] = [
                {"role": n.get("role", {}).get("value"), "name": n.get("name", {}).get("value")}
                for n in nodes[:15]
            ]
        except Exception as e:  # report, do not hide
            rep["ax_tree_error"] = repr(e)[:500]
    finally:
        env.close()
    with open(path, "w") as f:
        json.dump(rep, f, indent=1)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--screenshots", action="store_true", help="record screenshots every step (slower)")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    summary = {"args": vars(a), "tasks": {}}
    with open(os.path.join(a.outdir, "episodes.jsonl"), "w") as out:
        for task in a.tasks:
            try:
                rows = run_task(task, a.episodes, a.max_steps, a.seed0, a.screenshots, out)
                steps = [s for r in rows for s in r["step_seconds"]]
                summary["tasks"][task] = {
                    "episodes": len(rows),
                    "successes": sum(r["success"] for r in rows),
                    "mean_steps": statistics.mean(r["n_steps"] for r in rows),
                    "sec_per_step_mean": statistics.mean(steps) if steps else None,
                    "sec_per_step_median": statistics.median(steps) if steps else None,
                    "reset_sec_mean": statistics.mean(r["reset_seconds"] for r in rows),
                }
            except Exception as e:
                summary["tasks"][task] = {"error": "".join(traceback.format_exception_only(type(e), e))[:1500]}
            print(task, json.dumps(summary["tasks"][task]), flush=True)
    try:
        summary["formats_probe"] = probe_formats(os.path.join(a.outdir, "formats_probe.json"))
    except Exception as e:
        summary["formats_probe"] = {"error": "".join(traceback.format_exception_only(type(e), e))[:1500]}
    with open(os.path.join(a.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("formats_probe:", json.dumps(summary["formats_probe"])[:600])


if __name__ == "__main__":
    main()

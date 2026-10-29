"""Next-action choice by the local LLM (Qwen3.5-9B on llama-server), dev only.

usage: python -I llm_next_action.py --dataset abcd --n 500 --k 10 --out DIR [--url http://127.0.0.1:18082]
       [--no-guidelines] [--seed 1]

The prompt gives the LLM what a new human agent would get: the list of actions, the company's written guidelines
(ABCD only), k solved examples from the training split that are most similar to the current dialogue (TF-IDF cosine
on the same text the other arms see), and the current dialogue (ABCD: the full transcript so far, with the actions
taken; STAR: the previous actions, the recent exchange and the customer's utterances). Temperature 0. The answer is
mapped to an action name (exact, then normalised, then the closest name with difflib cutoff 0.6; else invalid).
Writes DIR/preds.jsonl (one line per example) and DIR/result.json. The test split is never opened.
"""
import os as _os  # published copy: paths are found from this file's place in the repository
_REPO_ROOT = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../.."))
import argparse
import difflib
import gzip
import json
import os
import random
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ft_baseline as F  # noqa: E402

GUIDE = f"{_REPO_ROOT}/studies/decision_layer/data/untrusted/abcd/guidelines.json"
STAR_TASKS = f"{_REPO_ROOT}/studies/decision_layer/data/untrusted/star/STAR-master/tasks"
DEMO_CHARS = 900
CUR_CHARS = 2500


def rows_of(dataset, split, allow_test=False):
    if split not in ("train", "dev") and not (allow_test and split == "test"):
        raise ValueError("only train and dev are allowed (test only under a frozen registration)")
    with gzip.open(os.path.join(F.DATA, dataset, f"examples_{split}.json.gz"), "rt") as h:
        d = json.load(h)
    return d["rows"], d["buttons"]


def dialogue_text(row, dataset, limit):
    if dataset == "abcd" and row.get("full"):
        return row["full"][-limit:]
    prev = row.get("prev") or []
    return (f"Actions taken so far: {', '.join(prev[-10:]) or 'none'}\n"
            f"Recent exchange: {row.get('recent') or ''}\n"
            f"All customer messages: {(row.get('cust') or '')[-limit // 2:]}")


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def map_answer(text, buttons):
    """The action name an answer gives, or None."""
    first = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    first = first.strip().strip("`*\"'. ").replace("Next action:", "").strip()
    if first in buttons:
        return first
    nb = {norm(b): b for b in buttons}
    if norm(first) in nb:
        return nb[norm(first)]
    for b in sorted(buttons, key=len, reverse=True):  # an exact name inside a longer answer
        if re.search(r"(?<![a-z0-9_-])" + re.escape(b) + r"(?![a-z0-9_-])", text or ""):
            return b
    m = difflib.get_close_matches(first, buttons, n=1, cutoff=0.6)
    return m[0] if m else None


def system_prompt(dataset, buttons, guidelines):
    who = {"abcd": "a customer-service agent of an online clothing store (the ABCD dataset)",
           "star": "a wizard who serves users of a task-oriented assistant (the STAR dataset); the actions are the "
                   "wizard's replies and API queries",
           "multiwoz": "the system of a travel information and booking assistant (the MultiWOZ 2.2 dataset); each "
                       "action is the set of dialogue acts of the system's next turn, written Domain-Act joined by +"}[dataset]
    s = (f"You are {who}. At each step you choose the agent's next action. Choose exactly one action from this list "
         f"and answer with the action name only, nothing else.\n\nActions:\n" + "\n".join(buttons))
    if guidelines is not None and dataset == "abcd":
        s += "\n\nThe company's written guidelines for agents (flows, subflows and the actions they need):\n" + guidelines
    elif guidelines is not None:
        s += ("\n\nThe task specifications (for each task: the wizard's replies by action name, and the graph of the "
              "usual next action):\n" + guidelines)
    return s


def star_specs():
    """All STAR task specifications (replies and graph), compact JSON, sorted by task (amendment AMEND_llm_star_specs)."""
    out = []
    for task in sorted(os.listdir(STAR_TASKS)):
        p = os.path.join(STAR_TASKS, task, f"{task}.json")
        if os.path.exists(p):
            with open(p) as h:
                d = json.load(h)
            out.append({"task": d.get("task", task), "replies": d.get("replies", {}), "graph": d.get("graph", {})})
    return json.dumps(out, ensure_ascii=False, separators=(",", ":"))


def ask(url, system, user, max_tokens=24, timeout=600, think=None):
    body = {"model": "qwen3.5-9b-q8_0", "temperature": 0.0, "max_tokens": max_tokens, "cache_prompt": True,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if think is not None:
        body["chat_template_kwargs"] = {"enable_thinking": bool(think)}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"]["content"], out.get("usage", {})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["star", "abcd", "multiwoz"], required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--url", default="http://127.0.0.1:18082")
    ap.add_argument("--no-guidelines", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shortlist", help="agent predictions (tm.npz): the LLM must choose among the agent's top k")
    ap.add_argument("--k-short", type=int, default=3)
    ap.add_argument("--exclude", help="preds.jsonl of an earlier run: its rows are not drawn (PLAN_hybrid_dev.md)")
    ap.add_argument("--think", action="store_true", help="ask for the template's thinking; the answer is the content")
    ap.add_argument("--max-tokens", type=int, default=None, help="default 24, or 4096 with --think")
    ap.add_argument("--head", type=int, default=None, help="answer only the first N drawn rows (sorted by row id)")
    ap.add_argument("--test-registration", help="answer TEST rows; needs this registration's .frozen.json")
    ap.add_argument("--only-rows", help="JSON list of row ids: answer only these of the drawn rows (re-asks)")
    ap.add_argument("--retries", type=int, default=2, help="re-sends of a failed request (30 s apart) before it is "
                    "recorded as an error")
    ap.add_argument("--max-consecutive-errors", type=int, default=10, help="stop the run (exit 4) after this many rows "
                    "in a row ended in a request error (a dead or hung server); a restart resumes and asks them again")
    ap.add_argument("--timeout", type=float, default=None, help="seconds per request; default 1800 with --think, "
                    "else 300 (the slowest L35k50 dev request took 41.3 s)")
    a = ap.parse_args(argv)
    from sklearn.feature_extraction.text import TfidfVectorizer

    tr, buttons = rows_of(a.dataset, "train")
    if a.test_registration:
        fr = os.path.splitext(a.test_registration)[0] + ".frozen.json"
        if not os.path.exists(fr):
            raise SystemExit(f"refused: no freeze record {fr}; the test split is read only under a frozen registration")
        dv, buttons2 = rows_of(a.dataset, "test", allow_test=True)
    else:
        dv, buttons2 = rows_of(a.dataset, "dev")
    assert buttons == buttons2
    excluded = set()
    if a.exclude:
        with open(a.exclude) as h:
            excluded = {json.loads(x)["i"] for x in h if x.strip()}
    pool = [i for i in range(len(dv)) if i not in excluded]
    idx = sorted(random.Random(a.seed).sample(pool, min(a.n, len(pool))))
    if a.head:
        idx = idx[: a.head]
    if a.only_rows:
        with open(a.only_rows) as h:
            only = set(json.load(h))
        assert only <= set(idx), "--only-rows holds rows outside the draw"
        idx = [i for i in idx if i in only]
    short = None
    if a.shortlist:
        import numpy as np
        S = np.load(a.shortlist)
        assert [int(v) for v in S["gold"]] == [int(r["y"]) for r in dv], "shortlist rows are not in dev order"
        short = S["top_ids"][:, : a.k_short]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform([F.render(r) for r in tr])
    guidelines = None
    if a.dataset == "abcd" and not a.no_guidelines:
        with open(GUIDE) as h:
            guidelines = json.dumps(json.load(h), ensure_ascii=False)
    elif a.dataset == "star" and not a.no_guidelines:
        guidelines = star_specs()
    system = system_prompt(a.dataset, buttons, guidelines)
    os.makedirs(a.out, exist_ok=True)
    done = set()
    pp = os.path.join(a.out, "preds.jsonl")
    if os.path.exists(pp):  # restart-safe: keep finished examples; rows that ended in a request error are asked again
        with open(pp) as h:
            old = [json.loads(x) for x in h if x.strip()]
        bad = [r for r in old if r.get("error") is not None]
        if bad:
            os.replace(pp, pp + time.strftime(".errors_%Y%m%dT%H%M%SZ", time.gmtime()))
            with open(pp, "w") as h:
                h.writelines(json.dumps(r) + "\n" for r in old if r.get("error") is None)
            print(f"restart: {len(bad)} rows with a request error are asked again", flush=True)
        done = {r["i"] for r in old if r.get("error") is None}
    t0 = time.time()
    timeout = a.timeout or (1800 if a.think else 300)
    consecutive = 0
    with open(pp, "a") as out:
        for n_done, i in enumerate(idx):
            if i in done:
                continue
            row = dv[i]
            sims = (Xtr @ vec.transform([F.render(row)]).T).toarray().ravel()
            demo_ids = sims.argsort()[::-1][:a.k][::-1]  # most similar last, next to the question
            demos = "\n\n".join(f"Example {j + 1}:\n{dialogue_text(tr[d], a.dataset, DEMO_CHARS)}\nNext action: "
                                f"{buttons[tr[d]['y']]}" for j, d in enumerate(demo_ids))
            cands = [buttons[int(c)] for c in short[i]] if short is not None else None
            cand_text = ("" if cands is None else "\nCandidate next actions proposed by the decision model (most likely "
                         f"first): {', '.join(cands)}. Choose exactly one of these candidates.")
            user = (f"Solved examples from earlier dialogues:\n\n{demos}\n\nNow the current dialogue:\n"
                    f"{dialogue_text(row, a.dataset, CUR_CHARS)}{cand_text}\nNext action:")
            t1 = time.time()
            for attempt in range(a.retries + 1):
                try:
                    text, usage = ask(a.url, system, user, max_tokens=a.max_tokens or (4096 if a.think else 24),
                                      timeout=timeout, think=True if a.think else None)
                    err = None
                    break
                except Exception as e:  # noqa: BLE001 - after the retries, recorded as an error (asked again on restart)
                    text, usage, err = "", {}, repr(e)[:300]
                    if attempt < a.retries:
                        time.sleep(30)
            fallback = None
            if cands is None:
                pred = map_answer(text, buttons)
            else:
                pred = map_answer(text, cands)
                fallback = pred is None
                if fallback:
                    pred = cands[0]
            rec = {"i": i, "gold": buttons[row["y"]], "pred": pred, "raw": text[:200], "error": err,
                   "candidates": cands, "fallback": fallback,
                   "s": round(time.time() - t1, 2), "prompt_tokens": usage.get("prompt_tokens"),
                   "completion_tokens": usage.get("completion_tokens"),
                   "demo_labels": [buttons[tr[d]["y"]] for d in demo_ids]}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if n_done % 50 == 0:
                print(n_done, rec["gold"], rec["pred"], rec["s"], flush=True)
            consecutive = consecutive + 1 if err is not None else 0
            if consecutive >= a.max_consecutive_errors:
                print(f"stop: {consecutive} rows in a row ended in a request error; a restart resumes", flush=True)
                break
    with open(pp) as h:
        recs = [json.loads(x) for x in h if x.strip()]
    acc = sum(r["pred"] == r["gold"] for r in recs) / len(recs)
    nn = sum(r["demo_labels"][-1] == r["gold"] for r in recs) / len(recs)
    maj = sum(max(set(r["demo_labels"]), key=r["demo_labels"].count) == r["gold"] for r in recs) / len(recs)
    res = {"dataset": a.dataset, "n": len(recs), "k": a.k, "guidelines": guidelines is not None, "accuracy": acc,
           "invalid": sum(r["pred"] is None for r in recs), "errors": sum(r["error"] is not None for r in recs),
           "nearest_demo_label_accuracy": nn, "demo_majority_accuracy": maj, "seconds": round(time.time() - t0, 1),
           "seed": a.seed, "argv": sys.argv, "utc_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "shortlist": a.shortlist, "k_short": a.k_short if a.shortlist else None, "excluded_rows": len(excluded),
           "fallbacks": sum(bool(r.get("fallback")) for r in recs), "think": a.think,
           "mean_completion_tokens": (sum(r.get("completion_tokens") or 0 for r in recs) / len(recs))}
    with open(os.path.join(a.out, "result.json"), "w") as h:
        json.dump(res, h, indent=1)
    print(json.dumps(res), flush=True)
    if res["errors"]:
        sys.exit(4)  # rows with a request error remain: the job fails; a restart asks them again


if __name__ == "__main__":
    main()

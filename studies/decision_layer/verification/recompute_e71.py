"""Independent tally of e71 v2 (tag e71v2) from results/explore/verify_e71.jsonl (read only).

Written from logs/plan_e71_v2_amendment.md: outcomes per (state, group); the headline is per state ("for X% of 200
sampled real dev states, no ... edit makes the agent choose a button the rules forbid in that state"), so a state is
certified at k when every one of its groups is certified at k. States whose u is outside the last 3 turns
(in_recent false) are reported separately. Also checks: one row per (state, group), zero-edit rows exact, and every
stored breaking example replayed through the real pipeline with the same choice and a broken rule.

    recompute_e71.py [tag] [seed] [file]   # default tag e71v2, seed 1, file verify_e71.jsonl (e76: verify_e76.jsonl)
"""
import collections
import json
import sys
from pathlib import Path

P = Path(str(Path(__file__).resolve().parents[1]) + "/results/explore/" + (sys.argv[3] if len(sys.argv) > 3 else "verify_e71.jsonl"))
TAG = sys.argv[1] if len(sys.argv) > 1 else "e71v2"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
K2, K1B, K1, OWN = ("certified at k = 2", "certified at k = 1; a 2-word edit breaks the rule",
                    "a 1-word edit breaks the rule", "own choice breaks the rule")

rows, seen, dup, bad_replay, not_exact = [], set(), 0, [], 0
for line in P.read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("tag") != TAG or int(r.get("seed", 1)) != SEED:
        continue
    key = (r["state"], r["group"])
    if key in seen:
        dup += 1
        continue
    seen.add(key)
    rows.append(r)
    not_exact += r.get("zero_edit_exact") is not True
    for kk in ("k1", "k2"):
        for e in (r.get(kk) or {}).get("examples", []):
            if not (e.get("replay_breaks_rule") is True and e.get("replay_choice") == e.get("choice")):
                bad_replay.append((r["state"], r["group"], kk, e.get("edited_utterance")))

pairs = collections.Counter(r["outcome"] for r in rows)
by_state = collections.defaultdict(list)
for r in rows:
    by_state[r["state"]].append(r)
print(f"tag {TAG} seed {SEED}: {len(rows)} (state, group) pairs over {len(by_state)} states; duplicates ignored {dup}; "
      f"zero-edit not exact {not_exact}; replay problems {len(bad_replay)}")
print("pairs:", dict(pairs))

for label, keep in (("u in the last 3 turns", True), ("u outside the last 3 turns", False), ("all", None)):
    st = {s: rs for s, rs in by_state.items() if keep is None or bool(rs[0].get("in_recent")) == keep}
    n = len(st)
    if not n:
        continue
    c2 = sum(all(r["outcome"] == K2 for r in rs) for rs in st.values())
    c1 = sum(all(r["outcome"] in (K2, K1B) for r in rs) for rs in st.values())
    own = sum(any(r["outcome"] == OWN for r in rs) for rs in st.values())
    one = sum(any(r["outcome"] == K1 for r in rs) for rs in st.values())
    print(f"{label:28s} states {n:3d}: certified at k = 1 {c1} ({100 * c1 / n:.1f}%), at k = 2 {c2} ({100 * c2 / n:.1f}%); "
          f"own choice breaks a rule {own}; some 1-word edit breaks a rule {one}")

groups = collections.defaultdict(collections.Counter)
for r in rows:
    groups[r["group"].split("(")[0]][r["outcome"]] += 1
print("by rule family:")
for g, c in sorted(groups.items()):
    print(f"  {g:10s} {dict(c)}")
if bad_replay:
    print("replay problems:", bad_replay[:10])

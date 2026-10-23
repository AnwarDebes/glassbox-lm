"""e73 code checks on STAR's TRAIN split only (no dev score): scripts/star_replies.py.

    HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 .venv-dl/bin/python -I tests/test_star_replies.py

1. walk("train") aligns with star_prepare's train examples: the same (convo_id, k, label) in the same order.
2. fill() with the logged labels reproduces the mapping file's train coverage (the sum of filled_exact).
3. The scoring path runs end to end on a train subset (the first 300 dialogues) with R21's STAR seed-1 model: the C4
   view's names equal the model's bit_names.json, and the CPU decision, the closed loop and the reply matching all run.
   The accuracies are printed only as a code check: train data, so they are inflated, and nothing is recorded.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT / "scripts"))

import star_replies as sr  # noqa: E402
from dl import count_text, cpu_tm_fast  # noqa: E402


def main() -> None:
    from dl.abcd import load_examples  # noqa: E402

    tr, blist = load_examples("train", "star")
    acts, free = sr.walk("train")
    assert len(acts) == len(tr), (len(acts), len(tr))
    bad = sum(not (a["convo_id"] == r["convo_id"] and a["k"] == r["k"] and a["label"] == r["button"]) for a, r in zip(acts, tr))
    assert bad == 0, bad
    print(f"1. walk(train) aligns with star_prepare's {len(tr):,} train examples; {sum(free.values()):,} free-text wizard replies")
    resp = sr.responses()
    mfile = json.loads(sr.MAP_USED.read_text())
    mapping, cov = mfile["mapping"], mfile["coverage_train"]
    exact = sum(int((r := sr.fill(a["label"], a, resp, mapping)) is not None and r.strip() == a["text"].strip())
                for a in acts if a["kind"] == "pick_suggestion" and (tt := sr.template_of(a["label"], a["tasks"], resp)) and sr.SLOT.findall(tt[1]))
    want = sum(c["filled_exact"] for c in cov.values())
    assert exact == want, (exact, want)
    print(f"2. fill() with the logged labels: {exact:,} slotted train picks exact, as the mapping file says")
    meta = count_text.patch("c4words-S1000-v1")
    from abcd_closed_loop import closed_loop  # noqa: E402
    from dl.abcd import ABCDView  # noqa: E402

    E = np.load(ROOT / "data" / "emb" / "star__none.npz")
    em = {f: E[f"train__{f}"] for f in ("cust", "recent", "full")}
    view = ABCDView(parts=("mem", "words", "xcust", "xrecent"), lex_k=1000, seed=0).fit(tr, em, blist)
    assert view.names == json.loads((sr.MODELS / "bit_names.json").read_text())
    convs = sorted({r["convo_id"] for r in tr})[:300]
    keep = set(convs)
    idx = [i for i, r in enumerate(tr) if r["convo_id"] in keep]
    sub, sacts = [tr[i] for i in idx], [acts[i] for i in idx]
    X = view.transform(sub, em)
    z = np.load(sr.MODELS / "model_seed1.npz")
    pack = cpu_tm_fast.pack(z["ta"], z["w"])
    fn = lambda Z: cpu_tm_fast.class_sums(pack, Z)  # noqa: E731
    for loop, p in (("open", fn(X).argmax(1)), ("closed", closed_loop(fn, X, sub, blist, 2 * len(blist) + 6))):
        lab = [blist[int(c)] for c in p]
        picks = [i for i, a in enumerate(sacts) if a["kind"] == "pick_suggestion"]
        ex = sum(int((r := sr.fill(lab[i], sacts[i], resp, mapping)) is not None and r.strip() == sacts[i]["text"].strip()) for i in picks)
        acc = float(np.mean([lab[i] == sacts[i]["label"] for i in range(len(sacts))]))
        print(f"3. train subset ({len(sub)} actions, {len(picks)} picks), {loop} loop: label accuracy {acc:.3f}, exact replies {ex} (code check only)")
    print("table", meta["table"], "ok")


if __name__ == "__main__":
    main()

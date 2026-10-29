#!/usr/bin/env python
"""Step 3a, the handover to the decision-layer agent (tu/PLAN_3a.md "Notes": on PASS, export the a3B table the way
c4words_S1000_v1 was exported, for R4). Runs only if tu/a3/verdict_a3.json says PASS. CPU only.

    python -I a3_export.py

Writes tu/a3/export/c4words_a3B_v1/:
- table.npz, byte-identical to tu/a3/tables/a3B_d768.npz (the registered table at lambda*);
- words.txt (the same 63,091 words, in row order; checked equal to c4words_S1000_v1/words.txt);
- vectors.npy (the rows, float16, 63,091 x 768);
- tokenize_ref.py, byte-identical to c4words_S1000_v1's (the same words in the same row order, so the same token ->
  row ids);
- README.md (provenance, the 3a numbers from verdict_a3.json, the wording rules), MANIFEST.sha256.
Refuses to overwrite.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent  # glassbox-lm/studies/text_understanding/tu
A3 = HERE / "a3"
OLD = HERE.parent / "w1" / "export" / "c4words_S1000_v1"
OUT = A3 / "export" / "c4words_a3B_v1"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    v = json.loads((A3 / "verdict_a3.json").read_text())
    assert v["verdict"] == "PASS", "3a did not pass: nothing to export"
    assert not OUT.exists(), f"{OUT} exists"
    fit = json.loads((A3 / "fit_meta.json").read_text())
    src = A3 / "tables" / "a3B_d768.npz"
    assert sha(src)[:16] == fit["sha256"]["a3B_d768.npz"], "the table is not the registered one"
    OUT.mkdir(parents=True)
    shutil.copyfile(src, OUT / "table.npz")
    z = np.load(OUT / "table.npz")
    words = [str(w) for w in z["words"]]
    (OUT / "words.txt").write_text("".join(w + "\n" for w in words), encoding="utf-8")
    assert (OUT / "words.txt").read_bytes() == (OLD / "words.txt").read_bytes(), "the word list differs from c4words_S1000_v1"
    np.save(OUT / "vectors.npy", z["codes"].astype(np.float16))
    shutil.copyfile(OLD / "tokenize_ref.py", OUT / "tokenize_ref.py")
    rows = []
    for d in ("clinc150", "banking77", "hwu64"):
        r = v[d]
        b = r["a3B_d768"]
        rows.append(f"| {d} | {r['tm_ppw']:.2f} | {b['tm']:.2f} ({b['tm_gain']:+.2f}) | {r['tm_potion']:.2f} | {r['lr_ppw']:.2f} | "
                    f"{b['lr']:.2f} ({b['lr_gain']:+.2f}) | {r['lr_potion']:.2f} |")
    readme = f"""# Handover: the C4-word table refined by a teacher (c4words_a3B_v1)

**For:** the decision-layer agent session (the decision_layer study), for R4 (the agent on STAR, ABCD and MultiWOZ dev) and then R5 (the
guards). **From:** step 3a of the text-understanding plan (glassbox-lm/studies/text_understanding/tu/PLAN_3a.md, LOG.md). **Status:** a
frozen artifact. Nothing here was run on ABCD, STAR or MultiWOZ.

## What it is

- The same 63,091 C4 words as c4words_S1000_v1. Each row was refitted so that the mean of a C4 sentence's word rows
  matches the sentence embedding of a TEACHER, bge-base-en-v1.5 (MIT; [CLS], L2-normalised). This is the least-squares
  recipe of Model2Vec / Tokenlearn, applied to our own C4 word list.
  - Fit: {fit['train']:,} C4 sentences, with {fit['held']:,} held out. The rows were free, lambda* = {fit['lambda_star']} on the
    held-out cosine. Mean student-teacher cosine on held-out sentences: {fit['heldout_cos'][str(fit['lambda_star'])]:.4f}
    (the linear map alone: {fit['heldout_cos']['W_A']:.4f}).
  - 768 dimensions. Unit rows. {fit['words_seen']:,} of {fit['words']:,} words occur in the fit sentences; the rest keep the
    linear map's row.
- **Provenance, in every report:** "network-free at run time, teacher at training time". The table learned from a
  neural teacher at training time, as potion did. NEVER write "no neural network at any stage" for this table. That
  phrase stays with c4words_S1000_v1. If it matches potion, it is a replication of potion's recipe on our word list,
  not a new method.

## The registered 3a result (VAL only; verdict {v['verdict']}, {v['sets_meeting_bar']} of 3 sets at +0.5 with both probes)

TM: s 8, the mean of seeds 4-6. LR: C in 1/10/100, best on val. Numbers from tu/a3/verdict_a3.json.

| set | TM C4 table | TM refined (gain) | TM potion | LR C4 table | LR refined (gain) | LR potion |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

## Files (sha256, first 16 hex; full list in MANIFEST.sha256)

- **table.npz:** byte-identical to tu/a3/tables/a3B_d768.npz, the table of the registered runs. Arrays: words, codes
  [63,091 x 768] float16 (unit rows), counts, vocab_ids, in the same format as c4words_S1000_v1.
- **words.txt / vectors.npy:** the same words and rows as plain files. words.txt is byte-identical to
  c4words_S1000_v1's.
- **tokenize_ref.py:** byte-identical to c4words_S1000_v1's. The word list and row order are the same, so its tokens
  and row ids hold for this table too. Its docstring names the older table.

## Use

- Exactly as c4words_S1000_v1, with the same tokenizer and the same existsT recipe: K 512 spherical k-means on YOUR
  train texts, max cosine, Thermo(8) fit on your train texts.
- Swap only the vectors. Every R4 run is your own registration, dev first.
- If the agent costs more than 0.5 points on dev (R4), the table is not taken into the agent. It may still be kept for
  classification, labelled so.

## Wording

- "network-free at run time, teacher at training time".
- "judged by an AI model, not a person" for any hand check.
- "we found no published study that ..."; never "first".
- The C4 counting table stays the reference with no neural network at any stage. Report it beside this table.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    names = ["table.npz", "words.txt", "vectors.npy", "tokenize_ref.py", "README.md"]
    (OUT / "MANIFEST.sha256").write_text("".join(f"{sha(OUT / n)}  {n}\n" for n in names))
    print(json.dumps({n: sha(OUT / n)[:16] for n in names + ["MANIFEST.sha256"]}))


if __name__ == "__main__":
    main()

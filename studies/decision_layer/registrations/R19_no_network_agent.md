# R19 registration: the decision agent with no neural network at run time (STAR and ABCD)

The plan is in the project notes (not published); it was fixed at about 20:43 UTC on 2026-10-17. Every gate is checked a second time, independently. Dev rules: `logs/plan_e60_static.md` (frozen 2026-10-17T21:09:33Z, sha256 d0a4eb820533ebed).

**The question.** The agent of R18 (STAR) and R9b (ABCD) reads its text through bge-large-en-v1.5 (335M parameters, run per decision). Can it decide with no neural network at run time, and how much accuracy does that cost?

**The chosen view: B, `--parts mem,words,xcust,xrecent`, `--encoder potion-32m`.** It was chosen by S2 on STAR dev at 16,000 clauses (the first of B, A, A+B within 0.5 points of the best). What runs per decision:
- a rule-based tokenizer;
- lookups in a fixed 63,091 x 512 table (potion-base-32M);
- a bitwise OR of precomputed per-token bit codes (existsT: "some word of the customer's words / last 3 turns is close to concept k at level j"; 512 concepts per field, fitted on train);
- the memory and word bits;
- Boolean clauses and an integer vote.

No neural network runs at decision time. The table was distilled once, offline, from a transformer (bge-base-en-v1.5, via Model2Vec). The existsT recipe is from the text-understanding line (project notes, not published), ported to `src/dl/static_text.py` with a CPU draw of the concepts, so the concepts are not those of that line. The speed-up of the text step is Model2Vec's, not this study's.

**What was known at the freeze:**
- the test numbers of R18 (STAR: TM-mem 0.6265, TM-nomem 0.4722, LR-mem 0.6591, MLP-mem 0.6580) and R9b (ABCD: TM-mem 0.8401, TM-nomem 0.6537, LR-mem 0.8390);
- every e60 dev number (table below) and the e61 numbers listed there;
- the text-understanding line's registered test rows (project notes, not published: T2, T3, T3b; read, not recomputed in this study);
- the probe in the plan (STAR: the table matches bge-large at nearest-centre reading; ABCD: 9.5 points lower on the customer-words-to-task probe);
- the rough speed smoke tests (n = 50).

Frozen by `scripts/freeze.py` before any R19 test run. **Second use of STAR's test split (after R18); eleventh use of ABCD's test split** (after R9, R9b, R10, R10c, R13, R14, R15, R16, R17, R16b).

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## Dev rows behind it (seed 1, `results/explore/abcd.jsonl`)

| Run | View | Clauses | TM dev | LR dev |
|---|---|---|---|---|
| e60_star_A_c4000 | A: mem,intent,step,words (table mean vectors) | 4,000 | 0.6126 | 0.6655 |
| e60_star_B_c4000 | B: mem,words,xcust,xrecent | 4,000 | 0.6074 | 0.6409 |
| e60_star_AB_c4000 | A+B: mem,intent,step,xrecent,words | 4,000 | 0.6042 | 0.6433 |
| e60_star_A_c16000 | A | 16,000 | 0.6296 | |
| **e60_star_B_c16000** | **B** | **16,000** | **0.6343** | |
| e60_star_AB_c16000 | A+B | 16,000 | 0.6378 | |
| e60_abcd_A_c4000 | A | 4,000 | 0.8382 | 0.8407 |
| **e60_abcd_B_c4000** | **B** | **4,000** | **0.8325** | 0.8111 |
| e60_abcd_AB_c4000 | A+B | 4,000 | 0.8341 | 0.8097 |
| bge-large references | e46_star_mem_c16000 / e45_star_mem (4,000) / e25_full (ABCD, 4,000) | | 0.6394 / 0.6131 / 0.8398 | 0.6657 (STAR) |

**e61 (the twins: view B without mem; reported with no rule):** ABCD (4,000 clauses) TM 0.6200 and LR 0.6547, so memory adds +21.25 points to the TM on dev (0.8325 against 0.6200). STAR (16,000 clauses) was still training at the freeze.

- **S1** (STAR at 4,000): best 0.6126, and none more than 3.0 below, so all three went to 16,000.
- **S2** (STAR at 16,000): best A+B 0.6378; B is 0.35 below, inside 0.5, and comes first, so B is chosen.
  - B's dev curve peaks at epoch 22 (0.6343) and drifts down to 0.6144 at epoch 40 (every 2 epochs: 0.509, 0.587, 0.608, 0.608, 0.620, 0.621, 0.633, 0.626, 0.632, 0.632, 0.634, 0.622, 0.625, 0.620, 0.621, 0.627, 0.609, 0.626, 0.618, 0.614).
  - Epoch-to-epoch swings are 1 to 2 points, so the 0.35-point gap to A+B is noise. The best-dev snapshot takes epoch 22.
- **S3:** STAR B 0.6343 >= 0.6094 and ABCD B 0.8325 >= 0.8098, so both datasets come to this test.
- B's dev cost against bge-large: STAR -0.51 points (16,000 clauses), ABCD -0.73 (4,000). Both are inside R19.1's 2.0-point margin on dev; a dev loss of 2 to 3 points would have passed S3 but would likely fail R19.1.
- **Selection is optimistic:** B was picked among three views on STAR dev (and the dev numbers are best-epoch numbers), so the test may come out lower than dev.
- **Reorder** (decided on 2026-10-18): R19 is frozen before e61's STAR twin finishes, so as to win back the hours lost in the night's outage. No rule depends on e61.

## Arms (test)

**STAR** (`results/confirm/R19star/`), at R18's settings: 16,000 clauses, T 16,000, s 8, 40 epochs, start state half, best-dev snapshot, one seed per job.
- `tm_mem_static`: view B, seeds 1 to 5.
- `tm_nomem_static`: view B without mem (`words,xcust,xrecent`), seeds 1 to 5.
- `lr_mem_static`: LR on view B's bits, seed 0, C from the dev grid.
- References copied from R18 by `scripts/r19_refs.py` (seed json and npz, bit_names; no models): `tm_mem_R18`, `tm_nomem_R18`, `lr_mem_R18`.

**ABCD** (`results/confirm/R19abcd/`), at R9b's settings (4,000 clauses, T 4,000, otherwise as above):
- the same three arms, seeds 1 to 5 (LR seed 0);
- references copied from R9b (10 seeds): `tm_mem_R9b`, `tm_nomem_R9b`, `lr_mem_R9b`.

Commands: `logs/jobs_R19.txt` (22 jobs, STAR TM first; the report commands are comments). Dry runs before the freeze (`logs/jobs_R19_dry.txt`; `--dry-run`, dev stands in for test; tiny TM with 200 clauses and 2 epochs, LR in full): their bits_sha256 equal the e60 B rows (STAR train a3ce2dd9, dev bc27a3c6; ABCD train 1f81cb23, dev 2f3223b2). All six dry runs ran.

| Dry run | Bits (train / dev) | Matches |
|---|---|---|
| STAR, with mem | a3ce2dd9 / bc27a3c6 | the e60 B rows |
| ABCD, with mem | 1f81cb23 / 2f3223b2 | the e60 B rows |
| ABCD, without mem | 00b387d7 / 60498079 | the e61 ABCD row |
| STAR, without mem | 1862407b / 0a9bbc27 | no dev row yet; compared when e61's STAR twin lands |

The two LR dry runs reproduce the e60 LR dev numbers exactly: STAR 0.640851, ABCD 0.811075.

## Pass rules (test, per dataset; both must hold)

- **R19.1 (no network, little cost):** `tm_mem_static` minus the bge-large reference (`tm_mem_R18`, `tm_mem_R9b`) is at least -2.0 points on the point estimate. It is "strong" if the interval's lower end is at least -2.0.
- **R19.2 (the memory still pays):** `tm_mem_static` minus `tm_nomem_static` is at least +5.0 points, with the interval above 0.

Statistic as R18 and R9b: per test action turn, the mean correctness over seeds; differences with a bootstrap over test dialogues (611 STAR, 1,004 ABCD), 10,000 resamples, seed 0, 95% interval (`scripts/report_abcd.py`). The references enter the report with their own seeds (R18: 5, R9b: 10).

## Reported with no rule

- `lr_mem_static` minus `lr_mem` (bge-large); TM minus LR on the static bits; R18's own memory pair recomputed from the copies (it must give +15.43).
- Accuracy by position.
- The AST-weighted accuracy for ABCD. This task has 30 buttons, not the 231 button-slot options of the ABCD paper's B-Slot, so it is not comparable with the paper's 89.9 / 93.6.
- The speed of one decision on one CPU thread (`scripts/static_speed.py`, STAR dev, n = 1,000; rough: shared machine).
  - Smoke test with n = 50, medians: the bge-large agent 850 ms per decision, of which 824 ms is encoding the two texts. The view-B agent takes 29 ms, of which 0.47 ms is existsT and 28.7 ms the TM decision in numpy.
  - The CPU decision reproduces both dev accuracies exactly.

## Hashes

- Caches:
  - data/emb/star__potion-32m.npz: 6772c124fd997462101e08e7bcdd81fc9cc90a60eb88752f7847c62dc3bb618d;
  - data/emb/abcd__potion-32m.npz: cae24286daca768c4a245a44b1c6b78a9f1ac187f095e695e1e6ecff51207b81 (provenance inside each).
- Table: 5e33b584aad09dc4c80d8c8ece6dc98768af7a3f4fe3a0eabcf4895bba3e0487.
- Pinned files of potion-base-32M at 1e5a03f8eeb2c98b928fbbd846f22f816360919f: model.safetensors 99f6c33204c9231a7391871b7a3c91409b532c8f587a9ea44fc282303d8dec28, tokenizer.json 7d75cbc54318138807c401b0f0c9721117c628b39de8e8e0edb6cb17e0ee7d18.
- Example files:
  - STAR: train 47735157f3e0035c, dev e562736dd9afa1b7, test f9cc8f150b25d94d;
  - ABCD: train c9e0f9b58a13a3dd, dev 5ccc737f5be90860, test 190312c67b2258b4.
- Copied references: 70 files, every SHA-256 in logs/r19_refs.out (that list: 37aba477b868eb24f269d669162ef960784bf0f0de48a9c2a706c77b6a88af4c).

## Code changed since R18 (`scripts/diff_frozen.py registrations/R18_star_replication.frozen.json`)

**On R19's path:**
- src/dl/static_text.py (new), src/dl/cpu_tm.py (new);
- src/dl/abcd.py (the PARTS check and xcust, xrecent: additions only);
- scripts/abcd_confirm.py and scripts/abcd_explore.py (opt-in DL_GPU_FRAC, bits_sha256, static_info, peak GPU memory: additions only);
- scripts/embed.py (the potion-32m entry);
- scripts/static_prepare.py, static_diag.py, static_speed.py, r19_refs.py (new);
- scripts/master_queue.sh (the job count: all of this study's python jobs, and those waiting for the GPU lock).

**Not on R19's path** (phase-3 dev files): scripts/abcd_bandit.py (`--dataset`, 14:27 on 2026-10-17), abcd_episodic.py, star_explore.py, star_gap_diagnose.py, star_vote_norm.py, src/dl/bitcache.py, tm_balance.py, tm_chunkbudget.py.

**Unchanged:** src/dl/tm.py, tm_init.py, boolean.py, data.py, scripts/report_abcd.py, abcd_prepare.py, star_prepare.py.

Tests:
- tests/test_static_text.py (CPU part and the GPU part: the default bge-large views of the pristine and the new abcd.py are byte-identical);
- tests/test_cpu_tm.py.

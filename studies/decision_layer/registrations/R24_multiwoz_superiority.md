# R24 registration: a registered superiority test on a third dataset (MultiWOZ 2.2), the agent with no neural network at any stage against the equally tuned bge-large agent

**Status:** draft (2026-10-19 21:57 UTC), written before e77's dev runs; design fixed (21:58 UTC). The places marked <...> are filled from e77's
picks (`scripts/e77_pick.py`). It is frozen with `scripts/freeze.py` after the dry runs
and before any R24 test run.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged. This draft was never frozen.

**The gate before the freeze** (18:51 UTC on 2026-10-20, fixed before any e77b 16k number;
`logs/plan_e77_note_2026-10-20.md`): this registration is frozen and run only if the tuned 16k dev gap (C4 at s*
minus bge-large at s*, seed 1, `scripts/e77_pick.py --stage final`) is at least -1.0. Below that, R24 waits and
MultiWOZ's test split stays unread. The gap is then a dev finding (FAILURE_MAP.md, docs/results.md), and R24 returns
only when fresh dev runs of a later stack pass the same gate, with every dev try listed.

Follows R22 (STAR: +1.64 [+0.94, +2.35], non-inferiority registered) and R23 (ABCD: -0.92 [-1.61, -0.26]). Dev rules:
`logs/plan_e77_multiwoz_dev.md` (frozen 2026-10-19T21:43:42Z). Scope: `logs/plan_e77_third_dataset_scope.md`.

**The question.** On a dataset this study has never tested on, is the agent with no neural network at any stage
better than the bge-large agent, when each agent's s is tuned on that dataset's dev by the same rule? Neither R22 nor
R23 tested superiority.

**Use of the test split:** MultiWOZ 2.2 test, 1st use. Its texts were read twice before this registration, both text
only: by the overlap check of the text-understanding line (2026-10-19; below) and by the a3 table's fit filter (a3_mwoz_filter.py,
2026-10-20T19:22Z; removal only, 68 + 5 C4 sentences removed; corrected 2026-10-21 per logs/plan_e82_note2_2026-10-21.md).
Its labels have never been read.

**What was known at the freeze:** every e77 dev row (4k screens, 16k step), e77's LR dev number, the dev breakdowns, the
dry runs', and the overlap numbers. Nothing else from MultiWOZ test.

## Data

- MultiWOZ 2.2 (budzianowski/multiwoz at fe0c8e65, MIT; data/untrusted/multiwoz22, MANIFEST.sha256 87d8b2cd).
- One example per system turn with an act annotation. The label is the sorted domain-act set, with combinations
  under 20 train occurrences mapped to OTHER (128 labels).
- **Test examples:** `scripts/multiwoz_prepare.py --with-test` writes data/multiwoz/examples_test.json.gz and
  labels_test.json, after the freeze. The label list comes from train, as for dev.
- **Overlap with the table's text** (the text-understanding line, overlap.json 52cb74eb in the project notes, not published): 385 of 14,744 test
  utterances (2.61%) match a C4 sentence (159 verbatim, 232 by an 8-gram). Only 6 of 11,715 utterances of 8 or more
  words equal a C4 sentence. These are stock service phrases. It is a small, declared overlap, and the table is not
  rebuilt.
- bge-large's pretraining data cannot be checked. That is a limit.

## Dev rows behind it (MultiWOZ dev, seed 1; `results/explore/abcd.jsonl`; e77_pick.py)

| Agent | 4k s 8 | 4k s 12 | 4k s 16 | 4k s 24 | 4k pick | 16k s 8 | 16k pick | s* |
|---|---|---|---|---|---|---|---|---|
| C4 view B | <> | <> | <> | <> | <> | <> | <> | **<s*C4>** |
| bge-large, R18's view | <> | <> | <> | <> | <> | <> | <> | **<s*bge>** |

- The tuned dev gap at 16k (C4 at s* minus bge-large at s*): <>.
- LR on view B's bits (wide grid, dev): <> at C <>. The majority label: 7.16%.

## Newest work: the baselines today (searched 2026-10-19 21:59 UTC to 2026-10-20 00:30 UTC; `multiwoz_newest.md` in the project notes, not published)

- **Not comparable:** no paper reports this task, the next system act SET as one of 128 classes with accuracy. So R24
  is a comparison between this study's two agents only, and no published number is claimed as beaten.
- **The official leaderboard** (budzianowski/multiwoz README, last changed 2025-01-14): DiactTOD (2023) is on top for
  MultiWOZ 2.2, with end-to-end Combined 104.4, and 110.3 in the policy table with ground-truth states.
- **Newer Combined claims:** MORE 107.35 (KDD 2026), DIMF 106.3 (2025 preprint), SUIT 105.02 (Findings of EMNLP 2024,
  standardized evaluator).
- **Next-act prediction:** TOD-Flow (Dec 2023, older), average F1 48.2 over 14 domains with GPT-turbo plus its graph.
  It was checked on the paper's page.
- **LLM agents:** CoALM (ACL 2025), GPT-4o Success 75.5, zero-shot on 2.4.
- **Interpretable policies:** CoDial (ACL 2026), rule code, Combined 79.5 on 2.2. No Tsetlin machine on MultiWOZ was
  found.
- **Caveat:** scores differ across evaluators (Nekvinda and Dušek 2021).

## Arms (MultiWOZ test, `results/confirm/R24mwoz/`; 16,000 clauses, T 16,000, 40 epochs, start state half, best-dev snapshot)

- `tm_mem_c4_s<s*C4>`: the C4 agent (view B on the C4 table; table_run.py, `--encoder none`), seeds 1 to 5.
- `tm_mem_bge_s<s*bge>`: the bge-large agent (R18's view, `--encoder bge-large --parts mem,intent,step,words`),
  seeds 1 to 5. Its test vectors come from `scripts/multiwoz_embed.py --add-test`. That run copies the train and dev
  arrays from the dev-stage file byte for byte.
- `lr_mem_c4`: LR on the C4 arm's bits, seed 0, C from R21's wide grid (ties to the smaller C). No rule.
- **Not run on test:** each agent at s 8. Decided at 21:58 UTC: those arms carry no rule, and the claims
  do not need them. The tuning gain is reported from e77b only, labelled "dev, one seed". This drops "each agent
  against its own s 8" from the no-rule list of the scope and of the frozen e77 plan (a dated line in the scope).

## Pass rules (test)

- **R24.1, superiority (the headline):** `tm_mem_c4_s<s*C4>` minus `tm_mem_bge_s<s*bge>` is above 0, and the whole
  95% interval is above 0.
- **R24.2, non-inferiority:** the same difference is at least -2.0 on the point estimate. It is "strong" if the
  interval's lower end is also at least -2.0.
- **Statistic,** as R21-R23: per test turn, each arm's correctness is averaged over its seeds. The difference's 95%
  interval comes from a bootstrap over the test dialogues (the number is counted at R24, after the freeze), 10,000 resamples, seed 0 (`scripts/report_abcd.py`).

## Reported with no rule

- TM minus LR on the same bits.
- Each arm against the majority label.
- First action against later actions, per arm.

## Wording

- **If R24.1 is met:** "On MultiWOZ 2.2, a third dataset (its test split's first use), the agent with no neural
  network at any stage is better than the equally tuned bge-large agent: +X [lo, hi]". Beside it go the STAR and ABCD
  results (+1.64 and -0.92), so the claim is for MultiWOZ only.
- **If R24.1 is not met:** it is reported as it is, with R24.2's outcome. "Better than" is not said.
- Never "beats the transformer" in general, and never "state of the art".

## Commands, checks, dry runs and hashes (filled before the freeze)

- **Jobs:** logs/jobs_R24.txt: the C4 seeds first, then bge-large, then LR. They run one at a time through
  scripts/seq_queue.sh, with its own log logs/R24_queue.log and cpu_load at most 4.5 cores. They run alone on the
  GPU (big registered jobs run alone).
- **Before the jobs, after the freeze:** `multiwoz_prepare.py --with-test` (CPU), then `multiwoz_embed.py --add-test`
  (GPU, minutes).
- **Dry runs:** tiny TM (200 clauses, 2 epochs), dev standing in for test, both arms and LR. Their train and dev bits
  must equal the e77 dev rows'.
- **Checker:** scripts/r24_check.py, adapted from r23_check.py, written before any test number. It checks:
  - the freeze;
  - the job set;
  - the cfg against the e77b rows;
  - the bits (train and dev equal the dry runs' and the dev rows'; one test hash per arm);
  - seed 1's dev accuracy equals its e77b row exactly;
  - the npz/json agreement and equal test rows;
  - examples_test written after the freeze, with train's label list;
  - the bge-large file's train and dev arrays equal to the dev-stage file's.

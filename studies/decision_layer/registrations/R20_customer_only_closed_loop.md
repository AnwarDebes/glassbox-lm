# R20 registration: the agent in a closed loop (STAR), and the agent that reads only the customer (ABCD)

**Outcome before the freeze: the question named as the headline was stopped at its dev gate.** R20.4 was not taken to STAR test. Arm 2 failed its dev gate S4 (+4.50 against +5.0, seed 1), so by the frozen plan it is a dev-only failure-map line. What remains for test:
- arm 1 on STAR test (R20.1 to R20.3, reported one by one);
- arm 2 on ABCD (R20.4), as supporting evidence only.

Follows R19 (met: the agent with no neural network at run time, open loop). Every gate is checked a second time, independently. Dev rules: `logs/plan_e65.md` (frozen 2026-10-18T20:23:31Z, sha256 a19d3f62e34580b0). The kinds of rule and the main question were fixed before e65 landed (20:55 UTC). The values were set after e64 to e67, and every one of their numbers is in "What was known at the freeze".

**The question.** R19's agent decides with no neural network at run time, but open loop: its memory bits are the logged presses, and it reads text that the human agent (ABCD) or the wizard (STAR) wrote. R20 asks:
- **Arm 1 (STAR, R19's and R18's registered models):** in a closed loop, with its own presses as memory, does the memory still pay? What do its own mistakes cost? How does the word-table agent compare with R18's bge-large agent in the same loop?
- **Arm 2 (the customer-only agent):** can the agent decide from the customer's words and its own earlier presses alone, in a closed loop, with no text that the human agent or the wizard wrote? Does memory still pay there?

**The main question**, named before e65's STAR rows landed, was R20.4 on STAR. It was stopped at its dev gate (the outcome above), so no test verdict exists for it.
- **Use of the test splits:** STAR's 4th (arm 1 only); ABCD's 12th (arm 2).
- R20.4 on ABCD was named as supporting evidence before STAR's rows landed. It is now arm 2's only test, and it stays supporting evidence.
- There is no combined "R20 met" line.

**The closed loop** (`scripts/abcd_closed_loop.py`, as in R13 and R17). In every test conversation the action turns are decided in order. At turn k the memory bits (`done(b)`, `last action = b`, `at least j actions so far`) come from the model's own choices at turns 0 to k-1. The text stays as logged. Accuracy is still against the logged button.

**View C (arm 2): `--parts mem,cwords,xcust,xcrecent --encoder potion-32m`**, fixed in `logs/plan_e65.md` before any e65 run. It is R19's view B with all human-agent and wizard text removed:
- mem: the memory bits (in the closed loop, the agent's own presses);
- cwords: the 1,000 most frequent 1-2-grams of the customer's last 3 utterances (`customer_recent` in `src/dl/abcd.py`);
- xcust: existsT bits of the customer's words so far, as in R19;
- xcrecent: existsT bits of the customer's last 3 utterances.

`tests/test_customer_view.py` checks that view C reads no human-agent text: replacing every row's `recent` and `full` text (both speakers) leaves its bits unchanged.

What runs per decision in arm 2:
- a rule-based tokenizer;
- lookups in the fixed 63,091 x 512 table (potion-base-32M);
- a bitwise OR of precomputed per-token bit codes;
- the customer's 1-2-gram bits and the memory bits from the agent's own presses;
- Boolean clauses and an integer vote.

No neural network runs at decision time. The table was distilled once, offline, from bge-base-en-v1.5 by Model2Vec.

**Limit (fixed wording).** The claim is "no text written by the human agent", not "no information about the human agent's actions". The customer's logged replies answer what the human agent really did and said. So they still carry information about the human agent's real actions, also at turns where the agent's own press differs from the logged one.

**What was known at the freeze:**
- the test numbers of:
  - R13 (ABCD closed loop, bge-large): TM-mem closed 0.8070, open 0.8401, TM-nomem 0.6537; closed minus open -3.31 [-3.95, -2.71];
  - R18 (STAR): TM-mem 0.6265, TM-nomem 0.4722, LR-mem 0.6591;
  - R19: STAR TM-mem 0.6271, TM-nomem 0.4766, LR-mem 0.6392; ABCD 0.8301, 0.6222, 0.8063;
- R19's post-hoc numbers (the no-memory pairs; STAR's LR refit at C 0.01, test 0.6493);
- every dev number of e64, e65, e66 and e67 (tables below), and the dry runs'.

Frozen by `scripts/freeze.py` before any R20 test run.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## Dev rows behind it

**e66: arm 1 on STAR dev** (`results/explore/abcd_closed_loop.jsonl`; open = logged memory, closed = own presses):

| Models | Open | Closed | Closed minus open |
|---|---|---|---|
| R19 tm_mem_static (word table), seeds 1 to 5 | 0.6345 | 0.5588 | -7.57 (per seed -8.14 to -6.82) |
| R19 lr_mem_static (C 0.1) | 0.6409 | 0.5860 | -5.49 |
| R18 tm_mem (bge-large), seeds 1 to 5 | 0.6395 | 0.5454 | -9.40 (per seed -10.05 to -8.95) |
| R18 lr_mem | 0.6657 | 0.5808 | -8.50 |

- Memory in the closed loop (the dev form of R20.1): 0.5588 against the twin without memory (e61, seed 1) 0.5010, +5.78.
- The word table against bge-large in the closed loop (the dev form of R20.3): +1.34.
- Own mistakes cost far more on STAR than on ABCD (e64: -2.77). R13.2's bar of -4.0 fails on STAR dev for both encoders.

**e64: R19's ABCD models in the closed loop on dev** (no rule): TM seeds 1 to 5 open 0.8318, closed 0.8041 (-2.77, per seed -2.93 to -2.61); LR open 0.8111, closed 0.7872 (-2.39).

**e65: view C and its twin without memory** (seed 1, R19's settings; `results/explore/abcd.jsonl` and `abcd_closed_loop.jsonl`):

| | STAR (16,000 clauses) | ABCD (4,000 clauses) |
|---|---|---|
| C: bits (train / dev hash) | 8,155 (2ab62d2d / 4bc07533) | 7,488 (78cfda79 / b3601375) |
| C: TM open (best epoch) | 0.5779 (14) | 0.7872 (20) |
| C: TM closed | 0.4767 | 0.7424 |
| C: own mistakes | -10.12 | -4.48 |
| C: LR open (C 0.1) | 0.6019 | 0.7600 |
| Twin without memory: bits | 7,775 (c0126c55 / 99e5bd17) | 7,422 (c4d98138 / c80a2bca) |
| Twin: TM (best epoch) | 0.4317 (26) | 0.4984 (40) |
| Twin: LR (C 0.1) | 0.4645 | 0.5350 |
| View B for comparison (R19, seed 1): open / closed / no memory | 0.6343 / 0.5530 / 0.5010 | 0.8325 / 0.8046 / 0.6200 |

- C's closed loop by position:
  - ABCD: 0.875, 0.797, 0.697, 0.610, 0.607, 6th and later 0.352;
  - STAR: 0.772, 0.630, 0.554, 0.489, 0.426, 6th and later 0.403.
- Dropping the human agent's or wizard's text costs, against view B:
  - ABCD: 4.53 points open loop, 6.22 closed loop, 12.16 without memory;
  - STAR: 5.65, 7.63 and 6.93.
- **The twins' dev curves are noisy** (every 2 epochs):
  - ABCD: 0.441, 0.465, 0.462, 0.476, 0.480, 0.476, 0.462, 0.490, 0.455, 0.466, 0.463, 0.482, 0.451, 0.497, 0.460, 0.441, 0.423, 0.387, 0.466, 0.498. Its best epoch (40) is noise, not a rising curve.
  - STAR: 0.397, 0.431, 0.419, 0.423, 0.424, 0.430, 0.412, 0.425, 0.409, 0.420, 0.404, 0.428, 0.432, 0.401, 0.425, 0.417, 0.415, 0.392, 0.386, 0.362.
  - The best-dev snapshot takes the highest point of a noisy curve, so the twin's number is optimistic. The size of the memory gain rests on a noisy twin; R20.4's sign does not.
- **S4** (frozen: arm 2 goes to a dataset's test only if C's closed-loop dev accuracy is at least 5.0 points above its twin's):
  - ABCD: 0.7424 - 0.4984 = **+24.40, passed** (checked independently, 22:32 UTC).
  - STAR: 0.4767 - 0.4317 = **+4.50, not passed**.
  - STAR's failure-map line (dev only): the customer-only agent on STAR in the closed loop. Without the wizard's text, a wrong press poisons the memory: the closed-loop memory gain falls from +14.6 (open loop) to +4.5, below the +5.0 gate, and its own mistakes cost 10.12 points.
  - S4 is applied as written, with the best-dev snapshot. The twin's typical level (about 0.42) would give about +5.7, but no extra seeds or epochs were run to recheck it.

**e67: LR on view B below the registered C grid** (dev; post hoc, after R19):

| C | 0.1 | 0.03 | 0.01 | 0.003 | 0.001 |
|---|---|---|---|---|---|
| STAR dev | 0.6409 | 0.6501 | 0.6555 | 0.6445 | 0.6209 |
| ABCD dev | 0.8111 | 0.8122 | 0.8086 | 0.7910 | 0.7503 |

So arm 2's LR searches a wider grid (below). Its dry run on ABCD picks C 0.03, dev 0.7666 (C 0.1: 0.7600).

## Arms (test)

**STAR** (`results/confirm/R20star/`), arm 1 only. No training: R19's and R18's registered models, `scripts/abcd_closed_loop.py --split test --out R20star`.
- `tm_mem_static_closed`: R19star tm_mem_static (view B), seeds 1 to 5, closed loop.
- `lr_mem_static_closed`: R19star lr_mem_static (C 0.1), closed loop.
- `tm_mem_closed`: R18 tm_mem (bge-large), seeds 1 to 5, closed loop.
- `lr_mem_closed`: R18 lr_mem, closed loop.
- References copied by `scripts/r20_refs.py` (seed json and npz, bit_names; no models): `tm_mem_static_R19`, `tm_nomem_static_R19`, `lr_mem_static_R19` (R19star, open loop; the twin has no memory bits, so its open and closed loop are the same) and `tm_mem_R18` (R18, open loop).

**ABCD** (`results/confirm/R20abcd/`), arm 2 (new models, `scripts/abcd_confirm.py`, R19's ABCD settings: 4,000 clauses, T 4,000, s 8, 40 epochs, start state half, best-dev snapshot, one seed per job):
- `tm_mem_cust`: view C, seeds 1 to 5 (open-loop test predictions). Each seed's job then runs `tm_mem_cust_closed`, its closed loop on test.
- `tm_nomem_cust`: view C without mem (`cwords,xcust,xcrecent`), seeds 1 to 5. It has no memory bits, so its open and closed loop are the same.
- `lr_mem_cust`: LR on view C's bits, seed 0. C is chosen on dev from 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100 (`--grid-c`), ties going to the first, the smaller C. Then `lr_mem_cust_closed`.
- No rule: `tm_mem_static_closed` (R19abcd tm_mem_static, view B, seeds 1 to 5, closed loop).
- References copied: `tm_mem_static_R19`, `tm_nomem_static_R19` (R19abcd, open loop) and `tm_mem_closed_R13` (R13's closed-loop arm: R9b's 10 bge-large models).

## Pass rules (test)

Statistic as R9b, R13, R18 and R19: per test action turn, the mean correctness over seeds; differences with a bootstrap over test dialogues (611 STAR, 1,004 ABCD), 10,000 resamples, seed 0, 95% interval (`scripts/report_abcd.py`). Each arm enters with its own seeds.

- **R20.1 (arm 1, STAR: memory still pays in the closed loop):** `tm_mem_static_closed` minus `tm_nomem_static_R19` has its interval above 0. No size bar.
- **R20.2 (arm 1, STAR: its own mistakes cost at most 4 points).** R13.2 in its exact form (`registrations/R13_closed_loop.md`, line 35): "TM-mem-short closed loop minus TM-mem-short open loop is at least -4.0 points." Here: `tm_mem_static_closed` minus `tm_mem_static_R19` is at least -4.0 points on the point estimate.
  - **Expected to fail.** On dev it is -7.57, and every seed is below -6.8. The bar is kept at R13's value and not loosened. A failure goes into the failure map.
- **R20.3 (arm 1, STAR: the word table in the closed loop):** `tm_mem_static_closed` minus `tm_mem_closed` (R18's bge-large agent in the same loop) is at least -2.0 points on the point estimate (R19.1's margin). It is "strong" if the interval's lower end is also at least -2.0.
- **R20.4 (arm 2, ABCD: memory pays for the customer-only agent in the closed loop):** `tm_mem_cust_closed` minus `tm_nomem_cust` has its interval above 0. No size bar.

## Reported with no rule

The full list, fixed now: the arms and pairs of the two report commands in `logs/jobs_R20.txt`. Nothing is added after the test read. For every arm the report gives accuracy, seed sd, accuracy by position and the AST-weighted accuracy (on ABCD, the paper's B-Slot measure).

STAR (`report_R20star.json`):
- TM against LR in the closed loop: `tm_mem_static_closed` minus `lr_mem_static_closed` (R19's LR at C 0.1, under-tuned on STAR: e67);
- what own mistakes cost R18's agent and R19's LR: `tm_mem_closed` minus `tm_mem_R18`, `lr_mem_static_closed` minus `lr_mem_static_R19`;
- the arm `lr_mem_closed` (R18's LR in the closed loop).

ABCD (`report_R20abcd.json`):
- dropping the human agent's text: `tm_mem_cust_closed` minus `tm_mem_static_closed`, `tm_mem_cust` minus `tm_mem_static_R19`, `tm_nomem_cust` minus `tm_nomem_static_R19`;
- own mistakes: `tm_mem_cust_closed` minus `tm_mem_cust`; `lr_mem_cust_closed` minus `lr_mem_cust`; `tm_mem_static_closed` minus `tm_mem_static_R19`;
- memory with view B in the closed loop: `tm_mem_static_closed` minus `tm_nomem_static_R19`;
- TM against LR: `tm_mem_cust_closed` minus `lr_mem_cust_closed`; `tm_mem_cust` minus `lr_mem_cust`;
- the full swap, no neural network and no human-agent text against bge-large with that text, both in the closed loop: `tm_mem_cust_closed` minus `tm_mem_closed_R13`.

**Why the cost of dropping the human agent's text carries no rule.** `logs/plan_e65.md` listed it as a planned kind of rule. By the time the values were set, its dev size was known (ABCD: -6.22 in the closed loop), so any margin would have been fitted to it. It is reported with its interval instead.

## Commands, checks and dry runs

- Jobs: `logs/jobs_R20.txt` (sha256 ff98fdb78ae18b8a). 16 job lines; the check and the two report commands are comment lines at its end.
  - First the five closed loops of existing models (minutes), then ABCD's arm 2. Cap 3 over all of this study's jobs.
  - Every `tm_mem_cust` and `lr_mem_cust` job runs its own closed loop on test right after training (`&&`).
  - A job that crashes is rerun once with the same command, and the amendment says so. Nothing else is rerun.
- After all runs, `scripts/r20_check.py` (written before this freeze, so its hash is frozen) checks the runs before any report is read:
  - the sources against the freeze;
  - the seed files against the job list (11 training and 23 closed-loop seeds), all written after the freeze;
  - the TM cfg against R19's (all but the seed);
  - LR's grid, C and dev accuracy against the dry run's;
  - the train and dev bit hashes against e65's rows, and seed 1's dev accuracy exactly;
  - one test-bit hash per view, and bit_names against the dry runs';
  - each closed-loop seed's open-loop accuracy against the same model's open-loop arm;
  - the copies against `logs/r20_refs.out` and their sources;
  - equal test rows in every arm.
  - A mock test of the check passed on this job list: a fake tree in a temporary folder with simulated outputs passes, and each of six planted faults fails.
- Dry runs (`logs/jobs_R20_dry.txt`; `--dry-run`, dev stands in for test; tiny TM with 200 clauses and 2 epochs, LR in full):

| Dry run | Bits (train / dev) | Matches |
|---|---|---|
| ABCD, view C (TM) | 78cfda79 / b3601375 | e65_abcd_C_c4000 |
| ABCD, twin (TM) | c4d98138 / c80a2bca | e65_abcd_Cnomem_c4000 |
| ABCD, LR on view C (wider grid) | 78cfda79 / b3601375 | e65_abcd_C_c4000; C 0.03, dev 0.766558 |
| STAR, view C (TM) | 2ab62d2d / 4bc07533 | e65_star_C_c16000 (arm 2 is not run on STAR) |
| STAR, twin (TM) | c0126c55 / 99e5bd17 | e65_star_Cnomem_c16000 (not run on STAR) |

  - STAR's LR dry run was stopped at 23:13:39 UTC, once S4 had failed on STAR.
  - The closed-loop path for view C: `abcd_closed_loop.py --split dev` on the dry-run models (tag R20_dry) gives open-loop accuracies equal to the dry runs' (ABCD TM 0.283659, ABCD LR 0.766558, STAR TM 0.070353). So it rebuilds view C's bits and the models exactly.
  - The default path of the changed `abcd_confirm.py`: two R19 dry runs rerun with default flags give the stored outputs. They match in every json field (all but name, time, peak GPU memory and training seconds), the bit_names bytes and the predictions, and also in the TM's class sums and model arrays and the LR's weights (C 0.1, dev 0.811075, as stored). With the flag unset, the json has no grid field.

## Hashes

- Caches, table, pinned potion-base-32M files and example files: as R19 (unchanged; `registrations/R19_no_network_agent.md`, "Hashes").
- e65 bits:
  - ABCD: C 78cfda79d077f0a4 / b3601375905f36bc, twin c4d9813803732dfc / c80a2bca3fcab463;
  - STAR (not run on test): C 2ab62d2d37a0609a / 4bc0753309077743, twin c0126c550cb1eaf1 / 99e5bd17b621e2ba.
- Copied references: 78 files, every SHA-256 in `logs/r20_refs.out` (that list: e6c255db13df1816bd68940a1d660cc136cf428303ccd4049b950137f18d81cb).

## Code changed since R19 (`scripts/diff_frozen.py registrations/R19_no_network_agent.frozen.json`)

**On R20's path** (pristine copies in `logs/pre_R20/`, equal to R19's frozen hashes; additions only):
- `src/dl/abcd.py` (a1bcb1d4 to a66c0c63): the parts cwords and xcrecent, and `customer_recent` (`logs/pre_R20/diff_abcd.txt`: -0 / +25 lines).
- `scripts/abcd_closed_loop.py` (388bc3f1 to ad3721be): `--dataset` (`diff_abcd_closed_loop.txt`: -0 / +6).
- `scripts/abcd_confirm.py` (a32b7fa2 to 0ae66772): `--grid-c`, and the grid in the json when it is set (`diff_abcd_confirm.txt`: -0 / +5).
- New: `scripts/r20_refs.py`, `scripts/r20_check.py`.

**Not on R20's path** (new since R19, dev and post-hoc work): `src/dl/cpu_tm_fast.py`, `scripts/closed_loop_explore.py`, `decision_speed.py`, `lr_c_check.py`, `lr_refit_posthoc.py`, `r19_check.py`.

**Unchanged:** `src/dl/tm.py`, `tm_init.py`, `boolean.py`, `data.py`, `static_text.py`, `linear.py`, `scripts/report_abcd.py`, `abcd_prepare.py`, `star_prepare.py`.

Tests:
- `tests/test_customer_view.py`:
  - the old views are byte-identical, the default bge-large views included (STAR and ABCD, on the GPU);
  - view B keeps R19's hashes;
  - view C reads no human-agent or wizard text.
- `tests/test_cpu_tm_fast.py`: the fast CPU decision used on dev; not on R20's test path.

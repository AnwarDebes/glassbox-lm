# R4: amendment written 2026-10-15 at 22:52Z, before any confirmatory run of R4

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged.

## Selections applied (rules of the frozen text)

**TM-single setting.** Val accuracy of the candidates (CLINC150, MiniLM, seed 1, `results/explore/tm.jsonl`):

| Candidate | Val accuracy |
|---|---|
| `repro_cdllm` | 0.9487 |
| `e1_thermo16` | 0.9500 |
| `e1_thermo8_c8k` | 0.9500 |
| `e1_thermo8_e80` | 0.9513 |
| `e5_ml4` | 0.9467 |
| `e5_ml8` | 0.9467 |
| `e5_ml16` | 0.9483 |
| `e5_cd25` | 0.9447 |
| `e5_cd50` | 0.9420 |
| `e5_s4` | 0.9440 |
| `e5_s16` | 0.9353 |

- The best is `e1_thermo8_e80`. No other candidate is within 0.10 points of it.
- **TM-single = thermo:8, 4,000 clauses, T 4,000, s 8, 80 epochs, batch 128, hardest negatives, no budget, no clause drop.**
- `repro_cdllm` ran on the first code version, which used "<=" in the clause selection; all others ran on the current "<". This does not change the choice.

**TM-readout arm.** In `e4_readout` the TM reached val 0.9457 and the better read-out ("clauses+bits") 0.9540, a gain of 0.83 points. The rule needs 0.5, so **the arm is included.** It runs with `--readout`, on the clause outputs of each seed's TM-single machine.

## Source changes since the freeze (`scripts/diff_frozen.py`)

- **`src/dl/tm.py`:**
  - field `budget_mode` (default "exact"); its branches are used only with a budget, and TM-single has none;
  - `stats()` has `p95_literals`.
- **`src/dl/linear.py`:** new `recall_at_precision`, reported without a rule.
- **`scripts/confirm_router.py`:**
  - options `--budget-mode`, `--lr-input` (default raw, as registered) and `--readout`;
  - writes the out-of-scope recall at 95% precision for both LR and TM, reported without a rule.
- **`src/dl/mdp.py`, `scripts/r6_run.py`, `scripts/tm_explore.py`:** not used by R4.
- No change touches the TM training rule or the LR fit used by the registered arms.

## Note added 2026-10-16 (internal audit 1)

- **Read-out variant level.** The frozen text says "the better of the two by val", without saying whether per seed or per cell. `scripts/rules_check.py` chooses per cell: the variant ("clauses" or "clauses+bits") with the larger sum of val accuracies over that cell's seeds. Val decides, never test.
- **The "clauses+bits" variant** gives weights to the TM's clause outputs and also to every input bit. It is a weighted vote of TM clauses and single-bit terms, not the TM's own vote. The write-up names it that way.
- **Interim look (corrected after internal audit 2).**
  - `rules_check.py` was run on partial R4 results before this note (first run 01:57:31Z; logged in the project notes, not published).
  - The per-cell selection of the read-out variant was in the file's first version. The edit at 02:02:48Z added the read-out-against-LR interval and the "ahead" flags, but did not change the selection.
  - The job log `logs/R4.out` had already printed both variants' test accuracies for every finished seed before this note was written. So the per-cell rule was not chosen blind to test numbers, though it was the rule already in the code.
  - Sensitivity: with the variant chosen per seed, the one "ahead" cell (Banking77 / bge-base) is +0.32 [-0.11, +0.75] instead of +0.45 [+0.02, +0.90]. The write-up reports both.

# e87: receipts, how readable the C4 agent's decisions are and what short rules cost, against LR and L1-LR on the same bits (dev only)

Written 2026-10-23, before any e87 number, with four fixes to the outline made before the freeze. No test split is read.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

**The questions (fixed now):**
- **Part A (no rule).** How short are the receipts behind the C4 agent's dev decisions, against LR and L1-LR on the same
  bits?
- **Part B (rules).** Can the C4 agent, trained with clauses of at most 3 literals, keep its dev accuracy within 2.0
  points of LR while 90% of its decisions have a receipt of at most 3 clauses of at most 3 literals? Can it also stay
  clear of L1-LR at the same budget?

**Why:** the lab report (project notes, not published; f79daa9e, sections 1 and 5) makes two points.
- The network-free agent and the MILP verifiability are not shown to be specific to the TM. LR on the same bits
  matches or beats it (R22 -0.09; R24 -1.55 to -2.49), and LR was never compared for verification.
- On the lab's text tasks (not this agent), the TM's clauses were long and weak: median about 270 literals, held-out
  precision about 0.59.
- So readable decisions, the claim that would be the TM's own, are unmeasured for the agent. e87 applies the lab's S1
  test to the agent.

## Part A

- **A0 (GPU, about 5 min, a slot in the GPU queue):** `scripts/e87_bits.py --dataset <ds>`, once for each of
  abcd, star and multiwoz. It builds the C4 view as table_run.py and abcd_explore.py do:
  - count_text.patch on c4words-S1000-v1;
  - ABCDView with mem, words, xcust and xrecent, lex_k 1000, seed 0, fitted on train, encoder none.
  - It refuses unless the train and dev bits' SHA-256 equal the registered arm's (bits_sha256 in seed1.json of R24mwoz,
    R22star or R23abcd).
  - It refits the L2 LR at the C of the dataset's LR dev row: dry_e77mwoz_lr_mem_c4 C 0.001, dry_R21star_lr_mem_c4
    C 0.01, dry_R21abcd_lr_mem_c4 C 0.03. It refuses unless that gives exactly the row's dev accuracy.
  - It writes results/explore/e87/<ds>_c4.npz.
- **A (CPU; 2 threads, nice 19, the CPU gate):** `scripts/e87_readability.py --dataset <ds>`, writing
  results/explore/e87/readability_<ds>.json.
- **The models (provenance).** All were trained on train and kept at their best-dev snapshot:
  - MultiWOZ: R24's tm_mem_c4_nrand, seeds 1-5 (16k, s 8, random negatives);
  - STAR: R22's tm_mem_c4_s16, seeds 1-5 (16k, s 16, hardest);
  - ABCD: R23's tm_mem_c4_s16, seeds 1-5 (4k, s 16, hardest).
  - Only model_seed<k>.npz is read, and from seed<k>.json only dev_acc and bits_sha256. No test file is read, and no
    test prediction: seed<k>.npz holds test predictions and is never opened.
  - Each seed's recomputed dev accuracy must equal its run's dev_acc exactly, or A stops.
- **LR:**
  - the L2 LR from A0;
  - L1-LR: sklearn 1.9.1 LogisticRegression with l1_ratio 1.0, solver saga, max_iter 200, tol 1e-3 and random_state 0,
    fitted on train with C in {0.01, 0.03, 0.1, 0.3, 1}.
  - Every C is reported. "Chosen on dev" means the most accurate, with ties going to the smaller C.

### Definitions (in class-sum terms)

- **Firing:** a clause fires on x when every included literal holds (a plain literal j: x_j = 1; a negated one:
  x_j = 0). An empty clause never fires, as in CoTM.clause_outputs at inference.
- **Class sums:** S_y is the sum of w[k, y] over the firing clauses k.
- **Decision and runner-up:** the decision c is argmax S, with ties going to the lower index. The runner-up r is the
  argmax over the other classes.
- **TM receipt:**
  - The candidates are the firing clauses with d_k = w[k, c] - w[k, r] > 0, ranked by d_k (descending; ties: the lower
    clause index first).
  - The receipt is the shortest prefix, at most 20 clauses, whose sums alone (w summed over the prefix) give argmax c.
  - Its size is counted in clauses, in total literals (plain + negated) and by its longest clause.
- **LR receipt:** for the L2 LR and for L1-LR at each C, each with its own c and r.
  - The candidates are the active bits with d_j = W[j, c] - W[j, r] > 0, ranked the same way.
  - The receipt is the shortest prefix, 0 to 20 bits, whose scores alone give argmax c. The intercept is included:
    b plus the sum of the prefix's W rows.
  - Its size is its number of bits, which equals its literals.
- **Equal budget:** a receipt's size is its total literals, for the TM and for LR alike.
  - Coverage at k is the share of dev decisions with a receipt of at most k literals, for k = 1 to 10.
  - S1's point: at most 3 clauses of at most 3 literals each for the TM; at most 9 bits for LR.
- **Clause precision:** for each clause that fires on at least 10 dev rows, the share of those rows whose label is the
  clause's strongest class (argmax over w[k]; ties to the lower index).

### Reported, per dataset (no pooled headline), per seed and as the mean over seeds; no rule

- The coverage curves (k = 1 to 10) for the TM, LR and L1-LR at each C. Also S1's point, and the share of decisions
  with no receipt within 20 items.
- **Clause statistics:**
  - literals per clause, plain and negated;
  - the share of clauses with only negated literals;
  - firing clauses per decision, and their literals;
  - clause precision (median and quartiles).
- **Accuracy with only short clauses:** dev accuracy using only the clauses of at most 3, and at most 9, literals. This
  is a short rule set made after training.

### Caveats (said with every Part A number)

- Dev chose each TM's snapshot epoch and LR's C, so dev is not fully held out. Clause precision on dev is optimistic.
- A receipt explains the model's decision, right or wrong.
- "Readable" here means short receipts. It is not a reading test with people.

## Part B (GPU, after its review and a slot in the GPU queue)

- **The cells:** the C4 agent on each dataset, at its size, s and negative rule:
  - MultiWOZ: 16k, s 8, random negatives;
  - STAR: 16k, s 16, hardest;
  - ABCD: 4k, s 16, hardest.
  - All three cells: max_literals 3, budget_mode "tmu", seeds 1-3, 40 epochs, start half, the best-dev snapshot, the
    model saved.
- **Why "tmu":** all three cells exceed tm.py's CHUNK_ELEMS (50M clause-literal pairs), so the update runs in chunks.
  Its "exact" budget raises there (line 315); "tmu" runs (lines 303-307).
- **"tmu" is a soft budget,** so a clause can end training above 3 literals.
  - The final clauses' literal-count distribution is reported.
  - The registered reading scores with every clause over 3 literals pruned (its weights set to 0). Unpruned scores are
    reported with no rule.
- **The bar (S1's, per dataset):** on at least 2 of 3 seeds, the pruned agent meets both conditions:
  - its dev accuracy is at least the L2 LR's dev accuracy (A0) minus 2.0;
  - its coverage at S1's point is at least 0.90.
- **The kill rule (per dataset):** "the TM adds nothing at this budget on <dataset>" if L1-LR at the same budget comes
  within 0.5 points.
  - L1-LR at the same budget is the most accurate grid C whose coverage at 9 bits is at least 0.90.
  - The rule fires if that L1-LR's dev accuracy is at least the pruned agent's mean over seeds 1-3, minus 0.5.
  - **The extension, fixed now (a fix made to the reviewed draft 0cd1a68e before the freeze):** if no grid C reaches
    0.90 at 9 bits, C 0.003, 0.001, 0.0003 and 0.0001 are added in that order. The extension stops at the first C that
    reaches 0.90, which is then the comparator. It also stops once a C's dev accuracy falls below the pruned agent's
    mean minus 5 points.
  - Only if no C reaches 0.90 is it written "L1-LR does not reach the budget". The kill rule is then "not testable",
    never "not killed".
  - The rule is scripts/e87_readability.py's l1_at_budget, tested with these cases.
- **What follows:**
  - **Met and not killed on a dataset:** a registered test may follow. It needs its own registration and a declared
    test use.
  - **Not met:** FAILURE_MAP gets "short receipts cost the agent its accuracy on <dataset>", with the numbers.
  - **Killed:** FAILURE_MAP gets "readable receipts are not specific to the TM on <dataset>".
- **Before B runs:** a dated note, frozen before any B job, with its tests and a dry run, adds:
  - the job lines;
  - the knob for budget_mode (scripts/knob_run.py has no --budget-mode; abcd_explore.py's --max-literals keeps
    budget_mode "exact");
  - the pruned scoring.
  - The rules above do not change.
- **Not in e87:** certification of single rules (S1's Benjamini-Yekutieli step). e87 makes no claim about a single rule.

## Checks and commands

- **tests/test_e87_readability.py PASSED** (logs/test_e87_readability_2.out; the first run, logs/test_e87_readability.out,
  was before the kill-rule fix). It covers:
  - Part B's kill rule with its extension: the most accurate C at the budget, killed at exactly -0.5; the extension
    stopping at the first C that reaches 0.90, or below the pruned agent minus 5; and "not testable";
  - firing with plain and negated literals and an empty clause;
  - TM receipts of one clause, of two (where the strongest clause alone favours a third class), with ties to the lower
    index, and with none within the cap;
  - LR receipts from the intercept alone and from two bits;
  - the coverage curve and S1's point;
  - clause precision with its 10-firing floor;
  - end to end on made-up files, including the dev-accuracy check, its refusal and no overwrite.
- **scripts/e87_bits.py has no unit test.** Its view and LR need the GPU. Its run refuses unless the bits equal the
  registered hashes and LR reproduces its dev row exactly; before the run it is only parse-checked. A0's outputs serve
  as Part A's dry run.
- **The commands**, one at a time:
  - A0 (GPU): `HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 DL_GPU_FRAC=0.25 .venv-dl/bin/python
    scripts/e87_bits.py --dataset <ds> >> logs/e87_bits.out 2>&1` for abcd, star and multiwoz;
  - A (CPU): `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 19 .venv-dl/bin/python
    scripts/e87_readability.py --dataset <ds> >> logs/e87_readability.out 2>&1`, only when `scripts/cpu_load.py 10`
    shows at most 4.5 cores.
- **Not measured yet:** the L1-LR fit times. saga on MultiWOZ's 56,581 x 6,580 train bits could take minutes to an
  hour per C. They are logged.

## Hashes (sha256sum, read 2026-10-23T01:49:54Z)

| File | SHA-256 |
|---|---|
| scripts/e87_bits.py | 3ad8dfae2d10c1d3be2a41994e6818da15aed58e823ae018ae43346913ba62d7 |
| scripts/e87_readability.py (read 01:53:07Z, after the kill-rule fix) | f156fcd10f798bac66a305f9b91773f8d48eb6df5c06cb21369ffcc61479b3a2 |
| tests/test_e87_readability.py (read 01:53:07Z) | 7365210eebd1dc3ffacd422ce96f8b6837e56a5b6dcd4bd2ce243099f67cd389 |
| logs/test_e87_readability_2.out (PASSED, with the kill-rule cases) | 8ee99d270b2eb35f44dcbf4667d6321736d10a838af472b00d6b4e4e85bc7b0c |
| logs/test_e87_readability.out (PASSED, before the fix) | 4719c291b540732f67c2796882100877cef5bf338d156ef35d8a38322c96f70d |

The models', the LR rows' and the input files (sha256sum, read 2026-10-23T01:50:54Z):

| Model, row or input file | SHA-256 |
|---|---|
| results/confirm/R24mwoz/tm_mem_c4_nrand/model_seed1.npz | c0c53f7cbaad806b1cdd8d6304a6ffdbee09bd4333580b56a2573d87ea9ee0e1 |
| results/confirm/R24mwoz/tm_mem_c4_nrand/seed1.json | 3d34ff250f73da501fe24baca6041822bb839eb0a644dbb744189636cade4697 |
| results/confirm/R24mwoz/tm_mem_c4_nrand/model_seed2.npz | af820861968029363721311d8427ed0c1b29abb80cd90c20e189c3166eaf0df3 |
| results/confirm/R24mwoz/tm_mem_c4_nrand/seed2.json | 7e84b053d3d5d9048d9230d05833a1f53d1ed28ac960e3651484bc3bf62e2c80 |
| results/confirm/R24mwoz/tm_mem_c4_nrand/model_seed3.npz | fb6956817eb74e5c10ec45f7e5faad7e15eb17929eea4f0d766b44100b6fc0d5 |
| results/confirm/R24mwoz/tm_mem_c4_nrand/seed3.json | d415b90a8dff4614502a7ce280ad796b6ea0fb3b951d03297b36d60b22509e2b |
| results/confirm/R24mwoz/tm_mem_c4_nrand/model_seed4.npz | 86889459b2a9811f3954cdc96813e03c19ae11fa845219e507727e1c324a6175 |
| results/confirm/R24mwoz/tm_mem_c4_nrand/seed4.json | b5f327f79bbf23a29f27d1b38f3c2d228039c3020aa532551983df1dd4f03b7e |
| results/confirm/R24mwoz/tm_mem_c4_nrand/model_seed5.npz | d15eee5e677c5931f5c94dfe60d61408039e1d345b1e1bb11550131f06031607 |
| results/confirm/R24mwoz/tm_mem_c4_nrand/seed5.json | f4d14476fece823543e5fde614c5ef2cd850c2590dce68609cfdc19d2f59e104 |
| results/confirm/R22star/tm_mem_c4_s16/model_seed1.npz | 48087be86b90e1709c0ffda4bdf74486b0cb95639c2b5faac0a9ea7d3f633d8f |
| results/confirm/R22star/tm_mem_c4_s16/seed1.json | dc637ac1d48a11fc1deeec50af2bb878175893bc4946dfe184407f537163318f |
| results/confirm/R22star/tm_mem_c4_s16/model_seed2.npz | bd14625ecc3d734eb556630267db8b104d82b77ac807f1042363a4ef32494291 |
| results/confirm/R22star/tm_mem_c4_s16/seed2.json | de18c2468124d63889fc69fc75e5ecf6ebf3af0f87b2f5df699fc2ec59b5a8fa |
| results/confirm/R22star/tm_mem_c4_s16/model_seed3.npz | 4a7d35e9ede223cbf826484b615233a368f4d8de2713ec88b0ea73c9a9957869 |
| results/confirm/R22star/tm_mem_c4_s16/seed3.json | 936fdc5e945e85b4190ca24cf6fcb14eb3505305b41b161f83edfee1618fc8ff |
| results/confirm/R22star/tm_mem_c4_s16/model_seed4.npz | 757d0924da55cdca4e8c811c79bf6d7df9d451c8ccf0578a9b620fa31d69cb4d |
| results/confirm/R22star/tm_mem_c4_s16/seed4.json | 90268e1aadfacc75d03c867737d886b5463e475d4684d97d90dea6072f7f019f |
| results/confirm/R22star/tm_mem_c4_s16/model_seed5.npz | d217e0c623925a6adc19cdedb039d515f4247c773a58670901015635a10893ad |
| results/confirm/R22star/tm_mem_c4_s16/seed5.json | b1ae42a5ba446ffd097c99791d93971e3198dc4045b0b848aace8b2f009eec08 |
| results/confirm/R23abcd/tm_mem_c4_s16/model_seed1.npz | 35ac9f979a71c20c24940786b24308b0c42768a1ed040fddbe040d78699548d2 |
| results/confirm/R23abcd/tm_mem_c4_s16/seed1.json | 4ba1dc75f8366a7aea3ed8a1283addaec4e3fc4fb3d3390b7f667e0c1707ced4 |
| results/confirm/R23abcd/tm_mem_c4_s16/model_seed2.npz | bf5048d28aef5d0c2b087b317b9f1cf5e52e533061b0283431b41bae4b4498d4 |
| results/confirm/R23abcd/tm_mem_c4_s16/seed2.json | 7b8a19d4a2e9946a8e97b53769dac338751edbb9a528bf09f2c4f09310bc12ca |
| results/confirm/R23abcd/tm_mem_c4_s16/model_seed3.npz | 491606f933e5793fd956920787b89f68fa3615e3422d872a0b27a56c27a6f533 |
| results/confirm/R23abcd/tm_mem_c4_s16/seed3.json | 48760cc11c616913a66c132cea9a70935856b331563c191a4b57189a0cb54790 |
| results/confirm/R23abcd/tm_mem_c4_s16/model_seed4.npz | 941e8de14a25520cbb42ddf306e09842a6b3befc1db741af22381f82188bbcf4 |
| results/confirm/R23abcd/tm_mem_c4_s16/seed4.json | e71caa75fd438a037140f296668a7308a9562cb997a5f01ebf7fd934b3b1929b |
| results/confirm/R23abcd/tm_mem_c4_s16/model_seed5.npz | 1aea85c44a80870f31e93e73f84b43bc3cdd70df0cd039f3f56f334b772da221 |
| results/confirm/R23abcd/tm_mem_c4_s16/seed5.json | 26f9e429446cf5e13b120e573fe237f18b6e82022833a35b3f7b0c6fa7705a23 |
| results/explore/dryrun/dry_e77mwoz_lr_mem_c4/seed0.json | f3abfcb8761ebb2a0d202a109731df4cddabb62916d43bdbfe2be25340b587cb |
| results/explore/dryrun/dry_R21star_lr_mem_c4/seed0.json | d64cbb4b8d4e2e83fafcaef9df43242ccd4e32b03b4a30004b72f48ccf213e8c |
| results/explore/dryrun/dry_R21abcd_lr_mem_c4/seed0.json | 98646174173b77ad533139f4dffded944eb4ca7782dbf278a904725f931e7f78 |
| data/emb/multiwoz__none.npz | fe366dc1df93b26fe29a55d9c4801b51bfeccd27ddf819250c738f8844fbf335 |
| data/emb/star__none.npz | fe366dc1df93b26fe29a55d9c4801b51bfeccd27ddf819250c738f8844fbf335 |
| data/emb/abcd__none.npz | fe366dc1df93b26fe29a55d9c4801b51bfeccd27ddf819250c738f8844fbf335 |
| data/tables/c4words_S1000_v1/table.npz | e1771d9eaf4a07f71c532fe4f3ef91909334cbdcb4d7785ff0bb7ed0e51c5c0a |
| data/multiwoz/examples_train.json.gz | 0e20a31434de10a5248f0b10ab01e7474ceb5e020cd5cbc7048bfe8f5db69e9b |
| data/multiwoz/examples_dev.json.gz | 3414cc8a6ecc64aafa0df070748b9d502a12e7996420b84a12df1b094a89fcd2 |
| data/star/examples_train.json.gz | 47735157f3e0035c59c8f88005ebe5457678083d60e653fbdca2f4df45c023df |
| data/star/examples_dev.json.gz | e562736dd9afa1b781d48bf78f43f23b1b8ceda17b3ddbf01557c59f0c8efb18 |
| data/abcd/examples_train.json.gz | c9e0f9b58a13a3dd9eff5821177cd74373cd1816465445a32bb6cc131933835f |
| data/abcd/examples_dev.json.gz | 5ccc737f5be9086060de4c28f560f1aa9b94566ed9599a98f44212c0e6cf2f17 |
| src/dl/abcd.py | a66c0c633ca227d24ec1f4b9ffcc97cf3b149c6fd6fc6e60919b9f9ed149cf63 |
| src/dl/count_text.py | 5f201106593c065f03a8f1bb843ae62ba08940422f682e0a76cdf064e9882d7e |
| src/dl/linear.py | e3ef06b96ef60740af7757c9f8c0672d7c6258404d67936bd4106723d3a0a65d |
| src/dl/tm.py | f649043f5801605b35174c526b290a0cfbadbdb66ade5bf04dafc122289d3424 |

The Part A chain (in the project notes, outside the repository; its stub test PASSED), read 2026-10-23T01:55:29Z:

| File | SHA-256 |
|---|---|
| e87_a_chain.sh (project notes, not published) | 34fbd45000ddab505ac774776560774d00bbeda7e4bf2f4f6ad10f484dd5e099 |
| test_e87_a_chain.sh (project notes, not published) | c45cccb669364dc0d2dfce983110f0377e5cf01072b42772f2c9ffc35a534414 |
| test_e87_a_chain.out (project notes, not published) | e4d11743486265588baac77eb7175e638310c91b537049106bcb501d0eff8c9a |

The chain runs the commands above: `e87_a_chain.sh a0` once the start rule holds (pids 3929234 and 3952401 gone, then 3 empty-GPU checks in a row), stopping at the first failing dataset and exiting when A0 is done, so the GPU is free; `e87_a_chain.sh a` after "e87 A0 end: done", each dataset behind the CPU gate. End lines: logs/e87_a0_end.log and logs/e87_a_end.log.

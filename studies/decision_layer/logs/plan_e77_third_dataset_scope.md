# Scope e77 (2026-10-19): a third dataset for a REGISTERED SUPERIORITY test (plan only; nothing downloaded, nothing run)

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged.

Decided at 20:54 UTC. **The claim still missing** is "better than
the equally tuned transformer agent", on a test split this study has never read. On STAR (R22.3) the C4 agent was
+1.64 [+0.94, +2.35] under a non-inferiority rule, and on ABCD (R23.1) it was -0.92 [-1.61, -0.26]. Neither was a
superiority test. Facts below come from web pages only (the repositories' READMEs; sources at the end). Anything not
read from a source is marked "to verify after the download".

## (1) Candidates and licences

| Dataset | Licence (from its repository) | System-action labels per turn | Fit |
|---|---|---|---|
| **MultiWOZ 2.2** (budzianowski/multiwoz, data/MultiWOZ_2.2) | MIT ("released under the MIT License"; the repository is labelled MIT) | yes: `dialog_acts.json`, per turn `dialogue_acts` (act name -> [slot, value] pairs) and `span_info`, "the same style as MultiWOZ 2.1" | **best** |
| Taskmaster-3 (google-research-datasets/Taskmaster, TM-3-2020) | CC BY 4.0 | no action layer: slot `segments` and an `apis` array of API calls per utterance | weak: only "which API call next", and most agent turns have none |
| Taskmaster-2 (TM-2-2020) | CC BY 4.0 | no: slot segments only; "does not indicate API calls" | no next-action task |
| Taskmaster-4 (TM-4-2024, coffee ordering) | CC BY 4.0 | none described | no |
| SGD (Schema-Guided Dialogue) | CC BY-SA 4.0 (share-alike; avoided) | yes: system actions per turn | kept as a fallback only |

**Choice: MultiWOZ 2.2 (MIT).** Its system turns carry dialogue-act labels. Its splits are fixed by its authors: the
README has train, dev and test folders; the sizes are to verify after the download (MultiWOZ 2.0 had 10,438 dialogues,
with 1,000 each for dev and test). None of its splits has been read by this study.

## (2) The task: one label per system turn, as STAR's next wizard action

- **One example per system turn,** in data/multiwoz/examples_{train,dev,test}.json.gz, in the ABCD / STAR format:
  - cust: the customer's utterances so far;
  - recent: the last 3 utterances with speaker tags;
  - prev: the system's earlier labels in the dialogue (the memory);
  - y: the label.
- **The label rule, fixed before any count is seen:** the sorted set of the turn's act names, joined by "+", with
  domain and act but no slots or values (for example "Restaurant-Inform+general-reqmore").
  - A combination with fewer than 20 train occurrences becomes "OTHER".
  - A turn with no act annotation is left out. The 2.2 README says 749 turns have none; the count is to verify.
- **Reported before any model:**
  - the number of labels and the train counts per label;
  - the share of "OTHER";
  - the majority-label accuracy on dev.
- A rule over the data is checked on train only; dev is used only by the agents' tuning.

## (3) The pipeline: new and reused

- **New:**
  - `scripts/multiwoz_prepare.py`, in the style of star_prepare.py, which writes the examples, the label list and the
    hashes;
  - the dataset name in abcd_explore.py's and abcd_confirm.py's `--dataset` choices. This is a one-line change to each
    (`["abcd", "star"]` to `["abcd", "star", "multiwoz"]`), done with the pre_R20 protocol: pristine copies, and the
    default path byte-identical;
  - the bge-large cache data/emb/multiwoz__bge-large.npz, from scripts/embed.py on the GPU;
  - data/emb/multiwoz__none.npz from none_prepare.py.
- **Reused unchanged:** ABCDView (view B and R18's bge-large view), dl.count_text with table_run.py, the TM and its
  settings, seq_queue.sh, report_abcd.py's statistic, and the checkers' pattern.
- **Contamination check before any model:**
  - The C4 slice behind the word table was deduped against ABCD and STAR dev/test turns, not MultiWOZ. Its dedupe rule (exact
    sentence and 8-gram) should be run on that slice against MultiWOZ dev and test turns, with the hits reported.
  - bge-large's pretraining data cannot be checked. That goes into the write-up as a limit.

## (4) Tuning on MultiWOZ dev for both agents, by the e74 rule

- A 4k screen with seed 1, at s 8, 12, 16 and 24, for both agents: the C4 agent with view B, and the bge-large agent
  with R18's view.
- The pick for each agent is the highest score, but it must be at least 0.3 above its own s 8, otherwise s 8; ties go
  to the first.
- Each pick then runs at 16k on dev, as STAR's.
- s* for each agent is its pick, unless its 16k run is below its own s 8 at 16k; then s 8.

## (5) The rule, fixed before any test read (R24)

- **R24.1 (superiority):** the C4 agent at its s* minus the bge-large agent at its s*, 5 seeds each, 16k: the point
  estimate above 0 and the whole 95% interval above 0. The bootstrap is over test dialogues, 10,000 resamples, seed 0,
  as before.
- **Beside it, R24.2 (non-inferiority):** at least -2.0, strong if the lower end is at least -2.0.
- **No rule:** each agent against its own s 8; TM against LR on the same bits (wide grid).
- **Wording:** "better than" is said only if R24.1 is met, and only for MultiWOZ. An R24.1 that is not met is reported
  as it is.

## (6) Budget and order (after the N9 study, about 6 to 7 h from 20:54 UTC)

- **GPU:**
  - the bge-large cache, about 20 to 30 min;
  - the screens: 4 x 4k C4 at about 5 min, plus 4 x 4k bge-large at about 2 min;
  - two 16k picks, about 35 min;
  - the test: 10 x 16k at 8 to 20 min each, about 2.5 h;
  - about 4 h in all.
- **CPU:** the prepare script and the C4 bits (minutes), the contamination check (in the text-understanding line, CPU), and the checkers.
- **Order:**
  1. this scope is checked independently;
  2. the download is checked independently first, with its licence and hash: the MultiWOZ repository at a pinned commit,
     data/MultiWOZ_2.2 only;
  3. prepare and the label counts (train);
  4. the contamination check;
  5. the e77 dev plan, frozen, then the screens;
  6. R24, frozen, then its runs.
- **Test use:** the 1st use of MultiWOZ's test split.

## Sources (web pages read 2026-10-19)

- github.com/budzianowski/multiwoz (README: MIT License; MultiWOZ 2.2 folder)
- raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/README.md (dialog_acts.json; train/dev/test folders; 749 turns without acts)
- github.com/google-research-datasets/Taskmaster, TM-2-2020, TM-3-2020 and TM-4-2024 (CC BY 4.0; annotations)
- arxiv.org/abs/2007.12720 (MultiWOZ 2.2 paper: annotation fixes, 8 domains)

## Additions after an independent check (21:14 UTC; scope fixed)

- **The memory bits** are the analogue of buttons pressed: the labels of the system's earlier turns in the dialogue,
  in order. That is the same combination labels, including OTHER, as STAR's wizard labels. So r["prev"] is the list of
  earlier labels, and the agent code is unchanged (done(label), last action = label, at least k actions so far).
  - A memory over single domain-acts (done(Restaurant-Inform) and so on) is not used. It would be a new view.
- **Reported before anything runs** (prepare's output):
  - the label count;
  - the train counts per label;
  - the share of OTHER on train and on dev;
  - the share of system turns left out for having no act;
  - the majority-label accuracy on dev.
  - The test split is not counted, and its labels are not read before R24.

## The download request (checked independently first)

- **Source:** github.com/budzianowski/multiwoz at commit fe0c8e65cfcd8462bd33c86e35f21addc84ca82b (master,
  2025-01-14).
- **Licence:** the repository's LICENSE says "The MIT License (MIT) Copyright (c) 2019 Paweł Budzianowski".
- **Only data/MultiWOZ_2.2, data files only:** 24 files, about 280 MB. The git blob SHA-1s come from the GitHub tree
  API at that commit.
  - README.md (7,230 B; 1e78846af6b9bce251d025e229be66ed21564379);
  - schema.json (18,123 B; 73776eb237d964f08dfc825a0239ce6d9856b80f);
  - dialog_acts.json (60,183,378 B; 8975b7888b7279387876ac8bce37825f5782ce40);
  - train/dialogues_001.json to dialogues_017.json (17 files; tree 5a2a9be02b502f1aa8ddee1a593cec7ad6b19cbf);
  - dev/dialogues_001.json and dialogues_002.json (tree 54ced820025827cc3b9445c07e2c17c453fca508);
  - test/dialogues_001.json and dialogues_002.json (tree c002d5a5eac976e8c056d026de799e8f424e3a15).
  - Not taken: convert_to_multiwoz_format.py and requirements.txt (code).
- **Method:**
  - one curl per file from raw.githubusercontent.com/budzianowski/multiwoz/<commit>/data/MultiWOZ_2.2/<path>, into a
    new empty folder data/untrusted/multiwoz22/;
  - each file's git blob SHA-1 is checked against the tree API's, and its SHA-256 is recorded in
    data/untrusted/multiwoz22/MANIFEST.sha256;
  - the files are read only as JSON with python -I, from the study folder.
- **Test:** the test files are downloaded with the rest. They are read only by the contamination check (text only, no
  labels) and by R24's registered runs.

## The counts (train and dev only; `scripts/multiwoz_prepare.py`, 21:19 UTC; logs/e77_prepare.out; data/multiwoz/labels.json)

- **Labels:** 128 (127 combinations with at least 20 train occurrences, plus OTHER), from 628 distinct raw
  combinations in train. 43 of the 128 labels have 20 to 50 train examples.
- **Train:** 8,437 dialogues and 56,581 examples. 195 system turns had no act (0.34%) and were left out. OTHER is
  3.03% of the examples.
- **Dev:** 1,000 dialogues and 7,359 examples. 15 system turns had no act (0.20%). OTHER is 3.38%.
- **Majority label on train:** general-bye. Its accuracy on dev is 7.16%.
- **The most frequent train labels:** general-bye 4,354; Train-Request 2,613; general-bye+general-welcome 2,594;
  Attraction-Inform 2,121; Booking-Book+general-reqmore 2,107; Restaurant-Inform 2,045.
- **Files:** data/multiwoz/examples_train.json.gz (0e20a314) and examples_dev.json.gz (3414cc8a). The test examples
  are not written; that is `--with-test`, at R24.
- dialog_acts.json is one file for all splits. The script keeps only the train and dev entries and discards the rest
  without inspecting them.

## Dated line (2026-10-19T21:57:15Z): R24 runs no s 8 arms on test

A note stamped 21:58 UTC (it arrived at 21:57 by this machine's clock) records the decision: "each agent against its own s 8" is dropped from R24's no-rule list. Those arms carry
no rule, and the claims do not need them. The tuning gain is reported from e77b only, labelled "dev, one seed". This
also applies to the same line in the frozen e77 dev plan (logs/plan_e77_multiwoz_dev.md, section "Then R24"), whose
text is not edited.

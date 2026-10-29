# Changelog

History of the programme, oldest first: one entry per week of the tm_language_model study, one for
each of hierarchical_tm and clause_lm, one for the publication of 10 and 11 Oct 2026, which also
covers the research notes, and one for each later step, phase of a study or publication. The studies are listed in
[studies/README.md](studies/README.md). A run id names a run folder and starts with the UTC time of
the run; run folders are not distributed. Every number and status below is repeated from the results
page that its entry links, where the file of the run and the full qualifiers are given.

## 19 to 25 Sep 2026: tm_language_model, from the pre-registration to the count-vote read-out

The pre-registration of the study was frozen on 2026-09-22; amendments A1 and A2 followed the same
day. Kill criterion KB1 was not triggered (`20260922T155509Z_parity_tmu`), on a statistic fixed
before that run, after a first comparison without a run folder had fallen outside the allowed band.
The planted-rules line of milestone M3 was met (`20260922T165439Z_planted_rules`) on the mean over
seeds, under a superset definition adopted post hoc, not under the strict one. The tiny language
model met H1.4 (`20260922T190301Z_tiny_lm`, `20260922T193047Z_tiny_lm`) on the validation split,
with settings chosen post hoc after two logged attempts had failed.

The anchor run `20260923T042350Z_CP_anchor` (causal, V = 8000, 1,048,576 clauses, seed 1) scores
2.2193 bits per byte on the WikiText-103 test split against 1.3719 for the in-house Kneser-Ney
5-gram on the same tokens: one seed, one read of the test split, and 0.114 of the test 8-grams also
occur in the training split. H1.1 is not met on seed 1 only, and kill criterion KB7 triggered;
seeds 2 and 3 were not run.

After the miss, amendments A3 to A10 (24 and 25 Sep) registered the families H8 to H17. The
count-vote read-out of A3 met H8.1 (`20260924T062704Z_cvtm_H8_1_causal_cp20M` and the masked runs
of seeds 2 to 5). Its threshold compares it with the signed votes of the same clause bank, not with
the n-gram baseline; it is at the level of a Kneser-Ney bigram, and the causal arm uses a bank
seen in the exploratory evaluation. The copy selector met H9.1, trails its best control (H9.2 not
met) and has no learning credit (`20260925T063437Z_cvtm_copy_selector_H9_verdicts`).
[Results page](studies/tm_language_model/docs/results.md).

## 26 Sep to 1 Oct 2026: tm_language_model, the self-supervised runs and their controls

The exploratory control study of the self-supervised family
(`20260926T204018Z_ssl_r1c_report_exploratory`) read "training does not add value". The
confirmatory text runs (seeds 1 to 3, 26 to 30 Sep) leave H17.1 to H17.4 and H17.11 not met
and H17.12 met (`20260930T163619Z_ssl_r1cf_report_confirmatory`); gate G-R1 was read on 2026-09-30
as "fail and below bag of words". The mechanism test (H17.5, both models at chance,
`20260930T203322Z_ssl_r2_hao_report_confirmatory`) and the image test (H17.10, trained code below
its young bank, `20260930T203322Z_ssl_r4_report_confirmatory`) are not met; the image controls of
1 Oct (`20261001T202442Z_ssl_r4c_report`) have no threshold. The study ended on 1 Oct and is
archived, with 12 of its 91 registered hypotheses met, 11 not met, 7 not judged and 61 not run.
[Results page](studies/tm_language_model/docs/results.md).

## 1 to 3 Oct 2026: the hierarchical_tm study

The study tested the hierarchical Tsetlin Machine procedure of the supervisor's project proposal
(unpublished). Its pass rules, [gates.md](studies/hierarchical_tm/gates.md), were frozen on 1 Oct
2026 before its runs; 27 exploratory probe runs of that day used pair data with a sampling bias
and are not used. Checks 1a, 1b, 3 and 4a are met. Check 4b is met on the s = 100 arm only, with a
small effect, and not met at s = 400 and at s = 200, an arm that was allowed after the results of
the other two were known. Checks 2a, 2b and 2c are not met, and no learned rule captured either of
the two sentence families of the proposal (Check 2b and the slide-6 paraphrase test); the results
page gives the qualifier of each. The one final test, `20261002T193513Z_final_test_main_table`
(mean of 5 seeds), has test AUC 0.735 on same-article and 0.750 on next-paragraph pairs, below word
overlap (0.817 and 0.857). [Results page](studies/hierarchical_tm/docs/results.md).

## 2 Oct 2026: preparation of the clause_lm study

The clause_lm study prepares a word-level clause language model. On Penn Treebank the n-gram
baselines, the pair data, the statistical pair baselines and one clause evaluator with k-means word
groups ran once each, without a Tsetlin Machine. For the one registered step,
[M5 step 1](studies/clause_lm/docs/prereg_M5_step1.md), Gate 0 (the reproduction of the lexical
Clause-Driven-LLM router on CLINC150) was not met in its one completed run,
`20261002T195255Z_m5_gate0_s8_seed42_sorted`; the cause is not confirmed. A retry was stopped
without a score, the registration was amended post hoc for a GPU rerun that did not take place,
and the grid and the pair model were not run. [Results page](studies/clause_lm/docs/results.md).

## 10 and 11 Oct 2026: preparation of the publication

The three studies were brought into one repository, each as a folder under `studies/`, with the
pages that span them in `docs/`. The wording of documents, comments and docstrings was edited,
absolute paths were replaced by paths relative to the study folders, and the registrations were
published as copies with a publication note; their registered hypotheses, thresholds, seeds and
dates are unchanged. Files that a record pins by hash and whose text changed were re-pinned, as the
[hash table](docs/hash-table.md) lists. Run folders, datasets and the files that carry sentences
of the project proposal are not distributed, and the edits were not exercised by a run
([known limitations](docs/known-limitations.md)).

On 11 Oct 2026 the folder [research/](research/README.md) was added: the ideas of the programme with
the pages of the writing board, the findings of a literature search of 2026-10-10 on each idea, the
design that follows from them and the list of open work. Nothing in that folder has been run. The
repository was given the licence CC BY-NC 4.0, and the upstream `setup.py` of the vendored package
was added to it.

## 14 Oct 2026: clause_lm, the amended Gate 0 and the grid of M5 step 1

The amended Gate 0 of [M5 step 1](studies/clause_lm/docs/prereg_M5_step1.md), the locked settings
with the torch backend on a GPU, was met in `20261014T221527Z_m5_gate0_s8_seed42_sorted_torchgpu`
(test accuracy 0.8293, inside the band of 0.005 around 0.82778). The amendment was written after
the Gate 0 result was known, so the pass is post hoc, and a first attempt ended at its start
without a score. The registered grid over s in {8, 25, 50, 100} with seeds 42, 123 and 456
followed. By the registered rules the question about s has a negative answer: s* = 8, and the mean
test accuracy at s* is 0.8201, below the 0.848 that the success rule needs
(`20261014T222730Z_m5_summary_sorted_torchgpu`). One grid run did not have the GPU to itself for
about 40 s. [Results page](studies/clause_lm/docs/results.md).

## 15 and 16 Oct 2026: decision_layer, phase 1: the router, the readable layer and learning from reward

The study decision_layer asks where Tsetlin Machines fail on language, part by part, and what works
instead when an LLM encoder does the language and a Tsetlin Machine makes the decision. Each of its
registered tests was frozen before its confirmatory runs. Phase 1 gave 17 verdicts on CLINC150,
Banking77 and HWU64, 9 met and 8 not met. A single Tsetlin Machine router trails logistic regression
on the same frozen embedding in every cell, by 0.56 to 2.41 points with 4,000 clauses (R4 not met)
and by 0.07 to 1.46 with 16,000 (R4b not met); after one temperature it is calibrated, and it is at least as good at flagging
out-of-scope queries (R1 met). The readable layer, whose inputs have names, comes within 0.3 points
of logistic regression on the same inputs (R7 met), and post hoc it restates the rule "route to the
nearest whitened centre". The literal budget lifts the word router by 4 to 6 points on the three
datasets; R2, R2c and R7b failed only their length rules, by one literal, through a tie in the
budget code, and the re-runs R2e, R2d and R7c are met. A policy learned with TD from delayed reward
met R6 and R8, with gains over the tuned rule (R6) and over always asking (R8) that come from the
out-of-scope episodes. Routing learned from reward alone met R5c and R5d with Boltzmann exploration
and 30 passes, after R5 and R5b were not met. [README](studies/decision_layer/README.md), [results
page](studies/decision_layer/docs/results.md).

## 15 to 17 Oct 2026: tally, the registration T0 and the first steps of the design

The study tally holds the first steps of the lines T, A and C of the [design](research/design.md).
Its one registration, T0 (counted votes against learned votes on the same clauses, with the Tsetlin
Machine library tmu, on Penn Treebank), was frozen on 2026-10-15 together with its tuning rule. Two
amendments of the same day fixed the settings by that rule, s = 20 and 9 epochs for the coalesced
machine and s = 5 and 1 epoch for the flat machine, and the final runs of seeds 1 to 10 for both
machines and the logistic-regression arm finished by 17 Oct. The verdict of T0 has not been drawn.
B1 and B6 ran as smoke and timing runs, A1 and A2 as a baseline check and a smoke run, and C3 as
LSTM preparation runs on the validation split. [README](studies/tally/README.md),
[results page](studies/tally/docs/results.md).

## 16 and 17 Oct 2026: decision_layer, phase 2: a dialogue workflow (ABCD, STAR) and a related dataset (MASSIVE)

Phase 2 used data that no test of phase 1 had read: ABCD (customer-service conversations; which of
30 action buttons the human agent presses next), STAR (wizard-of-oz dialogues, 187 actions) and
MASSIVE 1.1, a related dataset that shares about a third of its test texts with HWU64 train, with
different labels. On ABCD, memory literals lift the next-action accuracy of the Tsetlin Machine from
65.4% to 84.0% (R9b, +18.65 [+17.20, +20.10]). R9 itself is not met, because the control of its
memory rule read the history as words, a design error; its rules against logistic regression and an
MLP on the same named inputs, and against an opaque MLP on the raw embeddings, are met, and the rule
that the Tsetlin Machine beats logistic regression is not. From a 0/1 reward per press the Tsetlin
Machine reaches 0.988 of its supervised twin (R10). After the 15 registered passes it ends above
LinUCB (R10c, +5.51 and +3.00; registered when the results of R10 were known, as R14, R15, R16b and
R17 were registered after the results they build on), although LinUCB is ahead for the first 7 to 11 passes and earns more
reward while learning; it ends within the registered margin of a small neural bandit, judged on the
point estimate (R14, -0.45 [-1.04, +0.16]); and replay closes most of its slow-learning gap (R15).
With its own presses as memory and the text still replayed from the log, the supervised model keeps
80.7% (R13) and the reward-learned one 79.1% (R17). The hierarchical Tsetlin Machine of the package
PyHierarchicalTsetlinMachineCUDA, with 2 times the clauses and 16 times the automata, is 9.4 points
above the flat machine of the same package (R16) and 7.0 above a flat machine with as many automata
(R16b; T, s and the vote still differ), and 2.3 points below this study's coalesced flat machine. On
STAR the memory result replicates (R18, +15.43), and there the Tsetlin Machine is 3.3 points behind
logistic regression on the same bits. R11 (MASSIVE) and R12 (trained in English, scored in Norwegian
at 82.2%) are not met as a whole; R11.2, R12.1 and R12.3 are met.
[README](studies/decision_layer/README.md), [results page](studies/decision_layer/docs/results.md).

## 17 to 19 Oct 2026: decision_layer, phase 3: an agent with no neural network at run time, then at any stage

R19 replaced bge-large-en-v1.5, run per decision, by a fixed word table (potion-base-32M, distilled
once, offline, from bge-base-en-v1.5 via Model2Vec), lookups and a bitwise OR. With no neural
network at run time, an agent is level with the bge-large agent on STAR test (+0.06 [-0.63, +0.74])
and 1.0 point below it on ABCD (-1.00 [-1.62, -0.40]), inside the registered 2-point margin; R19.1
is met in the strong form on both datasets
(`studies/decision_layer/results/confirm/R19star/report_R19star.json`,
`studies/decision_layer/results/confirm/R19abcd/report_R19abcd.json`). One decision takes a median
of 29.5 ms on one CPU thread against 806 ms, a rough timing on a shared machine; the bge-large agent
would be faster on a GPU, which was not measured. R20 put the word-table agent in a closed loop on
STAR: memory still pays (R20.1 met), its own mistakes cost 7.51 points (R20.2 not met, as expected
from dev), and it is not worse than the bge-large agent in the same loop (R20.3 met, strong). The
question named as the headline of R20, an agent that reads only the customer, stopped at its dev
gate on STAR; on ABCD that agent, with no text written by the human agent, gives supporting evidence
(R20.4 met: memory +23.94 [+22.20, +25.67] points, at 72.6% in the closed loop). R21 replaced the table by one counted from C4 text, so that the C4 agent has no
neural network at any stage: it is within 0.10 (STAR) and 0.29 (ABCD) points of the potion agent,
met in the strong form on both datasets. With the Tsetlin Machine's s tuned on dev for both agents,
the C4 agent is +1.64 [+0.94, +2.35] against the bge-large agent on STAR (R22, all three rules met)
and -0.92 [-1.61, -0.26] on ABCD (R23.1 met, strong), both under non-inferiority rules. On dev,
under frozen plans and notes and with no registration, the raw TM policy of the ABCD agent had a
counterexample for every one of 50 guideline-derived order rules over a superset of real inputs, and
50 veto clauses made all 50 hold by construction; the STAR agent produced its replies from the
task's templates, equal to the wizard's word for word in 45.5% of the template turns against a
ceiling of 79.6%. [README](studies/decision_layer/README.md), [results
page](studies/decision_layer/docs/results.md).

## 19 to 23 Oct 2026: decision_layer, phase 4: a third dataset, MultiWOZ 2.2

On MultiWOZ 2.2 dev, with the training rule's hardest negative classes, the C4 agent, with no neural
network at any stage, fell 3.21 points behind the bge-large agent against a gate of -1.0 (e77b);
further stacks on dev did not pass the gate (e78, e79, e82), and lower vote thresholds T did not
help the reader (e81). Random negative classes, chosen on MultiWOZ dev (e83), lift the network-free
agents far more than bge-large but cost STAR -11.9 and ABCD -4.0 points on dev (measured on the refined agent), so the rule is a
MultiWOZ setting; with it both gates were met on dev (e84, e84b). R24, frozen before MultiWOZ's test
split was read, used that split for the first time: R24.1 is met in the strong form for the C4 agent
(+0.53 [-0.18, +1.23]) and for the refined-table agent, network-free at run time with bge-base-en-v1.5
as teacher at training time (+0.05 [-0.65, +0.72]), and R24.2,
superiority, is not met for either, so the study's only fresh split for a registered "better than"
is spent without one (`studies/decision_layer/logs/R24_report.out`, a run log, not distributed). On the
same bits logistic regression is ahead of the Tsetlin Machine for all three agents (C4 -1.55, refined
-2.49, bge-large -1.89; no rule). Two
independent recomputes agree, and a three-part review (leakage, selection and registration,
integrity; judged by an AI model, not a person) found no blocker. Two registration drafts were never
frozen: R25 (an agent on the teacher-refined table), whose dev go-bars failed, and rev 0 of R24, a
superiority test on MultiWOZ written on 2026-10-19, which the frozen R24 superseded. The dev plans
e86 and e87 were stopped before their analyses.
[README](studies/decision_layer/README.md), [results page](studies/decision_layer/docs/results.md).

## 21 Oct 2026: the glass box page

The folder [live/](live/README.md) holds a page that runs the count floor of
[research/design.md](research/design.md) in the browser, published at
https://anwardebes.github.io/glassbox-lm/. It reimplements the CVTM-core of tm_language_model in
JavaScript and fits it on ten public-domain books in the browser tab. `live/export.py` fits the same
model with the study's code (`tmlm.cvtm`, numpy backend) and writes the reference values that
`live/tests/model.test.mjs` checks; a workflow runs that test before each deployment. On 19 held-out
chapters the page measures a perplexity per token of 193.20 for the Kneser-Ney 5-gram and 165.91
for the CVTM-core, and the vote form and the recursion differ by at most 4.4e-15 in ln P. These are
numbers of the page's corpus, not results of a study, and no Tsetlin Machine runs on the page.

## 23 Oct 2026: publication of decision_layer and tally

The studies decision_layer and tally were added to the repository, each as a folder under
`studies/`, and the pages that span the studies, among them the roadmap in `research/`, were
brought up to date for five studies. The documents, comments and docstrings of the two studies were
edited for publication in wording and paths only. The registrations, dev plans and dated notes whose
wording was edited were published as copies with a publication note, the others as written; their
freeze records are unchanged, and the
[hash table](docs/hash-table.md) gives the registered and the published values. Independent
recomputations of results of decision_layer were added in `studies/decision_layer/verification/`,
and each was rerun on the published scripts. The tests of both studies were run on the published
files; those that need the GPU were not. Run folders, result files, run logs and datasets are not
distributed ([known limitations](docs/known-limitations.md)).

## 25 and 26 Oct 2026: the finish pass

A finish pass ran the registered and frozen work that could run on this machine in two days, fixed the bugs that an
inventory of the published tree had found, and added tests. Every new number below was recomputed independently from
its run files before it was written into its results page (the records of the pass are outside the repository).
Negative results are reported like positive ones. The open items are listed in the [roadmap](research/roadmap.md).

**tm_language_model.** Two amendments of the pre-registration were frozen before their runs. Under A11 the receipts of
the registered generations were re-checked with a checker that binds every decision to its run (seed, temperature,
mode and step order): all 11,400 receipts pass and all 1,412 tampered receipts fail, so H2.4 and Gate 3 are met
without the earlier qualifier on forged draws. Under A12 the WikiText-103 test split was read once (access 113): the
count-vote core, KN-5 plus document caches in clause form with no Tsetlin-learned clauses, scores 1.2478 bits per
byte, 0.1241 below KN-5 and 0.1054 below KN-5 plus a document unigram cache, so H8.3 and H8.4 are met. H3.5 is met at
the registered 200 steps, and H2.2 and H2.3 hold in a logged run of the parity tests (30 passed). Three rows are not
met: part (c) of H8.8 (a median of 879,895 training tokens per second against 1 million), H8.9 (pruning loses 0.0280
bits per byte at 22.2% of the bytes) and H9.6 (weighted R-squared -1.735), which closes the gate G1 T1. H8.11 and part
(a) of H8.8 were read as met from their run files, where no verdict had been recorded. Of the 91 registered
hypotheses, 15 are now met, 14 not met, 5 not judged and 57 not run. Fixes: `env/env.sh` puts the study's interpreter
first on `PATH` (the helper scripts had exited with status 127 on a machine with no `python3`); `make test` keeps off
the GPU and `make test-gpu` runs the GPU tests; `make smoke` runs a new end-to-end smoke test; the test-read driver
saves its outputs even when it fails after the read.

**tally.** The registration A2 was frozen and its one look at dev-test ran: the exchange-class code lowers the
perplexity of a Kneser-Ney 5-gram with a cache by 7.59% (95% interval over dev-test blocks 6.51% to 8.58%), and the
success line is met. The one code taken from a Tsetlin Machine (`orgroups`) added nothing, and that reading is
uninformative by construction. The A/A noise runs of A1 measure the train-side noise of the floor. The LSTM baseline
of C3 had been under-trained by its loss. The rerun with the paper's loss and clips gives 90.99 to 92.12 for the
medium model (paper 86.2) and, on 6 of 10 seeds, 86.22 to 87.05 for the large model (paper 82.2); the other 4 large
seeds collapsed in their first epoch, so the reference does not reproduce the paper. The B1 report no longer reads
censored runs as a kill, the launchers of B1 and B6 start jobs at nice 19 behind the CPU gate, and the run logger
marks killed runs and records the hashes of both Penn Treebank texts.

**clause_lm.** A frozen dev plan checked the cause of the failed Gate 0. The driver check is met (0.8186666666666667,
the recorded value, and a byte-identical model), but removing the negative-class erosion from the torch kernel did not
reproduce the numba model: arm A is mixed (0.7753, 0.7717 and 0.7747, healthy structure below the 0.78 bar), so there
is no cause statement, and the expectation written before the runs did not come true. The study gained plain-script
tests, a freeze tool and a launcher that names its interpreters.

**decision_layer.** The dev plan e86 (clause dropout) finished under a dated note and is closed as a dead end: dropout
is kept on no dataset. The dev plan e87 ran its Part B bar under a dated note: with clauses of at most 3 literals the
C4 agent stays more than 2.0 points below LR on dev on all three datasets (0 of 3 seeds each), so short receipts cost
it its accuracy; Part A and the kill rule have not run. The dev plan e80 (view B plus the customer's last 3
utterances) ran under a dated note and is not taken: STAR -0.40 points against a bar of +0.5, at view B's s 16. The
rule R2c.3 is now reported as not met, read from the rule output written with its run. The data card gives what the
local download cache shows for each encoder.

**hierarchical_tm.** Four bugs of the published tree were fixed after the study, with no run or registered rule
changed (the study README lists them): the launchers could not start on this workstation and now take their
interpreters from `src/interpreters.sh`; the length feature of `src/data_checks.py` wrapped around for unsigned
counts, and a dev re-run of the three length-only checks changes no verdict; the queue scripts quote their path
variables; the published copy of the deletion probe is annotated. Two new CPU tests cover the first two fixes and fail
on the versions before them. The freeze rule of gates.md was applied after the study ended, on 25 and 26 Oct 2026: 124
of the 683 numbers of the results page were re-checked on the GPU, all exactly, and the other 559 were not re-checked;
the two news retrains of Check 2c missed the tolerance on their time per text alone, so Check 2c is marked "not
reproduced in the sample" ([outcome page](studies/hierarchical_tm/docs/reproduction_after_the_study.md)). The
registered prediction for L2 with 8 options per OR-group, tested after the study ended (exploratory, 3 seeds), fails:
purity and recovery rise, but dev AUC falls by 0.25.

**The glass box page.** The page gave the 65th and later new words of a chapter one shared id; it now has 256 slots,
and the held-out perplexity per token of the CVTM-core moves from 165.91 to 170.38 (Kneser-Ney 5-gram: 193.20 to
193.29). The qualifier cells of `live/study.json` are parsed cell by cell, sampling no longer writes an empty slot,
the page gained keyboard, screen-reader and reduced-motion fixes, and new tests cover the data, the text handling, the
page and the export.

**Shared pages.** The licences and revisions of the cached datasets and models are on the data page; the known
limitations, the results overview, the roadmap and the counts of the README follow the results above, and the
README gives tally's two verdicts in its text, its status table and its results table; the [run
index](docs/run-index.md) lists the 109 run folders of 25 and 26 Oct; the [hash table](docs/hash-table.md) lists the
pinned files changed in the pass.

## 29 Oct 2026: four studies added from the research lines, and the finish pass's remaining files

The research lines that ran beside the studies were added as four studies, each with its code, its frozen
registrations and plans, its result files and a results page: [text_understanding](studies/text_understanding/README.md)
(16 to 25 Oct), [agent_guards](studies/agent_guards/README.md) (20 to 26 Oct), [language_part3](studies/language_part3/README.md)
(21 to 23 Oct) and [agent_vs_llm](studies/agent_vs_llm/README.md) (from 26 Oct). In agent_vs_llm's registered test the
readable agent of decision_layer, with no neural network at any stage, is not worse than the best local LLM configuration
by 3 points on STAR, ABCD and MultiWOZ 2.2 (MultiWOZ on seed 1 only) and better on none; a fine-tuned bge-base is ahead
on all three and TF-IDF logistic regression is level with the readable agent. M1 of text_understanding is met on three reused
intent test splits and M2's H1 is not met on MTOP English; E4 of agent_guards is not met and E2 is incomplete;
language_part3 stopped at its gates. decision_layer gained FastCoTM (`speed/`, exact speed-ups with byte-identical
results), its launch chains (`ops/`), more independent recomputes, four earlier script versions cited by hash and the run
logs its pages cite. The finish pass's remaining files of 25 Oct were added: the Gate 0 cause plan and its code
(clause_lm), A2's registration (tally), the notes of e80 and e87 (decision_layer), the HTM-01 and HTM-03 records
(hierarchical_tm) and the amendments A11 and A12 with their drivers (tm_language_model). The published copies of files
that ran find their paths from their own place in the repository and are edited in wording only; the hashes of the
files as they ran are in [docs/hash-table.md](docs/hash-table.md).

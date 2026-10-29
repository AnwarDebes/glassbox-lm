# Changes after the study

This page carries the dated note of 25 Oct 2026 on the code fixes made after the study ended (it ran from 1 to 3 Oct 2026), and on the pins of the registrations that the fixes break. The README section "Changes after the study (2026-10-25)" gives its short form. On 25 Oct 2026 Norwegian time is CET (UTC+1): summer time ended at 01:00 UTC that day. The UTC time is added in brackets in the heading.

## 25 Oct 2026, 17:31 CET (16:31 UTC): code fixes after the study, and the pins they break

*Publication note.* The text below is the frozen file (sha256 `0992997e618c59bed6089409207a0270f1ae919bb63399d920d3387db3a255e4`, frozen 2026-10-25T16:31:59Z; the frozen file itself is not distributed), edited for publication in wording only: the notes on its review and drafting are shortened and private paths are left out. Every rule and number is as frozen. workstation paths are shortened (the finish pass's folder is written `<finish-pass records>`, paths in this repository are relative to it, and the home folder is dropped from the others), and the headings move down two levels to fit this page. 11 of its 286 lines differ; nothing else changed.

### Frozen 2026-10-25T16:31:59Z (draft 3): dated note of 2026-10-25 on the code fixes of hierarchical_tm after the study, and the pins they break

Written in the finish pass of 2026-10-25. Draft 3 answers two pre-freeze reviews, both READY AFTER FIXES (judged by an
AI model, not a person).

**Times.** On 25 Oct 2026 times are CET = UTC+1:
summer time ended at 01:00 UTC that day, the last Sunday of October (the EU rule; the same offset was read from the tzdata 2026.4 file of
a study virtualenv).

#### What changed, who found it, and why

The study ended on 3 Oct 2026. Four of the issues were already in the published documents at HEAD 1bb3935: the
unquoted queue variables (docs/known-limitations.md line 49, HTM-B5), the launchers' need of a `python3` on `PATH`
(line 94, one half of HTM-B1), the wrap-around of `length_feats` (line 152: "The internal audit reports that it wraps
around for unsigned counts and found no effect on the reported results", HTM-B2), and the published deletion-probe copy
without the two-part rule (README line 117 at HEAD, HTM-B4). New on 2026-10-25, found by the inventory of the finish pass:
on this workstation `../tm_language_model/.venv-htm` does not exist and no `python3` is on `PATH`, so every launcher
stopped before its job started. The dev re-run below tests the audit's "no effect": the values move a little, no
verdict changes. No run folder and no registered rule was changed by the fixes.

- **HTM-B1, the launchers could not start.** `src/interpreters.sh` (new) sets `HTM_PY` (default: the old
  `../tm_language_model/.venv-htm/bin/python` when it is an executable file, else `../decision_layer/.venv-htm/bin/python`)
  and `HTM_TOOL_PY` (default: `../tm_language_model/.venv-tm/bin/python`). `src/launch.sh`, `queue/run_queue_mps.sh`,
  `queue/gpu_share_test.sh` and `src/feas_slide11/launch.sh` start `scripts/mem_guard.py` and `scripts/run_logged.py`
  through `HTM_TOOL_PY` and the job through `HTM_PY`, stop with exit status 127 before anything starts when either is
  not an executable file (`run_queue_mps.sh` before its MPS daemon), and write `HTM_PY` into the run config as
  `job_python`. `queue/run_queue.sh` reaches the fix through `src/launch.sh`. Behaviour that is new beyond the fix:
  `src/launch.sh` exits 2 with a usage line when it gets fewer than 3 arguments (before it sources `interpreters.sh`);
  `src/feas_slide11/launch.sh` exits 2 with no argument and copies `../interpreters.sh` into the run folder with its
  other script copies; the analysis step of `queue/gpu_share_test.sh` runs under `HTM_TOOL_PY` (the same interpreter as
  before, now named by the variable).
- **Bytecode in the borrowed virtualenv (round 2).** Since round 2, `src/interpreters.sh` also exports
  `PYTHONDONTWRITEBYTECODE=1` when `HTM_PY` lies in `../decision_layer/`, because that virtualenv held no `.pyc` for most
  of scipy and scikit-learn and the first job would have written `__pycache__` folders into it (review m8). The GPU job
  scripts of the finish pass export it as well.
- **The interpreter substitution.** `../decision_layer/.venv-htm` belongs to the decision_layer study, which built it for
  its registered R16. It holds PyHierarchicalTsetlinMachineCUDA at commit 5b4a223075c1e621577d508bc7c3c061230e29d5 (its
  `direct_url.json`), the commit of the frozen settings, and numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1, pycuda
  2025.1.3, scikit-image 0.26.0 and networkx 3.6.1, the versions of `../tm_language_model/env/htm.lock` (checked by
  `htm01/preflight.py`, output `test_preflight.txt`). It differs from that lock in imageio 2.38.1 (2.37.4), markupsafe
  3.0.4 (3.0.3), narwhals 2.27.1 (2.26.0), pillow 13.0.0 (12.3.0) and platformdirs 4.13.3 (4.11.12), and has no editable
  tmlm, which the drivers do not import. It is used by its path only. It was used for the finish pass only.
- **HTM-B2, `length_feats` wrapped around.** In `src/data_checks.py` the two paragraph lengths came as uint64 row sums
  of the uint8 matrices of `pairdata.ids_to_csr`, so |len A - len B| became about 1.8e19 whenever B was the longer one.
  The counts are now cast to int64 before the difference; the four other features are unchanged.
- **HTM-B5, unquoted path variables** in `queue/run_queue.sh`, `queue/run_queue_mps.sh` and `queue/gpu_share_test.sh`:
  quoted. shellcheck 0.11.0 (`shellcheck_b5.txt`) leaves 11 notes of unquoted expansions, all on purpose: the argument
  lists split into driver arguments (`$args` in both queues, `$A` in the sharing test), the job-name join `$(echo $open)`
  in `run_queue_mps.sh`, and the integers `$n` (in `$(seq 1 $n)`) and `$s` together with `${setting}` in job names of
  the sharing test. The launcher test also ran in a scratch path that contains a space (24 of 24 then, 26 of 26 in round 2).
- **HTM-B4**: `src/recovered/e3fb_deletion_probe.py` has a note at the top of its docstring on the two-part reading of
  its later version; its code is unchanged (an AST comparison with the HEAD version, docstring removed, is equal).

#### The pins: three values per file, and what the record now means

`frozen_settings_hashes.json` pins two of the changed files. The record was re-pinned once already, at publication on
2026-10-10/11, for wording-only edits (docs/hash-table.md lines 11-12: "the pin was replaced by the published value, so
that the checks in the code still compare equal values"). This code change is not re-pinned: the finish-pass assignment
said not to edit the JSON (the inventory, INVENTORY.md line 132, had proposed re-pinning). So:

| file | registered, and checked by the final test of 2 Oct 2026 | in the record since publication | now |
|---|---|---|---|
| `queue/run_queue_mps.sh` | 57c69a5585eaabd39eeeea04a21a5b6900e89e9a1992936b8b903cd2e45886e4 | dae7738a023d02d812ee38ac4fe583e565e67d463aca0c4f2c276fe61c9663d9 | 46e70e4ff077b2fbd06669a3ef346413ce8aab945b1bc75e14bcab078c9b6b81 |
| `src/data_checks.py` | 72b0d717d354c9e7e0e6dd2a8a3bf6c0fc285f43f6be4128d479cdabbd17ca4c | 896f2bdc7505fbd1dd965cd01c0da2e4876f7f4068d72c3971ca6b1395a26447 | 7523133e933459d552a355a28c0ac4967bcc9685be8d3c08d3a00fa42cbe6fe5 |

The registered values are the "as registered" column of docs/hash-table.md (lines 43 and 51); `report.json` of
`20261002T193513Z_final_test_main_table` records `src/data_checks.py` as 72b0d717... in its `inputs_sha256`. For these two
files the sentence of hash-table.md lines 11-12 no longer holds; the row edits are made in hash-table.md. Consequence:
`src/final_test_report.py --hashes frozen_settings_hashes.json` now refuses to run, because it hashes `src/data_checks.py`.
This was established by reading the code (lines 311-318 check the hashes and exit "refusing" on a difference), not by a
run; note that the script copies its four CODE files into `--out`, `$TMLM_RUN_DIR` or "." (lines 304-306) before it
refuses. The report uses none of the changed code (`length_feats` is not called by it).

#### What depends on the changed files

- **The freeze rule's check (HTM-01, not frozen)** re-ran both length-only checks of Check 1b and part 1 of Check 4b
  through the fixed `src/data_checks.py` in its step (iii), a reading made before its freeze (draft 3 of that plan gives
  the full outcome, section 11). Three logged values that the results page does not report move by more than 0.02 (the share of yes predictions, `pos_share`, of the length-only
  classifiers: ns boosted trees 0.5688 -> 0.533, ns_all logistic regression 0.4999 -> 0.6743, ns_all boosted trees
  0.5891 -> 0.6265). The fix was made before HTM-01, against the order that the inventory proposed (INVENTORY.md line 132:
  "Do this after HTM-01"), because the finish-pass assignment listed it first; HTM-01 declares the effect.
- **The pending GPU jobs** of HTM-01 and HTM-03 run through the changed `src/launch.sh` and `src/interpreters.sh` and the
  substituted interpreter (their job scripts set `HTM_PY` to it explicitly).

#### The length-only checks, re-run on dev

The three checks that call `length_feats` were re-run on the train and dev pairs only (no test file opened), first with
the version before the fix (896f2bdc..., the copy `head/src_head/data_checks.py`) and then with the fixed one, by
`rerun_length_checks.py` with the output `length_checks_rerun.json`, from 2026-10-25T11:30:47Z to 11:31:09Z (its
`started_utc` and `ended_utc`; the CPU gate read 3.35 cores at 11:30:46Z). The shell line itself was not saved. What the
output records: the interpreter `python` = `studies/tm_language_model/.venv-tm/bin/python`,
`threads` = OMP, OPENBLAS and MKL at 2, `versions_now` = numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1, and the sha256 of
its inputs (the four train and dev files) and of its code. An empty `CUDA_VISIBLE_DEVICES`, nice 19 and
`PYTHONDONTWRITEBYTECODE=1`, named in draft 2, are not recorded for this run (UNVERIFIED).
The env.json of the logged runs records numpy only, so the exact reproduction with the old function is what shows the
environment gives the same values. The version before the fix reproduced all 7 logged
records exactly (ns and ns_all, each logistic regression and boosted trees; Check 4b part 1 in its three runs). With the
fixed one (logged value first):

| check | logged in | logistic regression: accuracy [95% CI], AUC, pos_share | boosted trees: accuracy [95% CI], AUC, pos_share |
|---|---|---|---|
| Check 1b, `ns` | `20261001T094249Z_data_checks_final_exploratory` (`checks.json`; `checks.md` line 5) | 0.5017 [0.4947, 0.5086], 0.5012, 0.4812 -> 0.5017 [0.4947, 0.5086], 0.5021, 0.496 | 0.5 [0.4931, 0.507], 0.4966, 0.5688 -> 0.4989 [0.492, 0.5058], 0.4994, 0.533 |
| Check 1b, `ns_all` | `20261001T094347Z_data_checks_ns_all_v2_exploratory` (`checks.json`; `checks.md` line 3) | 0.5002 [0.4933, 0.5071], 0.5002, 0.4999 -> 0.5004 [0.4935, 0.5074], 0.4983, 0.6743 | 0.503 [0.4961, 0.5099], 0.5007, 0.5891 -> 0.5026 [0.4957, 0.5095], 0.503, 0.6265 |
| Check 4b part 1, `ns_all` (logistic regression, article bootstrap) | the three check4b runs (`metrics.json` `length_only`; `log.txt` line 84) | accuracy 0.5002 [0.493466, 0.506757], AUC 0.5002151 [0.492089, 0.507896] -> 0.50045 [0.493367, 0.507308], 0.49834707 [0.489589, 0.50642] | not part of this check |

Every interval still includes 0.50, so Check 1b and part 1 of Check 4b keep their verdicts. No run folder, registered rule
or verdict changed. The results page keeps the logged length-only values and, since this pass, says so next to them
(results.md line 47): at the page's three decimals the fixed function gives 0.499 [0.492, 0.506] for the boosted trees on
ns (the page has 0.500 [0.493, 0.507]) and an upper end of 0.509 for the boosted trees on ns_all (the page has 0.510);
nothing moves by more than 0.0011 in accuracy or 0.0028 in AUC. Rows with the wrapped value (`wrapped_rows.txt`, counted
with the study's loader): ns train 19,097 of 40,000, ns dev 9,629 of 20,000, ns_all train 19,233 of 40,000, ns_all dev
9,688 of 20,000. The study's own `test_wiki_groups.py` (`test_part1` runs the real ns_all length check) prints the same
fixed values.

#### Earlier builds made with the wrapped feature

Three earlier data-check runs used the same `length_feats`, and the results page quotes two of them (line 45):

- `20261001T092321Z_data_checks_exploratory` (first `ns` build): logistic regression 0.6788 [0.6723, 0.6852], boosted
  trees 0.6783, check1b_pass false (the page's "a length-only accuracy of 0.679"; also data card line 40).
- `20261001T094221Z_data_checks_ns_all_exploratory` (first `ns_all` build): logistic regression 0.5303 [0.5234, 0.5373],
  boosted trees 0.5319 [0.525, 0.5388], FAIL on length alone (1 skewed word); this FAIL led to the rebuild.
- `20261001T093113Z_data_checks_v2_exploratory` (second `ns` build): intervals [0.4875, 0.5014] and [0.4868, 0.5006].

Their data were overwritten by the rebuilds (`data/ns/*.npz` written 2026-10-01 09:38 UTC, `data/ns_all/*.npz` 09:43 UTC,
after those runs), so they cannot be re-run with the fixed function. The length FAIL of the first `ns_all` build is
therefore not re-checked. That it would still fail (the classifier also sees the two lengths and their minimum and
maximum) is UNVERIFIED.

#### Tests (done means tested)

All CPU, no GPU: `CUDA_VISIBLE_DEVICES` empty, 2 threads, nice 19, each test started after the CPU gate, scratch trees
outside the study. Every output named here is in `ws/htm`; its sha256 is in the block "Test outputs and records" below.
Times are the start and end lines of the output, or the runner's log.

- `src/test_data_checks.py` (new; synthetic rows): 1 of 5 passes and 4 fail on the version before the fix (exit 1;
  `test_data_checks_before.txt`, written 11:29:14Z); 5 of 5 pass on the fixed one (exit 0; `test_data_checks_after.txt`,
  11:29:16Z, and again `test_data_checks_r2b.txt`, 13:55:59Z to 13:56:03Z).
- `src/test_launchers.py` (new; stand-in jobs, fake `nvidia-smi` and `nvidia-cuda-mps-control`, every run folder in a
  scratch folder).
  - Round 1 (test file 71c9f266..., `src/interpreters.sh` 7bf7c936...; runner `run_launcher_tests.sh`, log
    `run_launcher_tests.log`, 11:42:16Z to 11:56:37Z): 24 of 24 on the fixed launchers with the current `env.sh`
    (`test_launchers_after.txt`) and with the HEAD `env.sh`, which puts no `python3` on `PATH`
    (`test_launchers_after_headenv.txt`); 24 of 24 in a path with a space (`test_launchers_after_spacepath.txt`,
    12:17:47Z to 12:20:39Z); on the HEAD launchers 3 pass and 16 fail with either `env.sh` (`test_launchers_before.txt`,
    `test_launchers_before_headenv.txt`).
  - Round 2 (test file 2154be76..., `src/interpreters.sh` 206f3017..., the current files, with two more checks for the
    bytecode export; runner `run_launcher_tests_r2.sh`, log `run_launcher_tests_r2.log`, 13:07:56Z to 13:23:25Z): 26 of
    26 in the same three settings (`test_launchers_r2_after.txt`, `test_launchers_r2_after_headenv.txt`,
    `test_launchers_r2_after_spacepath.txt`); on the HEAD launchers 3 pass and 18 fail with either `env.sh`
    (`test_launchers_r2_before.txt`, `test_launchers_r2_before_headenv.txt`). The round-2 numbers of this note come from
    these `_r2` files.
  - The failures on HEAD: the missing interpreter (exit 127 at `src/launch.sh` line 13 and `src/feas_slide11/launch.sh`
    line 8; with the HEAD `env.sh` also `/usr/bin/env: 'python3'`), plus checks of behaviour that HEAD never had (usage
    exit 2, a relative `HTM_PY`, a missing `HTM_PY` or `HTM_TOOL_PY`, the old virtualenv used when it exists, the
    bytecode export).
  - A third batch (`run_tests_r2b.sh`, log `run_tests_r2b.log`, started 13:55:49Z) repeated the data-check test (5 of 5)
    and the launcher tests with the current and the HEAD `env.sh` (26 of 26 each; `test_launchers_r2b_after.txt`,
    `test_launchers_r2b_after_headenv.txt`). It stopped during its space-path run: `test_launchers_r2b_after_spacepath.txt`
    (started 14:02:14Z, last written 14:03:02Z) has no summary line and no exit status, and the cause of the stop was
    not recorded. No result of this note rests on that file.
- The study's own CPU tests on the tree after the round-1 fixes: `src/test_e5a.py` 7 of 7 (`test_e5a_after.txt`,
  12:14:57Z to 12:15:22Z) and `src/test_wiki_groups.py` 10 of 10 (`test_wiki_groups_after.txt`, 12:15:22Z to
  12:23:00Z). Neither test uses the launchers or `src/interpreters.sh`.
- The AST check of the probe note (equal; `test_probe_docstring_only.txt`, written 12:01:35Z), and `shellcheck_b5.txt`
  (written 12:18:55Z).
- The second-round review re-ran these tests on the current files from 15:28:38Z to 15:49:55Z (judged by an AI model,
  not a person) and found every PASS and FAIL line equal to the saved outputs above. Its outputs were not kept; their
  sha256 are listed in its review file, which is not distributed.
- The `env.sh` of tm_language_model that the tests sourced: the current file (2bca37b6..., changed at 11:23:44Z for SHR-1 in the
  finish pass, uncommitted at the time) and its HEAD copy (fe72f61c...). A `__pycache__` folder in
  `../tm_language_model/src/tmlm/common/` was written at 11:24:02Z, before the first launcher test of this note (11:27Z);
  it is not from these tests.

Exact diffs of every changed file against HEAD and copies of the three new files are in `diffs_round2/` (with
`SHA256SUMS` and `file_times.txt`, the modification time of each change).

#### Hashes (pasted by command)

Study files: the 10 changed files at HEAD 1bb3935 (exported with git show) and now, then the 3 new files now (sha256sum output; read 2026-10-25T16:01:55Z). The three pages README.md, docs/results.md and docs/data_card_wikipedia_pairs.md change again when this note is linked and when the outcomes of HTM-01 and HTM-03 are added; their values here are those of the note's freeze.

```
778525b68fcea06aca53a807bfb4c32e7b6ab344d0703dfe813af9220b3d6afb  HEAD:studies/hierarchical_tm/src/launch.sh
dae7738a023d02d812ee38ac4fe583e565e67d463aca0c4f2c276fe61c9663d9  HEAD:studies/hierarchical_tm/queue/run_queue_mps.sh
48107df3f23ebb5e41d4096607810f9d7c095a1109d76c535ddc3a7c82c940ff  HEAD:studies/hierarchical_tm/queue/gpu_share_test.sh
5e7f4a0762f263cd6ac8bd017ee01cf2e6d6a60a94fc92eb56c75ae8623e4ab6  HEAD:studies/hierarchical_tm/src/feas_slide11/launch.sh
081533f102416c96fac0a1bf4e3df02f1ae7bdce955b019c691f8b5dfb6d8c99  HEAD:studies/hierarchical_tm/queue/run_queue.sh
896f2bdc7505fbd1dd965cd01c0da2e4876f7f4068d72c3971ca6b1395a26447  HEAD:studies/hierarchical_tm/src/data_checks.py
9925827b6b9ed147d0b5ff2afcd52b41f00f997a5e45d3c6185d1e4076536bd6  HEAD:studies/hierarchical_tm/src/recovered/e3fb_deletion_probe.py
d05202ab531bbc74f9daff4400675b97f03a9c1a0ff74fcac32650075554bae1  HEAD:studies/hierarchical_tm/README.md
a6b0fd9bf2cc84b802842849f169566f79cc25cda27392083785a50019ee4bde  HEAD:studies/hierarchical_tm/docs/results.md
93140ffcef6deb01c16e02214219728511a50eef59a2d727040956bd4854811d  HEAD:studies/hierarchical_tm/docs/data_card_wikipedia_pairs.md
faff0e7a7dcfad57733d6e792aa7003a460528f155f0f48a0e8345be8624c1f8  src/launch.sh
46e70e4ff077b2fbd06669a3ef346413ce8aab945b1bc75e14bcab078c9b6b81  queue/run_queue_mps.sh
459d45b71fb5f5b36fd890a2eccf721a5736d3acc3553cab40160e67d157ab7c  queue/gpu_share_test.sh
3d41c70a79ecfba6401314bbaee84805f60512908b83666baa25d6e92b7ba9b4  src/feas_slide11/launch.sh
900cc3b0ba2d307499a447526399c7c811d7b8353fea9b6ac1bfc98c8dfb64b4  queue/run_queue.sh
7523133e933459d552a355a28c0ac4967bcc9685be8d3c08d3a00fa42cbe6fe5  src/data_checks.py
1228a2a02610bf5a6e58e02018bd2c08f9ea717d06069f6b91610e72dd3b2976  src/recovered/e3fb_deletion_probe.py
39bc36451d5e16218c37955293efccd1ddee6a5b4b07e67760ba6984de52fc9a  README.md
e9ca0667dd9ec08e24663a30f3248598ac328f7db2ed603f548e5dc4ebd51562  docs/results.md
826fd22716157fc88a61d96d4a0066d07b77e5001f9df00d6d646bf2915f1ce4  docs/data_card_wikipedia_pairs.md
206f3017527cf5556580e5e3c43d8aa8e679d719139ad5764d735b06f4ee5f4e  src/interpreters.sh
2154be761c1d2b7ba248078554e1420bea55b55668ceff26ebed05a2ab1499f4  src/test_launchers.py
b928231ce0b206bd6e6d710ec1b6d39e04e24b02bdff5ef0529e59a888bc6e93  src/test_data_checks.py
2bca37b6d8c0277be2084b258e11b0de88e103119c320350893bd2f4275d9ab9  tm_language_model/env/env.sh
```

Test outputs and records (sha256sum output; files in ws/htm of the finish pass; test_launchers_r2b_after_spacepath.txt is the incomplete file named above):

```
322df08a62c7a052592ecce624b09ad04463257f9bb67491473acd032eb01620  test_data_checks_before.txt
5d5a76d3763365b6ea3f17dfaa45f5ee5165912b3c2a90cebeccc20094e4bba0  test_data_checks_after.txt
992c5a334a3a8dd16004e577045e5122255a8fc004da2baee179479dd0753df2  test_data_checks_r2b.txt
4fe084eef562403665a5dea087da41570f0944b4dce1ce7f0cb49e16b8140950  test_launchers_after.txt
ec1f91d9c7acb46977d450e94eb5ce1966e672c78ae9d87d3b6852507b2c139a  test_launchers_after_headenv.txt
518e1e0995d9cde27dc33bd8f8ead937c0a2367c04e8c22a22d389c1a915c6d6  test_launchers_after_spacepath.txt
188997eca901bc4fcc799225ccf8880d0228b086afc6e69bb81ea6b32cc8ea1a  test_launchers_before.txt
be5bca5dcbcdcbda3357fe25fb2ab1f657f897c99a2099d2a3dfeaec396505f6  test_launchers_before_headenv.txt
3baa8d0601a91a358c37a46bcbf7ca4c6b5c4b21ae46228e0a877a3b1800763b  test_launchers_r2_after.txt
faf818a6dc867dabf70b2c80288055a76e793ace048d2832b7aa2b1a79ff535b  test_launchers_r2_after_headenv.txt
42998c458e649a701ace539b77666845dfa7d6b17c02fd0092b5bab87d82089b  test_launchers_r2_after_spacepath.txt
af90859f34828d4d37e44565a411bcc22db22d94d4e69046d512f32092c4c2f4  test_launchers_r2_before.txt
3fcf9ae3503f52ab25fb51eb2c390fc5d5fcf1795e219e9e97b3151edc55602f  test_launchers_r2_before_headenv.txt
85333037864a097a5143f8fa72e549a5594f01a41afba73a0c0552054654b426  test_launchers_r2b_after.txt
6972c12afd912bc118c60c7d5ba5d23c9579caa32322a5cbade1b46b70837fb3  test_launchers_r2b_after_headenv.txt
b7a4ffaa1bd3c52c10c0814b44946c4b898dff56b5d08b54f1a226a696ce1fb9  test_launchers_r2b_after_spacepath.txt
16400e3c180fcc5035ea45bc9adcdf0e6b7a4ff2e16c51950f8ccff877883d66  run_launcher_tests.sh
1ae6ebfc5045cdbd54dd5a7f1c56d6ff3ff430f1e7a9bd4d919fdac0f1cd2228  run_launcher_tests.log
139cdedf5f3b800e28264467b8f8a99e898e351df1566dc5e14a5af661063e09  run_launcher_tests_r2.sh
7fff8fdafe99c88864df3c18117b194abbec0bde533ab6cce4aa3f41979e5278  run_launcher_tests_r2.log
7acb4cf0b79bfe778ea3c2bdf1f4b4e510d97c5abb8a20410663412268bf30bd  run_tests_r2b.sh
8edc680925378cebb80341f293162465feba9b4823f44474651f2ea4d7da09e7  run_tests_r2b.log
01b6a89f64f28211dedb9162f4262c84a2412df063829d62105031d72a2faae8  test_e5a_after.txt
3ebf194a8bb05b179e5665c5527b01af6ae8b9a4c20797ec0808f74806077f0a  test_wiki_groups_after.txt
4743b286fb0f0438da1a10ad29fad59955250aabe4e012647ea100e3ce347e90  test_probe_docstring_only.txt
4dabff8c53a2fdab7c7fbd8aa3aa06f0e855d28c2208f143f880ed04df5b0267  shellcheck_b5.txt
fd3049ae0205a1e129dc947f627be470129a02e4f343e2078141097eb8bd17a5  rerun_length_checks.py
f454440e3d92f9402196ba162fbe2e4c24abda5af2eb5e5947af1d9c943c69bf  length_checks_rerun.json
238216cda40f0de7d10d706bdaa56df05b7533d8a4abcd6e53ec7b0792085e76  length_checks_rerun.log
736d3d993b2043314e69b3dc29227aed746d1f6bd193d03dc577bed45c4460a7  wrapped_rows.py
b0868e6c869b075d5a2f92e57e58090da21566cbbf10ddc95a1e1aa587ad7603  wrapped_rows.txt
896f2bdc7505fbd1dd965cd01c0da2e4876f7f4068d72c3971ca6b1395a26447  head/src_head/data_checks.py
896f2bdc7505fbd1dd965cd01c0da2e4876f7f4068d72c3971ca6b1395a26447  head/data_checks_HEAD.py
9925827b6b9ed147d0b5ff2afcd52b41f00f997a5e45d3c6185d1e4076536bd6  head/e3fb_deletion_probe_HEAD.py
fe72f61cd184d9bec16e04a758fb39db9d0dfcd79fe0e90cb00d51863d958e6f  head/tlm_env_HEAD/env.sh
48107df3f23ebb5e41d4096607810f9d7c095a1109d76c535ddc3a7c82c940ff  head/launchers_HEAD/queue/gpu_share_test.sh
dae7738a023d02d812ee38ac4fe583e565e67d463aca0c4f2c276fe61c9663d9  head/launchers_HEAD/queue/run_queue_mps.sh
081533f102416c96fac0a1bf4e3df02f1ae7bdce955b019c691f8b5dfb6d8c99  head/launchers_HEAD/queue/run_queue.sh
5e7f4a0762f263cd6ac8bd017ee01cf2e6d6a60a94fc92eb56c75ae8623e4ab6  head/launchers_HEAD/src/feas_slide11/launch.sh
778525b68fcea06aca53a807bfb4c32e7b6ab344d0703dfe813af9220b3d6afb  head/launchers_HEAD/src/launch.sh
07c1c675cd1ff84b5cbc697ab4dee3ed75d450695c6d6cf6677922f0fa1a6720  diffs_round2/SHA256SUMS
94bb488980121d73b5b5ec4d82eab123ce626b33fbc5c330e6032bfb952fb550  diffs_round2/file_times.txt
```

#### Files outside the repository

Every record named above (test outputs, the re-run, the diffs) is kept in the admin folder of the finish pass
(`ws/htm`) and is not distributed; a repository page names them as "records of the finish pass, not distributed". No
test split was read: the re-run guards `np.load` against `test.npz` and opened only train and dev files. The final-test
split keeps its use count of 1.

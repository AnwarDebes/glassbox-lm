# language_part3: can a Tsetlin machine write text better than counting?

Two registered experiments of the third part of the project, each against counting systems tuned equally:

- **A-Struct** (`A/`): given a MultiWOZ 2.2 system turn's dialogue acts and the words written so far, does a TM that
  answers "is this candidate the next word?" beat the best conditional counting system and a counted decision tree on
  the same bits, in held-out bits per token?
- **C-core** (`C/`): is a TM that predicts the next word's class (65 classes) from 60 code bits as good as a counted
  class model on the same codes, inside KN5 + cache, on PTB?

**Status:** A-Struct stopped at its registered CPU gates; C-core passed its first gates, and its GPU step was never
frozen or run. Numbers in [docs/results.md](docs/results.md).

## Files

| Path | What |
|---|---|
| `A/REGISTRATION_R1.md`, `A/R1_A-Struct.frozen.json`, `A/MANIFEST_R1.sha256` | A-Struct's registration, freeze record and manifest |
| `A/astruct/`, `A/*.py`, `A/tests/` | its code (S0 data, S1 floors, S2 rows, arms, twins, gates; the K3c pilot) and tests |
| `A/out/` | the gate outputs of the runs that were made (`results.json`, `gates.json`) |
| `C/R1_C-core.md`, `C/R1_C-core.frozen.json`, `C/ccore/`, `C/scripts/`, `C/tests/` | C-core's registration, record, code and tests |
| `C/data/step6_decision.json` | C-core's gate decision |
| `chain/` | the wrapper that runs one Part 3 CPU job at a time across A and C, and its tests |

## Running

Every real job goes through `chain/part3_cpu_job.sh` (from this folder), which holds one lock for A and C and the CPU
gate; A and C each refuse to start without their frozen record and the study log line that holds its hash (`LOG.md`,
not distributed). The environments are `../tm_language_model/.venv-tm` and `../decision_layer/.venv-dl`. MultiWOZ 2.2
and the PTB splits are public; their copies, the caches and the run folders are not distributed.

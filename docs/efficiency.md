# Efficiency: time, power and energy

This page collects the cost figures of the programme and sorts them into what was measured, what is only bounded or
estimated, and what was never measured. Every value is read from the file named beside it. Paths are given from the
repository root; files under `studies/*/results/` and run folders are not distributed, the other files named here are.
All timings come from one shared workstation (one Tesla V100-SXM3-32GB GPU and two Intel Xeon Platinum 8168 CPU
packages) and are labelled rough in their files. No registered rule is about cost: these figures were recorded beside
the registered accuracy tests, not tested by them.

## In short

- The readable agent of [agent_vs_llm](../studies/agent_vs_llm/README.md), a Tsetlin Machine over a word table counted
  from C4 text with no neural network at any stage, chooses the next dialogue action in a median 0.50 to 2.98 ms on one
  CPU thread. LLM*, the prompted local 35B LLM that it matches within 3 points on three test sets (registered; the
  margin was set after a dev result, and on MultiWOZ it holds on seed 1 only), needs a median 10.6 to 11.63 s per
  decision on the V100: about 3,900 to 22,000 times more time.
- The energy of the readable agent was never measured: the container exposes no readable CPU energy counter. Charged
  the rated power of a whole CPU package, it would use 0.10 to 0.61 J per decision, against 1,993 to 2,175 J for LLM*
  (traced GPU board power times time), a ratio of about 3,500 to 21,000; charged both packages of the machine, about
  1,800 to 10,600.
- The speed is not specific to the Tsetlin Machine: TF-IDF logistic regression decides in 0.83 to 0.90 ms at the same
  accuracy. A fine-tuned transformer of about 110M parameters is 4.07 to 8.60 points more accurate and takes 195 to
  263 ms per decision on the same CPU thread, or about 1 ms in GPU batches; its power was not traced.
- Training is not cheaper: the Tsetlin Machine of the readable agent trains on the V100 in 140.5 to 1,202.5 s per
  seed, longer than fine-tuning the transformer on STAR and MultiWOZ and shorter on ABCD. The one measured comparison
  of training energy, in the language-model study, favours a small transformer by a factor of about 8.5.

## Measured

### Time per decision

| System | Hardware | STAR | ABCD | MultiWOZ 2.2 | Source |
|---|---|---|---|---|---|
| Readable agent, registered seed-1 models, median | one CPU thread, batch 1 | 2.71 ms | 0.50 ms | 2.98 ms | `studies/agent_vs_llm/runs/cost/agent_<dataset>_seed1_cpu.json` |
| The same, 90th percentile | one CPU thread, batch 1 | 2.98 ms | 0.55 ms | 3.29 ms | the same files |
| The same models with the C clause evaluator, median | one CPU thread, batch 1 | 0.96 ms | 0.30 ms | 1.07 ms | `studies/decision_layer/speed/cbench/bench_<dataset>.json` |
| TF-IDF logistic regression, median | one CPU thread, batch 1 | 0.90 ms | 0.83 ms | 0.88 ms | `studies/agent_vs_llm/runs/cost/tfidf_<dataset>_cpu.json` |
| Fine-tuned bge-base-en-v1.5, median | one CPU thread, batch 1 | 256 ms | 195 ms | 263 ms | `studies/agent_vs_llm/runs/cost/ft_<dataset>_cpu.json` |
| Fine-tuned bge-base-en-v1.5, dev pass divided by its rows | V100, batches of 128, float16, text tokenized before | 1.07 ms | 0.95 ms | 1.06 ms | `studies/agent_vs_llm/runs/ft_{star,abcd,mwoz}_bgebase_s1/result.json`, field `eval_s` of the last epoch |
| LLM* (Qwen3.6-35B-A3B at 4 bits, 50 retrieved examples), median per test row | V100 | 10.6 s | 11.13 s | 11.63 s | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| Qwen3.5-9B at 8 bits, 10 retrieved examples, mean per dev row | V100 | 0.8 s | 1.05 s | 1.06 s | dev table on the [results page of agent_vs_llm](../studies/agent_vs_llm/docs/results.md) |

- The readable agent was timed on 1,000 dev rows per dataset, and its dev accuracy on the CPU equals that of its run,
  so the timed models are the registered ones. They hold 15,997, 3,995 and 16,000 clauses with 816,158, 52,135 and
  936,967 literals, over a word table of 63,091 words by 500 dimensions (the same files).
- The C clause evaluator gives the same class sums as the NumPy path on all 6,908, 3,684 and 7,359 dev rows. Its
  figure is the sum of the medians of the text step, still in Python, and of the clause step; in the same run the NumPy
  path took 2.59, 0.44 and 2.93 ms. No registered run used it ([speed README](../studies/decision_layer/speed/README.md)).
- TF-IDF logistic regression was timed on 1,000 dev rows and the fine-tuned transformer on 300 dev rows (109.5M to
  109.6M parameters, field `n_params`). On the GPU the transformer took 7.4 s for 6,908 STAR dev rows, 3.5 s for 3,684
  ABCD rows and 7.8 s for 7,359 MultiWOZ rows.
- LLM* needs 24.4 GB of GPU memory ([energy note](../studies/agent_vs_llm/NOTE_energy.md)). The smaller Qwen3.5-9B
  configuration is about 11 to 13 times faster and less accurate: dev accuracy 0.528, 0.818 and 0.300 against 0.638, 0.864
  and 0.376 for LLM* on the same 500 dev rows.
- With older models on STAR (R19 of decision_layer), an agent that runs bge-large-en-v1.5 per decision took a median
  805.9 ms and an agent on the potion-base-32M word table 29.5 ms (`studies/decision_layer/logs/static_speed_R19.out`).

### Text to input bits

CLINC150 validation, 3,000 queries, one CPU thread, batches of 64; rough, read the ratios rather than the times.

| Encoder | Texts per second | Source |
|---|---|---|
| Lookup and OR of per-word bit codes (existsT) | 29,928 | `studies/text_understanding/speed_exists.json` |
| potion-base-32M, a static word table | 26,278 | `studies/text_understanding/speed.json` |
| all-MiniLM-L6-v2, a small transformer | 226 | `studies/text_understanding/speed.json` |
| bge-base-en-v1.5 | 34 | `studies/text_understanding/speed.json` |

### GPU board power under LLM*

nvidia-smi every 5 s for 3,600 s during the registered ABCD test run of LLM*: 720 samples, mean 188.5 W, median
164.2 W, minimum 89.2 W, maximum 337.9 W, the GPU busy 82% of the time on average, and no other job on the GPU
([energy note](../studies/agent_vs_llm/NOTE_energy.md)). It is the only power trace of agent_vs_llm. The trace
file is not distributed; the note gives the start of its SHA-256. No power was traced during the STAR and MultiWOZ runs.

### Training time on the V100

| Model | STAR | ABCD | MultiWOZ 2.2 | Source |
|---|---|---|---|---|
| Tsetlin Machine of the readable agent, 40 epochs, per seed (5 seeds) | 1,201.9 to 1,202.5 s | 140.5 to 142.9 s | 1,052.3 to 1,052.7 s | `studies/decision_layer/results/confirm/R22star/tm_mem_c4_s16/`, `R23abcd/tm_mem_c4_s16/` and `R24mwoz/tm_mem_c4_nrand/`, field `train_s` of `seed<k>.json` (not distributed) |
| Fine-tuned bge-base-en-v1.5, 4 epochs, seed 1, dev run | 835.6 s | 384.0 s | 859.6 s | `studies/agent_vs_llm/runs/ft_{star,abcd,mwoz}_bgebase_s1/result.json`, sum of `train_s` |

The Tsetlin Machine runs held at most 4,113, 1,942 and 3,880 MB of GPU memory; the fine-tuning runs spent another
23.4, 8.2 and 22.5 s on tokenizing. Training code that gives byte-identical results was made 6.8% to 16.9% faster
(FastCoTM, [speed README](../studies/decision_layer/speed/README.md)); no registered run used it.

### Training energy: the one measured comparison

From the first study, tm_language_model: the GPU's device-wide energy counter minus its idle power, on a shared GPU, so
both values are indicative only (their metric files say so).

| Model | Energy above idle | Validation bits per byte | Run id (not distributed) |
|---|---|---|---|
| Tsetlin Machine language model (V = 8000, 1,048,576 clauses), 4 epochs | 1,325,675 J | 2.1749 | `20260923T042350Z_CP_anchor` |
| Small transformer (6.85M parameters), 600 s of training | 155,520 J | 1.2341 | `20260922T172902Z_M2_baselines_transformer` |

The Tsetlin Machine model used about 8.5 times the energy for a model 0.94 bits per byte worse.

### The OR-group rule against the hierarchical package (or_group_tm, 2026-11-12)

From [studies/or_group_tm/docs/cost.md](../studies/or_group_tm/docs/cost.md), where every row names its source; one
CPU thread, rough. On the slide-10 sentences (direct labels, same files) the current version 0.3.4 of the
hierarchical package used a measured 105 to 120 J of GPU board energy per pass and 35 to 38 kJ per run of 300 passes,
and did not pass Check 2b on the DEV sentences (descriptive runs, checked only by their own scripts). The OR-group rule trained in 2.6 s on one CPU thread; charged a whole CPU
package (a bound, not a measurement) that is at most about 0.5 kJ per run, below 2% of the package's measured figure,
and its best rule met the bars of Check 2b with direct labels on 5 of 5 registered seeds (13 of 25 with the further seeds). On Wikipedia pairs it trains in 17.9 s per pass
over 40,000 pairs on one CPU thread (10 passes; the package needs 13.9 s per pass on the V100 for 100 passes), scores
a pair in 285 microseconds, and its model holds 22.1 kB of member words against 3.28 MB of include bits; per pair it
is not faster than the package's GPU bound. No energy of this rule was measured.

## Bounded or estimated, not measured

### Energy per decision

| | STAR | ABCD | MultiWOZ 2.2 |
|---|---|---|---|
| LLM*: traced GPU board power times mean time per dev row | 1,993 J | 2,175 J | 2,165 J |
| Readable agent charged one whole CPU package (205 W rated) | 0.556 J | 0.103 J | 0.611 J |
| Ratio | 3,588x | 21,184x | 3,542x |
| Readable agent charged both CPU packages of the machine (2 x 205 W) | 1.11 J | 0.21 J | 1.22 J |
| Ratio | 1,794x | 10,592x | 1,771x |

The first three rows are those of the [energy note](../studies/agent_vs_llm/NOTE_energy.md); the last two double its
figures for the readable agent and halve its ratios, because the workstation has two such packages (lscpu). What makes
them estimates:

- The figure of the readable agent is its median time times a rated power, not a reading: the container exposes no
  readable CPU energy counter. The rated power (TDP) is a sustained rating that short bursts can exceed, so the figure
  is not a strict upper bound; it charges the readable agent a whole package, or both, while it runs one thread.
- The figure of LLM* counts the GPU board only, not the host CPU, memory or power supply. Its power was traced during
  ABCD only, STAR and MultiWOZ reuse that mean, and its time is the mean of the dev rows, not the median of the test
  rows.
- With the 90th-percentile times of the readable agent the one-package ratios are 3,265x, 19,259x and 3,210x (energy
  note). The note also gives a per-core share of the package; that less conservative reading is not used here.

### The fine-tuned transformer on the GPU

No power was traced for it. For scale only: at the 89 to 338 W range of the LLM trace, its GPU time of about 1 ms per
row would come to roughly 0.1 to 0.4 J per decision, the same order as the estimate for the readable agent. A cost
advantage of the readable agent over a fine-tuned transformer is therefore not shown.

## Not measured

- The CPU energy of any system: the container exposes no readable CPU energy counter.
- The power or energy of the fine-tuned transformer, of TF-IDF logistic regression and of the training of the readable
  agent.
- The GPU power during the STAR and MultiWOZ runs of LLM*, and the power of the host beside the GPU board.
- A packed run-time size of the registered agents: only their clause and literal counts and the shape of the word
  table are recorded.
- Any low-power or embedded hardware.

## How to read these figures

What holds: for choosing among a fixed set of actions, once a few thousand logged dialogues exist, a small classifier
with no neural network at any stage matches a prompted local 35B LLM within the registered margins and decides about
3,900 to 22,000 times faster, on a CPU thread instead of a GPU.

What does not follow from them:

- An accuracy per joule: the energy of the readable agent is an estimate from a rated power.
- An advantage of the Tsetlin Machine: TF-IDF logistic regression has the same accuracy and the same speed
  ([agent_vs_llm results page](../studies/agent_vs_llm/docs/results.md)).
- An advantage over a fine-tuned transformer, which is 4.07 to 8.60 points more accurate and may cost about the same
  per decision on a GPU.
- A result with few logs: with 64 or 256 logged dialogues LLM* leads by 6.6 to 48.0 points on dev (log-size table of
  the same page).
- Text generation: the readable agent chooses among fixed actions and writes no text.
- A saving in training: the Tsetlin Machine is trained on the GPU and takes longer than fine-tuning the transformer on
  two of the three datasets.

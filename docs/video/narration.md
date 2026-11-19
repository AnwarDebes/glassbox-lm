# Video: final results

`glassbox-lm-final-results.mp4` is a short film of the final results of the project: 4 min 12 s, 1920 x 1080 pixels at 30 frames per second, H.264 video and AAC audio, in nine scenes. `glassbox-lm-final-results.srt` holds its captions as an SRT file (plain text with times); the same captions are also set in the picture, at the bottom of the frame. This page gives the narration as it is spoken, every number that appears on screen with the file of this repository it was read from, and how the video was made. The written report, with all numbers and their limits, is in [../report/](../report/).

## Narration

The narration is spoken in the first person by one synthetic voice. Numbers are read as words, and after a decimal point digit by digit (0.091 is read "zero point zero nine one").

### 1. The question

Could the knowledge of a language model be a list of rules that you can read, at a fraction of the power? In ten studies, I measured how far Tsetlin machines, which learn logical clauses, get on language. No model built here is called a large language model.

### 2. The clause form

The clause form comes from my supervisor's project proposal, which is unpublished. Inside a group, single words are joined by a logical OR. The groups are joined by a logical AND. The clause fires when every group finds at least one of its words in the text.

### 3. The learning rule

An earlier study, which ended mostly negative, tested the proposal's procedure with the package it names, the stock model. The latest study keeps the clause form and replaces the learning rule: one small automaton for every clause, group and word. It is a different model, and the earlier verdicts stand.

### 4. Test one: planted groups

Test one: artificial worlds that hide 8 groups of words. The registered bars are met on 5 of 5 seeds, with a group recovery of 1.0. But the rule was designed on worlds of the same generator.

### 5. Test two: slide 10, with direct labels

Test two: slide 10 of the proposal, one clause for a sentence family. With direct labels and a different model, the best single clause meets the bars of Check 2b on 5 of 5 registered seeds, on 8 of 20 further seeds, and on 13 of 25 pooled: about one half. The pair version, the procedure of the proposal, met none of the bars on the DEV sentences.

### 6. Test three: Wikipedia paragraph pairs

Test three: Wikipedia paragraph pairs. On held-back test pairs, the three rules of the registered test are met, as predicted from DEV: the OR-group rule is above the stock model in its registered setting by 0.091, 0.101 and 0.098 AUC of the vote sum. It is not a comparison at matched size or cost, so it does not tell what makes the difference, and the test has no rule against word overlap. The vote sum minus PPMI cosine, a baseline built from word counts, has its interval below zero on all three kinds of pairs. Minus logistic regression, the interval is below zero on the two kinds where that baseline is above chance.

### 7. Cost

The cost: the nine models of the design stage have a median size of about 22 kilobytes. Each registered test run took 192 to 230 seconds on one CPU thread. The energy is bounded, not measured. In the one measured comparison of training energy, indicative only, the Tsetlin machine language model used about 8.5 times the energy of a small transformer, and scored worse.

### 8. An earlier result: the readable agent

A readable agent, with no neural network at any stage, chooses the next action in logged dialogues. Within the registered margin of 3 points, it is not worse than a prompted local LLM on three test sets; on one of them, on seed 1 only, not on the 5-seed mean. The timings are rough: it decides in milliseconds on one CPU thread, against seconds on a GPU for the LLM. But it is better on no test set, a fine-tuned transformer is ahead, and TF-IDF logistic regression is level or ahead, and faster on two of the three.

### 9. Where everything is

Code, registrations and results pages are in the repository. No number of the studies can be recomputed from a clone alone. The registered rules were fixed before their runs. The negative results are reported like the others. The Tsetlin machine language model, in a single-seed run, missed its main accuracy goal against a 5-gram model. And the proposal's own procedure ended below word overlap. So the opening question stays open: no power saving was measured.

## Numbers on screen

Every number that the film shows, in the order of the scenes, with the text it stands in and the file of this repository it was read from (names that hold a digit, such as E5, R1 or Check 2b, are not listed). Numbers that the narration says are shown on screen while they are spoken. Scale ticks of the charts (0, 0.5 and 1 for the AUC; -4 to +4 points; 1 ms to 10 s; 31, 0 and -32 for the state of an automaton, whose range is given in `studies/or_group_tm/README.md`) are axis labels, not results.

| scene | number on screen | in the text | source file |
|---|---|---|---|
| 1 | 10 | 10 studies, 19 Sep to 12 Nov 2026 | `README.md` |
| 1 | 19 | 10 studies, 19 Sep to 12 Nov 2026 | `README.md` |
| 1 | 12 | 10 studies, 19 Sep to 12 Nov 2026 | `README.md` |
| 1 | 2026 | 10 studies, 19 Sep to 12 Nov 2026 | `README.md` |
| 3 | 12 | The learning rule of the latest study (or_group_tm, 12 Nov 2026) | `README.md` |
| 3 | 2026 | The learning rule of the latest study (or_group_tm, 12 Nov 2026) | `README.md` |
| 3 | 1984 | Clause form and training signal: known from earlier work (Valiant 1984; Littlestone 1988; Brown et al. 1992; Solan et al. 2005). | `studies/or_group_tm/README.md` |
| 3 | 1988 | Clause form and training signal: known from earlier work (Valiant 1984; Littlestone 1988; Brown et al. 1992; Solan et al. 2005). | `studies/or_group_tm/README.md` |
| 3 | 1992 | Clause form and training signal: known from earlier work (Valiant 1984; Littlestone 1988; Brown et al. 1992; Solan et al. 2005). | `studies/or_group_tm/README.md` |
| 3 | 2005 | Clause form and training signal: known from earlier work (Valiant 1984; Littlestone 1988; Brown et al. 1992; Solan et al. 2005). | `studies/or_group_tm/README.md` |
| 3 | 31 | 31 | `studies/or_group_tm/README.md` |
| 3 | 0 | 0 | `studies/or_group_tm/README.md` |
| 3 | -32 | -32 | `studies/or_group_tm/README.md` |
| 3 | 0 | member at 0 or more | `studies/or_group_tm/README.md` |
| 3 | 1 | recruit: should fire, does not fire. In the false group, the words of the text gain 1. | `studies/or_group_tm/README.md` |
| 4 | 64 | 64 or 256 words in 8 hidden groups | `studies/or_group_tm/docs/results.md` |
| 4 | 256 | 64 or 256 words in 8 hidden groups | `studies/or_group_tm/docs/results.md` |
| 4 | 8 | 64 or 256 words in 8 hidden groups | `studies/or_group_tm/docs/results.md` |
| 4 | 5 | 5 of 5 seeds | `studies/or_group_tm/verification/recompute_e1.out` |
| 4 | 1.0 | 1.0 | `studies/or_group_tm/verification/recompute_e1.out` |
| 4 | 0.80 | group recovery (bar: at least 0.80) | `studies/or_group_tm/docs/results.md` |
| 5 | 10 | Test two: slide 10, with direct labels (E3, registered) | `studies/or_group_tm/README.md` |
| 5 | 5 | 5 of 5 registered seeds [0.478, 1.000] | `studies/or_group_tm/logs/e3_report.json` |
| 5 | 0.478 | 5 of 5 registered seeds [0.478, 1.000] | `studies/or_group_tm/logs/e3_report.json` |
| 5 | 1.000 | 5 of 5 registered seeds [0.478, 1.000] | `studies/or_group_tm/logs/e3_report.json` |
| 5 | 20 | The model as a whole (at least one of its 20 clauses fires) accepts 0.117 to 0.304 of the deletion texts (bracket 6: 0.485 to 0.911). | `studies/or_group_tm/logs/e3_report.json` |
| 5 | 0.117 | The model as a whole (at least one of its 20 clauses fires) accepts 0.117 to 0.304 of the deletion texts (bracket 6: 0.485 to 0.911). | `studies/or_group_tm/logs/e3_report.json` |
| 5 | 0.304 | The model as a whole (at least one of its 20 clauses fires) accepts 0.117 to 0.304 of the deletion texts (bracket 6: 0.485 to 0.911). | `studies/or_group_tm/logs/e3_report.json` |
| 5 | 6 | The model as a whole (at least one of its 20 clauses fires) accepts 0.117 to 0.304 of the deletion texts (bracket 6: 0.485 to 0.911). | `studies/or_group_tm/logs/e3_report.json` |
| 5 | 0.485 | The model as a whole (at least one of its 20 clauses fires) accepts 0.117 to 0.304 of the deletion texts (bracket 6: 0.485 to 0.911). | `studies/or_group_tm/logs/e3_report.json` |
| 5 | 0.911 | The model as a whole (at least one of its 20 clauses fires) accepts 0.117 to 0.304 of the deletion texts (bracket 6: 0.485 to 0.911). | `studies/or_group_tm/logs/e3_report.json` |
| 5 | 8 | 8 of 20 further seeds [0.191, 0.639] | `studies/or_group_tm/logs/e3_report.json` |
| 5 | 20 | 8 of 20 further seeds [0.191, 0.639] | `studies/or_group_tm/logs/e3_report.json` |
| 5 | 0.191 | 8 of 20 further seeds [0.191, 0.639] | `studies/or_group_tm/logs/e3_report.json` |
| 5 | 0.639 | 8 of 20 further seeds [0.191, 0.639] | `studies/or_group_tm/logs/e3_report.json` |
| 5 | 13 | 13 of 25 pooled [0.313, 0.722] | `studies/or_group_tm/logs/e3_report.json` |
| 5 | 25 | 13 of 25 pooled [0.313, 0.722] | `studies/or_group_tm/logs/e3_report.json` |
| 5 | 0.313 | 13 of 25 pooled [0.313, 0.722] | `studies/or_group_tm/logs/e3_report.json` |
| 5 | 0.722 | 13 of 25 pooled [0.313, 0.722] | `studies/or_group_tm/logs/e3_report.json` |
| 5 | 95% | [exact 95% interval] | `studies/or_group_tm/logs/e3_report.json` |
| 6 | 0.735 | 0.735 | `studies/or_group_tm/logs/e5_report.json` |
| 6 | 0.750 | 0.750 | `studies/or_group_tm/logs/e5_report.json` |
| 6 | 0.614 | 0.614 | `studies/or_group_tm/logs/e5_report.json` |
| 6 | 0.826 | 0.826 | `studies/or_group_tm/logs/e5_report.json` |
| 6 | 0.852 | 0.852 | `studies/or_group_tm/logs/e5_report.json` |
| 6 | 0.711 | 0.711 | `studies/or_group_tm/logs/e5_report.json` |
| 6 | 95% | difference [95% interval] | `studies/or_group_tm/logs/e5_report.json` |
| 6 | +0.091 | +0.091 [+0.082, +0.099] | `studies/or_group_tm/logs/e5_report.json` |
| 6 | +0.082 | +0.091 [+0.082, +0.099] | `studies/or_group_tm/logs/e5_report.json` |
| 6 | +0.099 | +0.091 [+0.082, +0.099] | `studies/or_group_tm/logs/e5_report.json` |
| 6 | +0.101 | +0.101 [+0.093, +0.110] | `studies/or_group_tm/logs/e5_report.json` |
| 6 | +0.093 | +0.101 [+0.093, +0.110] | `studies/or_group_tm/logs/e5_report.json` |
| 6 | +0.110 | +0.101 [+0.093, +0.110] | `studies/or_group_tm/logs/e5_report.json` |
| 6 | +0.098 | +0.098 [+0.085, +0.110] | `studies/or_group_tm/logs/e5_report.json` |
| 6 | +0.085 | +0.098 [+0.085, +0.110] | `studies/or_group_tm/logs/e5_report.json` |
| 6 | +0.110 | +0.098 [+0.085, +0.110] | `studies/or_group_tm/logs/e5_report.json` |
| 6 | 0 | both intervals above 0 in each row: paired article bootstrap (shown) and seeds resampled | `studies/or_group_tm/logs/e5_report.json` |
| 6 | -0.026 | Vote sum minus PPMI cosine: -0.026, -0.034, -0.100 (sa, np, ns_all); minus logistic regression: -0.044, -0.051 (sa, np); each with its interval below 0. On ns_all the logistic regression is at chance (AUC 0.494). | `studies/or_group_tm/logs/e5_report.json` |
| 6 | -0.034 | Vote sum minus PPMI cosine: -0.026, -0.034, -0.100 (sa, np, ns_all); minus logistic regression: -0.044, -0.051 (sa, np); each with its interval below 0. On ns_all the logistic regression is at chance (AUC 0.494). | `studies/or_group_tm/logs/e5_report.json` |
| 6 | -0.100 | Vote sum minus PPMI cosine: -0.026, -0.034, -0.100 (sa, np, ns_all); minus logistic regression: -0.044, -0.051 (sa, np); each with its interval below 0. On ns_all the logistic regression is at chance (AUC 0.494). | `studies/or_group_tm/logs/e5_report.json` |
| 6 | -0.044 | Vote sum minus PPMI cosine: -0.026, -0.034, -0.100 (sa, np, ns_all); minus logistic regression: -0.044, -0.051 (sa, np); each with its interval below 0. On ns_all the logistic regression is at chance (AUC 0.494). | `studies/or_group_tm/logs/e5_report.json` |
| 6 | -0.051 | Vote sum minus PPMI cosine: -0.026, -0.034, -0.100 (sa, np, ns_all); minus logistic regression: -0.044, -0.051 (sa, np); each with its interval below 0. On ns_all the logistic regression is at chance (AUC 0.494). | `studies/or_group_tm/logs/e5_report.json` |
| 6 | 0 | Vote sum minus PPMI cosine: -0.026, -0.034, -0.100 (sa, np, ns_all); minus logistic regression: -0.044, -0.051 (sa, np); each with its interval below 0. On ns_all the logistic regression is at chance (AUC 0.494). | `studies/or_group_tm/logs/e5_report.json` |
| 6 | 0.494 | Vote sum minus PPMI cosine: -0.026, -0.034, -0.100 (sa, np, ns_all); minus logistic regression: -0.044, -0.051 (sa, np); each with its interval below 0. On ns_all the logistic regression is at chance (AUC 0.494). | `studies/or_group_tm/logs/e5_report.json` |
| 7 | 10% | One shared machine; repeats spread by up to about 10%. | `studies/or_group_tm/docs/cost.md` |
| 7 | 22.1 | 22.1 kB | `studies/or_group_tm/docs/cost.md` |
| 7 | 192 | 192 to 230 s | `studies/or_group_tm/logs/e5_report.json` |
| 7 | 230 | 192 to 230 s | `studies/or_group_tm/logs/e5_report.json` |
| 7 | 205 | Energy: bounded, not measured. The bound is rated power (205 W) times the time. | `studies/or_group_tm/docs/cost.md` |
| 7 | 1,325,675 | Tsetlin machine language model: 1,325,675 J | `docs/efficiency.md` |
| 7 | 6.85M | small transformer (6.85M parameters): 155,520 J | `docs/efficiency.md` |
| 7 | 155,520 | small transformer (6.85M parameters): 155,520 J | `docs/efficiency.md` |
| 7 | 8.5 | about 8.5 times the energy, for a model 0.94 bits per byte worse | `docs/efficiency.md` |
| 7 | 0.94 | about 8.5 times the energy, for a model 0.94 bits per byte worse | `docs/efficiency.md` |
| 8 | 95% | readable agent minus LLM*, accuracy in points [95% interval], seed 1 | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | 1 | readable agent minus LLM*, accuracy in points [95% interval], seed 1 | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | 35B | LLM*: a prompted local 35B LLM | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | +1.50 | STAR +1.50 [-0.02, +3.05] | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | -0.02 | STAR +1.50 [-0.02, +3.05] | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | +3.05 | STAR +1.50 [-0.02, +3.05] | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | +0.00 | ABCD +0.00 [-1.17, +1.14] | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | -1.17 | ABCD +0.00 [-1.17, +1.14] | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | +1.14 | ABCD +0.00 [-1.17, +1.14] | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | -1.12 | MultiWOZ -1.12 [-2.78, +0.57] | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | -2.78 | MultiWOZ -1.12 [-2.78, +0.57] | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | +0.57 | MultiWOZ -1.12 [-2.78, +0.57] | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | 3 | registered margin: 3 points | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | 3-point | The 3-point margin was set after a dev result. | `studies/agent_vs_llm/docs/results.md` |
| 8 | 1 | Rule NI: met on all three on seed 1; on the 5-seed mean not on MultiWOZ | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | 5-seed | Rule NI: met on all three on seed 1; on the 5-seed mean not on MultiWOZ | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | 0.50 | readable agent, one CPU thread: 0.50 to 2.98 ms | `studies/agent_vs_llm/runs/cost/agent_<dataset>_seed1_cpu.json` |
| 8 | 2.98 | readable agent, one CPU thread: 0.50 to 2.98 ms | `studies/agent_vs_llm/runs/cost/agent_<dataset>_seed1_cpu.json` |
| 8 | 10.6 | LLM*, on a GPU: 10.6 to 11.63 s | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | 11.63 | LLM*, on a GPU: 10.6 to 11.63 s | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | 3,900 | about 3,900 to 22,000 times the time of the readable agent | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | 22,000 | about 3,900 to 22,000 times the time of the readable agent | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | 3 | Better than LLM* by 3 points: met on no dataset. | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | 110M | Fine-tuned transformer (about 110M parameters): ahead by +4.07 to +8.60 points. | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | +4.07 | Fine-tuned transformer (about 110M parameters): ahead by +4.07 to +8.60 points. | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | +8.60 | Fine-tuned transformer (about 110M parameters): ahead by +4.07 to +8.60 points. | `studies/agent_vs_llm/results_REG_agent_vs_llm/results.json` |
| 8 | 0.94 | TF-IDF logistic regression: level, or 0.94 points ahead on ABCD, and faster on STAR and MultiWOZ. | `studies/agent_vs_llm/docs/results.md` |
| 9 | 4.0 | Code and documents: CC BY-NC 4.0 | `README.md` |
| 9 | 2.2193 | Language model: H1.1 not met (2.2193 bits per byte, 5-gram 1.3719; on seed 1 only) | `README.md` |
| 9 | 5-gram | Language model: H1.1 not met (2.2193 bits per byte, 5-gram 1.3719; on seed 1 only) | `README.md` |
| 9 | 1.3719 | Language model: H1.1 not met (2.2193 bits per byte, 5-gram 1.3719; on seed 1 only) | `README.md` |
| 9 | 1 | Language model: H1.1 not met (2.2193 bits per byte, 5-gram 1.3719; on seed 1 only) | `README.md` |
| 9 | -0.082 | The proposal's procedure with its package: final test below word overlap on both Wikipedia tasks (-0.082 [-0.093, -0.071] and -0.106 [-0.116, -0.096] AUC) | `docs/results-overview.md` |
| 9 | -0.093 | The proposal's procedure with its package: final test below word overlap on both Wikipedia tasks (-0.082 [-0.093, -0.071] and -0.106 [-0.116, -0.096] AUC) | `docs/results-overview.md` |
| 9 | -0.071 | The proposal's procedure with its package: final test below word overlap on both Wikipedia tasks (-0.082 [-0.093, -0.071] and -0.106 [-0.116, -0.096] AUC) | `docs/results-overview.md` |
| 9 | -0.106 | The proposal's procedure with its package: final test below word overlap on both Wikipedia tasks (-0.082 [-0.093, -0.071] and -0.106 [-0.116, -0.096] AUC) | `docs/results-overview.md` |
| 9 | -0.116 | The proposal's procedure with its package: final test below word overlap on both Wikipedia tasks (-0.082 [-0.093, -0.071] and -0.106 [-0.116, -0.096] AUC) | `docs/results-overview.md` |
| 9 | -0.096 | The proposal's procedure with its package: final test below word overlap on both Wikipedia tasks (-0.082 [-0.093, -0.071] and -0.106 [-0.116, -0.096] AUC) | `docs/results-overview.md` |
| 9 | 0.7983 | Synthetic sequences (E6): not met (test accuracy at most 0.7983, bar 0.95) | `studies/or_group_tm/README.md` |
| 9 | 0.95 | Synthetic sequences (E6): not met (test accuracy at most 0.7983, bar 0.95) | `studies/or_group_tm/README.md` |

## How the video was made

The video is rendered from code. A storyboard holds every sentence of the narration and every text and number on screen, each with the fact it states and the file of this repository that fact was read from; a program draws every frame from it. The narration is one synthetic voice, generated on this machine. Each sentence was generated in several takes, and a take was used only when measurements showed it complete: silence at both ends, a pace inside fixed bounds and an automatic transcription that matches the script word for word (for the name Tsetlin, a transcription given the name as a hint). Two sentences ("Could the knowledge of a ..." and "No number of the studies ...") came out faster than those bounds in every take; their takes were slowed by 10 to 12 percent, with the pitch kept, and then measured like the others. The takes were aligned word by word, so that each figure and each caption appears with the words that say it, and the narration was brought to -16 LUFS. There is no music and no sound effect. The build scripts are not part of the repository.

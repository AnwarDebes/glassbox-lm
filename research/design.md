# Design: a count floor and a test bench for Tsetlin learning

Working name: Tally. This page is a design. Nothing in it has been run, and every number marked as expected is an estimate from published tables. It follows from the [findings](findings.md) and is the source of the [roadmap](roadmap.md).

## What it is, and what it is not

Tally is two things that are kept apart.

1. **A count floor.** An exact index of the text, word-category codes learned from counts, counted patterns over words and categories, a cache and a copy rule. It contains no Tsetlin Machine. It is a careful rebuild of the count-based mixtures of 2010 to 2011, and every Tsetlin language model has to beat it to matter.
2. **A test bench.** Places in that model, called slots, where a Tsetlin Machine can be put and compared, on the same bits, with a twin that only counts and with standard learners. The Tsetlin result of the programme is the measured difference in each slot.

Tally is not the logic based LLM. It is the measuring ground on which one can be built.

## Expected numbers

Penn Treebank, perplexity, lower is better. All values in the first column are estimates.

| What | Expected | Basis |
|---|---|---|
| Count floor with fitted mixing weights | 110 to 118 | published: 141.2 for a Kneser-Ney 5-gram, 125.7 with a cache, 115.6 with a cache and a class model, 108.7 for the best mix without a neural network in the same 2011 table |
| Count floor with no fitted number at all | unknown, possibly no better than 125.7 | such weights drift to the single best member |
| What Tsetlin learning adds on top | unknown; the prior is small | in the exploratory run of tm_language_model the learned clauses added 0.0002 bits per byte over the best count stack |
| Tuned LSTMs and Transformer-XL on the same data | 54.5 to 59.6 | published; the floor would be at about twice their perplexity |

The first Tsetlin steps on text (B2, B3 and B5 below) are more likely than not to give another negative result. The design looks for positive Tsetlin results in line T.

## The slots

- **Arbiter.** Several simple members propose the next word, and a Tsetlin Machine decides which proposal to trust. This is a yes/no classification with frequent inputs, the kind of task Tsetlin Machines handle. A warning from this repository: in the final test of hierarchical_tm a logistic regression is well above the Tsetlin model (0.869 against 0.735).
- **Category predictor.** Predict the category of the next word from the categories of the last 5 to 32 words. Three learners side by side: a flat Tsetlin Machine, the Hierarchical Tsetlin Machine with the categories as its OR-groups, and plain counts.
- **Members.** The machine types of the author's earlier repositories (the Multi-HGTM head, ConfTM, the router) and the Graph Tsetlin Machine as team members that stay only if they save bits.

## Three learning rules

All three are known methods; only their use with Tsetlin Machines was not found. Each is tested where it is well defined.

- **Team margin.** A Tsetlin Machine in a team learns from an example with a probability equal to the error of the rest of the team (boosting by filtering). Tested on Tsetlin Machine composites (B6). Known weak point: where the team is already right on the training data, members stop learning.
- **Marginal bit.** The private reward of a member is the number of bits the team saved because it spoke (difference rewards). Tested first on a synthetic team with known answers (B1).
- **Dual-state trial.** An old rule and its changed version are both kept and counted until a small counter decides; keeping both is allowed. This is a sequential test (Wald; Hellman and Cover 1970), not a state of an automaton. Tested on the same synthetic team.

## Where the three ideas end up

- **NTMA** becomes members that each learn from the real target, seen as different views of the input, with a gate; the team-margin rule is its learning signal.
- **CDE** becomes the category code of the count floor, many bits per word learned from counts, and test A2 measures which code helps.
- **QTMA** is not in the first build. Ignoring a feature and abstaining are standard parts. One cheap test is parked (B8): one signed counter per feature against the two standard automata.

## Rules for every experiment

- The pass and fail numbers are written down before the run.
- Five data splits, cut by document.
- A model never sees the token it is scored on.
- The baseline runs first, and at full strength.
- A Tsetlin part must beat a twin that only counts.
- At least 5 seeds, and 10 for a pass or fail claim. One seed is exploratory.

## Experiment lines

**Week 0, reading and measuring.** Read the papers listed as not read on the findings page. Measure how much CPU is free on the shared machine. Install a reference n-gram toolkit.

**Line T, Tsetlin first.** It shares no code with the language model and can start together with line A.

- T0: with the CAIR library, repeat the finding of tm_language_model that counted votes beat learned votes on the same clauses for next-word prediction. Either outcome is a result about the Tsetlin Machine.
- B1: the synthetic team game for the marginal bit and the dual-state trial.
- B6: team margin against blind composites on Fashion-MNIST and AG News, with the controls (one machine with the same total clauses, no communication, a fully coupled team).

**Line A, the count floor (about 7 to 9 weeks for one person).**

- A1: protocol, noise, and a fair measure of the room an arbiter has.
- A2: the CDE test. Which category code helps: learned classes, the k-means classes already measured in clause_lm, predefined categories, random codes, the OR-groups of hierarchical_tm. Pass: at least 5% better than without classes. Word similarity (SimLex-999) is measured as well.
- A3: the full floor against Kneser-Ney plus cache.
- A4: raw text with unknown words, and a small test of copying a name never seen in training.

**Line B, the slots (after A3).** B2 the arbiter; B3 a window test for the category predictor, with the Hierarchical Tsetlin Machine as one arm; B4 a search for pattern structure; B5 the machine types as members; B7 category-coded hypervectors in the Graph Tsetlin Machine; B8 (parked) the three-action automaton. The arms with the Hierarchical and the Graph Tsetlin Machine need a GPU; the published demos of both packages are reproduced on the machine first.

**Line C, scale and value.** Larger data; tests for cost, trace, removal of a document and abstaining; a neural comparison as a curve over training budgets.

**Line D, the bridge (diagnostic only, needs a GPU).** A small transformer labels each position with a short code, Tsetlin clause banks learn the codes, and counts read out the next word.

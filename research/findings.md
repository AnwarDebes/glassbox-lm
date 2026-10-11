# What the literature says about the three ideas

Literature search of 2026-10-10 on the ideas of the [ideas page](ideas.md). Nothing on this page has been run. The limits of the search are on the [index page](README.md#how-far-to-trust-these-pages); a statement about a study of this repository repeats its results page.

## In short

- **NTMA.** A team of Tsetlin Machines is shown to work in one form only: every member learns from a real target, and the outputs are combined at the end. Stacks, in which one machine reads the clause outputs of another, failed or lost to a single layer.
- **QTMA.** Qubits do not let a machine go through all possibilities at once. The useful part of the idea is classical: a third state, abstaining, and a probability.
- **CDE.** One predefined category per word is too weak, and the basic idea was built in 2015. The form with evidence is many yes/no category bits per word, learned from counts, with the word kept beside them.
- **The bottleneck behind all three** is the learning signal: how each part learns without backpropagation. No method without it has been shown competitive for language models at scale.
- **The yardstick.** On Penn Treebank (perplexity, lower is better) a Kneser-Ney 5-gram scores 141.2, with a cache 125.7, and the best mix without a neural network in the same 2011 table 108.7 (Mikolov et al. 2011). No Tsetlin language model with a proper probability score against a count model was found in print.

## NTMA: a team of Tsetlin Machine types as neurons

### Tested before

Partly. No published team that mixes different Tsetlin Machine types was found. Teams and stacks of one type exist.

| What | Who | Result |
|---|---|---|
| Specialists trained separately, class sums added (TM Composites) | Granmo 2023; follow-up 2024 | CIFAR-10: 75.1% with 4 members in 2023 (best single member 63.5%); 82.8% with 22 members in 2024 (best single member 75.4%). Doubling the clauses moves 82.7% to 82.8% |
| Weighted composites on multi-sensor data | two journal papers of 2026 | not read; titles only |
| A gated mixture of Tsetlin experts (MixCTME) | IEEE Internet of Things Journal, 2025 | not read |
| 49 small machines passing clause outputs, two layers | Rudin et al. 2026 (preprint) | MNIST 94.40%, below a small single machine (96.73%) |
| Layers trained one after another on the labels (Multi-Layer TM) | Tarasyuk et al., ISTM 2024 | second layer plus 0.3 to 3.7 points, third layer nothing; no control with the same total number of clauses |
| Depth inside the clause (Graph TM) | Granmo et al. 2025/2026 | depth 2 against 1: virus genomes 59.81% to 95.14%; sentiment gains inside the error bars |
| AND-OR trees inside a clause (Hierarchical TM) | CAIR, code 2026 | demos only: noisy parity 99.50% (average of 10 runs), Hex 99.45% |
| Recurrent clause feedback | Kumar et al. 2026 | no better than the control without feedback on a Boolean finite-state benchmark (abstract only) |
| Brain-inspired binary or logical systems aiming at language | many | none found that reached a competitive language model with its own local learning rule; the larger systems kept gradient training |
| Mixtures of expert language models | BTM 2022, c-BTM 2023 | work when the experts start from a shared model, are split by learned clusters and are blended (perplexity 13.82 to 13.05 with the top 4 of 32); a hard choice of one expert loses on unseen domains |
| Many simple predictors, each trained on the target, mixed by a gated mixer (PAQ, cmix, gated linear networks) | Mahoney 2005; Veness et al. 2017, 2019 | enwik8 in bits per character: PPMd 1.71, paq8hp12 1.30, cmix 1.17 (cmix has a neural part) |

In the author's earlier repositories ([upstream page](../docs/upstream-repositories.md)), read from their files and not run again:

- Works: Multi-HGTM. Each tower learns from the label and a Tsetlin head reads the clause bits of the towers. It scores 0.968 against 0.727 for adding the votes, on a synthetic task built to need an AND across views (5 seeds). On the one real data set the gain is not significant (0.928 against 0.925).
- Fails or loses to a single layer: every scheme that feeds the clause outputs of one machine into another in depth (small synthetic tasks or MUTAG; 1 to 5 seeds). The joint two-layer HGTM stayed at the majority rate in all 18 two-layer runs, and the one-layer runs of the same sweeps were mostly at chance too.
- The working head and the failing stacks were never run on the same task.

### What the brain offers

- "Three types of neurons" (sensory, inter, motor) says where a cell sits. It does not say how the cell computes or learns, and brain atlases count thousands of cell types.
- What can be mapped onto a Tsetlin Machine, as an analogy and not a tested result: clause = dendritic branch (an AND), vote sum = cell body, positive and negative clauses = excitation and inhibition, feedback = a local rule plus one broadcast signal.
- The team of the brain is a gate that switches sub-circuits on, not a vote among equals. It also has a fast store for single episodes next to a slow learner of structure.

### Limits others met

- The composite gain flattens (82.7% to 82.8% when the clauses double).
- Greedy stacking loses to one machine or overfits. A stacked result without a control of the same size proves nothing.
- One reward shared by many units does not scale. The convergence proofs of the Tsetlin Machine cover single flat machines only.
- Language models trained without backpropagation show large gaps (perplexity 52.0 against 29.8 with direct feedback alignment; 387 against 150 with forward gradients).

### The form with evidence

1. Every member votes on a real target. If nothing is stacked, a team of types is one Tsetlin Machine whose clauses are split into groups, each group seeing the input in a different way.
2. Types are views, plus a gate that decides which groups and experts are used.
3. Depth inside the clause (Graph TM, AND-OR), or at most a second level trained on the same real targets, always with a control of the same total size.
4. Experts: one global model first, specialists on learned clusters, a blend of the top few, always mixed with the global model.
5. The six challenges named in the README of the Hierarchical Tsetlin Machine repository (conceptualization, introspection, transparency, lifelong learning, generalization, reactive and reflective thinking) are a better frame than neuron types.

## QTMA: an automaton that is both states at once, on qubits

### Tested before

Yes, in several different senses.

| What | Who | Result |
|---|---|---|
| Tsetlin logic over quantum measurements, automata classical, simulated on at most 6 qubits | Bhatia 2026 (preprint) | 1.000 when the right measurement basis is given by hand, 0.062 otherwise; the paper says itself that it is no claim of quantum advantage, and ridge regression gets the same 1.000 on the same features |
| Tsetlin Machine on noisy thermodynamic gates | Bristol 2026 (preprint) | level with a normal machine at threefold redundancy (93.7 against 94.0), 48.1% without |
| Fuzzy Tsetlin automata that emulate qubits and quantum gates classically | Kagan and Rybalov 2025/2026 | read in part: an emulation of quantum gates with no learning, no reward and no classifier |
| Automaton as a probability over its states | Probabilistic TM, 2024 | calibration on a toy set only |
| One clause voting for one class and against another | Coalesced TM 2021 | Fashion-MNIST 71.99% to 89.66% at 50 clauses per class |
| Three-valued systems at scale | BitNet b1.58 and others | ternary weights match full precision in large language models, but are trained by backpropagation with hidden real-valued weights |
| Quantum language processing on real hardware | several groups | toy scale (130 sentences, 17 words); a simulated quantum transformer scores 122.0 on Penn Treebank against 97.0 to 100.1 for a one-layer transformer |

### Why qubits do not do what the idea hopes

- A measurement returns one outcome. For search without structure the proven best is a square-root gain (Bennett, Bernstein, Brassard and Vazirani 1997; Grover), not "all at once".
- For learning from ordinary examples with no assumption on the distribution, quantum and classical learners need the same amount of data up to a constant.
- The known quantum gains for learning Boolean rules need the data itself as a quantum state and a uniform or product distribution. Text is neither.
- On early fault-tolerant machines a square-root gain is not expected to pay, because of the cost of error correction (Babbush et al. 2021).

### What is useful in it

- "True, false or both" is a logic with three or four values, not quantum mechanics. A standard Tsetlin Machine already has four states per feature and clause (x, NOT x, both, neither), and its "both" state is dead: a clause with x AND NOT x never fires.
- The form with evidence: drop the qubits and the word "quantum" for the language model. One signed counter per feature (include x, include NOT x, or do not care) was not found in the Tsetlin Machine literature; rule conditions over 0, 1 and "do not care" are standard in learning classifier systems. Use the third value where it earns something, for example a clause that abstains instead of guessing, and take the probability from counts and not from the vote sum.
- If a quantum angle is wanted for its own sake: a small theory study on a simulator, or Tsetlin Machines as readable rule learners for quantum measurement data.

## CDE: every word into categories, every category a dimension

### Is the diagnosis right

Partly. Boolean input is a real, named limit of the Tsetlin Machine (Graph TM paper). The notes of the CAIR repositories list other things as open as well: hierarchy above the word, flat against deep, relative position, one representation per word in context, less random hypervectors. The Tsetlin embedding papers name training time as their limit. So the embedding is one open problem among several.

### Tested before

Yes. As a word representation it was built once (Faruqui and Dyer 2015); categories were used as an add-on many more times.

| What | Who | Result |
|---|---|---|
| Binary category vectors for words from lexicons (172,418 features, 119,257 words) | Faruqui and Dyer 2015 | better than word2vec on true similarity (SimLex 56.6 against 43.6), much worse on relatedness (44.6 against 65.6); best when joined with word2vec |
| Class-based count language models | Brown 1992; Goodman 2001; Model M 2009 | small gains over a word-count baseline that shrink with data |
| Word tree made by experts on the output side | Morin and Bengio 2005 | faster, but worse than the flat model; trees learned from data removed the loss (Mnih and Hinton 2008, second-hand) |
| Class first, then word, in a neural model | Chen et al. 2016 | Penn Treebank 138.2 against 123.8 for the flat output |
| Cache, skip n-grams and classes together, no neural network | Momtazi et al. 2010, in Mikolov et al. 2011 | Penn Treebank 116.6 alone and 108.7 when mixed with Kneser-Ney and a cache, against 141.2 for Kneser-Ney |
| Long-distance trigger pairs | Rosenfeld 1996 | 32 to 39% lower perplexity for the whole adaptive model (abstract only) |
| Tsetlin word embeddings (TM-AE and successors) | CAIR 2023 to 2026 | similarity average 0.52 against 0.58 for word2vec; SimLex 0.14 against 0.26; slow to train |
| Position-marked context words and part-of-speech classes as Tsetlin literals | Abdelwahab, Granmo and Jiao, ISTM 2025 | text classification only (IMDB 90%); no language-model score |
| Tsetlin text predictors | Gorji, Granmo and Goodwin 2023; Tsetlin.jl demos 2025 to 2026 | next letter inside a word, compared with small neural nets only; text samples without a score |
| Word classes in this repository | clause_lm | k-means classes move a Kneser-Ney trigram from 161.72 to 157.24 ([results](../studies/clause_lm/docs/results.md)) |

### What neural language models do with categories

- A token carries many features at once: millions of features exist in a large model, fewer than 300 active per token. Categories are real inside the models, but the features are graded and the granularity is not fixed.
- A fixed vector per word explains under 5% of how the representation of the word varies in context.
- Copying from the context and binding need pointers, not categories.

### The step after the lookup

This is the question the idea left open. Three things, each with a logic-friendly form; only the first is in the design.

1. **More bits per word.** A code of yes/no bits from coarse to fine, learned from counts, with hand-made categories as extra bits.
2. **Context decides (open).** The bits that hold for this occurrence are what the word allows AND what the context predicts.
3. **Categories on the output side (open).** Predict the category of the next word first, then the word inside it. In the neural test above this was worse than the flat output.

## What the studies of this repository already say

- **The token-level model lost to counts.** 2.2193 bits per byte against 1.3719 for a Kneser-Ney 5-gram, on one seed ([tm_language_model](../studies/tm_language_model/docs/results.md)).
- **Counting beats voting.** A count read-out of the same clause bank scores 1.8280 against 2.1749 for its signed votes, still above the n-gram baseline.
- **The clauses chosen by learning added little over count models.** In the exploratory feasibility run the mixture with learned clauses is 0.0002 bits per byte ahead of the best count stack; in the confirmatory self-supervised runs the trained banks equalled their own barely trained bank (plus 0.09 points) and stayed 5.85 points below bag of words.
- **The hierarchical model is below word overlap** on its one final test: AUC 0.735 against 0.817 on same-article pairs and 0.750 against 0.857 on next-paragraph pairs ([hierarchical_tm](../studies/hierarchical_tm/docs/results.md)).

Three lessons for any new design:

1. Let counts give the probabilities. Whether Tsetlin learning chooses useful features is not shown; it is the question to test, not a result to build on.
2. The training signal decides what a learned group means.
3. Keep the untrained, random, bag-of-words and count-vector controls in every test. So far learning has not beaten them by much.

## The board questions, short answers

- **Must all rules be visited for each token (photo 27)?** No. Rules about exact word sequences are rows in a table, and the right row is found by lookup. Only a few thousand clauses are evaluated, and those can be indexed.
- **Constant token cost, all tokens at once (photo 19)?** Nearly constant for scoring one word. All at once: no, because each word depends on the previous ones. That has nothing to do with interpretability.
- **A context of 10 million tokens (photo 30)?** Stored and searched exactly: yes, cheaply. Understood: no. "Rules do not forget" is true for an index and unproven for learned rules.
- **Compress the rules like TurboQuant (photos 31, 32)?** TurboQuant does not apply to yes/no rules. The better answer is not to store word-sequence rules at all and to keep the text in an index.
- **Many small logic based LLMs (photo 29)?** As different views that all see all the data and are mixed with a global member: yes. As small separate models with one picked per prompt: the evidence is mixed.
- **True, false or both (photos 25, 26)?** Not quantum. Useful as "ignore", as abstaining, and as keeping two versions of a rule until the evidence decides. One correction to the board: AND narrows inside a rule, votes are added across rules, and OR does not eliminate.

## What is new and what is not

Survives in narrow form, where a few searches found no prior work:

1. A Tsetlin Machine as the arbiter in a count-and-lookup language model.
2. A measurement protocol: Tsetlin Machines inside a language model, reported in bits per token against Kneser-Ney, a cache and standard learners, each against a twin that only counts.
3. Team-margin fine-tuning of separately trained Tsetlin Machines (the method itself is known as boosting by filtering).
4. Marginal-bit rewards for Tsetlin automata (the mathematics is known as difference rewards).
5. One three-action automaton per feature inside a Tsetlin Machine.

Not new, to be cited and not claimed: the whole count floor, class-then-word prediction, matching at category level, abstaining, a second layer trained on the labels, mixtures of experts, Tsetlin text generation as such, and category features as Tsetlin input. At least three Tsetlin text predictors exist, so "the first Tsetlin language model" is not a claim to make.

## Not read

Three papers on teams of Tsetlin Machines were not read: MixCTME (2025), WCoCo-WTM (2026) and a 2026 Sensors Journal paper on composites. Kagan and Rybalov was read in part. They are first on the [roadmap](roadmap.md).

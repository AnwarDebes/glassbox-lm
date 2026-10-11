# References

The paper files that were read for this programme are not distributed. This page is the bibliography in their place.

The full bibliography of the tm_language_model study is its [docs/09](../studies/tm_language_model/docs/09-bibliography.md). It gives a DOI, an arXiv id or a URL wherever one was recorded and an evidence label for every entry, and it is not repeated here. On this page "docs/09", "docs/13" and the other numbered names mean the documents of that study in `studies/tm_language_model/docs/`. This page adds three things: the record of the literature sweep of 2026-09-25, to which docs/09 and amendment A8 of [docs/13](../studies/tm_language_model/docs/13-preregistration.md) point; the references of the hierarchical_tm study, which tests the hierarchical Tsetlin Machine of the supervisor's project proposal (unpublished); and the references of the clause_lm study.

An identifier is given only where a record of the project holds it. Otherwise the entry says "identifier not recorded". Authors, titles, venues and years are given as the records hold them. Where the notes name a work only by a short name, the entry gives the full title as it was read from the arXiv or ACL Anthology page of the work on 2026-10-02, and no authors, because none are recorded. The few entries that still carry "(short name)" or "title not recorded" have no recorded title. Where two records differ on a venue or a year, both are given. The commits of the code repositories that were read are in [upstream-repositories.md](upstream-repositories.md), and the datasets are in [data.md](data.md).

## Verification labels

The studies labelled their sources in two ways.

The tm_language_model study uses the evidence labels in square brackets that its [docs/00](../studies/tm_language_model/docs/00-README.md) defines. For a reference the relevant ones are "verified in paper" (read in the cited paper; the label says how much was read, for example the abstract or the full text), "verified in source" (read in the repository, the package or the official page that is named), "verified: exists" (the reference exists; nothing else about it was checked), "stated" (taken over and not re-verified) and "unverified". In the section of docs/09 that the sweep added, a label can also say "corrected", followed by what was fixed at the second check, or mark the entry as a missed-paper entry that was added at the second check. This page refers to those labels as docs/09 has them.

The clause_lm study labels each claim that it takes from a source by the way the claim was checked.

| Label | Meaning |
|---|---|
| checked against the source | The primary source was opened and the claim was compared with it. |
| checked twice | The primary source was read a second time, independently of the first reading, and the quotation or number agreed with it. A qualifier that the second reading attached stays with the label. |
| reported, not re-checked | The claim comes from a literature report that quotes the source; the source was not opened again for it. |
| corrected at the second check | The second reading found the reported claim wrong in the point that the entry names. |
| numbers checked only | The second reading compared the numbers and not the wording. |
| searched, not found (an absence is not proven) | The outcome of a search. A claim of this kind stays at this level, and a statement about novelty that rests on it names the scope of the search. |

A label is bounded by the level of the note that holds it. Where a report of the level "reported, not re-checked" calls one of its own claims verified, the claim stays "reported, not re-checked" on this page, and no label is raised. A few figures that the notes themselves mark as taken from a secondary source are marked "unverified". By the rule of the study, only claims of the first two levels are meant to be quoted in a write-up; an entry of the level "reported, not re-checked" is a lead.

The hierarchical_tm study recorded four citations as checked on 2 Oct 2026. Its section below gives them with the labels of the clause_lm scheme, because the same passages were entered in the clause_lm notes on the same day.

## The literature sweep of 2026-09-25

The sweep followed the decision of 2026-09-25 to check the literature before expensive runs (docs/13 A8). Its purpose was to set the registered hypotheses of docs/13 against published evidence. It is exploratory: it changed no threshold, gate or prediction of docs/13, and a proposal in it that touches a registered run needs a dated amendment first. Amendment A8 of the same day names the sweep as one of the three findings that motivate it. The entries that came out of it are the section "Added 2026-09-25 (verified literature sweep)" of docs/09.

An earlier check of prior art and of the existence of every cited arXiv id, made on 2026-09-22, is recorded in docs/09 itself, in its paragraph "Existence check" and its section "Citations added by the prior-art check".

### Method

The literature was searched from six angles:

1. Tsetlin Machine theory and learning dynamics;
2. Tsetlin Machines for text and for self-supervision;
3. hierarchy, logic networks and learnability;
4. counting and training-free methods;
5. energy and efficiency;
6. the published programme of Granmo and co-authors on logical and hierarchical Tsetlin Machines (papers, talks, a book chapter and public repositories).

For each angle the papers that were found were listed with their key findings, a prediction was written for every registered hypothesis that the angle bears on, and design implications were drawn. Every citation was then checked a second time, separately from the first pass, against the paper or its record (PDF, publisher page, Crossref, OpenAlex, arXiv). The second check gave each citation one of four statuses: verified, corrected, unverifiable or wrong. It also reviewed every prediction, with reasons for agreeing or disagreeing, and listed papers that the first pass had missed.

### Counts

| Quantity | Count | Basis |
|---|---|---|
| Papers listed in the first pass | 118 | record of the sweep |
| Citation checks in the second pass | 123 | record of the sweep; 5 of them concern papers that are cited only in design implications |
| Checks with the status verified | 97 | record of the sweep |
| Checks with the status corrected | 25 | record of the sweep |
| Checks with the status unverifiable | 1 | record of the sweep |
| Checks with the status wrong | 0 | record of the sweep |
| Missed papers listed at the second check | 48 | record of the sweep |
| Missed papers marked verified at the second check | 8 | record of the sweep |
| Papers added to docs/09 | 77 | counted in docs/09 |
| Correction notes on rows that docs/09 already had | 8 | counted in docs/09 |
| Leads that are named and not verified | 30 | counted in the list below |

The counts with the basis "record of the sweep" were taken at the time from a working file that is not part of the run folders. They were not recounted for this page. The four statuses add up to the 123 checks. The 77 papers are what remained after duplicates across angles and papers that docs/09 already listed had been removed; docs/09 holds them in six groups of 7, 6, 18, 22, 16 and 8 entries. docs/09 labels four of its entries as missed-paper entries (Gorji 2019, Morris 2022, Hill 2015, Hewitt 2019); the record does not say which of the eight missed papers that were marked verified were removed as duplicates or as already listed.

Two kinds of entries were dropped. The one unverifiable entry is Bhattarai, Granmo and Jiao, "An Interpretable Knowledge Representation Framework for Natural Language Processing with Cross-Domain Application", ECIR 2023 (identifier not recorded): the record states that the paper exists and that its abstract could not be obtained, so nothing is attributed to it. Every missed-paper entry that the second check did not mark verified was dropped as well. Those that duplicate a verified paper of another angle are covered by that paper; the remaining 30 are the leads below.

### Limits

- Three of the six angles (hierarchy, training-free methods, and the programme of Granmo and co-authors) were worked without a web search. Their sources were fetched by identifier (arXiv listings, Crossref, OpenAlex, Semantic Scholar, ACL Anthology, publisher pages), and the text of the PDF files was extracted locally. Coverage of work from 2025 and 2026 on n-gram language models, on hyperdimensional language models and on Tsetlin Machines for text is therefore weaker than intended.
- Several entries were read as abstracts only; the label in docs/09 says so. Such an entry supplies no number beyond its abstract.
- Tsetlin Machine theory and most published Tsetlin Machine results come from supervised machines with unit votes. The engine of the tm_language_model study differs from them in its feedback step, its literal budget and its integer vote weights (its docs/11), so bounds and coverage formulas from that literature transfer to it as orders of magnitude at most. The second check attached this caveat to every angle.
- The probabilities that the sweep gave for the registered hypotheses are judgements and carry the label "unverified" in its record. They are not reproduced here.
- The leads below were named at the second check and were not verified. Nothing in the sweep rests on them.

Leads, to be checked before they are cited. On Tsetlin Machines: the Convolutional Tsetlin Machine (arXiv 1905.09688); the Integer-Weighted Tsetlin Machine (arXiv 2005.05131); Tarasyuk et al., ISTM 2023; the multi-step deterministic automaton (arXiv 2007.02114); Wheeldon et al. 2020, Phil. Trans. R. Soc. A; Abeyrathna, ISTM 2025; the Probabilistic Tsetlin Machine (arXiv 2410.17851); Bhattarai et al. 2020 on text novelty (arXiv 2011.08755). Others: Neubig and Dyer 2016; Bakhtin et al. 2018; Gated Linear Networks; context-tree weighting; Teh 2006; Shareghi et al. 2016; Zhang et al. 2021; Heafield et al. 2013; Pibiri and Venturini 2019; Chelba et al. 2010; Muennighoff et al. 2023; ML.ENERGY; Zeus; Blier and Ollivier 2018; Goldman, Kearns and Schapire 1993; Angluin, Hellerstein and Karpinski 1993; Hancock and Mansour 1991; Mossel, O'Donnell and Servedio 2004; Arora et al. 2014; Wang et al. 2021 (RRL); Chatterjee 2018; the IWLS 2020 contest. Apart from the five arXiv ids, no identifier is recorded for a lead.

### Use of each paper

docs/09 gives every paper with what was read in it. The tables below say where the sweep used it. A paper was used in one or more of four ways: as evidence for the prediction on a registered hypothesis (the ids are those of docs/13; "s2" is the context scaling experiment with H3.3 and H3.4), as support for one of five mechanisms, as a constructive lead, or in the list of prior art that is cited and not claimed. The five mechanisms are what the sweep took from the literature to explain negative results of learning by the Tsetlin automata: the settings of s and T; noisy next-token labels, under which the automata do not reach an absorbing state; clause length and negation; class balance; and the equilibrium of the votes, which is a quantised linear estimate. All five come from supervised Tsetlin Machines with unit votes and apply to the engine of the study by analogy. What the study measured is on its [results page](../studies/tm_language_model/docs/results.md).

"docs/09 only" marks a paper that the sweep's evidence table, mechanisms and leads do not name; its use is then the one that its docs/09 entry states, repeated here in short form. Entries are named by first author and year, with a short tag where two entries share both.

Tsetlin Machine theory and dynamics (7 entries):

| Entry of docs/09 | Use in the sweep |
|---|---|
| Helin 2025 | Mechanism: equilibrium of the votes; together with Sharma 2022 it predicts a tie with logistic regression on the same literals (H9.3). Prior art for a map from votes to probabilities. |
| Sharma 2022 | H17.11, and H9.3 through the mechanism above: with fixed clauses, vote learning is perceptron learning, which bears on every comparison of a Tsetlin Machine with logistic regression on the same literals. |
| Belaid 2023 | docs/09 only: support for count-seeded clauses as the baseline without training. |
| Gorji 2019 | docs/09 only: specificity built into each clause, an alternative to a sweep over s. |
| Morris 2022 | docs/09 only: prior art for updates without the class sum and for a cap on literals. |
| Glimsdal 2022 | docs/09 only: the comparator for negative classes drawn by frequency. |
| Granmo 2023 (Markov boundary pruning) | Mechanism: clause length and negation, as one of the three remedies that the sweep names (with Sharma 2023 and Yadav 2024). Candidate gating in the lead on a positive-literal arm. |

Tsetlin Machines for text and self-supervision (6 entries):

| Entry of docs/09 | Use in the sweep |
|---|---|
| Yadav 2022 | Mechanisms: settings of s and T (text optima at low s), and clause length and negation (negated literals on sparse bag-of-words input). |
| Kadhim 2024 | The same two mechanisms. Prior art for negated literals as reasoning by elimination. |
| Sharma 2023 | Mechanism: clause length and negation (remedy). Listed among the lossy speed-ups that need a paired check. |
| Yadav 2024 | Mechanism: clause length and negation (remedy: pruning of literals after training). |
| Halenka 2024 | docs/09 only: an input encoding against negation drift. |
| Berge 2019 | H17.11: a prior at full supervision for a Tsetlin Machine against linear baselines. Mechanism: settings of s and T (a published text setting). |

Hierarchy, logic networks and learnability (18 entries):

| Entry of docs/09 | Use in the sweep |
|---|---|
| Valiant 1984 | H17.5: Theorem A (learning by elimination) as the frame for clauses of the form "group absent". Lead: elimination as an oracle baseline that is not a Tsetlin Machine. |
| Pitt 1988 | H17.5: worst-case hardness of proper learning; found at the second check to fit the test only loosely. |
| Malach 2019 (Boolean circuits) | H17.5: layer-wise learning of tree circuits under a local correlation assumption; a loose fit. |
| Malach 2019 (Is Deeper Better) | H14.2 and H17.5: the conjecture that deep models are learned mainly where shallow models already do well. |
| Abbe 2021 | H17.5: the staircase property; a loose fit. |
| Abbe 2022 | H17.5: companion of Abbe 2021; existence and title checked only. |
| Abbe 2023 (leap complexity) | H17.5: learning time grows with the leap of the target; an analogy. |
| Abbe 2023 (generalization on the unseen) | docs/09 only: accuracy on seen and on unseen combinations is reported separately (H17.5). |
| Abbe 2023 (curriculum) | Lead: a curriculum without distractors for the positive-literal arm. |
| Cornacchia 2023 | The same lead. |
| Mossel 2016 | docs/09 only: OR-groups built from unlabelled co-occurrence and not from label feedback. |
| Allen-Zhu 2020 | docs/09 only: context for greedy and then joint training of hierarchies. |
| Barak 2022 | Lead: a warning against an early stopping rule on targets that show long flat phases before they are learned. |
| An 2026 | H14.2: deep logic gate networks stall unless every gate also sees the input. |
| Tarasyuk 2024 | H14.2: a published multi-layer Tsetlin Machine, read as an abstract only. |
| Duan 2026 | H17.10: withdrawn as support at the second check, because its finding points the other way. |
| Susskind 2023 | docs/09 only: an energy reference for weightless networks on FPGA. |
| Aleksander 1984 | docs/09 only: the one-shot n-tuple baseline behind Duan 2026; existence checked only. |

Counting and training-free methods (22 entries):

| Entry of docs/09 | Use in the sweep |
|---|---|
| Khandelwal 2020 | H9.3: at the second check its n-gram interpolation was found to measure the reverse direction. |
| Li 2022 | H9.3: its residual gain had been read the wrong way round, which the second check corrected. docs/09: calibrates the Kneser-Ney 5-gram of the study. |
| Mahoney (benchmark page) | H15.1: its timings include online learning on mixed hardware and are not costs of batched inference. Lead: context mixing as the model for a Tsetlin Machine that selects contexts. |
| Chelba 2014 | Lead (lossy speed-ups): entropy pruning degrades Kneser-Ney tables. |
| Brants 2007 | H17.8: count models gain steadily with data. Lead (lossy speed-ups): Stupid Backoff is unnormalised and gives no bits per byte. |
| Nguyen 2024 | Lead: a Tsetlin Machine as a selector of context over count estimators. |
| Grave 2017 | s2: cache gains extend to caches of thousands of words. |
| Levy 2015 | H17.3: tuned count-based word vectors as the comparison for concept scores. |
| Sahlgren 2016 | H17.3 and H17.8: factorised counts on small corpora. |
| Hill 2015 | H17.3: SimLex-999 scores similarity and not association, so it is reported apart from the other two sets. |
| Wieting 2019 | H17.1 and H17.7: random encoders come close to trained ones. |
| Coates 2011 | H17.7 and H17.10: learning matters more for small dictionaries than for large ones. |
| Hewitt 2019 | docs/09 only: the method behind the control banks. |
| Ng 2001 | H17.2 and H17.11: with few labels the choice of probe moves the result. |
| Wang 2012 | H17.2: multinomial naive Bayes and NBSVM as count baselines that the control set lacked. Lead: such probes as unregistered rows. |
| Opitz 2023 | H17.2: a correctly scored bag-of-words classifier as the bar. |
| Jiang 2023 | docs/09 only: to be cited only with its tie-breaking caveat. |
| Rahimi 2016 | docs/09 only: the efficiency framing for a model that only matches counts. |
| Joshi 2014 and 2017 | docs/09 only: a one-pass hyperdimensional language model. |
| Tunstall 2022 | docs/09 only: the ceiling of the few-label protocol with a pretrained encoder. |
| Warstadt 2023 | H9.4 and H17.9: its BLiMP is filtered by vocabulary; transformers on small corpora. |
| Lan 2020 | H17.6: next-sentence prediction learns topic and not coherence. Lead: order-swap or topic-matched negatives for the same-document task of the project proposal. |

Energy and efficiency (16 entries):

| Entry of docs/09 | Use in the sweep |
|---|---|
| Heafield 2011 | H15.1 and H15.4: the cost per query of a Kneser-Ney 5-gram. Leads: KenLM as the energy comparator, and lossless count structures. |
| Luccioni 2024 | docs/09 only: its joules per query cannot anchor a ratio per token. |
| Samsi 2023 | docs/09 only: joules per token of large models, not comparable with a row for a small model. |
| Deletang 2024 | Lead: a size-inclusive row (description length) for a large count model. |
| Yang 2024 | H9.7, H15.1 and H15.7: coverage of the GPU power sensor, and the minimum length and the repeats of timed passes. |
| Elsworth 2025 | H9.7: the accounting boundary of energy figures. |
| Wu 2022 | docs/09 only: inference dominates the footprint, so energy per token is the headline. |
| Ma 2025 | H15.3: its latency as a crude upper anchor. docs/09: its energy per token is estimated and not measured. |
| Geiping 2023 | H17.9 and H9.7: under a tight budget the training recipe decides, so the comparator states its recipe. |
| Clark 2020 | H17.9: a plain masked-language-model comparator is conservative. |
| Jordan (repository) | H9.7: recipe and software changes move the time to a target, so the comparison family is called lightly tuned. |
| Horowitz 2014 | docs/09 only: a count model of several gigabytes is bound by memory access. |
| Rahman 2024 | docs/09 only: the hardware line of Tsetlin Machines, with no result on language. |
| Zeng 2025 | Lead (exact speed-ups): bitwise evaluation with early exit and reordering of literals. |
| Lin 2025 | docs/09 only: read as part of the second check of the Tunheim entry. |
| Duan 2025 | Mechanism: noisy labels (positive and negative clauses of a class include the same literals). |

Prior art from CAIR that is cited and not claimed (8 entries):

| Entry of docs/09 | Use in the sweep |
|---|---|
| Phoulady 2020 | Its sampling scheme narrows the novelty of the exact Binomial draw of Track A. Lead (exact speed-ups). |
| Przybysz 2023 | An exact SAT encoding of Tsetlin Machines exists, so only the receipt per token is claimed. |
| Granmo 2023 (TMComposites) | The mixture of count experts is structurally a composite and is cited as such. |
| Abouzeid 2022 | Label-free self-supervision of Tsetlin Machines exists already. |
| Yadav 2021 | H17.11: a prior at full supervision. Distributional neighbours as literals, next to the cluster literals of the study. |
| Bhattarai 2022 | H17.10: position coordinates as an encoding. Prior art for subword, position-aware text features. |
| Maheshwari 2023 | Lead: bytes per clause after include-encoding as the memory metric of a clause bank. |
| Granmo 2022 (book, chapter 1) | The framing against which the count core is read: summation-based like naive Bayes and logistic regression, with a low footprint in energy and memory. |

Correction notes on rows that docs/09 already had (8 rows; docs/09 states each correction):

| Row of docs/09 | Use in the sweep |
|---|---|
| Bhattarai et al., TM-AE (arXiv 2301.00709) | H17.3. Mechanism: settings of s and T. A correction recommended for docs/01. |
| Gao, Yadav et al., LLM-Guided Semantic Bootstrapping (arXiv 2604.12223) | Prior art: pretraining of a Tsetlin Machine from a teacher. |
| Kadhim et al., Omni TM-AE (arXiv 2505.16386) | H17.3 and H17.11; its classifier table is not a comparison on the same features. Lead: a read-out of automaton states. |
| Gao, Yadav et al., Clusters are All You Need (arXiv 2606.19815) | Prior art: pretraining of a Tsetlin Machine from a teacher. |
| Abeyrathna et al., CSC-TM (arXiv 2301.08190) | Mechanism: clause length and negation. H13.1 and H17.7. Listed among the lossy speed-ups (a literal budget). |
| Glimsdal and Granmo, CoTM (arXiv 2108.07594) | H17.1, H17.4 and H17.7. Mechanism: class balance. Names language modelling as an intended use. |
| Tarasyuk et al., Pattern Recognition 174 (2026) | H17.1, H17.4 and H17.10. Mechanisms: settings of s and T, and class balance. |
| Zhang, Jiao, Granmo, ICLR 2026 | H13.1. Mechanisms: settings of s and T, and noisy labels. |

The sweep also drew on entries that docs/09 listed before it, among them FastOmniTMAE, the Contracting Tsetlin Machine, clause indexing (Gorji et al., arXiv 2004.03188), the Sparse Tsetlin Machine, BLiMP, Infini-gram and Granmo's original paper (arXiv 1804.01508).

## References of the hierarchical_tm study

The study is described in its [README](../studies/hierarchical_tm/README.md). The project proposal that it tests carries the title "Logical self-supervised learning of language with hierarchical Tsetlin machines". It is not distributed, and no identifier is recorded for it.

The study's record lists the first four entries of the table as checked on 2 Oct 2026; they were collected for the write-up of the study. The labels in the table are those that the clause_lm notes give to the same passages, after the second check where there was one. The fifth entry is cited in the study's [registrations](../studies/hierarchical_tm/docs/registrations.md) and is not among the four.

| Work | Venue and year | Identifier | Label | Use in the study |
|---|---|---|---|---|
| Bhattarai, Granmo et al. "Tsetlin Machine Embedding" (title as recorded in docs/09; the study's record calls the work TM-AE) | Findings of EACL 2024, pp. 1512 to 1522 | arXiv 2301.00709 | checked twice | Prior work on word embeddings with a Tsetlin Machine. Cited for the stated purpose of ORing documents ("The purpose of ORing multiple documents is to increase the frequency of rare context words."), for its SimLex-999 comparison with GloVe and word2vec, for its training settings, and for the future work it names ("how sentence-level and document-level embedding can be created using clauses"). |
| Granmo. The Tsetlin Machine book, chapter 2 (chapter title not recorded; chapter 1 is the entry Granmo 2022 of docs/09) | draft of 13/11/22 | https://tsetlinmachine.org/wp-content/uploads/2022/11/Tsetlin_Machine_Book_Chapter_2.pdf | checked twice | p. 49: the "Forget Value 1/s", which controls the frequency of the patterns that are learnt. p. 51: a worked example at s = 15, in which patterns with a frequency above 1/15 are found; the record marks it as a worked example and not as a general law. The registration of 2 Oct 2026, 03:05 CEST, cites p. 49 for its hypothesis that 1/s is also the rate of forgetting. |
| cair/PyHierarchicalTsetlinMachineCUDA (package), `kernels.py`, lines 140 to 162 of the copy that was read; the commit of that copy is not recorded with the citation | public repository | commit of the runs in the study README and in docs/upstream-repositories.md | checked against the source | Type I feedback with boost 0: a word that is present is pushed toward inclusion with probability (s-1)/s, and a word that is absent toward exclusion with probability 1/s. An empty option therefore takes up a word of frequency p only if s > 1/p. The registrations derive their values of s from this rule. The runs started from Granmo's MultiConceptDemo script of the package, whose `--vanilla` arm fixes the size of the flat controls. |
| cair/HierarchicalTsetlinMachine (C reference implementation), README, section "Further Work"; and slide 37 of Granmo's talk of 2026-02-06 | public repository; talk of 2026 | commit of the clone in docs/upstream-repositories.md | checked against the source | The place where a natural-language application of the hierarchical Tsetlin Machine is named as further work: "Proof of concept applications (image understanding, natural language understanding, signal processing, etc.)". The README of the CUDA package does not carry the passage. |
| Gorji, Granmo, Phoulady, Goodwin. "A Tsetlin Machine with Multigranular Clauses" | arXiv preprint, 2019 | https://arxiv.org/abs/1909.07310 | not among the four checked citations of this study; labelled in docs/09 and in the clause_lm section | Named in the notes that follow the registration of 2 Oct 2026, 03:05 CEST, as an option for later: a different s per clause. It was not run. |

One search record belongs to this study. Its registrations fix the wording for novelty: no published Tsetlin Machine work on paragraph pairs or on matching without word overlap was found, in a search of OpenAlex, ISTM 2022 to 2025, arXiv to September 2026 and the ACL Anthology. Label: searched, not found (an absence is not proven).

The datasets of the study (WikiText-103, AG News, WordNet 3.0) are described in [data.md](data.md) and in the two data cards that the study README names.

## References of the clause_lm study

The study is preparation for a word-level language model with readable clauses; its [README](../studies/clause_lm/README.md) gives the milestones M1 to M5. Its literature notes were written on 2026-10-02. They consist of a collection of findings on the specificity s and on the closest Tsetlin Machine work on language, two reports on the state of the art (Tsetlin Machine language modelling; interpretable, neurosymbolic and count-based language models), a close reading of six papers, and two second checks of the same day.

Counted from the tables of the two second checks, 38 claims were read a second time, 16 in the first table and 22 in the second. Of these, 30 agreed with the source (13 and 17), several with a qualifier; 6 were corrected (1 and 5); for 1 only the numbers were compared; and 1 was not found in the source in the part that concerns calibration. A claim that neither check covers keeps the level of the note that holds it.

The measurements of the study are on its [results page](../studies/clause_lm/docs/results.md) and come from run files. The works below supply context: the rule for s behind the planned comparisons of M1 and the question of M5 step 1, the nearest prior work, and the baselines that later milestones would be compared with. Result figures of the cited works are not reproduced, with one exception in the entry of Xu and Jelinek.

Identifiers of the form NNNN.NNNNN that the notes give as bare numbers in lists of arXiv references are written here as arXiv ids. "(short name)" means that no record holds the full title of the work. An entry that gives a title without authors has its title from the arXiv or ACL Anthology page of the work, and its authors are not recorded.

### The specificity s and the frequency of literals

| Work | Venue and year | Identifier | Label | Use in the study's notes |
|---|---|---|---|---|
| Granmo. "The Tsetlin Machine" (title as recorded in docs/09) | arXiv, 2018 | arXiv 1804.01508 | checked twice (qualified: the targeted frequency lies just above 1/s) | s sets the frequency of the sub-patterns that a clause targets, which agrees with the rule s > 1/p that was read from the kernel of the package. |
| Granmo. The Tsetlin Machine book, chapter 2 (chapter title not recorded) | draft of 13/11/22 | https://tsetlinmachine.org/wp-content/uploads/2022/11/Tsetlin_Machine_Book_Chapter_2.pdf | checked twice | The "Forget Value 1/s" (p. 49) and a worked example at s = 15 (p. 51), not a general law. |
| Zhang, Jiao, Granmo, Goodwin. "On the Convergence of Tsetlin Machines for the IDENTITY- and NOT Operators" | arXiv, 2020 | arXiv 2007.14268 | checked twice | Rare features need a sufficiently large s for convergence. |
| Kadhim, Granmo, Jiao, Shafik. "Exploring State Space and Reasoning by Elimination in Tsetlin Machines" | ISTM 2024, pp. 1 to 8 | https://arxiv.org/abs/2407.09162 | checked twice (qualified: s and T change together and boosted true-positive feedback is on, so the effect is not isolated to s) | A larger s goes with more literals that are not negated. That the method uses one global s is reported, not re-checked. |
| Yadav, Jiao, Granmo, Goodwin. "Robust Interpretable Text Classification against Spurious Correlations Using AND-rules with Negation" | IJCAI 2022, pp. 4439 to 4446 | https://doi.org/10.24963/ijcai.2022/616 | checked twice | On a sparse bag-of-words input most literals of a clause are negated. |
| Gorji, Granmo, Phoulady, Goodwin. "A Tsetlin Machine with Multigranular Clauses" | arXiv preprint, 2019 (docs/09); the notes give SGAI 2019 (reported, not re-checked) | https://arxiv.org/abs/1909.07310 | checked twice | Specificity that varies per clause, not per literal and not by frequency. It is listed under the search for an s that is matched to frequency. |
| Berge, Granmo, Tveit, Goodwin, Jiao, Matheussen. "Using the Tsetlin Machine to Learn Human-Interpretable Rules for High-Accuracy Text Categorization with Medical Applications" | IEEE Access 7, pp. 115134 to 115146, 2019 | https://arxiv.org/abs/1809.04547 | reported, not re-checked | Typical settings of s and T of a text classifier with negated literals. |
| Sharma, Yadav, Granmo, Jiao. "Drop Clause: Enhancing Performance, Robustness and Pattern Recognition Capabilities of the Tsetlin Machine" | AAAI 2023, 37(11), pp. 13547 to 13555 | https://doi.org/10.1609/aaai.v37i11.26588 | reported, not re-checked | The same. |
| Bhattarai, Granmo, Jiao. "ConvTextTM: An Explainable Convolutional Tsetlin Machine Framework for Text Classification" | LREC 2022, pp. 3761 to 3770 | https://aclanthology.org/2022.lrec-1.401/ | reported, not re-checked | The same. |

### Tsetlin Machine work on language and embeddings

| Work | Venue and year | Identifier | Label | Use in the study's notes |
|---|---|---|---|---|
| Rahimi Gorji, Granmo, Goodwin. "Natural Language Modeling with the Tsetlin Machine" | IEA/AIE 2023, LNAI 13926, pp. 141 to 150 | DOI 10.1007/978-3-031-36822-6_12 | checked against the source | The nearest prior work to a Tsetlin Machine language model: it predicts the next character of a word, is compared with two small MLPs, and reports no perplexity and no n-gram baseline. One s is tuned per model, not per feature. |
| Abdelwahab, Granmo, Jiao. "Interpretable Contextual Word Prediction Utilizing Tsetlin Machine" | ISTM 2025 (IEEE), pp. 72 to 79 | DOI 10.1109/istm67926.2025.00019 | reported, not re-checked; key facts checked against the source | The nearest prior work at word level: one Tsetlin Machine per target word, evaluated by classification accuracy on nine datasets, with no next-word metric and no perplexity. |
| Bhattarai, Granmo et al. "Tsetlin Machine Embedding" (title as recorded in docs/09) | Findings of EACL 2024, pp. 1512 to 1522 | arXiv 2301.00709 | checked twice | Word embeddings from the prediction of a target word out of its bag-of-words context; evaluated as embeddings and not as a language model. |
| Kadhim, Jiao, Shafik, Granmo. Omni TM-AE: A Scalable and Interpretable Embedding Model Using the Full Tsetlin Machine State Space | 2025 | arXiv 2505.16386 | checked twice | The same family of embedding models. |
| Kadhim, Jiao, Shafik, Granmo, Shende. "FastOmniTMAE: Parallel Clause Learning for Scalable and Hardware-Efficient Tsetlin Embeddings" | arXiv, 2026 | arXiv 2605.06982 | checked twice (qualified: the stated speed-up applies to classification and not to the similarity benchmark); the close reading is reported, not re-checked | The same family: static word embeddings with clause weights and no probabilities. |
| Glimsdal, Granmo. Coalesced Multi-Output TM (title as recorded in docs/09) | 2021 | arXiv 2108.07594 | corrected at the second check (the pair of accuracies that the note had assigned to Fashion-MNIST belongs to Kuzushiji-MNIST; the quotation on language models agreed with the source) | Clause sharing between outputs; the paper names language models and auto-encoding as applications that it expects. |
| Yadav, Jiao, Granmo, Goodwin. "Enhancing Interpretable Clauses Semantically using Pretrained Word Representation" | BlackboxNLP 2021, pp. 265 to 274 | https://arxiv.org/abs/2104.06901 | reported, not re-checked | Neighbours from GloVe added to the bag-of-words input of a Tsetlin Machine. |
| Saha, Granmo, Zadorozhny, Goodwin. "A Relational Tsetlin Machine with Applications to Natural Language Understanding" | arXiv, 2021 (index of the tm_language_model study); the notes give 2022 | arXiv 2102.10952 | reported, not re-checked | Clauses with variables for entities, applied to question answering. |
| Granmo et al. "The Tsetlin Machine Goes Deep: Logical Learning and Reasoning With Graphs" | arXiv, version 2 of 2026-05-06 | arXiv 2507.14874 | checked twice (qualified: the comparisons are confounded by different input encodings) | Deep clauses over graphs, on supervised text classification among other tasks. |
| Gao, Yadav et al. LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines | Findings of ACL 2026 | arXiv 2604.12223 | checked twice (the BERT baselines of the paper are taken from the literature and were not re-run there) | A Tsetlin Machine that learns from examples written by a large language model and runs without it: a hybrid by other authors. |
| Gao, Yadav et al. Clusters are All You Need: Pre-Training the Tsetlin Machine with Semantic Clusters from Language Models for Interpretability | June 2026 | arXiv 2606.19815 | checked twice (qualified: trained on a subset; the BERT numbers of the paper are copied from another paper) | A Tsetlin Machine without negated literals that is pretrained from clusters. Its setting of the Type I reward is reported, not re-checked. |
| Yadav and Nicolae (title not recorded) | Algorithms, 2022 | doi 10.3390/a15050143 | reported, not re-checked | Clause scores of a Tsetlin Machine as input to an attention network. |
| Kadhim et al. Adversarial Attacks on AI-Generated Text Detection Models: A Token Probability-Based Approach Using Embeddings | 2025 | arXiv 2501.18998 | reported, not re-checked | Tsetlin Machine embeddings used against a detector of machine-written text. |
| Tarasyuk, Gorbenko, Rahman, Jiao, Granmo, Shafik, Yakovlev. "Learning dynamics, pattern recognition capability and interpretability of the Tsetlin Machine" | Pattern Recognition 174 (2026) 113028 | identifier not recorded | reported, not re-checked; one quotation checked against the source | The trade-off between s and T, which are global hyperparameters, and the statement that hierarchical abstraction is not native to the standard Tsetlin Machine. |
| Zhang, Jiao, Granmo. "Convergence Analysis of Tsetlin Machines Under Noise-Free and Noisy Training Conditions: From 2 Bits to k Bits" | ICLR 2026 | OpenReview feOrSQdD9Y | reported, not re-checked | Convergence theory with a global s and T. |
| Granmo et al. "A Tsetlin Machine for Logical Learning and Reasoning With AND-OR Hierarchies" | 2026, forthcoming, as cited in the README of the C reference implementation and in the README of the CUDA package | identifier not recorded | the citation in the first README is checked against the source, the one in the second is reported, not re-checked; a preprint was searched, not found (an absence is not proven) | The paper of the hierarchical package. Nothing is quoted from it. |
| Tarasyuk, Gorbenko, Yakovlev, Shafik. "Multi-Layer Tsetlin Machine: Architecture and Performance Evaluation" | ISTM 2024 | https://doi.org/10.1109/istm62799.2024.10931518 | reported, not re-checked (abstract only) | Related work on more than one layer. |
| Granmo. Keynote at the NORA annual conference of 1 June 2026, and the news article about it on nora.ai | talk, 2026 | identifier not recorded | reported, not re-checked | A public statement on AND-OR hierarchies; neither the talk nor the article is reported to mention language. |

### Scaling, inference and hardware of Tsetlin Machines

| Work | Venue and year | Identifier | Label | Use in the study's notes |
|---|---|---|---|---|
| Ostby, Brambo, Glimsdal. "The Sparse Tsetlin Machine: Sparse Representation with Active Literals" | arXiv | arXiv 2405.02375 | numbers checked only | Memory and time of a sparse representation with a large vocabulary. |
| Contracting Tsetlin Machine with Absorbing Automata | not recorded | arXiv 2310.11481 | reported, not re-checked | Named among the recent advances that matter for scaling. |
| Literal streaming with absorbing states (short name) | ISTM 2024 | identifier not recorded | reported, not re-checked (a speed claim from a search snippet) | The same. |
| Green Tsetlin Redefining Efficiency in Tsetlin Machine Frameworks | not recorded | arXiv 2405.04212 | reported, not re-checked | The same. |
| Glimsdal, Saha, Bhattarai, Giri, Sharma, Tunheim, Yadav. "Focused Negative Sampling for Increased Discriminative Power in Tsetlin Machines" | ISTM 2022, pp. 73 to 80 | https://doi.org/10.1109/ISTM54910.2022.00021 | reported, not re-checked | Fewer epochs through focused sampling of negative classes. |
| Granmo. "TMComposites: Plug-and-Play Collaboration Between Specialized Tsetlin Machines" | arXiv preprint, 2023 | https://arxiv.org/abs/2309.04801 | reported, not re-checked | Collaboration of specialised machines. |
| Hnilov. Fuzzy-Pattern Tsetlin Machine (title as recorded in docs/09) | not recorded | arXiv 2508.08350 | checked twice (qualified: the largest stated speed-up compares 32 threads with a single thread) | Text classification with very few clauses per class. |
| Helin, Granmo, Shende, Jiao, Zadorozhny, Dumbre, Shafik, Yakovlev. "Uncertainty Quantification in the Tsetlin Machine" | arXiv preprint, 2025 | https://arxiv.org/abs/2507.04175 | not found in the source as far as calibration is concerned; the report was corrected accordingly at the second check (the scores are not calibrated probabilities) | Whether the class sums of a Tsetlin Machine can serve as probabilities for perplexity. As they stand they cannot, which is why the planned model lets smoothed counts vote. |
| Granmo, Andersen, Jiao, Zhang, Blakely, Berge, Tveit. "Learning Minimalistic Tsetlin Machine Clauses with Markov Boundary-Guided Pruning" | ISTM 2023, pp. 1 to 8 | https://doi.org/10.1109/istm58889.2023.10454914 (arXiv 2309.06315) | reported, not re-checked | An extra automaton removes literals; s is unchanged. |
| Zeng, Duan, Shafik, Yakovlev. "Fast and Compact Tsetlin Machine Inference on CPUs Using Instruction-Level Optimization" | arXiv preprint, 2025 | https://arxiv.org/abs/2510.15653 | reported, not re-checked | Faster inference on CPUs. |
| Tunheim, Zheng, Jiao, Shafik, Yakovlev, Granmo. "An All-Digital 8.6-nJ/Frame 65-nm Tsetlin Machine Image Classification Accelerator" | IEEE TCAS-I 73(2), Feb 2026 (docs/09); the notes give IEEE TCAS-I 2025 | arXiv 2501.19347 | checked twice | The measured energy of a fabricated Tsetlin Machine chip, for image classification and not for language. |
| Duan, Shafik, Yakovlev. "ETHEREAL: Energy-efficient and High-throughput Inference using Compressed Tsetlin Machine" | arXiv preprint, 2025 | https://arxiv.org/abs/2502.05640 | reported, not re-checked | Energy against binarised networks on small tasks. |
| Low-Energy Reduced RISC-V Instruction Subset Processor for Tsetlin Machine Inference at the Edge | ISVLSI 2026 | arXiv 2606.19964 | reported, not re-checked | The same. |

### Interpretable language models

| Work | Venue and year | Identifier | Label | Use in the study's notes |
|---|---|---|---|---|
| Backpack Language Models | ACL 2023 | arXiv 2305.16765 | corrected at the second check (wording of the comparison with a small GPT-2; the numbers agreed with the source) | A white-box language model near a small transformer in loss, weaker in syntax. |
| Concept Bottleneck Large Language Models | ICLR 2025 | arXiv 2412.07992 | corrected at the second check (the classification figure lies below the black-box model, and the perplexity is that of generated sentences, not of a test set) | A concept-bottleneck model; its concepts are labelled by a large language model. |
| Scaling and evaluating sparse autoencoders | not recorded | arXiv 2406.04093 | reported, not re-checked | Interpretation after training only. |
| AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders | ICML 2025 | arXiv 2501.17148 | reported, not re-checked | A benchmark on which sparse autoencoders are reported as not competitive. |
| Negative results on sparse autoencoders, GDM (short name) | Alignment Forum, March 2025 | identifier not recorded | reported, not re-checked | The same debate. |
| Steering LLMs? Actually, Sparse Autoencoders can outperform simple baselines | not recorded | arXiv 2605.31183 | reported, not re-checked | The same debate. |
| Weight-sparse transformers have interpretable circuits | 2025 | arXiv 2511.13653 | reported, not re-checked | Readable circuits; scale is the open point. |
| Prototype Language Models | July 2026 | arXiv 2607.00510 | reported, not re-checked | An interpretable model that is reported close to dense models; it gives attributions, not rules. |
| Scaling Inherently Interpretable Language Models | August 2026 | arXiv 2608.07594 | reported, not re-checked | The same. |
| Kolmogorov-Arnold Networks for Small Language Models | July 2026 | arXiv 2607.15525 | reported, not re-checked | No consistent advantage over MLP baselines is reported. |
| Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures | April 2026 | arXiv 2604.16042 | reported, not re-checked | Five paradigms, none of which is a language model of symbolic rules. |

### Logic-based and neurosymbolic models

| Work | Venue and year | Identifier | Label | Use in the study's notes |
|---|---|---|---|---|
| Petersen et al. "Deep Differentiable Logic Gate Networks" | NeurIPS 2022 | arXiv 2210.08277 | reported, not re-checked | Logic gate networks; vision only. |
| Convolutional Differentiable Logic Gate Networks | NeurIPS 2024 | arXiv 2411.04732 | checked twice | The same; vision only. |
| Buehrer, Plesner, Aczel, Wattenhofer. "Recurrent Deep Differentiable Logic Gate Networks" | arXiv, 2025 | arXiv 2508.06097 | checked twice (with the caveat that the collapsed model, which is the deployable one, scores lower); the close reading is reported, not re-checked | The nearest logic network on language. It is not a Tsetlin Machine, and its perplexity is that of conditional translation, not of a language model. |
| Logical Neural Networks | 2020 | arXiv 2006.13155 | reported, not re-checked | Neurons with a meaning as parts of a formula; not a language model. |
| Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning | Findings of EMNLP 2023 | arXiv 2305.12295 | reported, not re-checked | Logic outside the language model. |
| Adaptable Logical Control for Large Language Models | NeurIPS 2024 | arXiv 2406.13892 | reported, not re-checked | The same. |
| Interpretable Next-token Prediction via the Generalized Induction Head | NeurIPS 2025 | arXiv 2411.00066 | corrected at the second check (the test corpus and the two tokenizers behind the reported accuracies) | Rule-like next-token prediction that uses a neural similarity metric; it reports no perplexity. |
| Augmenting Interpretable Models with LLMs during Training | Nat. Commun. 2023 | arXiv 2209.11799 | reported, not re-checked | Interpretable models for classification only. |
| Neuron-Anchored Rule Extraction for Large Language Models via Contrastive Hierarchical Ablation | KDD 2026 | arXiv 2605.03058 | reported, not re-checked | The same. |

### Count, n-gram and class-based models, and baselines

| Work | Venue and year | Identifier | Label | Use in the study's notes |
|---|---|---|---|---|
| Liu, Min, Zettlemoyer, Choi, Hajishirzi. Infini-gram (title as recorded in docs/09) | COLM 2024 | arXiv 2401.17377 | checked twice (caveats: the figures are validation figures, and the relative improvement is defined on the gap to a perplexity of 1) | An n-gram model of unbounded order that is combined with neural models: part of the revival of count models. |
| Khandelwal, Levy, Jurafsky, Zettlemoyer, Lewis. "Generalization through Memorization: Nearest Neighbor Language Models" | ICLR 2020 | https://arxiv.org/abs/1911.00172 | checked twice (caveat: the quoted perplexity includes the continuous cache) | A retrieval hybrid; entry of the baseline table. |
| KNN-LM Does Not Improve Open-ended Text Generation | not recorded | arXiv 2305.14625 | reported, not re-checked | No gain of the hybrid in open-ended generation is reported. |
| Nguyen. "Understanding Transformers via N-gram Statistics" | NeurIPS 2024 | https://arxiv.org/abs/2407.12034 | checked twice (caveat: the agreement is with the best rule per prediction, an upper bound, not with a predictor that stands alone) | Support for next-token prediction by rules. |
| Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models | January 2026 | arXiv 2601.07372 | corrected at the second check (one benchmark gain in the abstract differs from the table of the paper) | A hashed n-gram lookup inside a large model. |
| Neubig and Dyer. Generalizing and Hybridizing Count-based and Neural Language Models | EMNLP 2016 | arXiv 1606.00499 | checked twice (caveat: sentence-level figures, "not directly comparable" to standard Penn Treebank) | A hybrid of count and neural models on Penn Treebank. |
| Svete et al. Can Transformers Learn n-gram Language Models? | EMNLP 2024 | arXiv 2410.03001 | reported, not re-checked | Smoothed counts against transformers on n-gram languages. |
| Stratos. Mutual Information Maximization for Simple and Accurate Part-Of-Speech Induction | NAACL 2019 | arXiv 1804.07849 | reported, not re-checked | Induction of word classes, scored against the tags of Penn Treebank. |
| Goodman. A Bit of Progress in Language Modeling | 2001 | arXiv cs/0108005 | checked twice | The gain from word clustering in n-gram models. |
| Scaling Hidden Markov Language Models | EMNLP 2020 | arXiv 2011.04640 | checked twice | A class-like hidden Markov model; the source of the perplexities of that model and of a Kneser-Ney 5-gram on Penn Treebank and WikiText-2 in the baseline table. |
| A Generalized Language Model as the Combination of Skipped n-grams and Modified Kneser-Ney Smoothing | ACL 2014 | arXiv 1404.3377 | checked twice (caveat: the stated gains hold for models of order 5) | Skip-n-grams against modified Kneser-Ney. |
| WSI (short name) | Findings of ACL 2025 | identifier not recorded | reported, not re-checked | The remark that no unsupervised method surpasses a baseline with one cluster per lemma. |
| Bahl et al. (title not recorded) | IEEE TASSP, 1989 | identifier not recorded | reported, not re-checked; its figures are unverified (secondary source) | A language model as a tree of questions about the preceding words: an early model of rules with count-based distributions. |
| Xu and Jelinek. Random Forests in Language Modeling (the first; the title of the second is not recorded) | EMNLP 2004, and CSL 2007 | ACL Anthology W04-3242 (the first) | checked twice | Random-forest language models, in which averaged leaf distributions act as votes of count-based rules. The notes also record a cross-check against this paper: its held-out set has 73,760 words, the same number as the validation stream of run `20261002T170451Z_ngram_baselines` (73,760 tokens, `config.json`), and its interpolated Kneser-Ney trigram has a perplexity of 160.1 on that set, next to 161.72 in that run (`metrics.json`, one run). |
| Brown et al. Class-Based n-gram Models of Natural Language | 1992 | https://aclanthology.org/J92-4003/ | reported, not re-checked | Class-based n-grams. |
| Dagan, Lee, Pereira. Similarity-Based Models of Word Cooccurrence Probabilities | 1999 | arXiv cs/9809110 | title read from the arXiv page; the content was not labelled in the notes | Similarity-based smoothing for unseen bigrams, the family of the similarity-smoothing baseline on the clause_lm results page. |
| Li, Cai, Xu, Watanabe. "Residual Learning of Neural Text Generation with n-gram Language Model" | Findings of EMNLP 2022, pp. 1523 to 1533 | https://aclanthology.org/2022.findings-emnlp.109/ (arXiv 2210.14431) | checked twice | The source of the perplexity of a KenLM 5-gram on WikiText-103 in the baseline table. |
| Regularizing and Optimizing LSTM Language Models | not recorded | arXiv 1708.02182 | checked twice | A neural baseline of the baseline table. |
| Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context | not recorded | arXiv 1901.02860 | checked twice | The same. |
| A BabyLM paper (short name) | not recorded | ACL Anthology 2025.babylm-main.16 | reported, not re-checked | A known artifact of the BLiMP evaluation of BabyLM, noted as an evaluation rule. |

The baseline table of the notes also holds rows for a small LSTM, the Mogrifier LSTM and pretrained models without fine-tuning. These rows name no work with an identifier, and two of them are marked as taken from secondary sources, so they have no entry here.

### Routers, guardrails and regulation

| Work | Venue and year | Identifier | Label | Use in the study's notes |
|---|---|---|---|---|
| RouteLLM: Learning to Route LLMs with Preference Data | ICLR 2025 | arXiv 2406.18665 | reported, not re-checked | A router benchmark; the router itself is opaque. |
| IRT-Router: Effective and Interpretable Multi-LLM Routing via Item Response Theory | ACL 2025 | arXiv 2506.01048 | reported, not re-checked | A router with interpretable latent parameters, not rules. |
| vLLM Semantic Router: Signal Driven Decision Routing for Mixture-of-Modality Models | 2026 | arXiv 2603.04444 | checked twice (caveat: some of its signals are learned classifiers) | Routing policies as Boolean formulas that are written by hand; the paper names learned policies as future work. The notes set the learned, readable clauses of milestone M5 against this. |
| Arch-Router: Aligning LLM Routing with Human Preferences | not recorded | arXiv 2506.16655 | reported, not re-checked | Routing policies in natural language. |
| R^2-Guard: Robust Reasoning Enabled LLM Guardrail via Knowledge-Enhanced Logical Reasoning | ICLR 2025 | arXiv 2407.05557 | checked twice | A guardrail with rules. For this entry and the next two together, the notes state that the rules are written by hand or by a large language model. |
| ShieldAgent: Shielding Agents via Verifiable Safety Policy Reasoning | ICML 2025 | arXiv 2503.22738 | reported, not re-checked | The same. |
| AgentSpec: Customizable Runtime Enforcement for Safe and Reliable LLM Agents | ICSE 2026 | arXiv 2503.18666 | checked twice | The same. |
| RouterBench: A Benchmark for Multi-LLM Routing System | not recorded | arXiv 2403.12031 | reported, not re-checked | A benchmark for controllers. |
| EU AI Act, Articles 13(1) and 14(4); Regulation (EU) 2026/1744 | official texts | CELEX 32024R1689 and 32026R1744 | checked twice | The requirements on transparency and on human oversight that a readable router would have to meet. |
| Nannini et al. AI Agents Under EU Law | April 2026 | arXiv 2604.04604 | reported, not re-checked | A statement on traceability in high-risk systems. |

### Energy of small language models

| Work | Venue and year | Identifier | Label | Use in the study's notes |
|---|---|---|---|---|
| Ma, Wang, Huang, Zhang, Hu, Song, Xia, Wei. "BitNet b1.58 2B4T Technical Report" | arXiv preprint, 2025 | https://arxiv.org/abs/2504.12285 | corrected at the second check (the energy figure is an estimate for a sequence, not a measurement per token) | An example of an energy claim that is not measured. |
| Zhu et al. "Scalable MatMul-free Language Modeling" | arXiv | arXiv 2406.02528 | reported, not re-checked | An efficiency claim for a language model without matrix multiplication. |

The entries on Tsetlin Machine hardware are in the table on scaling above.

### Search records

Each of these is an absence that was searched for. All carry the label "searched, not found (an absence is not proven)". The scope of the search is recorded for the second one only.

- No word-level Tsetlin Machine language model that is scored by perplexity against n-gram baselines, on Penn Treebank or WikiText.
- No Tsetlin Machine work on pairs of paragraphs or sentences, on the prediction of the same document or the next paragraph, or on matching without word overlap. Scope: OpenAlex, ISTM 2022 to 2025, arXiv to September 2026, ACL Anthology.
- No specificity that is matched to frequency or set per literal.
- No hierarchical AND-OR Tsetlin Machine that learns word classes from raw text, and no preprint of the paper on AND-OR hierarchies.
- No Tsetlin Machine router or guard by other groups.
- No measured energy per token for a Tsetlin Machine language model or a logic-gate language model.

### Software, repositories and documentation

| Item | Identifier | Label | Use in the study's notes |
|---|---|---|---|
| cair/PyHierarchicalTsetlinMachineCUDA | commits in docs/upstream-repositories.md | checked against the source | `kernels.py`, lines 140 to 162: the Type I feedback from which the rule s > 1/p follows. The examples of the package hold no language demo. `studies/clause_lm/src/pair_tm.py` imports the package. |
| cair/HierarchicalTsetlinMachine (C reference implementation) | commit in docs/upstream-repositories.md | checked against the source | The README names natural language understanding under "Further Work" and cites the paper on AND-OR hierarchies as forthcoming. Its only demo is on noisy parity. |
| cair/LogicalTransformer and cair/LogicalTransformerWithTsetlinMachine | commits in docs/upstream-repositories.md | checked against the source | One Tsetlin Machine per token and a composite; demos of word prediction for a few target words. |
| BooBSD/Tsetlin.jl | github.com/BooBSD/Tsetlin.jl | checked against the source | A text example at character level, with generated samples and timings and with no accuracy, perplexity or bits per character; one global s. |
| cair/pyTsetlinMachine | commit not recorded | reported, not re-checked | The package exposes a global s and `s_range` only. |
| torchtsetlin (PyPI 0.3.0, September 2026) | github.com/vlbthambawita/torchtsetlin | reported, not re-checked | An engineering option that was left to be checked. |
| NVIDIA, documentation of the Multi-Process Service (R610) and the CUDA 8.0 programming guide | identifier not recorded | reported, not re-checked | Whether work from several processes can overlap on one GPU: background for the GPU sharing of the hierarchical_tm study (its study rule 4). |
| Clause-Driven-LLM | https://github.com/AnwarDebes/Clause-Driven-LLM | checked against the source (the files of the downloaded copy were read); the commit is recorded in project notes, not checked against upstream | The router that M5 step 1 reproduces. The study README describes the copy that was used and the two changes to it. |

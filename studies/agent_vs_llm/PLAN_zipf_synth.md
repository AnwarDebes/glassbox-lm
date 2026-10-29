# Plan ZIPF-synth: does a Tsetlin Machine learn only patterns above a frequency threshold? (synthetic, CPU)

Exploratory, synthetic data only. Written and frozen before the first number.

## Why

Eight registered tests of the programme found Tsetlin Machines learning language statistics worse than counting or a
linear model. One candidate cause: a clause converges on sub-patterns whose frequency within the class is above about
1/s (Granmo's original analysis of the specificity s), and a finite clause budget is spent on the most frequent
patterns first; language is Zipfian, so most informative patterns are rare. This plan measures the two parts directly.

## Data (generator `zipf_synth.py`, numpy seed fixed per draw)

- d = 300 binary features; background bits on with probability 0.05 (sparse, like words).
- K = 60 planted patterns, each a conjunction of 3 distinct features (no feature shared between patterns). A positive
  example carries exactly one pattern; pattern k is chosen with probability proportional to 1/k (Zipf, exponent 1).
- Negative examples are background only, with every planted pattern absent (a full pattern is broken by clearing one of
  its bits). 20,000 training and 20,000 test examples, half positive.
- The label is decided by the planted pattern only, so a perfect learner reaches 100%.

## Learners

- TM: tmu TMClassifier (vanilla, CPU, the project's pinned tmu 5605ff07 in .venv-tm), T = clauses, 30 epochs,
  s in {2, 4, 8, 16, 32, 64}, clauses in {20, 80, 320}, seeds 1 to 3.
- Twin: logistic regression on the same bits (sklearn, C from {0.1, 1, 10} chosen on a validation slice of train).
  (A counting twin over all 3-feature co-occurrences was dropped before the first number: background triples also
  occur in positive examples, so it is not well defined without its own design.)

## Measurements

- Per pattern k: its share of positive training examples (f_k) and its test recall (share of test positives carrying k
  that are classified positive). The false-positive rate on negatives beside it.
- Recall as a function of f_k for each s and clause budget; the frequency f50 at which recall falls below 50%.

## Reading (fixed now; no rule beyond this)

- "Frequency threshold set by s" is supported if, at the largest budget (320 clauses), f50 decreases as s grows and
  f50 * s stays within a factor of 4 across the s values where f50 is defined.
- "Budget crowding" is supported if, at fixed s, f50 rises when the clause budget falls from 320 to 20.
- Either reading is reported as measured if not supported. Nothing here is a claim about language until a separate
  plan measures it on text.

# Plan FT-dev: the network-free agent against transformers fine-tuned end to end (STAR and ABCD dev)

Dev only, exploratory. No test file is opened. Written and frozen before the first number of this plan.

## Question

The registered tests of decision_layer (R21 to R24) compare the network-free agent (C4 counting table, no neural
network at any stage, then a Tsetlin Machine) with agents that feed frozen transformer features to the same learner.
They never compare it with a transformer fine-tuned end to end on the same examples. How large is that gap on STAR and
ABCD dev? The answer decides whether "a network-free agent matches transformers" can be a headline, or only "matches
frozen transformer features".

## Data and inputs

- `glassbox-lm/studies/decision_layer/data/{star,abcd}/examples_{train,dev}.json.gz` (read only): STAR 54,913 train and
  6,908 dev rows, 187 actions; ABCD 29,190 train and 3,684 dev rows, 30 actions. The test files are not opened.
- The same information as the agents (memory of previous actions, the recent exchange, the customer's utterances),
  rendered as text: `actions: <last 10 previous actions> | recent: <recent> | customer: <last 800 characters of the
  customer's utterances>`, cut at 256 tokens from the end.

## Arms

- FT-RoBERTa: `roberta-base`, lr 3e-5.
- FT-BERT: `bert-base-uncased`, lr 5e-5.
- FT-bge-base: `BAAI/bge-base-en-v1.5` (a BERT encoder; the teacher of potion and of the refined table), lr 5e-5.
- Common: AdamW, weight decay 0.01, batch 32, linear schedule with 6% warmup, 4 epochs, fp16 autocast, max length 256,
  seed 1 for the screen. Dev accuracy after every epoch.
- If the best fine-tuned arm is within 3.0 points of the C4 agent on both datasets, FT-bge-large (`BAAI/bge-large-en-v1.5`,
  lr 2e-5) runs too, so that the strongest affordable encoder is tried before any parity reading.
- The best arm then runs seeds 2 and 3.

## References (existing rows, seed 1, best dev epoch; docs/results.md e74c and e75)

- STAR dev: C4 agent (16,000 clauses, s 16) 0.6545; bge-large agent (s 16) 0.6507; LR on the C4 bits 0.6558.
- ABCD dev: C4 agent (4,000 clauses, s 16) 0.8417; bge-large agent (s 24) 0.8472.

## Statistic and decision (fixed now)

The same selection for every arm: the best dev epoch (optimistic for all arms alike); the final epoch is reported
beside it. G = (best fine-tuned arm, mean over its seeds) minus the C4 agent's reference, per dataset, in points.

- G at most 1.0 on both datasets: the parity line. A confirmatory test follows (fresh data preferred; a reused test
  split only through a frozen registration with its use count).
- G above 1.0 and at most 3.0 on both: the "within 3 points at a fraction of the cost" line; cost is measured next.
- G above 3.0 on either dataset: no parity claim; the network-free result stays "matches frozen transformer features",
  and the next step is chosen from the other candidates.

## Limits known now

- Dev numbers with best-epoch selection are optimistic for every arm; the references are seed 1 only.
- One learning rate per model, no tuning grid: a fine-tuned arm could gain with tuning, which can only widen G. A G at
  most 1.0 would therefore need a tuned check before any claim.
- The agents' memory bits may encode the action history more directly than text does; the text rendering keeps the
  same information.

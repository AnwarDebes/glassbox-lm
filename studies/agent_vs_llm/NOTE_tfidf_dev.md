# Note to PLAN_ft_dev.md: a classical network-free arm on the same text (dev only, descriptive)

Written and frozen before its first number, while the fine-tuning queue of PLAN_ft_dev.md runs.

- Arm TFIDF-LR: the text that ft_baseline.py renders for each example (same function `render`), TF-IDF over word
  1- and 2-grams (lowercase, sublinear tf, min_df 2), then multinomial logistic regression (saga, max_iter 200),
  C in {1, 4, 16}; the best C on dev is reported, with every C beside it (optimistic, like the best epoch of the
  other arms). STAR and ABCD; train on train, score on dev; the test files are not opened.
- No rule: the arm is descriptive. It tells whether a classical model with no neural network at any stage, on the same
  information, is closer to the fine-tuned transformers than the network-free agent is. It does not change the decision
  rule of PLAN_ft_dev.md, which compares the fine-tuned arms with the C4 agent.
- CPU only, nice 19, 2 threads, after the CPU gate.

# Note to PLAN_ft_dev.md: saving dev predictions (written 2026-10-26T20:52:56Z)

From the second job of the queue on (ft_abcd_bert_s1 and later), ft_baseline.py also writes dev_preds.npz: for the best
dev epoch, the top-5 label ids and probabilities of every dev example and the gold labels. Training, selection and
scoring are unchanged (the previous version is kept as ft_baseline.v1_8c328b8a.py). The first job, ft_star_bert_s1, ran
with the previous version and has no predictions file. The predictions are for error analysis and later dev work; no
rule of the plan reads them.

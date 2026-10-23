# B6: team margin on composites

Question (design step B6): does rule T beat a blind composite of Tsetlin machines at equal
clauses? Rule T: a member learns from an example with probability equal to the error of the
rest of the team on that example, the error computed out of fold. Library: tmu at commit
5605ff0, CPU backend, one single-threaded process per job. Budget: 100 clauses per class per
member.

## Files

| file | what |
|---|---|
| prepare.py | splits and booleanized views, written to data/b6/ (git-ignored), with manifest.json |
| config.py | member recipes, budget, epochs, folds, refresh, head and gate settings |
| data.py | loading of the prepared views |
| engine.py | thin layer over tmu's TMClassifier: training pass over chosen examples, class sums, clause outputs, coupled team |
| combine.py | composite rules (Granmo 2023 normalization and variants), team error |
| run.py | one job: one (arm, dataset, seed), or one member of it |
| timing.py | one epoch of one member on the real train split (timing only) |
| launch.py | the whole grid through cpu_slot.sh 1, dependencies respected |
| report.py | verdict from the run folders |

## Data

Fashion-MNIST: the Hugging Face parquet files of zalando-datasets/fashion_mnist (revision
531be5e2). The train file is in the shared Hugging Face cache of the tm_language_model study;
the test file was downloaded into data/fashion_mnist_hf/ (sha256 in the manifest). AG News:
fancyzhx/ag_news (revision eb185aad) from the same cache.

Splits, seeded (20261015) and stratified by class, cut from the official training file:

| dataset | train | calib (held out) | valid | devtest (official test) |
|---|---|---|---|---|
| Fashion-MNIST | 50,000 | 5,000 | 5,000 | 10,000 |
| AG News | 108,000 | 6,000 | 6,000 | 7,600 |

Views. Fashion-MNIST, the four specialists of Granmo 2023 as in the CAIR scripts: adaptive
Gaussian thresholding with a 10x10 convolution; 8-level thermometer with 3x3 and with 4x4
convolution; HOG (flat). AG News, three views after the CAIR IMDb team script: binary word
unigrams (chi2 top 5,000), word bigrams (chi2 top 5,000), char_wb 4-grams (chi2 top 3,000),
vocabulary and selection fitted on the train split only.

Preparation needs OpenCV 4 (OpenCV 5 has no HOGDescriptor), so it runs in its own environment:

    uv venv .venv-b6 --python 3.11
    uv pip install --python .venv-b6/bin/python opencv-python-headless==4.10.0.84 numpy==2.4.6 \
        pyarrow==26.0.0 scikit-learn==1.9.1 scipy==1.17.1
    PYTHONPATH=src .venv-b6/bin/python -m tally.b6.prepare

Everything else runs with the tm_language_model environment (tmu 5605ff0).

## Arms

M members (Fashion-MNIST 4, AG News 3), 100 clauses per class each, the same T, s and epochs in
every arm. Final composites use the Granmo 2023 rule with alpha taken on the calib split.

| arm | jobs per seed and dataset | what |
|---|---|---|
| member | M | one blind member, trained alone on every example |
| best member | 0 | the blind member with the best validation accuracy |
| blind composite | 0 | the M blind members combined (Granmo 2023) |
| single | 1 | one machine with M x 100 clauses per class (Fashion-MNIST: the configuration of one member, fixed by tuning; AG News: the three views concatenated) |
| ruleT | 1 | M members and K out-of-fold shadow copies of each, refreshed every epoch |
| nocomm | M | each member at the rule T rates of the same seed, shuffled over examples |
| coupled | 1 | all members vote into one sum from the start (update probability from the joint vote) |
| seqboost | 1 | member k learns with probability equal to the out-of-fold error of members 1..k-1 |
| fusion | 1 | a TM head (100 clauses per class) over the clause outputs of the blind members |
| gate | 1 | the neural gate of tmu.composite (MLP 128-64-softmax over members), trained on calib |

Team error of rule T: for member i, S is the sum of the other members' out-of-fold class sums,
each divided by its alpha; error = 1 - softmax(S / tau)[true class], tau fitted by log loss on
the same out-of-fold scores (option --error 01: 1 if the rest of the team is wrong). Out of fold:
the train split is cut into K = 3 stratified folds; shadow copy k of every member trains on the
other folds with the same rule and scores fold k. The main members train on the whole train
split. Epoch 1 has no information: error 1 - 1/classes.

## Running

All from the study folder, PYTHONPATH=src, Python of the tm_language_model environment.

Smoke (small subsets, dev-test replaced by a validation slice, label must start with b6smoke):

    python -m tally.b6.run --arm member --dataset fmnist --seed 1 --member hog --label b6smoke --smoke 400 --epochs 2

Timing (one member, one epoch on the full train split):

    nice -n 10 ../tm_language_model/scripts/cpu_slot.sh 1 -- python -m tally.b6.timing --dataset fmnist --member thresh10

Tuning on validation, seed 1 (label must start with b6tune; --t-scale only here):

    python -m tally.b6.run --arm member --dataset fmnist --seed 1 --member hog --label b6tune_t1 --epochs 60 --eval-every 5

The grid (after the registration is frozen):

    python -m tally.b6.launch --label b6 --seeds 1-10 --registration registrations/<file>.md --dry-run
    python -m tally.b6.launch --label b6 --seeds 1-10 --registration registrations/<file>.md

Verdict:

    python -m tally.b6.report --label b6 --seeds 1-10

## Notes

- The paper normalizes with alpha over the whole scored set (the CAIR scripts use the test file);
  here alpha comes from the calib split. tmu's TMComposite divides per sample instead; that
  variant is a secondary row of the report.
- tmu's neural gate imports keras, which the environment does not have, and the library does not
  define what the gate is trained to predict. The gate arm uses the same architecture in
  scikit-learn (Dense 128 relu, Dense 64 relu, softmax over members, Adam), trained on the calib
  split to predict the member with the largest normalized margin of the true class.
- Runs never print dev-test accuracy; report.py computes it once from the saved class sums.

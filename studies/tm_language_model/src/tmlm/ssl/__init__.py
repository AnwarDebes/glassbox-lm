"""Logical self-supervised learning read-outs (docs/12 F23): what the frozen, Tsetlin-automata-learned,
self-supervised clause banks are good for as REPRESENTATIONS (concepts, transfer, patterns).

Modules (each documents itself; nothing is imported here, so the modules stay independent):
    features   document features from frozen banks (transfer tests)
    transfer   few-label transfer protocol on frozen representations
    concepts   word-concept read-outs of a clause bank (count-vote and literal profiles), teacher-free
               baselines on the same tokens, word-similarity evaluation, WordNet concept purity

Everything here is EXPLORATORY until docs/13 amendment A6 registers a confirmatory test.
"""

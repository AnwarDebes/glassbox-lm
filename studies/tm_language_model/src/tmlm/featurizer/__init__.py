"""BPE windows and the parser-derived polarity features (ablation, from MASKED text only).

Normative text: docs/03 'Featurizer spec'; docs/06 leakage rule.

Planned modules (written at the milestone named above; nothing here is implemented yet):
- build_windows(corpus, tokenizer, window=24, mask_rate=0.15) -> List[Window]   # Track A
- polarity: spaCy neg arcs, scope adverbs, discourse connectives, computed in .venv-dl and exchanged through files
"""

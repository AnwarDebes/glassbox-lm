# E0: feasibility of an OR-group learning rule on the planted-group tests (exploratory, design data)

Publication note: the text below is the plan as it was fixed on 2026-11-12T15:53:26Z (SHA-256 of that file: `d5f19fb9d86e0bbcd6f806e2213f0f830eac7cbd0cad9d4b003a016218463dfa`), with folder and file names changed to those of this repository. Rules, bars, settings and numbers are unchanged.

Written before any run of this study. Dev / design work only: no test split is read.

## Question

Check 2a of the hierarchical_tm study (L2 borderline after 1,000 passes; L3 group recovery 0 on every seed) was run with the stock feedback of PyHierarchicalTsetlinMachineCUDA 5b4a223. Does a feedback rule that treats an OR-group as a set of member words, with one Tsetlin automaton per (clause, group, word), recover the hidden groups of L2 and L3 within the stock budget of 100 passes?

## Why the stock rule is expected to struggle (reading of the package source, a hypothesis until tested)

1. An option with no included word is true, so a group with one unused option is always true (the study counted 57.5% of the OR-groups of L2 seed 1 as always true).
2. One number, `s`, sets both how rare an option may be and how many co-occurring words an option absorbs: an option must fire in more than 1/s of its feedback events to survive, and every word present in more than 1/s of its firings is included. Rare options need a large `s`; a large `s` makes options long and mixed.
3. All options of a group get the same feedback, so nothing assigns a missed positive to one free option.

## The rule under test (module `src/orgroup_tm.py`)

A clause is an AND of G groups (K per paragraph side). Each group has one automaton per word; the word is a member when the state is at least 0. A group is true when a member word is present; a group with no member is false; a clause with no member at all is blank.

For a clause picked by the usual vote gate (probability (T - v) / 2T when it should fire, (T + v) / 2T when it should not; v = clipped sum of the votes):

- should fire, blank or all false groups empty: **seed**: the present words are dealt at random to the empty groups of their side and become members;
- should fire, fires: **reinforce**: present members +1;
- should fire, does not fire: **recruit**: present non-members +1 in the false groups picked by the gating variant;
- should not fire, fires: **reject**: present members -r in one group chosen at random;
- should not fire, does not fire: **repress**: present non-members -1 in the false groups picked by the gating variant.

Gating variants: `near` (exactly one false group), `side` (every group of the other side true), `all` (every false group). No word becomes a member without having been present in an example on which the clause should fire.

## Data (design data)

`../hierarchical_tm/data/pcp/pairs_{L2,L3}_s{1,2,3}.npz` and `pairs_L3_s{1,2,3}_unseen.npz`, read only; hidden groups from `pcp_world.World(rung, seed)`. The stock results were read on these files and the rule is designed on them, so E0 is exploratory. A confirmation needs worlds of new seeds under a plan frozen first (E1), and is only run if E0 meets its bar.

## Bar (Check 2a of gates.md, unchanged)

Per rung, on at least 2 of 3 seeds, after at most 100 passes:

- dev accuracy at least 0.90, read as PLAIN accuracy at the natural cut-off (predict "same" when the vote sum is at least 1), the strictest of the readings the study reported; tuned accuracy and AUC are reported beside it;
- group recovery at least 0.80, scored by the study's own `pcp_world.recovery` (Jaccard at least 0.8, groups of at least 2 words);
- L3 only: plain accuracy on the unseen word pairs at least 0.85.

Purity and coverage of all groups (definition of `pcp_purity.py`) are reported, no bar.

## Arms and settings, fixed now

First look: 40 clauses, all of positive polarity, K = 1 group per side, T = 4, initial state -1 (one recruit event makes a word a member), seed state 0, r = 2, recruit and reinforce probability 1.0, repress probability 1.0, 5 passes, seeds 1 to 3, the three gating variants. Because E0 is design work, further arms may follow (initial state -3, r in {1, 4}, 20 or 100 passes, K = 2, half the clauses of negative polarity); every arm that is run is logged in `logs/RUNS.md` with its numbers, whatever it shows.

## Reference numbers (stock rule; read from `../hierarchical_tm/docs/results.md`, Check 2a table)

| rung, setting | tuned accuracy, seeds 1 / 2 / 3 | plain accuracy | group recovery |
|---|---|---|---|
| L2, s 21.1, 100 passes | 0.847 / 0.846 / 0.8755 | 0.523 / 0.522 / 0.572 | 0.625 / 0.5 / 0.75 |
| L2, s 21.1, 1,000 passes | 0.9215 / 0.897 / 0.9015 | 0.911 / 0.5025 / 0.634 | 0.875 / 1.0 / 0.875 |
| L3, s 800, 300 passes | 0.8365 / 0.821 / 0.821 | 0.8365 / 0.821 / 0.821 | 0.0 / 0.0 / 0.0 |
| L3 unseen pairs, s 800, 300 passes | 0.788 / 0.74 / 0.752 | not given | not applicable |

## What a miss means

- No gating variant meets the L2 bar: the rule is a dead end; logged, the line stops.
- L2 met, L3 not: one revision of the rule is allowed, labelled post hoc; if that misses too, the line stops at "L2 only".
- Both met: write E1 (new seeds, frozen first), then the slide-10 check (Check 2b, unchanged bars) and the Wikipedia pairs on dev.

## Not claimed whatever the outcome

Nothing about real text, synonyms, energy or the state of the art; nothing about a fault of the package (its behaviour may be intended); no "first" or "new" before a literature check.

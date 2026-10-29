# Addendum 1 to NOTE_review_fixes.md (written 2026-10-27T08:42:35Z)

Item 4 is replaced. The agent is not retrained for the registration: AGENT's test predictions are the registered,
verified seed files of R22 (STAR), R23 (ABCD) and R24 (MultiWOZ), seeds 1-5, at exactly the registered settings;
their seed-1 dev accuracies are the dev references of this line (0.6544585987261147, 0.8417480998914224,
0.35004756080989263). So no new agent code reads a test split, and the dry runs of agent_preds.py on STAR and
MultiWOZ are not needed: they were removed from the queue (jobs_ft.txt) before they started. agent_preds.py keeps
its new --expect-dev-acc guard (stop before any test row if the dev accuracy differs), unused here.
The four decision_layer source files that changed since R24's freeze changed in comments and docstrings only
(code_since_R24.diff: abcd.py, count_bits.py, count_text.py, static_text.py against the originals kept in
admin/publish/pass_2026-10-23/originals/).
Seen since the note: L35k50 on MultiWOZ, 343 of 500 rows answered at 08:38Z, accuracy 0.364 on them (listed by
select_llm_star.py as not eligible); TF-IDF dry runs on STAR and MultiWOZ dev (0.6528662420382165 and
0.3507270009512162) and on the 500 LLM dev rows 0.672 (STAR) and 0.384 (MultiWOZ).

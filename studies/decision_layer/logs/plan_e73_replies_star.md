# Note e73 (2026-10-19): the STAR agent produces its replies from the task's templates (dev only; rules fixed before the first dev score)

Decided at 07:44 UTC as the next agent step after R21's write-up. Source:
a plan in the project notes (not published). This note fixes the rules. It is frozen with `scripts/freeze.py`
before the first dev score. No test split is read. No rule decides a pass; it is a measurement.

**Publication note:** the wording of this copy was edited for publication on 2026-10-23. The registered question, rules, thresholds, seeds, analysis plans, amendments, dates and times are unchanged; the freeze record next to this file holds the SHA-256 of the text as registered.

## What is measured

For every labelled wizard action of STAR's dev split (star_prepare's examples, 10% of dialogues by hash), the
measure is how often the agent's reply text equals what the wizard really sent.
- **The agent:** R21's STAR models (`results/confirm/R21star/tm_mem_c4`, seeds 1 to 5). View B on the C4-word table,
  with the bits built through dl.count_text.patch and the empty "none" cache. The decision is the CPU path
  (cpu_tm_fast), so there is no neural network at any stage and no GPU.
- **The reply** is fixed by the rules below. For a query label there is no reply text (the API name only, scored as
  label accuracy).

## Rules (`scripts/star_replies.py`)

1. **Template:** the template of label L comes from the first task in the dialogue's WizardCapabilities whose
   `tasks/<task>/responses.json` has L.
2. **Slots** ({name} or {name:spec}):
   - filled from the latest knowledge-base return_item before the turn whose APIName is the template's task;
   - the slot-to-field mapping is the derived one (below);
   - the value string is inserted as is: str(v), or ", ".join for a list, and the format spec is ignored (the
     database stores display strings);
   - a slot that cannot be filled gives no reply text, and so no match.
3. **The mapping**, derived from train before any dev score, by the positional rule:
   - a train pick whose logged text (stripped) fully matches the template's pattern gives each slot its own text;
   - each slot is mapped to the Item field whose value string equals that text most often, ties to the field name
     that sorts first;
   - the file is `results/explore/star_reply_mapping_positional.json`, sha256 4dd5e768c06ada65 (`logs/e73_mapping_positional.out`).
   - The literal rule ("the field whose value string appears most often in the logged text") is kept as
     `results/explore/star_reply_mapping.json` (af465936ae283e59). It cannot tell slots apart, because every slot of
     a template got the same field (0.406 of slotted train picks filled exactly against 0.749). It is reported once
     as that, and is not used for scoring.
   - A hand-written mapping, if ever run, is a separate, labelled variant.
   - The positional mapping was chosen at 07:48 UTC; the literal rule was wrong for templates
     with more than one slot. Both files and hashes predate any dev score.

### Known limits of the mapping (listed before any dev score; train coverage from the mapping file)

Overall, 8,486 of 11,325 slotted train picks (0.749) are filled exactly. 25 of 36 slotted templates are at 0.9 or
more. These 11 are not, so the dev numbers cannot be read as having chosen the mapping:
- **trip_directions step lists** (1,920 train picks; exact 0): trip_inform_simple_step_ask_proceed (1,211),
  trip_inform_last_step_and_done (388) and trip_inform_detailed_step (321). The instruction comes from a list of
  steps in the item, so no single field holds it, and simple_instruction / detailed_instruction stay unmapped.
- **Interface text that differs from responses.json** (0 pattern matches):
  - doctor_schedule's doctor_inform_booking_available (207) and doctor_inform_booking_unavailable (106): the
    template says "{time:d} o'clock", the logged text says e.g. "12 pm";
  - hotel_service_request's hotel_inform_service_request_successful (108).
- **Unmapped or boolean slots:**
  - plane_inform_flight_details (177; clazz unmapped);
  - apartment_inform_search_result (155; pois, has_balcony, has_elevator unmapped, booleans shown as phrases);
  - spaceship_life_support's spaceship_inform_outcome (95; message unmapped);
  - hotel_ask_customer_request (12; hotel_name unmapped).
- **Partly filled:** ride_confirm_booking (64; 0.344 exact).

4. **Exact match:** the reply equals the logged text after str.strip(); case and inner spaces are kept.
5. **The ceiling:** the logged label's own template, filled the same way. It does not depend on the model.
6. **Loops:**
   - open: memory from the logged history;
   - closed: memory from the agent's own presses (`abcd_closed_loop.closed_loop`, as in R20).
   - The text stays as logged in both. A wrong reply is not fed back to the customer (R20's limit).

## Reported (per seed and as the mean over seeds 1 to 5)

- label accuracy on all actions, on template picks and on queries;
- the agent's exact matches: their share of template picks, their share of all wizard replies (picks and free
  text), and plain and slotted separately;
- the ceiling, overall and plain / slotted, and the agent's exact matches over the ceiling's;
- the share of the wizard's replies that are free text (no template), which no template agent can match;
- the train coverage of the mapping per template (from the mapping file).

## Wording

- Say "produces its replies from the task's templates". Never write "generates language" or "writes language".
- The claim is about STAR's dev split, with the text as logged. It is not a live dialogue with users.

## Run

- **When:** after R21's write-up; CPU only.
- **Command:** `HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv-dl/bin/python -I scripts/star_replies.py --mode score --seeds 1 2 3 4 5 --tag e73`
- **Rows:** results/explore/star_replies.jsonl. The script refuses to score before this note's freeze, checks that the raw dev actions align with star_prepare's dev examples (convo_id, k, label), and records the mapping's sha256 in every row.
- **Script at the freeze:** scripts/star_replies.py a90f1701ecfe5b57 (its mapping modes ran before the freeze: logs/e73_mapping.out, logs/e73_mapping_positional.out).

# B1: planted team game

Does marginal-bit feedback (each member is paid the bits the team saved because it spoke) find a
planted rule set with fewer samples than a shared reward, and does the gap grow with team size?
Second question: does the dual-state trial (old and changed clause both kept and counted until a
sequential test decides) beat applying a change directly? The registration in
`registrations/` is the binding text; this file says how to run the code.

## Files

| file | what it does |
|---|---|
| `game.py` | planted game: n = 256 random bits, R = M / 4 pairwise exclusive rules of k = log2(M) - 1 literals, P(y=1) = 0.9 if a rule fires else 0.1; entropy Hb(0.1) bits exactly |
| `team.py` | team of M one-clause members (Tsetlin automata, 64 states, 512 literals), counted log-odds sum, the three rewards, reward to Type I / II feedback, direct changes, dual-state trial, fixed-sample A/B test (numba) |
| `search.py` | greedy and random search on the same team at equal evaluations (numba) |
| `flat.py` | tuned flat Tsetlin machine (tmu TMClassifier, CPU) with a logistic calibration of the class sums |
| `calib.py` | fixes the member constants s and RHO on a one-member game, where the rewards cannot favour an arm |
| `run.py` | one run = one (arm, M, seed); writes a run folder with the excess-bits curve |
| `selfcheck.py` | checks: exclusive rules, exact entropy, oracle at zero, duplicate gets zero, closed-form marginal = recomputation, probabilities, evaluation count |
| `report.py` | verdict from the run folders |
| `../../../scripts/b1_launch.sh` | the full grid through cpu_slot.sh (needs a frozen registration) |

## Run

From the study folder, with the study's Python:

    export PYTHONPATH=src NUMBA_NUM_THREADS=1
    python -m tally.b1.selfcheck
    python -m tally.b1.run --arm marginal --M 8 --seed 0 --label smoke --cap 30000
    python -m tally.b1.run --arm flat_tune --M 8 --seed 10000 --label smoke --cap 40000
    python -m tally.b1.run --arm flat --M 8 --seed 0 --label smoke --cap 30000
    python -m tally.b1.run --arm dual --M 8 --seed 0 --label smoke --cap 30000
    python -m tally.b1.run --arm ab --M 8 --seed 0 --label smoke --cap 30000
    python -m tally.b1.report --label smoke

Labels: `smoke` and `timing` for tests, `reg` only for the registered grid (`--registration`
required, caps fixed in `run.CAPS`; a `reg` run refuses to start while `run.CAPS_FROM`, the run
id of the timing run that fixed the caps, is not set). Longer runs go through
`nice -n 19 ../tm_language_model/scripts/cpu_slot.sh 1 -- ...` with the thread pools capped at 1,
and start only when `../decision_layer/scripts/cpu_load.py 10` reads at most 4.5 cores; the grid
launcher `scripts/b1_launch.sh` does this before every start (`B1_DRY_RUN=1` prints its jobs).

## Open before a registration (2026-10-25)

No B1 registration exists. The smoke and calibration runs show four problems that a
registration has to settle first:

- The shared reward never reached the target in the member calibration: in
  `20261015T074132Z_b1_smoke_calib2`, which ran the current `calib.py`, all 60 shared runs were
  censored at 1,000,000 samples, while the selfish and marginal rules reached it in all their
  runs at the 8 grid points with s at most 3.9. An earlier calibration with three rules
  (`20261015T072321Z_b1_smoke_calib`, an earlier `calib.py`) has the first five runs of every grid
  point censored (45 of 45); which rule they used is not recorded (UNVERIFIED). A primary ratio
  shared / marginal would then be decided by a comparator that does not learn on this game.
- The team constants of `team.default_params()` (s 3.9, RHO 0.25) were not chosen by a
  calibration: the three calibrations chose s 2.5 / RHO 1.0, s 3.9 / RHO 1.0 and s 2.5 /
  RHO 0.5, and RHO 0.25 is in none of their grids.
- The caps (`run.CAPS`, 2,000,000 samples for every M) are provisional; no timing run has fixed
  them (`run.CAPS_FROM` is None).
- The dual-state success line has no interval condition (reduction at least 0.25 and never
  worse); the primary line has one.

Fixed on 2026-10-25: the dual-state line of `report.py` no longer reads "kill" when the runs
are censored (both arms at the cap gave a reduction of 0); success is blocked when dual runs at
M = 128 are censored and kill when marginal runs are (`tests/test_b1_report.py`).

## Arms

| arm | what it does |
|---|---|
| `shared` | every member gets the team's log-loss gain over the background predictor on the sample |
| `selfish` | each member gets its own gain as a lone predictor (background plus its evidence) |
| `marginal` | each member gets team loss without it minus team loss with it (closed form); changes applied directly |
| `dual` | `marginal`, but a changed clause is tried beside the old one (both counted) until a Wald test over old / new / both decides |
| `ab` | `dual` with a fixed-length test instead of the sequential one; its length is the mean trial length of the `dual` run with the same M and seed |
| `greedy` | one member at a time adds the literal that saves the most bits on a fresh batch; same evaluations per sample as a team |
| `random` | one member at a time tries random clauses on a fresh batch; same evaluations per sample |
| `flat` | tmu TMClassifier with M clauses, tuned per size on a tuning game, probability by logistic calibration on 2,000 separate samples |
| `flat_tune` | the tuning of `flat` for one size (seed 10000) |
| `oracle` | the planted clauses as members, only counting (no learning); the true probability has excess 0 by construction |

Reward to feedback is the same in every Tsetlin arm: with probability min(1, |r| / RHO), a
member whose clause fires gets Type Ia when r > 0 and Type II when r < 0; a member whose clause
does not fire gets Type Ib when r > 0.

## Output

Each run folder (`results/<UTC>_b1_<label>_<arm>_M<M>_s<seed>/`) holds config, source hashes,
log, and `metrics.json` with `curve` (samples, evaluations, excess bits over the entropy on the
20,000-sample hold-out, log-loss minus entropy), `samples_to_target` (first checkpoint where the
excess is within 0.01 bits there and at the next checkpoint) and `censored`.

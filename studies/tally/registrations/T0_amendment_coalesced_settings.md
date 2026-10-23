# T0 amendment, 2026-10-15: settings of the coalesced machine

Written at 2026-10-15T18:24:29Z by `scripts/t0_auto.py` from the tuning runs, by the rule of `T0.md` (lowest learned-arm bits per word on valid, seed 1; ties to the lower s), before any final run of this machine.

- s = 5: run `20261015T072026Z_t0_tune_coal_c2000_s5_seed1`, best learned valid 9.243282 bits per word at epoch 1
- s = 10: run `20261015T072026Z_t0_tune_coal_c2000_s10_seed1`, best learned valid 9.084931 bits per word at epoch 4
- s = 20: run `20261015T072026Z_t0_tune_coal_c2000_s20_seed1`, best learned valid 9.057047 bits per word at epoch 9

Chosen: s = 20, epochs = 9 (run `20261015T072026Z_t0_tune_coal_c2000_s20_seed1`).

Code change after the freeze of T0.md, log text only: run.py printed 418 dropped positions per epoch also for the flat machine, whose trainer drops none; it now prints 0 for flat.

# T0 amendment, 2026-10-15: settings of the flat machine

Written at 2026-10-15T21:14:27Z by `scripts/t0_auto.py` from the tuning runs, by the rule of `T0.md` (lowest learned-arm bits per word on valid, seed 1; ties to the lower s), before any final run of this machine.

- s = 5: run `20261015T113259Z_t0_tune_flat_c100_s5_seed1`, best learned valid 10.634485 bits per word at epoch 1
- s = 10: run `20261015T072535Z_t0_tune_flat_c100_s10_seed1`, best learned valid 11.115641 bits per word at epoch 1
- s = 20: run `20261015T181024Z_t0_tune_flat_c100_s20_seed1`, best learned valid 11.441973 bits per word at epoch 1

Chosen: s = 5, epochs = 1 (run `20261015T113259Z_t0_tune_flat_c100_s5_seed1`).

Code change after the freeze of T0.md, log text only: run.py printed 418 dropped positions per epoch also for the flat machine, whose trainer drops none; it now prints 0 for flat.

# R2d and R7c amendment, written 2026-10-16T12:05:17Z. One R2d result file existed by then (banking77_B_trim32_rank seed 1, written 11:56:40Z); it had not been opened

- The registration calls the re-runs the test splits' "second use". For R2d it is the second use of the Banking77 and HWU64 test splits by a lexical router (after R2c). For R7c it is the third use of the CLINC150, Banking77 and HWU64 test splits with this readable view (after R7 and R7b).
- Nothing else changes.

## Note added 2026-10-16T14:31:51Z, after the results

- The R7c job log (logs/R2d_R7c.out), which prints each finished run's test accuracy, was read at 14:07 to find the out-of-memory crashes, before the relaunch. The relaunch repeats the 6 registered commands, and R7c needs all 10 seeds, so no choice depended on what the log showed.
- 16 of R2d's 20 models and 14 of R7c's 30 models are bit-identical to R2c's and R7b's (their runs met no tie). R2d and R7c check the length bound; they are not new tests of the accuracy results.

# R4b: amendment written 2026-10-15 at 23:00Z, before any confirmatory run of R4b

## Selection applied (rule of the frozen text)

Candidates, val accuracy (CLINC150, MiniLM, seed 1):

| Candidate | Val accuracy |
|---|---|
| `e5_c16k` (16,000 clauses, 40 epochs) | 0.9563 |
| `e14_c32k` (32,000 clauses, 40 epochs) | 0.9547 |
| `e14_c16k_e80` (16,000 clauses, 80 epochs) | 0.9567 |

- The best is `e14_c16k_e80`. `e5_c16k` is within 0.10 points of it (0.04 lower) and costs half as much (clauses × epochs), so the tie rule chooses it.
- **TM-single = thermo:8, 16,000 clauses, T 16,000, s 8, 40 epochs, batch 128, hardest negatives.**

## Source changes since the freeze

`scripts/diff_frozen.py` lists `scripts/r6_run.py` (fitted-Q option, R6 only) and new helper scripts (`r6_seed.sh`, `rules_check.py`, `select_r5r6.py`). None is used by R4b's arms.

## Addition at 23:04Z, still before any R4b TM run

- At 16,000 clauses, the float32 count tensors of one update take 3 to 7 GB per job on the shared V100 (bge-base: 16,000 × 10,752 literals). The first R4b queue (launched 23:00:59Z) was stopped at once.
- Not one R4b TM run had started; only logistic-regression jobs could have run.
- `src/dl/tm.py` now does the automaton update in clause chunks when clauses × literals exceed 50,000,000 (`CHUNK_ELEMS`).
  - The update rule is the same, and every Binomial draw keeps its distribution. Only the order of the random stream differs.
  - A check on a small task, forcing chunks: mean accuracy over 4 seeds 0.563 unchunked against 0.571 chunked.
- Every R4b TM run uses the chunked path (16,000 × 5,376 and 16,000 × 10,752 are both above the threshold).
- R4 (4,000 clauses: at most 43,008,000 pairs) and R2 (16,000,000) stay on the unchunked path, unchanged.

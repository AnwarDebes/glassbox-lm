# R5b amendment, written 2026-10-16 at 06:03:31Z, before any R5b result exists (results/confirm/R5b holds no result file)

Correction of a sentence in the frozen text. It says "A 0.05 share of uniform random actions remains (the epsilon of the code's default; --eps 0.05)". That is not what the code does.

In `src/dl/bandit.py`, `TMBandit.act` returns the Boltzmann sample as soon as `boltzmann > 0`, so the epsilon branch is never reached. Exploration is pure Boltzmann sampling over the class sums. The exploratory runs that chose tau (e16, e17) ran the same code, so the registered arm is what was explored. Only the description was wrong. Nothing else changes.

"""What a decoder records per generated token (the input of `tmlm.certify.receipt`).

A `StepRecord` is everything needed to re-derive one token without the model: the window
(tokens and hidden flags) the prediction was computed on, the target position, the clause ids
that fired, the exact class sums, the decision mode, the random draw and the survivor mask of
the elimination decoder. `Generation` is the ordered list of records plus the final window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sampling import Draw


@dataclass
class StepRecord:
    step: int
    position: int
    claimed: int
    mode: str  # "sample" | "greedy" | "eliminate"
    tokens: np.ndarray  # u16 [L] window at prediction time (the value at hidden positions is irrelevant)
    hidden: np.ndarray  # u8 [L]
    fired: np.ndarray  # u32 ascending clause ids
    sums: np.ndarray  # int32 [V]
    temperature: float
    draw: Draw | None
    licensing: np.ndarray | None = None  # eliminate: TRUE clauses with positive votes
    eliminating: np.ndarray | None = None  # eliminate: TRUE clauses with negative votes
    pending: int = 0  # eliminate: clauses whose truth is still unknown
    survivors: np.ndarray | None = None  # eliminate: bool [V]
    conflict: bool = False  # eliminate: survivors were empty, fell back to every class


@dataclass
class Generation:
    prompt: np.ndarray  # u16 [P]
    tokens: np.ndarray  # u16 [L] final window (prompt + generated)
    generated_positions: np.ndarray  # int64, ascending
    records: list[StepRecord] = field(default_factory=list)
    seconds: float = 0.0
    engine_calls: int = 0
    info: dict = field(default_factory=dict)

    @property
    def generated(self) -> np.ndarray:
        return self.tokens[self.generated_positions]

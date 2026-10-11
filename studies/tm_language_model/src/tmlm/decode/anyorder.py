"""Any-order autoregressive sampling over the masked predictor (docs/03 "Track C: decoders",
docs/04 M8, the RADD reading: a per-position masked predictor supports any-order decoding).

`anyorder_generate` fills the hidden positions of a window one at a time. The order is
`left_to_right`, `random` (a counter-based key per position) or `confidence` (the open position
with the largest vote margin, ties toward the lower position). Each token is drawn from the
calibrated distribution of `predict_scores` (docs/03 "Probabilities and perplexity") through the
counter-based sampler, or taken greedily (exact integer argmax of the logit numerators).

`sequential_nll` is the exact log-likelihood of a complete window under the chosen order:
sum over positions of -log p(x_p | the positions before it in the order), the rest hidden.
It is a proper likelihood of the any-order factorisation (not the causal TM perplexity of
protocol CP, which needs the causal training), reported next to the diffusion ELBO.
"""

from __future__ import annotations

import time

import numpy as np

from .conftm import logit_numerators
from .sampling import DRAW_TOKEN, draw, order_keys, sample_token
from .trace import Generation, StepRecord

ORDERS = ("left_to_right", "random", "confidence")


def _margin(row: np.ndarray) -> int:
    """top-1 minus top-2 class sum (docs/03: unmasking order by clause-vote margin)."""
    if row.shape[0] < 2:
        return int(row.max())
    part = np.partition(row, -2)
    return int(part[-1] - part[-2])


def prepare_window(scorer, prompt: np.ndarray, n_new: int) -> np.ndarray:
    """Write the prompt, hide `n_new` positions after it; returns the generated positions."""
    prompt = np.asarray(prompt, dtype=np.uint16)
    L = int(prompt.shape[0]) + int(n_new)
    if scorer.L != L:
        raise ValueError(f"the scorer is bound to length {scorer.L}, the window needs {L}")
    tokens = np.zeros(L, dtype=np.uint16)
    tokens[: prompt.shape[0]] = prompt
    hidden = np.zeros(L, dtype=np.uint8)
    hidden[prompt.shape[0]:] = 1
    scorer.set_window(tokens, hidden)
    return np.arange(prompt.shape[0], L, dtype=np.int64)


def choose_position(scorer, order: str, seed: int, step: int, open_pos: np.ndarray, scores_all=None) -> int:
    if order == "left_to_right":
        return int(open_pos.min())
    if order == "random":
        keys = order_keys(seed, open_pos, step)
        best = np.flatnonzero(keys == keys.min())
        return int(open_pos[best].min())
    if order == "confidence":
        rows = scores_all if scores_all is not None else scorer.scores(open_pos)
        margins = np.array([_margin(r) for r in rows])
        best = np.flatnonzero(margins == margins.max())
        return int(open_pos[best].min())
    raise ValueError(f"order must be one of {ORDERS}")


def anyorder_generate(scorer, prompt: np.ndarray, n_new: int, seed: int, order: str = "left_to_right",
                      temperature: float = 1.0, greedy: bool = False, record: bool = True) -> Generation:
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    model = scorer.model
    if model.cfg.mode == "causal" and order != "left_to_right":
        raise ValueError("a causal model only supports left_to_right decoding")
    t0 = time.time()
    calls0 = getattr(scorer, "n_engine_calls", 0)
    gen_pos = prepare_window(scorer, prompt, n_new)
    out = Generation(prompt=np.asarray(prompt, dtype=np.uint16), tokens=np.zeros(scorer.L, dtype=np.uint16),
                     generated_positions=gen_pos)
    cal, bias = model.calibration, model.class_bias
    for step in range(int(n_new)):
        open_pos = scorer.open_positions()
        p = choose_position(scorer, order, seed, step, open_pos)
        row = scorer.scores(np.array([p]))[0]
        num = logit_numerators(row[None, :], bias, cal.a_q16, cal.b_q16)[0]
        if greedy:
            d, claimed, mode = None, int(num.argmax()), "greedy"
        else:
            d = draw(seed, step, p, DRAW_TOKEN)
            claimed, mode = sample_token(num, temperature, d), "sample"
        if record:
            out.records.append(StepRecord(step=step, position=p, claimed=claimed, mode=mode,
                                          tokens=scorer.tokens.copy(), hidden=scorer.hidden.copy(),
                                          fired=np.asarray(scorer.fired(p), dtype=np.uint32), sums=row.copy(),
                                          temperature=float(temperature), draw=d))
        scorer.set_token(p, claimed)
    out.tokens = scorer.tokens.copy()
    out.seconds = time.time() - t0
    out.engine_calls = getattr(scorer, "n_engine_calls", 0) - calls0
    out.info = {"decoder": "anyorder", "order": order, "temperature": float(temperature), "greedy": bool(greedy), "seed": int(seed)}
    return out


def sequential_nll(scorer, tokens: np.ndarray, order: str = "left_to_right", seed: int = 0) -> dict:
    """Exact NLL (nats) of the complete window `tokens` under the any-order factorisation."""
    tokens = np.asarray(tokens, dtype=np.uint16)
    L = int(tokens.shape[0])
    if scorer.L != L:
        raise ValueError("scorer length mismatch")
    model = scorer.model
    scorer.set_window(tokens, np.ones(L, dtype=np.uint8))
    total = 0.0
    for step in range(L):
        open_pos = scorer.open_positions()
        p = choose_position(scorer, order, seed, step, open_pos)
        lp = model.calibration.logprobs(scorer.scores(np.array([p])), model.class_bias)[0]
        total += float(-lp[int(tokens[p])])
        scorer.set_token(p, int(tokens[p]))
    return {"nll_nats": total, "n_tokens": L, "nats_per_token": total / L, "perplexity_per_token": float(np.exp(total / L)),
            "bits_per_token": total / L / np.log(2.0), "order": order}

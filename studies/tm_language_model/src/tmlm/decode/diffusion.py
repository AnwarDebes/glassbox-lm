"""Masked-diffusion sampler and ELBO (docs/03 "Diffusion sampler spec"; docs/05 E4; docs/04 M8;
MDLM and RADD [verified in paper, docs/09]).

Sampling: start with every generated position hidden; at each of `steps` iterations score all
hidden positions, draw a token per position from the calibrated distribution, commit the
positions with the largest clause-vote margin (highest confidence first, ties toward the lower
position) so that the number still hidden follows the linear schedule
`n_hidden(s) = round(n_new * (1 - s / steps))`; then, if `remask_frac > 0`, re-hide that
fraction of the generated visible positions with the lowest calibrated log-probability of their
current token (a re-hidden position is re-sampled in a later step). Every draw is counter-based.

ELBO: for the absorbing-state diffusion with the linear schedule the negative ELBO of a window x is

    NELBO(x) = E_{t ~ U(0,1)} [ (1 / t) * sum_l 1[x_t^l = MASK] * (-log p_theta(x^l | x_t)) ]

(MDLM eq. for alpha_t = 1 - t, where -alpha_t' / (1 - alpha_t) = 1 / t) with x_t obtained by
hiding every position independently with probability t, which is exactly the `diffusion`
training corruption of docs/11 section 1.3. `elbo_nll` estimates it with `n_samples` stratified
rates t_k = (k + u_k) / K and the counter-based RNG for u_k and the masks; the estimate is
unbiased for the bound, and the bound is an UPPER bound on the true NLL (reported as such,
docs/06). ELBO perplexity = exp(NELBO / L) per token. The kill criterion of docs/04 M8 compares
it with the causal TM perplexity of protocol CP, which needs the causal training run.
"""

from __future__ import annotations

import time

import numpy as np

from .anyorder import _margin, prepare_window
from .conftm import logit_numerators
from .sampling import DRAW_TOKEN, ELBO_MASK, ELBO_RATE, TWO32, draw, sample_token
from .trace import Generation, StepRecord


def _schedule(n_new: int, steps: int) -> list[int]:
    """Number of generated positions still hidden after step s = 1..steps (last is 0)."""
    return [int(round(n_new * (1.0 - s / steps))) for s in range(1, steps + 1)]


def diffusion_generate(scorer, prompt: np.ndarray, n_new: int, seed: int, steps: int | None = None,
                       temperature: float = 1.0, remask_frac: float = 0.0, record: bool = True) -> Generation:
    model = scorer.model
    if model.cfg.mode == "causal":
        raise ValueError("diffusion sampling needs a masked or diffusion model (both sides visible)")
    steps = int(steps) if steps else int(n_new)
    if steps < 1 or not 0.0 <= remask_frac < 1.0:
        raise ValueError("steps >= 1 and 0 <= remask_frac < 1")
    t0 = time.time()
    calls0 = getattr(scorer, "n_engine_calls", 0)
    gen_pos = prepare_window(scorer, prompt, n_new)
    gen_set = set(int(p) for p in gen_pos)
    out = Generation(prompt=np.asarray(prompt, dtype=np.uint16), tokens=np.zeros(scorer.L, dtype=np.uint16),
                     generated_positions=gen_pos)
    cal, bias = model.calibration, model.class_bias
    targets = _schedule(int(n_new), steps)
    step = 0
    while True:
        open_pos = np.array(sorted(p for p in scorer.open_positions() if int(p) in gen_set), dtype=np.int64)
        if open_pos.shape[0] == 0:
            break
        want_hidden = targets[min(step, steps - 1)] if step < steps else 0
        n_commit = max(1, open_pos.shape[0] - want_hidden)
        rows = scorer.scores(open_pos)
        tokens_snapshot, hidden_snapshot = scorer.tokens.copy(), scorer.hidden.copy()
        proposals = []
        for i, p in enumerate(open_pos):
            num = logit_numerators(rows[i][None, :], bias, cal.a_q16, cal.b_q16)[0]
            d = draw(seed, step, int(p), DRAW_TOKEN)
            proposals.append((-_margin(rows[i]), int(p), sample_token(num, temperature, d), d, i))
        proposals.sort()  # most confident first, ties toward the lower position
        for neg_margin, p, claimed, d, i in proposals[:n_commit]:
            if record:
                out.records.append(StepRecord(step=step, position=p, claimed=claimed, mode="sample",
                                              tokens=tokens_snapshot, hidden=hidden_snapshot,
                                              fired=np.asarray(scorer.fired(p), dtype=np.uint32), sums=rows[i].copy(),
                                              temperature=float(temperature), draw=d))
        for neg_margin, p, claimed, d, i in proposals[:n_commit]:
            scorer.set_token(p, claimed)
        # remasking of low-confidence generated tokens (never in the final step)
        if remask_frac > 0.0 and step < steps - 1:
            visible = np.array(sorted(p for p in gen_set if scorer.hidden[p] == 0), dtype=np.int64)
            n_remask = int(np.floor(remask_frac * visible.shape[0]))
            if n_remask > 0:
                lp = cal.logprobs(scorer.scores(visible), bias, temperature)
                conf = lp[np.arange(visible.shape[0]), scorer.tokens[visible].astype(np.int64)]
                order = np.lexsort((visible, conf))  # lowest confidence first, ties toward the lower position
                for p in visible[order[:n_remask]]:
                    scorer.clear(int(p))
        step += 1
    out.tokens = scorer.tokens.copy()
    out.seconds = time.time() - t0
    out.engine_calls = getattr(scorer, "n_engine_calls", 0) - calls0
    out.info = {"decoder": "diffusion", "steps": steps, "temperature": float(temperature), "remask_frac": float(remask_frac),
                "seed": int(seed), "iterations": step}
    return out


def elbo_nll(scorer, tokens: np.ndarray, n_samples: int, seed: int) -> dict:
    """Monte Carlo estimate of the negative ELBO (nats) of the complete window `tokens`."""
    tokens = np.asarray(tokens, dtype=np.uint16)
    L = int(tokens.shape[0])
    if scorer.L != L:
        raise ValueError("scorer length mismatch")
    model = scorer.model
    K = int(n_samples)
    terms = []
    for k in range(K):
        u = draw(seed, k, 0, ELBO_RATE).u32
        t_q32 = (k * TWO32 + u) // K  # stratified rate in [k/K, (k+1)/K)
        t_q32 = max(1, t_q32)
        hidden = np.array([1 if draw(seed, k, l, ELBO_MASK).u32 < t_q32 else 0 for l in range(L)], dtype=np.uint8)
        masked = np.flatnonzero(hidden)
        if masked.shape[0] == 0:
            terms.append(0.0)
            continue
        scorer.set_window(tokens, hidden)
        lp = model.calibration.logprobs(scorer.scores(masked), model.class_bias)
        nll = float(-lp[np.arange(masked.shape[0]), tokens[masked].astype(np.int64)].sum())
        terms.append(nll / (t_q32 / TWO32))
    nelbo = float(np.mean(terms))
    return {"nelbo_nats": nelbo, "n_tokens": L, "n_samples": K, "nats_per_token": nelbo / L,
            "elbo_perplexity_per_token": float(np.exp(nelbo / L)), "bits_per_token": nelbo / L / np.log(2.0),
            "sample_sd_of_terms": float(np.std(terms, ddof=1)) if K > 1 else float("nan")}

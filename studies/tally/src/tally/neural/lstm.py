"""Exploratory neural reference for line C3: word LSTM language model on the tally PTB splits.

    python -m tally.neural.lstm --size medium|large --seed 1 [--epochs N] [--loss sum_steps|mean_tokens]
        [--clip C]

Zaremba et al. 2014 settings (arXiv 1409.2329, section 4.1): 2 layers, tied embeddings off,
plain SGD with lr 1, BPTT 35, batch 20, dropout 0.5 medium / 0.65 large on the non-recurrent
connections, init uniform 0.05 / 0.04, lr divided by 1.2 / 1.15 per epoch after epoch 6 / 14,
39 / 55 epochs, gradient norm clipped at 5 (medium) / 10 (large).
Valid perplexity after every epoch with the number of training tokens seen, for the budget
curve, and the mean gradient norm before clipping with the share of clipped steps. Dev-test is
not scored. No claim: preparation of the C3 comparison.

Loss (changed 2026-10-25, finish pass): the reference code of the paper (github.com/wojzaremba/
lstm, main.lua at 76870253) adds the gradients of the 35 unrolled steps, each step's loss being
Torch's ClassNLLCriterion, which averages over the 20 rows of the minibatch (sizeAverage true by
default); it clips the norm of that summed gradient and updates with lr 1. The loss is therefore
the sum over the time steps of the batch-mean cross-entropy, the default here (--loss sum_steps:
summed cross-entropy divided by the batch size). The 20 runs of 2026-10-15 used the mean over all
35 x 20 tokens (--loss mean_tokens), a gradient 35 times smaller, and clip 5 for both sizes
(--clip 5): they are under-trained, and those two options reproduce their setting.
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn

from tally import ptb, runlog

CFG = {"medium": dict(h=650, drop=0.5, epochs=39, decay_from=6, decay=1.2, init=0.05, clip=5.0),
       "large": dict(h=1500, drop=0.65, epochs=55, decay_from=14, decay=1.15, init=0.04, clip=10.0)}
LOSSES = ("sum_steps", "mean_tokens")
BPTT, BATCH = 35, 20


class LM(nn.Module):
    def __init__(self, V, h, drop, init):
        super().__init__()
        self.emb = nn.Embedding(V, h)
        self.rnn = nn.LSTM(h, h, 2, dropout=drop)
        self.drop = nn.Dropout(drop)
        self.out = nn.Linear(h, V)
        for p in self.parameters():
            nn.init.uniform_(p, -init, init)

    def forward(self, x, state):
        y, state = self.rnn(self.drop(self.emb(x)), state)
        return self.out(self.drop(y)), state


def batchify(ids, bsz, dev):
    n = len(ids) // bsz
    return torch.from_numpy(ids[: n * bsz].astype(np.int64)).view(bsz, n).t().contiguous().to(dev)


def chunk_loss(z, y, mode):
    """Training loss of one chunk; z (steps, batch, V) logits, y (steps, batch) targets.
    sum_steps: sum over the steps of the batch-mean cross-entropy (Zaremba et al.);
    mean_tokens: mean over all steps and rows (the runs of 2026-10-15)."""
    ce = nn.functional.cross_entropy(z.reshape(-1, z.size(-1)), y.reshape(-1), reduction="sum")
    if mode == "sum_steps":
        return ce / y.size(1)
    if mode == "mean_tokens":
        return ce / y.numel()
    raise ValueError(mode)


def train_epoch(model, opt, data, loss_mode, clip, bptt=BPTT):
    """One pass over the batchified train stream with truncated BPTT; returns (tokens seen,
    mean gradient norm before clipping, share of steps whose norm was clipped)."""
    model.train()
    state, seen, norms = None, 0, []
    for i in range(0, data.size(0) - 1, bptt):
        x, y = data[i:i + bptt], data[i + 1:i + 1 + bptt]
        x = x[: len(y)]
        state = tuple(s.detach() for s in state) if state else None
        z, state = model(x, state)
        loss = chunk_loss(z, y, loss_mode)
        opt.zero_grad()
        loss.backward()
        norms.append(float(nn.utils.clip_grad_norm_(model.parameters(), clip)))
        opt.step()
        seen += y.numel()
    norms = np.asarray(norms)
    return seen, float(norms.mean()), float((norms > clip).mean())


def evaluate(model, data, bptt=BPTT):
    model.eval()
    tot, n, state = 0.0, 0, None
    with torch.no_grad():
        for i in range(0, data.size(0) - 1, bptt):
            x, y = data[i:i + bptt], data[i + 1:i + 1 + bptt]
            x = x[: len(y)]
            z, state = model(x, state)
            tot += nn.functional.cross_entropy(z.view(-1, z.size(-1)), y.reshape(-1), reduction="sum").item()
            n += y.numel()
    return tot / n


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=CFG, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=None, help="stop after this many epochs (smoke); "
                    "the lr schedule is that of the full run")
    ap.add_argument("--loss", choices=LOSSES, default="sum_steps")
    ap.add_argument("--clip", type=float, default=None, help="default: 5 medium, 10 large")
    a = ap.parse_args(argv)
    if a.clip is None:
        a.clip = CFG[a.size]["clip"]
    return a


def main():
    a = parse()
    c = CFG[a.size]
    n_ep = min(a.epochs or c["epochs"], c["epochs"])
    name = f"c3prep_lstm_{a.size}_seed{a.seed}_valid_{a.loss}_clip{a.clip:g}" + (f"_smoke{n_ep}ep" if n_ep < c["epochs"] else "")
    run = runlog.Run(name, dict(vars(a), **{k: v for k, v in c.items() if k != "clip"}, epochs_run=n_ep, bptt=BPTT,
                                batch=BATCH, lr=1.0, torch=torch.__version__, note="exploratory, valid only"),
                     data_files=[ptb.path("train"), ptb.path("valid")])
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    vocab, _, ids, _ = ptb.load()
    tr, va = batchify(ids["train"], BATCH, dev), batchify(ids["valid"], 10, dev)
    model = LM(len(vocab), c["h"], c["drop"], c["init"]).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    lr, seen, hist = 1.0, 0, []
    for ep in range(1, n_ep + 1):
        if ep > c["decay_from"]:
            lr /= c["decay"]
            for g in opt.param_groups:
                g["lr"] = lr
        t0 = time.time()
        n, gnorm, clipped = train_epoch(model, opt, tr, a.loss, a.clip)
        seen += n
        v = evaluate(model, va)
        rec = dict(epoch=ep, tokens_seen=seen, lr=lr, valid_ppl=math.exp(v), valid_bpw=v / math.log(2),
                   grad_norm_mean=gnorm, clipped_share=clipped, epoch_s=round(time.time() - t0, 1))
        hist.append(rec)
        run.print("EPOCH", rec)
    best = min(hist, key=lambda r: r["valid_ppl"])
    run.finish(dict(history=hist, best_valid_ppl=best["valid_ppl"], best_epoch=best["epoch"]))


if __name__ == "__main__":
    main()

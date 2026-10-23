"""Exploratory neural reference for line C3: word LSTM language model on the tally PTB splits.

    python -m tally.neural.lstm --size medium|large --seed 1

Zaremba et al. 2014 settings (2 layers, tied embeddings off, SGD lr 1, clip 5, BPTT 35, batch 20,
dropout 0.5 medium / 0.65 large, lr divided by 1.2 / 1.15 per epoch after epoch 6 / 14).
Valid perplexity after every epoch with the number of training tokens seen, for the budget
curve. Dev-test is not scored. No claim: preparation of the C3 comparison.
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn

from tally import ptb, runlog

CFG = {"medium": dict(h=650, drop=0.5, epochs=39, decay_from=6, decay=1.2, init=0.05),
       "large": dict(h=1500, drop=0.65, epochs=55, decay_from=14, decay=1.15, init=0.04)}


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


def evaluate(model, data, bptt=35):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=CFG, required=True)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    c = CFG[a.size]
    run = runlog.Run(f"c3prep_lstm_{a.size}_seed{a.seed}_valid", dict(vars(a), **c, bptt=35, batch=20, lr=1.0, clip=5,
                     torch=torch.__version__, note="exploratory, valid only"), data_files=[ptb.path("train"), ptb.path("valid")])
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    vocab, _, ids, _ = ptb.load()
    tr, va = batchify(ids["train"], 20, dev), batchify(ids["valid"], 10, dev)
    model = LM(len(vocab), c["h"], c["drop"], c["init"]).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    lr, seen, hist = 1.0, 0, []
    for ep in range(1, c["epochs"] + 1):
        if ep > c["decay_from"]:
            lr /= c["decay"]
            for g in opt.param_groups:
                g["lr"] = lr
        model.train()
        t0, state = time.time(), None
        for i in range(0, tr.size(0) - 1, 35):
            x, y = tr[i:i + 35], tr[i + 1:i + 36]
            x = x[: len(y)]
            state = tuple(s.detach() for s in state) if state else None
            z, state = model(x, state)
            loss = nn.functional.cross_entropy(z.view(-1, z.size(-1)), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            seen += y.numel()
        v = evaluate(model, va)
        rec = dict(epoch=ep, tokens_seen=seen, lr=lr, valid_ppl=math.exp(v), valid_bpw=v / math.log(2),
                   epoch_s=round(time.time() - t0, 1))
        hist.append(rec)
        run.print("EPOCH", rec)
    best = min(hist, key=lambda r: r["valid_ppl"])
    run.finish(dict(history=hist, best_valid_ppl=best["valid_ppl"], best_epoch=best["epoch"]))


if __name__ == "__main__":
    main()

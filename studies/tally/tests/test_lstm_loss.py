"""Loss scaling of the C3 LSTM (src/tally/neural/lstm.py) on a tiny model, on the CPU.

Run: CUDA_VISIBLE_DEVICES= PYTHONPATH=src ../clause_lm/.venv-cdllm-gpu/bin/python tests/test_lstm_loss.py

Needs torch (the C3 runs record torch 2.6.0+cu124; .venv-tm has no torch). Checks:
1. The default loss is the sum over the time steps of the batch-mean cross-entropy, as in the
   reference code of Zaremba et al. 2014, written out here with explicit loops; the old loss
   (--loss mean_tokens) is the mean over all tokens, smaller by the number of steps.
2. The default clip is 5 for the medium and 10 for the large model (paper, section 4.1).
3. On a tiny LM (dropout 0), the gradient of the default loss is the number of steps times the
   gradient of the old loss, and one SGD step of train_epoch with lr 1 and a clip that does not
   act moves the parameters that many times further.
4. With a clip that acts, both steps are cut to the clip norm.
5. 30 epochs of train_epoch on a short repeating stream lower its cross-entropy below half.
"""
import sys

import torch
import torch.nn as nn

from tally.neural import lstm

FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f" ({detail})" if detail and not cond else ""), flush=True)
    if not cond:
        FAILS.append(name)


def params_vec(m):
    return torch.cat([p.detach().reshape(-1).clone() for p in m.parameters()])


def grads_vec(m):
    return torch.cat([p.grad.detach().reshape(-1).clone() for p in m.parameters()])


def main():
    torch.manual_seed(0)
    if not hasattr(lstm, "chunk_loss") or not hasattr(lstm, "train_epoch"):
        check("lstm.py has a loss-scaling option (chunk_loss, train_epoch)", False,
              "missing: the loss is the mean over all tokens, fixed in main()")
        print(f"FAIL: {len(FAILS)} checks")
        sys.exit(1)
    T, B, V = 7, 3, 11
    z = torch.randn(T, B, V, dtype=torch.float64)
    y = torch.randint(0, V, (T, B))
    ref = sum(sum(-torch.log_softmax(z[t, b], -1)[y[t, b]] for b in range(B)) / B for t in range(T))
    got = lstm.chunk_loss(z, y, "sum_steps")
    check("default loss = sum over steps of the batch-mean cross-entropy (explicit loops)",
          torch.allclose(got, ref, rtol=1e-12, atol=1e-12), f"{got} vs {ref}")
    old = lstm.chunk_loss(z, y, "mean_tokens")
    check("old loss = mean over all tokens = default / steps",
          torch.allclose(old, nn.functional.cross_entropy(z.reshape(-1, V), y.reshape(-1)), rtol=1e-12)
          and torch.allclose(got, T * old, rtol=1e-12))
    a = lstm.parse(["--size", "medium"])
    b = lstm.parse(["--size", "large"])
    check("defaults: loss sum_steps, clip 5 medium and 10 large",
          a.loss == "sum_steps" and b.loss == "sum_steps" and a.clip == 5 and b.clip == 10, f"{a} {b}")
    c = lstm.parse(["--size", "large", "--loss", "mean_tokens", "--clip", "5"])
    check("the 2026-10-15 setting is still available (--loss mean_tokens --clip 5)",
          c.loss == "mean_tokens" and c.clip == 5)

    # gradients and one SGD step on a tiny LM
    V, H, steps, batch = 23, 8, 35, 4
    data = torch.randint(0, V, (steps + 1, batch))
    grads, moves = {}, {}
    for mode in lstm.LOSSES:
        torch.manual_seed(1)
        m = lstm.LM(V, H, 0.0, 0.1).double()
        z, _ = m(data[:-1], None)
        lstm.chunk_loss(z, data[1:], mode).backward()
        grads[mode] = grads_vec(m)
        torch.manual_seed(1)
        m = lstm.LM(V, H, 0.0, 0.1).double()
        p0 = params_vec(m)
        opt = torch.optim.SGD(m.parameters(), lr=1.0)
        lstm.train_epoch(m, opt, data, mode, clip=1e9, bptt=steps)
        moves[mode] = params_vec(m) - p0
    check("gradient of the default loss = 35 x gradient of the old loss",
          torch.allclose(grads["sum_steps"], 35 * grads["mean_tokens"], rtol=1e-9, atol=1e-14))
    check("one SGD step (lr 1, no clipping) moves the parameters 35 times further",
          torch.allclose(moves["sum_steps"], 35 * moves["mean_tokens"], rtol=1e-6, atol=1e-12),
          f"{float(moves['sum_steps'].norm()) / float(moves['mean_tokens'].norm()):.4f}")
    clip = 1e-3
    norms = {}
    for mode in lstm.LOSSES:
        torch.manual_seed(1)
        m = lstm.LM(V, H, 0.0, 0.1).double()
        p0 = params_vec(m)
        opt = torch.optim.SGD(m.parameters(), lr=1.0)
        _, gnorm, share = lstm.train_epoch(m, opt, data, mode, clip=clip, bptt=steps)
        norms[mode] = (float((params_vec(m) - p0).norm()), gnorm, share)
    # clip_grad_norm_ scales by clip / (norm + 1e-6), so the step is a hair under the clip
    check("with a clip that acts both steps have the clip norm",
          all(abs(v[0] - clip) < 1e-4 * clip and v[0] <= clip and v[2] == 1.0 for v in norms.values()), f"{norms}")

    # a few epochs on a short synthetic stream (a repeating pattern) lower the loss
    torch.manual_seed(2)
    pat = torch.tensor([1, 5, 2, 7, 3, 9, 4, 6] * 40)
    stream = pat.view(-1, 4)
    m = lstm.LM(10, 16, 0.0, 0.1)
    opt = torch.optim.SGD(m.parameters(), lr=1.0)
    before = lstm.evaluate(m, stream, bptt=10)
    for _ in range(30):
        lstm.train_epoch(m, opt, stream, "sum_steps", clip=5.0, bptt=10)
    after = lstm.evaluate(m, stream, bptt=10)
    check("30 epochs on a repeating stream lower the cross-entropy below half", after < 0.5 * before,
          f"{before:.3f} -> {after:.3f}")
    print(f"     synthetic stream: cross-entropy {before:.3f} -> {after:.3f} nats")
    print("PASS" if not FAILS else f"FAIL: {len(FAILS)} checks")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

"""E11 receipts (G5) and deterministic replay (G6) on the tiny LM (docs/05 E11; docs/04 M8 ACCEPT;
docs/13 H2.2, H2.4, H6.4; config experiments/configs/E11.json).

Steps, all on the CPU engine (docs/04 lane 3: "M8 decoders and receipts against the CPU engine"):
1. train the tiny-LM engine (config `tiny_lm.json` engine block, `--tokens` first tokens of the
   WikiText-103 BPE-2000 train split, `--epochs`) or load a saved model (`--model`);
2. calibrate `logprobs` on the FROZEN validation eval set: two scalars fitted on the first half of
   `--cal-targets` targets, ECE and split-conformal sets measured on the second half (Gate 2, ECE);
3. pre-registered prompts: the first `--prompts` TEST documents (access counted, docs/06), prompt =
   their first `--prompt-len` tokens, `--n-new` generated tokens, sampling seeds = the config's;
   three decoders (any-order AR with confidence order, masked diffusion with remasking, elimination),
   each on the C engine; any-order and diffusion also on the incremental evaluator, whose tokens
   must be identical (T2.3) and whose ms per token is reported next to the engine's (H6.4);
4. receipts for every generated token; the bank exported to `bank.npz`; `checker_standalone.py`
   run as a SEPARATE PROCESS on every receipt file (H2.4); tampering in every mode on up to
   `--tamper-per-mode` tokens, all of which must fail;
5. replay: the any-order generation of seed 1 repeated in two fresh subprocesses with
   OMP_NUM_THREADS 1 and 2 from the saved model; receipts must be byte-identical (H2.2);
6. the diffusion ELBO (upper bound) and the exact any-order NLL on the first `--elbo-docs`
   validation documents, labelled as such (the causal perplexity of protocol CP is a different run).

    scripts/cpu_slot.sh 6 -- scripts/run_logged.py --name E11 --config experiments/configs/E11.json \\
        -- .venv-tm/bin/python experiments/e11_receipts_replay.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from engine_common import ROOT, EngineConfig, load_json, make_engine, measured, write_metrics_or_print
from m3_tiny_lm import slice_train
from tmlm.certify.receipt import TAMPER_MODES, build_receipts, receipt_text, tamper, write_receipts
from tmlm.data.corpus_store import load_split
from tmlm.data.evalsets import load_evalset
from tmlm.decode.anyorder import anyorder_generate, sequential_nll
from tmlm.decode.conftm import Calibration, conformal_qhat, conformal_report, ece, fit_two_scalar, mean_nll
from tmlm.decode.diffusion import diffusion_generate, elbo_nll
from tmlm.decode.eliminate import elimination_generate
from tmlm.decode.incremental import IncrementalEvaluator
from tmlm.decode.scorer import TrainedModel

CORPUS = "wikitext103"
CHECKER = ROOT / "src" / "tmlm" / "certify" / "checker_standalone.py"


def derived(value, **extra) -> dict:
    d = {"value": value, "label": "derived"}
    d.update(extra)
    return d


def out_dir() -> Path:
    d = os.environ.get("TMLM_RUN_DIR")
    if d:
        return Path(d)
    d = Path(os.environ.get("TMLM_SCRATCH", ROOT / "results" / "_scratch_e11"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_text(vocab: int):
    try:
        from tmlm.data.bpe import load_tokenizer
        tok = load_tokenizer(vocab)
        return lambda ids: tok.decode([int(t) for t in ids])
    except Exception:  # the tokenizer is a rendering convenience, not a dependency of the receipts
        return lambda ids: " ".join(str(int(t)) for t in ids)


def generate(model: TrainedModel, decoder: str, prompt: np.ndarray, n_new: int, seed: int, incremental: bool):
    L = int(prompt.shape[0]) + n_new
    scorer = IncrementalEvaluator(model, L) if incremental else model.bind(L)
    if decoder == "anyorder":
        return anyorder_generate(scorer, prompt, n_new, seed=seed, order="confidence")
    if decoder == "diffusion":
        return diffusion_generate(scorer, prompt, n_new, seed=seed, steps=max(1, n_new // 2), remask_frac=0.1)
    if decoder == "eliminate":
        return elimination_generate(scorer, prompt, n_new, seed=seed)
    raise ValueError(decoder)


def replay_worker(model_path: str, prompts_path: str, out_path: str, n_new: int, seed: int) -> int:
    """Subprocess entry: regenerate (any-order, confidence order) from the saved model."""
    model = TrainedModel.load(model_path)
    prompts = [np.asarray(p, dtype=np.uint16) for p in json.loads(Path(prompts_path).read_text())]
    docs = [build_receipts(generate(model, "anyorder", p, n_new, seed, False), model) for p in prompts]
    write_receipts(out_path, {"runs": docs})
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "experiments" / "configs" / "E11.json"))
    ap.add_argument("--engine-config", default=str(ROOT / "experiments" / "configs" / "tiny_lm.json"))
    ap.add_argument("--model", default=None, help="saved TrainedModel .npz (skips training)")
    ap.add_argument("--tokens", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--clauses", type=int, default=None)
    ap.add_argument("--prompts", type=int, default=20)
    ap.add_argument("--prompt-len", type=int, default=16)
    ap.add_argument("--n-new", type=int, default=32)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--cal-targets", type=int, default=8000)
    ap.add_argument("--elbo-docs", type=int, default=20)
    ap.add_argument("--elbo-samples", type=int, default=8)
    ap.add_argument("--tamper-per-mode", type=int, default=100)
    ap.add_argument("--dev", action="store_true", help="small everything, for development")
    ap.add_argument("--replay-worker", nargs=5, metavar=("MODEL", "PROMPTS", "OUT", "N_NEW", "SEED"), default=None)
    args = ap.parse_args()
    if args.replay_worker:
        m, p, o, n, s = args.replay_worker
        return replay_worker(m, p, o, int(n), int(s))

    cfg_json, eng_json = load_json(args.config), load_json(args.engine_config)
    eng_over = dict(eng_json.get("engine", {}))
    vocab = int(eng_over.get("vocab", 2000))
    tokens = args.tokens or (300_000 if args.dev else int(eng_json["params"].get("tokens", 5_000_000)))
    epochs = args.epochs or (1 if args.dev else int(eng_json["params"].get("epochs", 4)))
    if args.clauses or args.dev:
        eng_over["n_clauses"] = args.clauses or 4096
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else ([1] if args.dev else cfg_json["seeds"])
    n_prompts = 3 if args.dev and args.prompts == 20 else args.prompts
    n_new = 12 if args.dev and args.n_new == 32 else args.n_new
    cal_targets = 1000 if args.dev and args.cal_targets == 8000 else args.cal_targets
    elbo_docs = 3 if args.dev and args.elbo_docs == 20 else args.elbo_docs
    tamper_per_mode = 20 if args.dev and args.tamper_per_mode == 100 else args.tamper_per_mode
    odir = out_dir()
    metrics: dict = {}
    render = token_text(vocab)
    t_all = time.time()

    # ---------------------------------------------------------------- 1. model
    train_full = load_split(CORPUS, vocab, "train")
    train = slice_train(train_full, tokens)
    val = load_split(CORPUS, vocab, "validation")
    cfg = EngineConfig(**{**eng_over, "vocab": vocab, "seed": int(eng_over.get("seed", 1))})
    if args.model:
        model = TrainedModel.load(args.model)
        cfg = model.cfg
        metrics["model_source"] = measured(args.model)
    else:
        t0 = time.time()
        eng = make_engine(cfg, train)
        n_ex = 0
        for e in range(epochs):
            for b in range(eng.n_batches()):
                n_ex += eng.train_batch(e, b)["n_examples"]
        wall = time.time() - t0
        model = TrainedModel.from_engine(eng, train.tok)
        metrics["train_tokens"] = measured(int(train.tok.shape[0]))
        metrics["train_epochs"] = measured(epochs)
        metrics["train_examples_per_s"] = measured(n_ex / max(1e-9, wall), unit="examples/s")
        metrics["train_wall_s"] = measured(wall)
        metrics["alive_clauses"] = measured(int(model.state["alive"].sum()))
        del eng
    metrics["engine_config"] = measured(cfg.to_dict())
    metrics["state_hash"] = measured(model.state_hash)
    print(f"model: C={cfg.n_clauses} state_hash {model.state_hash}", flush=True)

    # ---------------------------------------------------------------- 2. calibration on validation
    ev = load_evalset(CORPUS, vocab, "validation", "subword")
    targets = ev["targets"].astype(np.uint32)[:cal_targets]
    hidden = ev["hidden"].astype(np.uint8)
    gold = val.tok[targets].astype(np.int64)
    probe = make_engine(cfg, val)
    probe.load_state(model.state)
    t0 = time.time()
    scores = np.concatenate([probe.predict_scores(0, targets[i:i + 2048], hidden) for i in range(0, targets.shape[0], 2048)])
    metrics["calibration_scoring_wall_s"] = measured(time.time() - t0)
    half = targets.shape[0] // 2
    cal, rep = fit_two_scalar(scores[:half], gold[:half], model.class_bias)
    model = model.with_calibration(cal)
    lp_held = cal.logprobs(scores[half:], model.class_bias)
    e_rep = ece(np.exp(lp_held), gold[half:])
    e_id = ece(np.exp(Calibration.identity().logprobs(scores[half:], model.class_bias)), gold[half:])
    qhat = conformal_qhat(np.exp(cal.logprobs(scores[:half], model.class_bias)), gold[:half], alpha=0.1)
    conf = conformal_report(np.exp(lp_held), gold[half:], qhat)
    metrics.update({
        "calibration_a": measured(cal.a, note="a_q16 / 2^16, fitted on the first half of the validation targets"),
        "calibration_b": measured(cal.b), "calibration_a_q16": measured(cal.a_q16), "calibration_b_q16": measured(cal.b_q16),
        "calibration_fit_targets": measured(half), "calibration_heldout_targets": measured(int(targets.shape[0] - half)),
        "calibration_nll_fit_nats": measured(rep["nll_quantised"]), "calibration_nll_identity_fit_nats": measured(rep["nll_identity"]),
        "calibration_nll_unigram_only_fit_nats": measured(rep["nll_unigram_only"]),
        "calibration_nll_heldout_nats": measured(mean_nll(scores[half:], gold[half:], model.class_bias, cal)),
        "calibration_top1_heldout": measured(e_rep["accuracy"]),
        "ece_heldout": measured(e_rep["ece"], note="15 equal-width bins, top-1 confidence, calibrated"),
        "ece_heldout_identity": measured(e_id["ece"], note="a = 1, b = 0 (uncalibrated raw sums as logits)"),
        "gate2_ece_below_0.05": measured(bool(e_rep["ece"] < 0.05), note="docs/04 M8 ACCEPT, on the held-out validation half"),
        "conformal_alpha": measured(0.1), "conformal_qhat": measured(conf["qhat"]),
        "conformal_coverage_heldout": measured(conf["coverage"]), "conformal_mean_set_size_heldout": measured(conf["mean_set_size"]),
        "evalset_sha256": measured(ev["meta"].get("sha256")),
    })
    print(f"calibration a={cal.a:.4f} b={cal.b:.4f} nll {rep['nll_quantised']:.4f} (identity {rep['nll_identity']:.4f}) "
          f"held-out ECE {e_rep['ece']:.4f} top-1 {e_rep['accuracy']:.4f} conformal coverage {conf['coverage']:.3f} size {conf['mean_set_size']:.1f}", flush=True)
    del probe
    model_path = odir / "model.npz"
    model.save(model_path)
    bank = model.bank()
    bank_path = odir / "bank.npz"
    bank_hash = bank.save(bank_path)
    metrics["bank_sha256"] = measured(bank_hash)
    metrics["bank_alive_with_anchor"] = measured(int(((bank.alive != 0) & (bank.anchor != 0xFFFFFFFF)).sum()))

    # ---------------------------------------------------------------- 3. pre-registered prompts and generation
    test = load_split(CORPUS, vocab, "test", purpose=f"E11 pre-registered prompts: first {n_prompts} test documents, {args.prompt_len} tokens each")
    prompts, prompt_docs = [], []
    for d in range(test.doc_start.shape[0] - 1):
        lo, hi = int(test.doc_start[d]), int(test.doc_start[d + 1])
        if hi - lo >= args.prompt_len:
            prompts.append(np.ascontiguousarray(test.tok[lo:lo + args.prompt_len]))
            prompt_docs.append(d)
        if len(prompts) >= n_prompts:
            break
    (odir / "prompts.json").write_text(json.dumps([[int(t) for t in p] for p in prompts]))
    metrics["prompts"] = measured(len(prompts), note=f"first test documents with at least {args.prompt_len} tokens: docs {prompt_docs}")
    metrics["prompt_len"] = measured(args.prompt_len)
    metrics["n_new"] = measured(n_new)
    metrics["sampling_seeds"] = measured(seeds)
    decoders = ("anyorder", "diffusion", "eliminate")
    receipt_files: list[Path] = []
    samples_log = []
    per_dec: dict[str, dict] = {d: {"tokens": 0, "engine_s": 0.0, "engine_calls": 0, "inc_s": 0.0, "inc_tokens": 0, "identical": 0, "compared": 0,
                                    "fired_total": 0, "conflicts": 0, "licensed_steps": 0, "eliminated_steps": 0} for d in decoders}
    for seed in seeds:
        for dec in decoders:
            docs = []
            for pi, prompt in enumerate(prompts):
                g = generate(model, dec, prompt, n_new, seed, incremental=False)
                st = per_dec[dec]
                st["tokens"] += int(g.generated.shape[0])
                st["engine_s"] += g.seconds
                st["engine_calls"] += g.engine_calls
                st["fired_total"] += sum(int(r.fired.shape[0]) for r in g.records)
                if dec == "eliminate":
                    st["conflicts"] += int(g.info.get("conflicts", 0))
                    st["licensed_steps"] += sum(1 for r in g.records if r.licensing is not None and r.licensing.shape[0] > 0)
                    st["eliminated_steps"] += sum(1 for r in g.records if r.eliminating is not None and r.eliminating.shape[0] > 0)
                if dec != "eliminate":
                    gi = generate(model, dec, prompt, n_new, seed, incremental=True)
                    st["inc_s"] += gi.seconds
                    st["inc_tokens"] += int(gi.generated.shape[0])
                    st["compared"] += 1
                    st["identical"] += int(np.array_equal(g.tokens, gi.tokens))
                doc = build_receipts(g, model, bank)
                docs.append(doc)
                text = render(g.generated)
                samples_log.append({"decoder": dec, "seed": seed, "prompt_index": pi, "test_doc": prompt_docs[pi],
                                    "prompt_text": render(prompt), "generated_ids": [int(t) for t in g.generated], "generated_text": text})
                print(f"[{dec} seed {seed} prompt {pi}] {render(prompt)!r} -> {text!r}", flush=True)
            path = odir / f"receipts_{dec}_seed{seed}.json"
            write_receipts(path, {"runs": docs})
            receipt_files.append(path)
    (odir / "samples.json").write_text(json.dumps(samples_log, indent=1, ensure_ascii=False))
    for dec, st in per_dec.items():
        metrics[f"{dec}.tokens_generated"] = measured(st["tokens"])
        metrics[f"{dec}.engine_ms_per_token"] = measured(1000.0 * st["engine_s"] / max(1, st["tokens"]), note="C engine full re-evaluation per step (the H6.4 reference)")
        metrics[f"{dec}.engine_calls_per_token"] = measured(st["engine_calls"] / max(1, st["tokens"]))
        metrics[f"{dec}.fired_clauses_per_token"] = measured(st["fired_total"] / max(1, st["tokens"]))
        if st["compared"]:
            metrics[f"{dec}.incremental_ms_per_token"] = measured(1000.0 * st["inc_s"] / max(1, st["inc_tokens"]), note="Python incremental evaluator (T2.3)")
            metrics[f"{dec}.incremental_identical_generations"] = measured(f"{st['identical']}/{st['compared']}")
            metrics[f"{dec}.H6_4_incremental_faster"] = measured(bool(st["inc_s"] / max(1, st["inc_tokens"]) < st["engine_s"] / max(1, st["tokens"])))
        if dec == "eliminate":
            metrics["eliminate.conflicts"] = measured(st["conflicts"])
            metrics["eliminate.steps_with_licensing_clauses"] = measured(st["licensed_steps"])
            metrics["eliminate.steps_with_eliminating_clauses"] = measured(st["eliminated_steps"])
    (odir / "receipt_example.txt").write_text("\n\n".join(receipt_text(t, lambda i: render([i])) for t in json.loads(receipt_files[0].read_text())["runs"][0]["tokens"][:5]))

    # ---------------------------------------------------------------- 4. standalone checker (separate process) and tampering
    n_tok = n_pass = 0
    for path in receipt_files:
        doc_all = json.loads(path.read_text())
        for k, doc in enumerate(doc_all["runs"]):
            single = odir / "_single.json"
            write_receipts(single, doc)
            r = subprocess.run([sys.executable, str(CHECKER), "--bank", str(bank_path), "--receipts", str(single), "--json", str(odir / "_report.json"), "--quiet"],
                               capture_output=True, text=True)
            rep_j = json.loads((odir / "_report.json").read_text())
            n_tok += rep_j["n_tokens"]
            n_pass += rep_j["n_pass"]
            if r.returncode != 0:
                print(f"CHECKER FAILED on {path.name} run {k}: {[x['errors'] for x in rep_j['results'] if not x['pass']][:3]}", flush=True)
    (odir / "_single.json").unlink(missing_ok=True)
    (odir / "_report.json").unlink(missing_ok=True)
    metrics["receipts_checked"] = measured(n_tok)
    metrics["receipts_pass"] = measured(n_pass)
    metrics["receipts_pass_fraction"] = measured(n_pass / max(1, n_tok), note="checker_standalone.py run as a separate process")
    metrics["H2_4_all_receipts_pass"] = measured(bool(n_tok > 0 and n_pass == n_tok))
    spec = importlib.util.spec_from_file_location("checker_standalone", CHECKER)
    chk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chk)
    lb = chk.load_bank(str(bank_path))
    all_docs = [(path.name, doc) for path in receipt_files for doc in json.loads(path.read_text())["runs"]]
    tamper_tot = tamper_fail = 0
    t0 = time.time()
    for mode in TAMPER_MODES:
        cands = [(name, doc, i) for name, doc in all_docs for i in range(len(doc["tokens"]))]
        stride = max(1, len(cands) // max(1, tamper_per_mode))
        n_try = n_fail = 0
        for name, doc, i in cands[::stride]:
            t = tamper(doc, i, mode, bank, seed=1)
            if t is None:
                continue
            n_try += 1
            res = chk.check_receipts(lb, t)
            n_fail += int(not res[i]["pass"])
            if n_try >= tamper_per_mode:
                break
        metrics[f"tamper.{mode}"] = measured(f"{n_fail}/{n_try} failed")
        tamper_tot += n_try
        tamper_fail += n_fail
    metrics["tamper_wall_s"] = measured(time.time() - t0)
    metrics["tampered_total"] = measured(tamper_tot)
    metrics["tampered_failed"] = measured(tamper_fail)
    metrics["H2_4_all_tampered_fail"] = measured(bool(tamper_tot > 0 and tamper_fail == tamper_tot))
    print(f"receipts {n_pass}/{n_tok} pass; tampered {tamper_fail}/{tamper_tot} fail", flush=True)

    # ---------------------------------------------------------------- 5. replay in fresh subprocesses (threads 1 and 2)
    replay_prompts = odir / "replay_prompts.json"
    replay_prompts.write_text(json.dumps([[int(t) for t in p] for p in prompts[:3]]))
    ref = odir / "replay_ref.json"
    docs_ref = [build_receipts(generate(model, "anyorder", p, n_new, seeds[0], False), model) for p in prompts[:3]]
    write_receipts(ref, {"runs": docs_ref})
    hashes = {"in_process": sha256_file(ref)}
    for n_threads in (1, 2):
        out = odir / f"replay_threads{n_threads}.json"
        env = dict(os.environ, OMP_NUM_THREADS=str(n_threads))
        r = subprocess.run([sys.executable, __file__, "--replay-worker", str(model_path), str(replay_prompts), str(out), str(n_new), str(seeds[0])],
                           env=env, capture_output=True, text=True, cwd=str(ROOT / "experiments"))
        if r.returncode != 0:
            print("replay worker failed:", r.stderr[-2000:], flush=True)
            hashes[f"threads{n_threads}"] = "FAILED"
        else:
            hashes[f"threads{n_threads}"] = sha256_file(out)
    identical = len(set(hashes.values())) == 1
    metrics["replay_receipt_sha256"] = measured(hashes)
    metrics["H2_2_replay_byte_identical"] = measured(bool(identical), note="any-order generation, seed %d, first 3 prompts, in-process vs OMP_NUM_THREADS 1 and 2" % seeds[0])
    print(f"replay byte-identical: {identical} {hashes}", flush=True)

    # ---------------------------------------------------------------- 6. ELBO and any-order NLL on validation documents
    L = args.prompt_len + n_new
    el_terms, seq_terms, n_docs = [], [], 0
    for d in range(val.doc_start.shape[0] - 1):
        lo, hi = int(val.doc_start[d]), int(val.doc_start[d + 1])
        if hi - lo < L:
            continue
        window = np.ascontiguousarray(val.tok[lo:lo + L])
        el_terms.append(elbo_nll(model.bind(L), window, args.elbo_samples, seed=1))
        seq_terms.append(sequential_nll(model.bind(L), window))
        n_docs += 1
        if n_docs >= elbo_docs:
            break
    if n_docs:
        nelbo = float(np.mean([t["nats_per_token"] for t in el_terms]))
        seq = float(np.mean([t["nats_per_token"] for t in seq_terms]))
        metrics["elbo_docs"] = measured(n_docs, note=f"first validation documents with at least {L} tokens, window = first {L} tokens")
        metrics["elbo_samples_per_doc"] = measured(args.elbo_samples)
        metrics["elbo_nats_per_token"] = measured(nelbo, note="negative ELBO, an UPPER bound on the NLL (MDLM linear schedule), masked-trained model")
        metrics["elbo_perplexity_per_token"] = measured(float(np.exp(nelbo)), note="exp of the per-token negative ELBO; not the CP causal perplexity")
        metrics["anyorder_l2r_nats_per_token"] = measured(seq, note="exact NLL of the left-to-right any-order factorisation of the masked model")
        metrics["anyorder_l2r_perplexity_per_token"] = measured(float(np.exp(seq)))
        metrics["elbo_over_anyorder_nll_ratio"] = derived(nelbo / seq, note="the docs/04 M8 kill criterion compares ELBO perplexity with the CAUSAL TM perplexity (protocol CP), which needs the causal run; this ratio is against the any-order NLL of the same masked model and is not that criterion")
        print(f"ELBO {nelbo:.4f} nats/token (ppl {np.exp(nelbo):.2f}); any-order l2r NLL {seq:.4f} (ppl {np.exp(seq):.2f}) on {n_docs} docs", flush=True)
    metrics["total_wall_s"] = measured(time.time() - t_all)
    metrics["omp_num_threads"] = measured(os.environ.get("OMP_NUM_THREADS"))
    write_metrics_or_print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

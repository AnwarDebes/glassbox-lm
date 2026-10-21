// The browser model against the study's code: tests/golden.json is written by export.py from
// tmlm.cvtm (KNModel, SuffixClauses, doc_clauses, mixture) on the same tokens.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import zlib from 'node:zlib';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { KN, Core, DocState, bucketOf, coreDist, topK } from '../model.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const live = path.join(here, '..');
const corpus = JSON.parse(fs.readFileSync(path.join(live, 'corpus.json'), 'utf8'));
const core = JSON.parse(fs.readFileSync(path.join(live, 'core.json'), 'utf8'));
const golden = JSON.parse(fs.readFileSync(path.join(here, 'golden.json'), 'utf8'));
const buf = zlib.gunzipSync(fs.readFileSync(path.join(live, 'tokens.u16.gz')));
const tokens = new Uint16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2);
const V = corpus.vocab.length;

function split(name) {
  const docs = corpus.docs.filter((d) => d.split === name);
  const n = docs.reduce((a, d) => a + d.length, 0);
  const tok = new Uint16Array(n), ds = new Int32Array(docs.length + 1);
  let o = 0;
  docs.forEach((d, i) => { tok.set(tokens.subarray(d.start, d.start + d.length), o); o += d.length; ds[i + 1] = o; });
  return { tok, ds };
}

const train = split('train');
const held = split('heldout');
const t0 = performance.now();
const kn = KN.fit(train.tok, train.ds, V);
const fitMs = performance.now() - t0;
const model = new Core(kn, core.weights);

test('the corpus file is the one export.py wrote', async () => {
  const { createHash } = await import('node:crypto');
  assert.equal(tokens.length, corpus.n_tokens);
  assert.equal(createHash('sha256').update(buf).digest('hex'), corpus.tokens_sha256);
});

test('counting gives the type counts and discounts of KNModel', () => {
  assert.deepEqual(kn.nTypes.slice(1), golden.n_types.slice(1));
  for (let k = 1; k <= 5; k++) {
    assert.deepEqual(kn.raw[k].d, golden.discounts[k].raw, `raw discounts of level ${k}`);
    if (k < 5) assert.deepEqual(kn.cont[k].d, golden.discounts[k].cont, `continuation discounts of level ${k}`);
  }
  console.log(`fit: ${train.tok.length} tokens in ${fitMs.toFixed(0)} ms`);
});

// score the held-out chapters once, keeping every position
const rows = [];
let o = 0;
for (let d = 0; d + 1 < held.ds.length; d++) {
  model.scoreDoc(held.tok.subarray(held.ds[d], held.ds[d + 1]), (r) => { rows.push({ ...r, gpos: o + r.p }); });
  o += held.ds[d + 1] - held.ds[d];
}

test('suffix clauses, document clauses, buckets and the core match the study code position by position', () => {
  assert.equal(rows.length, golden.n_positions);
  let maxLn = 0, maxCore = 0;
  golden.positions.forEach((p, i) => {
    const r = rows[p];
    maxLn = Math.max(maxLn, Math.abs(r.lnpKn - golden.ln_p_kn[i]));
    maxCore = Math.max(maxCore, Math.abs(r.pCore - golden.p_core[i]) / golden.p_core[i]);
    assert.equal(r.kStar, golden.k_star[i], `k* at ${p}`);
    assert.deepEqual(r.f.n, golden.n_m[i], `n_m at ${p}`);
    assert.equal(r.f.ptrLen, golden.ptr_len[i], `pointer length at ${p}`);
    assert.equal(r.f.ptrCand, golden.ptr_cand[i], `pointer candidate at ${p}`);
    assert.equal(r.bucket, golden.bucket[i], `bucket at ${p}`);
  });
  console.log(`max |d ln P_KN| ${maxLn.toExponential(2)}, max relative |d P_core| ${maxCore.toExponential(2)}`);
  assert.ok(maxLn < 1e-12);
  assert.ok(maxCore < 1e-12);
});

test('the vote form equals the recursion on every held-out position (H8.2 on this corpus)', () => {
  let m = 0;
  for (const r of rows) m = Math.max(m, Math.abs(r.lnpKn - Math.log(r.pRec)));
  console.log(`max |vote sum - ln recursion| = ${m.toExponential(2)} over ${rows.length} positions`);
  assert.ok(m < 1e-12);
});

test('held-out bits per token of KN orders 1 to 5 and of the core', () => {
  let core5 = 0;
  for (const r of rows) core5 -= Math.log2(r.pCore);
  assert.ok(Math.abs(core5 - golden.core_bits) / golden.core_bits < 1e-9, `core ${core5} vs ${golden.core_bits}`);
  const bits = [0, 0, 0, 0, 0, 0];
  for (const r of rows) for (let K = 1; K <= 5; K++) bits[K] -= Math.log2(kn.prob(r.ctx, r.gold, K, r.cids));
  for (let K = 1; K <= 5; K++) {
    const g = golden.order_bits[K];
    assert.ok(Math.abs(bits[K] - g) / g < 1e-9, `order ${K}: ${bits[K]} vs ${g}`);
  }
});

test('full distributions: the top 8 of the core match dense_pass', () => {
  // rebuild the document state at each requested position
  const docOf = (gp) => { let d = 0; while (held.ds[d + 1] <= gp) d++; return d; };
  golden.top_positions.forEach((gp, i) => {
    const d = docOf(gp);
    const doc = new DocState();
    for (let q = held.ds[d]; q < gp; q++) doc.push(held.tok[q]);
    const st = model.next(doc, 8);
    const ids = st.top;
    ids.forEach((id, j) => {
      const want = golden.top_p[i][j];
      assert.ok(Math.abs(st.mix.p[id] - want) / want < 1e-12, `p at ${gp} rank ${j}`);
    });
    assert.deepEqual(ids, golden.top_ids[i], `ids at ${gp}`);
  });
});

test('every core distribution sums to one', () => {
  const doc = new DocState();
  for (let q = 0; q < 200; q++) doc.push(held.tok[q]);
  const st = model.next(doc, 5);
  let s = 0;
  for (const x of st.mix.p) s += x;
  assert.ok(Math.abs(s - 1) < 1e-12);
  assert.equal(bucketOf(st.kStar, st.f.n, st.f.ptrLen), st.bucket);
  assert.ok(topK(coreDist(st.base, st.f, st.w).p, 3).length === 3);
});

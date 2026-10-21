// The CVTM-core of the tm_language_model study (docs/11 section 11), counted in the browser.
//
// KN: interpolated modified Kneser-Ney of orders 1..5 with the conventions of
// studies/tm_language_model/src/tmlm/eval/baselines/ngram.py (KNModel): every document padded with
// 4 BOS tokens (BOS = V, never a target), raw counts at the top level, continuation counts below,
// three discounts per level from its counts of counts, the unigram level interpolated with the
// uniform distribution over the V tokens. Types are numbered like numpy.unique over the keys
// (prefix id * 65536 + token), found by a stable two-pass counting sort, so the arithmetic of a
// probability runs in the same order as in the study's code.
//
// Suffix clauses (section 11.2): ln P_KN(w | h) = b(w) + sum over the fired contexts u of
// [d(u) + v(u, w)], with b = ln p_1, d = ln gamma and v = ln(1 + alpha / (gamma p_k(w | u'))).
// Document clauses D0..D4, the pointer P and the bucketed mixture follow tmlm/cvtm/doc_clauses.py
// and tmlm/cvtm/mixture.py; the weights of the mixture come from core.json (fitted by export.py).

export const ORDER = 5;
export const PTR_MIN = 5;
export const PTR_MAX = 64;
const S = 65536;
const M = ORDER - 1;

// ------------------------------------------------------------------ discounts
function modifiedDiscounts(cnt) {
  let n1 = 0, n2 = 0, n3 = 0, n4 = 0;
  for (let i = 0; i < cnt.length; i++) {
    const c = cnt[i];
    if (c === 1) n1++; else if (c === 2) n2++; else if (c === 3) n3++; else if (c === 4) n4++;
  }
  const y = (n1 + 2.0 * n2) > 0 ? n1 / (n1 + 2.0 * n2) : 0.0;
  const ratio = (a, b) => (b > 0 ? a / b : 0.0);
  const d1 = 1.0 - 2.0 * y * ratio(n2, n1);
  const d2 = 2.0 - 3.0 * y * ratio(n3, n2);
  const d3 = 3.0 - 4.0 * y * ratio(n4, n3);
  return [Math.min(Math.max(d1, 0.0), 1.0), Math.min(Math.max(d2, 0.0), 2.0), Math.min(Math.max(d3, 0.0), 3.0)];
}

// statistics of one level for one count variant: per context total, N1, N2, N3+
function levelStats(k, cnt, keys, nCtx) {
  const total = new Float64Array(nCtx);
  const n1 = new Int32Array(nCtx), n2 = new Int32Array(nCtx), n3 = new Int32Array(nCtx);
  for (let t = 0; t < cnt.length; t++) {
    const c = cnt[t];
    if (c === 0) continue;
    const ctx = k === 1 ? 0 : Math.floor(keys[t] / S);
    total[ctx] += c;
    if (c === 1) n1[ctx]++; else if (c === 2) n2[ctx]++; else n3[ctx]++;
  }
  return { cnt, total, n1, n2, n3, d: modifiedDiscounts(cnt) };
}

const discountOf = (d, c) => (c >= 3 ? d[2] : c === 2 ? d[1] : c === 1 ? d[0] : 0.0);
const gammaMass = (lv, c) => lv.d[0] * lv.n1[c] + lv.d[1] * lv.n2[c] + lv.d[2] * lv.n3[c];

function lowerBound(a, x) {
  let lo = 0, hi = a.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (a[mid] < x) lo = mid + 1; else hi = mid; }
  return lo;
}
function lookup(a, x) {
  const i = lowerBound(a, x);
  return i < a.length && a[i] === x ? i : -1;
}

// ------------------------------------------------------------------ the n-gram counts
export class KN {
  // tokens: Uint16Array of the training documents, docStart: Int32Array (n_docs + 1), V: vocabulary
  static fit(tokens, docStart, V) {
    const g = KN.fitSteps(tokens, docStart, V);
    let r = g.next();
    while (!r.done) r = g.next();
    return r.value;
  }

  // the same fit, awaiting `step(info)` after every level (lets a page draw between levels)
  static async fitAsync(tokens, docStart, V, step) {
    const g = KN.fitSteps(tokens, docStart, V);
    let r = g.next();
    while (!r.done) { await step(r.value); r = g.next(); }
    return r.value;
  }

  static *fitSteps(tokens, docStart, V) {
    const n = ORDER;
    const nDoc = docStart.length - 1;
    const nTok = docStart[nDoc];
    const L = (n - 1) + nTok + (n - 1) * nDoc;
    const stream = new Uint16Array(L).fill(V);
    const real = new Uint8Array(L);
    for (let d = 0; d < nDoc; d++) {
      for (let i = docStart[d]; i < docStart[d + 1]; i++) {
        const pos = i + (n - 1) * (d + 2);
        stream[pos] = tokens[i];
        real[pos] = 1;
      }
    }
    const kn = new KN(V);
    kn.nTrainTokens = nTok;
    const keys = [null, null], raw = [null, null], suf = [null, null];
    keys[1] = new Float64Array(V + 1).map((_, i) => i);
    raw[1] = new Int32Array(V + 1);
    for (let i = 0; i < L; i++) if (real[i]) raw[1][stream[i]]++;
    let prev = new Int32Array(L);
    for (let i = 0; i < L; i++) prev[i] = stream[i];
    let nPrev = V + 1;
    const bucketTok = new Int32Array(V + 2);
    let byTok = new Int32Array(L), order = new Int32Array(L);
    for (let k = 2; k <= n; k++) {
      // stable counting sort of the positions by (prefix id, token): numpy.unique over the keys
      const prefixAt = (i) => (i === 0 ? 0 : prev[i - 1]);
      bucketTok.fill(0);
      for (let i = 0; i < L; i++) bucketTok[stream[i] + 1]++;
      for (let b = 1; b < bucketTok.length; b++) bucketTok[b] += bucketTok[b - 1];
      for (let i = 0; i < L; i++) byTok[bucketTok[stream[i]]++] = i;
      const bucketPre = new Int32Array(nPrev + 1);
      for (let i = 0; i < L; i++) bucketPre[prefixAt(i) + 1]++;
      for (let b = 1; b < bucketPre.length; b++) bucketPre[b] += bucketPre[b - 1];
      for (let r = 0; r < L; r++) { const i = byTok[r]; order[bucketPre[prefixAt(i)]++] = i; }
      const ids = new Int32Array(L);
      const kk = new Float64Array(L), first = new Int32Array(L);
      let nt = -1, last = -1;
      for (let r = 0; r < L; r++) {
        const i = order[r];
        const key = prefixAt(i) * S + stream[i];
        if (key !== last) { nt++; kk[nt] = key; first[nt] = i; last = key; }
        ids[i] = nt;
      }
      nt++;
      keys[k] = kk.slice(0, nt);
      const cnt = new Int32Array(nt);
      for (let i = 0; i < L; i++) if (real[i]) cnt[ids[i]]++;
      raw[k] = cnt;
      const s = new Int32Array(nt);
      for (let t = 0; t < nt; t++) s[t] = prev[first[t]];
      suf[k] = s;
      prev = ids;
      nPrev = nt;
      yield { stage: 'level', k, types: nt };
    }
    // continuation counts: cont[k][g] = number of level-(k+1) types with a positive count whose suffix is g
    const cont = new Array(n + 1).fill(null);
    for (let k = n - 1; k >= 1; k--) {
      const upper = k + 1 === n ? raw[k + 1] : cont[k + 1];
      const c = new Int32Array(keys[k].length);
      const s = suf[k + 1];
      for (let t = 0; t < upper.length; t++) if (upper[t] > 0) c[s[t]]++;
      cont[k] = c;
    }
    yield { stage: 'continuation' };
    kn.keys = keys;
    kn.nTypes = keys.map((a) => (a ? a.length : 0));
    kn.raw = [null];
    kn.cont = [null];
    for (let k = 1; k <= n; k++) {
      const nCtx = k === 1 ? 1 : keys[k - 1].length;
      kn.raw[k] = levelStats(k, raw[k], keys[k], nCtx);
      kn.cont[k] = k < n ? levelStats(k, cont[k], keys[k], nCtx) : null;
    }
    kn.p1 = [null];
    for (let K = 1; K <= n; K++) kn.p1[K] = kn.unigramFull(K);
    return kn;
  }

  constructor(V) {
    this.V = V;
    this.bos = V;
  }

  level(k, order = ORDER) {
    return k === order ? this.raw[k] : this.cont[k];
  }

  unigramFull(order) {
    const lv = this.level(1, order);
    const total = lv.total[0];
    const gamma = total > 0 ? gammaMass(lv, 0) / total : 1.0;
    const p = new Float64Array(this.V + 1);
    const add = gamma / this.V;
    for (let w = 0; w < this.V + 1; w++) {
      const c = lv.cnt[w];
      const alpha = total > 0 ? Math.max(c - discountOf(lv.d, c), 0.0) / total : 0.0;
      p[w] = alpha + add;
    }
    p[this.bos] = 0.0;
    return p;
  }

  // ids of the contexts of length 1..4 (index = length), -1 when absent; ctx: 4 ids, last nearest the target
  contextIds(ctx) {
    const out = [-1, -1, -1, -1, -1];
    for (let len = 1; len <= M; len++) {
      let idx = ctx[M - len];
      for (let j = M - len + 1; j < M && idx >= 0; j++) {
        const depth = j - (M - len) + 1;
        idx = lookup(this.keys[depth], idx * S + ctx[j]);
      }
      out[len] = idx;
    }
    return out;
  }

  typeId(level, ctxId, w) {
    return ctxId < 0 ? -1 : lookup(this.keys[level], ctxId * S + w);
  }

  // P(w | ctx) of the order-`order` model, the operations of KNModel.prob
  prob(ctx, w, order = ORDER, cids = this.contextIds(ctx)) {
    let p = this.p1[order][w];
    for (let k = 2; k <= order; k++) {
      const lv = this.level(k, order);
      const cid = cids[k - 1];
      const total = cid >= 0 ? lv.total[cid] : 0.0;
      const seen = total > 0;
      const tid = seen ? this.typeId(k, cid, w) : -1;
      const c = tid >= 0 ? lv.cnt[tid] : 0;
      const alpha = seen ? Math.max(c - discountOf(lv.d, c), 0.0) / total : 0.0;
      const gamma = seen ? gammaMass(lv, cid) / total : 1.0;
      p = alpha + gamma * p;
    }
    return p;
  }

  // the clauses behind ln P(w | ctx) under the order-5 model (section 11.2): bias, then one entry per
  // fired suffix clause, shortest context first; lnp is the vote sum, in the order of SuffixClauses.logprob
  votes(ctx, w, cids = this.contextIds(ctx)) {
    const b = Math.log(this.p1[ORDER][w]);
    let lnp = b;
    let pPrev = this.p1[ORDER][w];
    const clauses = [];
    for (let j = 2; j <= ORDER; j++) {
      const lv = this.level(j, ORDER);
      const cid = cids[j - 1];
      const total = cid >= 0 ? lv.total[cid] : 0.0;
      const fired = total > 0;
      const tid = fired ? this.typeId(j, cid, w) : -1;
      const c = tid >= 0 ? lv.cnt[tid] : 0;
      const alpha = fired ? Math.max(c - discountOf(lv.d, c), 0.0) / total : 0.0;
      const gamma = fired ? gammaMass(lv, cid) / total : 1.0;
      const pj = alpha + gamma * pPrev;
      if (fired) {
        const d = Math.log(gamma);
        const v = c > 0 ? Math.log1p(alpha / (gamma * pPrev)) : 0.0;
        lnp = lnp + d;
        if (tid >= 0) lnp = lnp + v;
        clauses.push({ k: j - 1, level: j, cid, total, count: c, followers: lv.n1[cid] + lv.n2[cid] + lv.n3[cid],
          stat: j === ORDER ? 'count' : 'continuation count', alpha, gamma, d, v, pLower: pPrev, p: pj });
      }
      pPrev = pj;
    }
    return { b, lnp, p: pPrev, clauses };
  }

  // k*: the longest context whose clause fires (0 when only the bias fires)
  kStar(cids) {
    let ks = 0;
    for (let k = 2; k <= ORDER; k++) {
      const cid = cids[k - 1];
      if (cid >= 0 && this.level(k, ORDER).total[cid] > 0) ks = k - 1;
    }
    return ks;
  }

  // full distribution P_KN(. | ctx), length V (the operations of KNModel.dist)
  dist(ctx, cids = this.contextIds(ctx)) {
    const p = this.p1[ORDER].slice(0, this.V);
    for (let k = 2; k <= ORDER; k++) {
      const lv = this.level(k, ORDER);
      const cid = cids[k - 1];
      if (cid < 0 || !(lv.total[cid] > 0)) continue;
      const total = lv.total[cid];
      const gamma = gammaMass(lv, cid) / total;
      for (let w = 0; w < p.length; w++) p[w] *= gamma;
      const lo = lowerBound(this.keys[k], cid * S), hi = lowerBound(this.keys[k], (cid + 1) * S);
      for (let t = lo; t < hi; t++) {
        const c = lv.cnt[t];
        if (c === 0) continue;
        const w = this.keys[k][t] - cid * S;
        if (w >= this.V) continue;
        p[w] += Math.max(c - discountOf(lv.d, c), 0.0) / total;
      }
    }
    return p;
  }
}

// ------------------------------------------------------------------ document clauses
const keyOf = (t, a, b) => { let s = ''; for (let i = a; i < b; i++) s += t[i] + ','; return s; };

// Streaming document state: feed tokens one by one; `features()` describes the next position.
export class DocState {
  constructor() {
    this.t = [];
    this.tables = [new Map(), new Map(), new Map(), new Map(), new Map()];
    this.ptr = new Map();
  }

  features(gold = -1) {
    const t = this.t, p = t.length;
    const n = [0, 0, 0, 0, 0], g = [0, 0, 0, 0, 0], rows = [null, null, null, null, null];
    for (let m = 0; m <= 4; m++) {
      if (p < m) continue;
      const e = this.tables[m].get(m === 0 ? '' : keyOf(t, p - m, p));
      if (!e) continue;
      n[m] = e.n;
      rows[m] = e.follow;
      if (gold >= 0) g[m] = e.follow.get(gold) || 0;
    }
    let ptrLen = 0, ptrPos = -1;
    if (p >= PTR_MIN) {
      const list = this.ptr.get(keyOf(t, p - PTR_MIN, p));
      if (list) {
        for (let r = list.length - 1; r >= 0; r--) {
          const q = list[r];
          let len = PTR_MIN;
          while (len < PTR_MAX && q - 1 - len >= 0 && t[p - 1 - len] === t[q - 1 - len]) len++;
          if (len > ptrLen) { ptrLen = len; ptrPos = q; }
          if (ptrLen === PTR_MAX) break;
        }
      }
    }
    return { n, g, rows, ptrLen, ptrPos, ptrCand: ptrPos >= 0 ? t[ptrPos] : -1 };
  }

  push(tok) {
    const t = this.t, p = t.length;
    for (let m = 0; m <= 4; m++) {
      if (p < m) continue;
      const key = m === 0 ? '' : keyOf(t, p - m, p);
      let e = this.tables[m].get(key);
      if (!e) { e = { n: 0, follow: new Map() }; this.tables[m].set(key, e); }
      e.n++;
      e.follow.set(tok, (e.follow.get(tok) || 0) + 1);
    }
    if (p >= PTR_MIN) {
      const key = keyOf(t, p - PTR_MIN, p);
      let list = this.ptr.get(key);
      if (!list) { list = []; this.ptr.set(key, list); }
      list.push(p);
    }
    t.push(tok);
  }
}

// ------------------------------------------------------------------ the mixture of section 11.4
const N_COMP = 7, N_MC = 26, N_L = 4;
export const COMPONENTS = ['S', 'D0', 'D1', 'D2', 'D3', 'D4', 'P'];
const countBucket = (n) => (n <= 0 ? -1 : n === 1 ? 0 : n <= 3 ? 1 : n <= 7 ? 2 : n <= 15 ? 3 : 4);
const ptrBucket = (L) => (L < PTR_MIN ? 0 : L <= 8 ? 1 : L <= 16 ? 2 : 3);

export function bucketOf(kStar, n, ptrLen) {
  let mStar = -1, c = 0;
  for (let m = 0; m <= 4; m++) if (n[m] >= 1) { mStar = m; c = countBucket(n[m]); }
  const mc = mStar >= 0 ? 1 + 5 * mStar + c : 0;
  return (kStar * N_MC + mc) * N_L + ptrBucket(ptrLen);
}

export function activeOf(f) {
  return [true, f.n[0] >= 1, f.n[1] >= 1, f.n[2] >= 1, f.n[3] >= 1, f.n[4] >= 1, f.ptrLen >= PTR_MIN];
}

// P(gold) of the core: sum over active i of w_i p_i / sum over active i of w_i, in the fixed order
export function goldProb(pKn, f, gold, w) {
  const act = activeOf(f);
  let num = 0.0, den = 0.0;
  for (let i = 0; i < N_COMP; i++) {
    if (!act[i]) continue;
    let pi;
    if (i === 0) pi = pKn;
    else if (i <= 5) pi = f.g[i - 1] / Math.max(f.n[i - 1], 1);
    else pi = f.ptrCand === gold ? 1.0 : 0.0;
    num = num + w[i] * pi;
    den = den + w[i];
  }
  return num / den;
}

// the full distribution of the core (mixture.full_distribution), split into the part counted from
// the books (S) and the part remembered from the document (D0..D4, P)
export function coreDist(base, f, w) {
  const V = base.length;
  const out = new Float64Array(V);
  const fromDoc = new Float64Array(V);
  for (let i = 0; i < V; i++) out[i] = w[0] * base[i];
  let den = 0.0 + w[0];
  for (let m = 0; m <= 4; m++) {
    const row = f.rows[m];
    if (!row || f.n[m] < 1) continue;
    for (const [tok, c] of row) {
      const add = w[1 + m] * (c / f.n[m]);
      out[tok] = out[tok] + add;
      fromDoc[tok] += add;
    }
    den = den + w[1 + m];
  }
  if (f.ptrLen >= PTR_MIN && f.ptrCand >= 0) {
    out[f.ptrCand] = out[f.ptrCand] + w[6] * 1.0;
    fromDoc[f.ptrCand] += w[6];
    den = den + w[6];
  }
  const p = new Float64Array(V), fromBooks = new Float64Array(V);
  for (let i = 0; i < V; i++) { p[i] = out[i] / den; fromDoc[i] /= den; fromBooks[i] = (w[0] * base[i]) / den; }
  return { p, fromBooks, fromDoc, den };
}

export function topK(p, k) {
  const idx = [];
  for (let i = 0; i < p.length; i++) {
    if (idx.length < k) { idx.push(i); idx.sort((a, b) => p[b] - p[a] || a - b); continue; }
    const last = idx[k - 1];
    if (p[i] > p[last] || (p[i] === p[last] && i < last)) { idx[k - 1] = i; idx.sort((a, b) => p[b] - p[a] || a - b); }
  }
  return idx;
}

// ------------------------------------------------------------------ the whole model
export class Core {
  constructor(kn, weights) {
    this.kn = kn;
    this.weights = weights;
  }

  static contextOf(tokens) {
    const ctx = [];
    for (let j = 0; j < M; j++) {
      const q = tokens.length - M + j;
      ctx.push(q >= 0 ? tokens[q] : -1);
    }
    return ctx;
  }

  // the readable state of the next position of a document
  next(doc, kTop = 10) {
    const kn = this.kn;
    const ctx = Core.contextOf(doc.t).map((x) => (x < 0 ? kn.bos : x));
    const cids = kn.contextIds(ctx);
    const kStar = kn.kStar(cids);
    const f = doc.features();
    const bucket = bucketOf(kStar, f.n, f.ptrLen);
    const w = this.weights[bucket];
    const base = kn.dist(ctx, cids);
    const mix = coreDist(base, f, w);
    const top = topK(mix.p, kTop);
    return { ctx, cids, kStar, f, bucket, w, base, mix, top };
  }

  // score whole documents token by token: per-position records (for the measurement and the tests)
  scoreDoc(tokens, onPos) {
    const kn = this.kn;
    const doc = new DocState();
    for (let p = 0; p < tokens.length; p++) {
      const gold = tokens[p];
      const ctx = Core.contextOf(doc.t).map((x) => (x < 0 ? kn.bos : x));
      const cids = kn.contextIds(ctx);
      const vt = kn.votes(ctx, gold, cids);
      const kStar = kn.kStar(cids);
      const f = doc.features(gold);
      const bucket = bucketOf(kStar, f.n, f.ptrLen);
      const pCore = goldProb(Math.exp(vt.lnp), f, gold, this.weights[bucket]);
      if (onPos) onPos({ p, gold, ctx, cids, lnpKn: vt.lnp, pRec: vt.p, kStar, f, bucket, pCore });
      doc.push(gold);
    }
  }
}

// ------------------------------------------------------------------ text
export const TOKEN_RE = /[a-z0-9]+(?:'[a-z0-9]+)*|[.,;:!?]/g;

export function normalise(text) {
  return text
    .replace(/[\u2019\u2018]/g, "'").replace(/[\u201c\u201d]/g, '"')
    .replace(/_/g, ' ').replace(/--/g, ' ').replace(/[\u2014\u2013]/g, ' ')
    .normalize('NFKD').replace(/[\u0300-\u036f]/g, '')
    .replace(/\u00e6/g, 'ae').replace(/\u00c6/g, 'Ae').replace(/\u0153/g, 'oe').replace(/\u0152/g, 'Oe')
    .toLowerCase();
}

export function tokenize(text) {
  return normalise(text).match(TOKEN_RE) || [];
}

// word ids with document-word slots for words the vocabulary does not hold
export class Vocab {
  constructor(words, nSlots) {
    this.words = words;
    this.nWords = words.length;
    this.nSlots = nSlots;
    this.V = words.length + nSlots;
    this.index = new Map(words.map((w, i) => [w, i]));
  }

  encode(tokens, slots = []) {
    const ids = [];
    for (const t of tokens) {
      const i = this.index.get(t);
      if (i !== undefined) { ids.push(i); continue; }
      let j = slots.indexOf(t);
      if (j < 0) { slots.push(t); j = slots.length - 1; }
      ids.push(this.nWords + Math.min(j, this.nSlots - 1));
    }
    return { ids, slots };
  }

  text(id, slots = []) {
    if (id < this.nWords) return this.words[id];
    const j = id - this.nWords;
    return slots[j] !== undefined ? slots[j] : `<w${j}>`;
  }
}

// lowercase tokens back to readable text
export function detok(words) {
  let out = '';
  let cap = true;
  for (const w of words) {
    const punct = /^[.,;:!?]$/.test(w);
    let s = w === 'i' ? 'I' : w.startsWith("i'") ? 'I' + w.slice(1) : w;
    if (cap && !punct) { s = s.charAt(0).toUpperCase() + s.slice(1); cap = false; }
    if (punct) out += s; else out += (out ? ' ' : '') + s;
    if (/^[.!?]$/.test(w)) cap = true;
  }
  return out;
}

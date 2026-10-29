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

// The code points that unicodedata.combining() of the Python that runs export.py (Python 3.11,
// Unicode 14.0.0) calls combining, as hexadecimal ranges: normalise() removes exactly these after
// NFKD, as export.py does (912 code points; tests/test_export.py checks the list against Python).
// The class \p{Mn} of Node 26 (Unicode 17) would differ from it on 1,197 code points, the former
// range U+0300 to U+036F on 802.
const COMBINING =
  '300-34E,350-36F,483-487,591-5BD,5BF,5C1-5C2,5C4-5C5,5C7,610-61A,64B-65F,670,6D6-6DC,6DF-6E4,6E7-6E8,' +
  '6EA-6ED,711,730-74A,7EB-7F3,7FD,816-819,81B-823,825-827,829-82D,859-85B,898-89F,8CA-8E1,8E3-8FF,93C,94D,' +
  '951-954,9BC,9CD,9FE,A3C,A4D,ABC,ACD,B3C,B4D,BCD,C3C,C4D,C55-C56,CBC,CCD,D3B-D3C,D4D,DCA,E38-E3A,E48-E4B,' +
  'EB8-EBA,EC8-ECB,F18-F19,F35,F37,F39,F71-F72,F74,F7A-F7D,F80,F82-F84,F86-F87,FC6,1037,1039-103A,108D,' +
  '135D-135F,1714-1715,1734,17D2,17DD,18A9,1939-193B,1A17-1A18,1A60,1A75-1A7C,1A7F,1AB0-1ABD,1ABF-1ACE,' +
  '1B34,1B44,1B6B-1B73,1BAA-1BAB,1BE6,1BF2-1BF3,1C37,1CD0-1CD2,1CD4-1CE0,1CE2-1CE8,1CED,1CF4,1CF8-1CF9,' +
  '1DC0-1DFF,20D0-20DC,20E1,20E5-20F0,2CEF-2CF1,2D7F,2DE0-2DFF,302A-302F,3099-309A,A66F,A674-A67D,' +
  'A69E-A69F,A6F0-A6F1,A806,A82C,A8C4,A8E0-A8F1,A92B-A92D,A953,A9B3,A9C0,AAB0,AAB2-AAB4,AAB7-AAB8,' +
  'AABE-AABF,AAC1,AAF6,ABED,FB1E,FE20-FE2F,101FD,102E0,10376-1037A,10A0D,10A0F,10A38-10A3A,10A3F,' +
  '10AE5-10AE6,10D24-10D27,10EAB-10EAC,10F46-10F50,10F82-10F85,11046,11070,1107F,110B9-110BA,11100-11102,' +
  '11133-11134,11173,111C0,111CA,11235-11236,112E9-112EA,1133B-1133C,1134D,11366-1136C,11370-11374,11442,' +
  '11446,1145E,114C2-114C3,115BF-115C0,1163F,116B6-116B7,1172B,11839-1183A,1193D-1193E,11943,119E0,11A34,' +
  '11A47,11A99,11C3F,11D42,11D44-11D45,11D97,16AF0-16AF4,16B30-16B36,16FF0-16FF1,1BC9E,1D165-1D169,' +
  '1D16D-1D172,1D17B-1D182,1D185-1D18B,1D1AA-1D1AD,1D242-1D244,1E000-1E006,1E008-1E018,1E01B-1E021,' +
  '1E023-1E024,1E026-1E02A,1E130-1E136,1E2AE,1E2EC-1E2EF,1E8D0-1E8D6,1E944-1E94A';
const COMBINING_RE = new RegExp(
  `[${COMBINING.split(',').map((r) => r.split('-').map((h) => `\\u{${h}}`).join('-')).join('')}]`, 'gu');

export function normalise(text) {
  return text
    .replace(/[\u2019\u2018]/g, "'").replace(/[\u201c\u201d]/g, '"')
    .replace(/_/g, ' ').replace(/--/g, ' ').replace(/[\u2014\u2013]/g, ' ')
    .normalize('NFKD').replace(COMBINING_RE, '')
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

  // the ids at or above this one are slots that hold no word of a text with these slots
  limit(slots) {
    return this.nWords + Math.min(slots.length, this.nSlots);
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

// ------------------------------------------------------------------ sampling and number formats of the page
// a small seeded generator (mulberry32), so that a seed always writes the same words
export function mulberry32(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// one id drawn from p ** (1 / T) over the ids below `limit` (Vocab.limit): the ids at or above it
// are slots that hold no word of the text and have nothing to write
export function sample(p, rng, T, limit = p.length) {
  const n = Math.min(limit, p.length);
  const q = new Float64Array(n);
  let z = 0, last = -1;
  for (let i = 0; i < n; i++) { q[i] = Math.pow(p[i], 1 / T); z += q[i]; if (q[i] > 0) last = i; }
  if (last < 0) throw new Error('nothing to sample: every allowed id has probability 0');
  let u = rng() * z;
  for (let i = 0; i < n; i++) { u -= q[i]; if (u < 0) return i; }
  return last;
}

// a probability as a percentage, as HTML (below 0.001 percent it says so instead of printing 0.000%)
export function pct(p) {
  if (!(p > 0)) return '0%';
  const x = p * 100;
  if (x < 0.001) return '&lt;0.001%';
  return (p >= 0.1 ? x.toFixed(1) : p >= 0.01 ? x.toFixed(2) : x.toFixed(3)) + '%';
}

// x as m × 10^e with one decimal of m, as HTML (m is between 1.0 and 9.9 after rounding)
export function sci(x) {
  if (x === 0) return '0';
  let e = Math.floor(Math.log10(Math.abs(x)));
  if (Math.abs(Number((x / 10 ** e).toFixed(1))) >= 10) e += 1;
  return `${(x / 10 ** e).toFixed(1)} × 10<sup>${e < 0 ? '−' : ''}${Math.abs(e)}</sup>`;
}

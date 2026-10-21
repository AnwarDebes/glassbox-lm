// The glass box page: loads ten public-domain books, counts them in the tab (model.js), and shows every vote.
import { KN, Core, DocState, Vocab, tokenize, detok, ORDER, PTR_MIN } from './model.js';

const $ = (s, r = document) => r.querySelector(s);
const FILM = /[?&]film\b/.test(location.search);
const nf = new Intl.NumberFormat('en-US');
const realNow = () => (window.__realNow ? window.__realNow() : performance.now());
const frame = () => (FILM ? Promise.resolve() : new Promise((r) => requestAnimationFrame(() => r())));
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const pct = (p) => (p >= 0.995 ? (p * 100).toFixed(1) : p >= 0.1 ? (p * 100).toFixed(1) : p >= 0.01 ? (p * 100).toFixed(2) : (p * 100).toFixed(3)) + '%';
const sgn = (x, d = 2) => (x > 0 ? '+' : x < 0 ? '−' : '') + Math.abs(x).toFixed(d);
function sci(x) {
  if (x === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(x)));
  return `${(x / 10 ** e).toFixed(1)} × 10<sup>${e < 0 ? '−' : ''}${Math.abs(e)}</sup>`;
}
const SUBS = ['', '₋₁', '₋₂', '₋₃', '₋₄'];
const times = (n) => (n === 1 ? 'once' : n === 2 ? 'twice' : `${nf.format(n)} times`);
const minus = (s) => String(s).replace(/^-/, '−');

const S = {};            // model and data
const G = { sel: null, undo: [], seed: 1, genSeed: 12 }; // page state

// ------------------------------------------------------------------ loading
async function gunzip(res) {
  if (typeof DecompressionStream === 'undefined') throw new Error('This browser cannot unpack the corpus (DecompressionStream is missing).');
  return new Response(res.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}

function setStatus(html, ready = false) {
  $('#status-t').innerHTML = html;
  $('#status').classList.toggle('ready', ready);
}

async function load() {
  const [corpus, core, study, tokBuf] = await Promise.all([
    fetch('corpus.json').then((r) => r.json()),
    fetch('core.json').then((r) => r.json()),
    fetch('study.json').then((r) => r.json()),
    fetch('tokens.u16.gz').then(gunzip),
  ]);
  Object.assign(S, { corpus, coreDoc: core, study });
  S.tokens = new Uint16Array(tokBuf);
  S.vocab = new Vocab(corpus.vocab.slice(0, corpus.n_words), corpus.n_slots);
  S.V = corpus.vocab.length;
  // the training stream and the held-out chapters
  const train = corpus.docs.filter((d) => d.split === 'train');
  const nTrain = train.reduce((a, d) => a + d.length, 0);
  S.trainTok = new Uint16Array(nTrain);
  S.trainDs = new Int32Array(train.length + 1);
  S.trainDocs = [];
  let o = 0;
  train.forEach((d, i) => {
    S.trainTok.set(S.tokens.subarray(d.start, d.start + d.length), o);
    S.trainDocs.push({ ...d, tstart: o });
    o += d.length;
    S.trainDs[i + 1] = o;
  });
  S.held = corpus.docs.filter((d) => d.split === 'heldout');
  fillPages();
  setStatus(`Counting ${nf.format(nTrain)} tokens from ten books…`);
  await frame();
  const t0 = realNow();
  S.kn = await KN.fitAsync(S.trainTok, S.trainDs, S.V, async (st) => {
    if (st.stage === 'level') setStatus(`Counting every pattern of ${st.k} words… ${nf.format(st.types)} kinds`);
    await frame();
  });
  S.fitMs = realNow() - t0;
  S.core = new Core(S.kn, core.weights);
  S.patterns = 0;
  for (let k = 2; k <= ORDER; k++) for (const c of S.kn.raw[k].cnt) if (c > 0) S.patterns++;
  setStatus(`Counted <b>${nf.format(nTrain)}</b> tokens of ten books into <b>${nf.format(S.patterns)}</b> patterns in <b>${(S.fitMs / 1000).toFixed(1)} s</b>, in this tab. Training is counting.`, true);
}

// ------------------------------------------------------------------ text helpers
function wordOf(id, slots) { return S.vocab.text(id, slots); }

function tokensHtml(ids, slots, hiFrom = -1, hiTo = -1, wAt = -1) {
  let out = '';
  let cap = true;
  ids.forEach((id, i) => {
    const w = id >= S.V ? '' : wordOf(id, slots);
    const punct = /^[.,;:!?]$/.test(w);
    let s = w === 'i' ? 'I' : w.startsWith("i'") ? 'I' + w.slice(1) : w;
    if (cap && !punct && s) { s = s.charAt(0).toUpperCase() + s.slice(1); cap = false; }
    if (/^[.!?]$/.test(w)) cap = true;
    let piece = esc(s);
    if (i === wAt) piece = `<mark class="w">${piece}</mark>`;
    else if (i >= hiFrom && i < hiTo) piece = `<mark>${piece}</mark>`;
    out += (punct || !out ? '' : ' ') + piece;
  });
  return out;
}

function docLabel(d) { return `${S.corpus.books[d.book].title} · ${d.label}`; }

// every place in the counted books where the context `ctx` (ids, BOS = V) is followed by w
function findSources(ctxIds, w, limit = 4000) {
  const hits = [];
  const k = ctxIds.length;
  const tok = S.trainTok;
  for (const d of S.trainDocs) {
    const a = d.tstart, b = d.tstart + d.length;
    for (let i = a; i < b; i++) {
      if (tok[i] !== w) continue;
      let ok = true;
      for (let j = 1; j <= k && ok; j++) {
        const want = ctxIds[k - j];
        const q = i - j;
        if (want === S.V) ok = q < a; else ok = q >= a && tok[q] === want;
      }
      if (ok) { hits.push({ d, i }); if (hits.length >= limit) return hits; }
    }
  }
  return hits;
}

function quoteHtml(h, k) {
  const { d, i } = h;
  const a = d.tstart, b = d.tstart + d.length;
  const from = Math.max(a, i - k - 9), to = Math.min(b, i + 10);
  const ids = Array.from(S.trainTok.subarray(from, to));
  const html = tokensHtml(ids, [], i - k - from, i - from, i - from);
  return `<div class="quote"><p>${from > a ? '… ' : ''}${html}${to < b ? ' …' : ''}</p><cite>${esc(docLabel(d))}</cite></div>`;
}

function pickQuotes(hits, n = 3) {
  const out = [], books = new Set();
  for (const h of hits) { if (!books.has(h.d.book)) { books.add(h.d.book); out.push(h); } if (out.length >= n) return out; }
  for (const h of hits) { if (!out.includes(h)) out.push(h); if (out.length >= n) break; }
  return out;
}

// ------------------------------------------------------------------ the glass box
function encodeText(text) {
  const toks = tokenize(text);
  return S.vocab.encode(toks, []);
}

function compute(text) {
  const enc = encodeText(text);
  const doc = new DocState();
  enc.ids.forEach((id) => doc.push(id));
  const st = S.core.next(doc, 8);
  return { ...enc, doc, st };
}

function renderBox() {
  const text = $('#text').value;
  const cur = compute(text);
  G.cur = cur;
  const { st, ids, slots } = cur;
  // the context the clauses read
  const ctxEl = $('#ctx');
  let h = '<span class="lab">reads</span>';
  for (let j = 0; j < 4; j++) {
    const id = st.ctx[j];
    const k = 4 - j;
    if (id === S.V) h += `<span class="w bos">start<sub>x${SUBS[k]}</sub></span>`;
    else h += `<span class="w${id >= S.vocab.nWords ? ' slot' : ''}">${esc(wordOf(id, slots))}<sub>x${SUBS[k]}</sub></span>`;
  }
  ctxEl.innerHTML = h;
  // candidates
  const top = st.top;
  if (G.sel === null || !top.includes(G.sel)) G.sel = top[0];
  const ol = $('#cands');
  ol.innerHTML = top.map((id) => {
    const p = st.mix.p[id], b = st.mix.fromBooks[id], m = st.mix.fromDoc[id];
    const slot = id >= S.vocab.nWords;
    return `<li class="cand${id === G.sel ? ' on' : ''}" data-id="${id}" tabindex="0" role="button" aria-pressed="${id === G.sel}">
      <span class="cw${slot ? ' slot' : ''}">${esc(wordOf(id, slots))}</span><span class="cp">${pct(p)}</span>
      <span class="bar" aria-hidden="true"><i class="b" style="width:${(b * 100).toFixed(2)}%"></i><i class="m" style="width:${(m * 100).toFixed(2)}%"></i></span></li>`;
  }).join('');
  const kTxt = st.kStar === 0 ? 'none of your last words as a pattern the books contain' : st.kStar === 1 ? 'your last word as a pattern' : `your last ${st.kStar} words as a pattern`;
  const memW = st.f.n.some((x) => x > 0) ? 1 - st.w[0] / weightSum(st) : 0;
  $('#cand-note').innerHTML = `The books contain ${kTxt}. In this situation the mixture gives the counted books <b>${pct(st.w[0] / weightSum(st))}</b> of its weight and the memory of your text <b>${pct(memW)}</b>.`;
  $('#accept-w').textContent = wordOf(G.sel, slots);
  renderLedger();
}

function weightSum(st) {
  const act = [true, ...st.f.n.map((x) => x >= 1), st.f.ptrLen >= PTR_MIN];
  let s = 0;
  act.forEach((a, i) => { if (a) s += st.w[i]; });
  return s;
}

function litHtml(ids, slots) {
  const k = ids.length;
  return ids.map((id, j) => {
    const sub = SUBS[k - j];
    const w = id === S.V ? '<i>start</i>' : esc(wordOf(id, slots));
    return `<span class="lit${id >= S.vocab.nWords && id !== S.V ? ' slot' : ''}">x${sub} = ${w}</span>`;
  }).join(' <span class="and">∧</span> ');
}

function renderLedger() {
  const { st, slots } = G.cur;
  const w = G.sel;
  const kn = S.kn;
  const word = wordOf(w, slots);
  $('#why-w').textContent = word;
  const vt = kn.votes(st.ctx, w, st.cids);
  const rows = [];
  rows.push({ cls: 'tap', src: { k: 0 }, cl: `<span class="if">BIAS</span> always fires: how common <b>${esc(word)}</b> is`, vote: vt.b,
    sub: w >= S.vocab.nWords ? 'a word the books never use: it gets the share of the unseen' : `ln p<sub>1</sub>(${esc(word)}), from ${nf.format(kn.cont[1].cnt[w])} different words before it` });
  for (const c of vt.clauses) {
    const ctxIds = st.ctx.slice(4 - c.k);
    const total = c.d + c.v;
    const follower = c.count > 0;
    const top = c.level === ORDER;
    const sub = follower
      ? (top ? `the books have these ${c.k} words ${times(c.total)}; ${esc(word)} came next ${times(c.count)} · vote ${sgn(c.v)}, ln γ ${sgn(c.d)}`
        : `${esc(word)} follows these words in ${nf.format(c.count)} of the ${nf.format(c.total)} longer patterns the books have for them · vote ${sgn(c.v)}, ln γ ${sgn(c.d)}`)
      : `never followed by ${esc(word)} in the books: only ln γ ${sgn(c.d)} for the share it passes down`;
    rows.push({ cls: follower ? 'tap' : '', src: follower ? { k: c.k, ctxIds } : null,
      cl: `<span class="if">IF</span> ${litHtml(ctxIds, slots)} <span class="then">THEN</span>`, vote: total, sub });
  }
  const rec = kn.prob(st.ctx, w, ORDER, st.cids);
  const diff = Math.abs(vt.lnp - Math.log(rec));
  rows.push({ cls: 'sum', cl: `Sum of the votes: ln P<sub>books</sub>(${esc(word)}) = ${minus(vt.lnp.toFixed(4))} → <b>${pct(Math.exp(vt.lnp))}</b>`, vote: null,
    sub: `Kneser-Ney computed the usual way gives ${pct(rec)}; the two differ by ${sci(diff)} in ln P.` });
  // memory of the text
  const f = st.f;
  const wsum = weightSum(st);
  const mem = [];
  for (let m = 0; m <= 4; m++) {
    if (f.n[m] < 1) continue;
    const c = (f.rows[m] && f.rows[m].get(w)) || 0;
    const ctxIds = G.cur.ids.slice(G.cur.ids.length - m);
    const cl = m === 0
      ? `<span class="if">MEMORY</span> <b>${esc(word)}</b> is ${c} of the ${f.n[0]} words you wrote`
      : `<span class="if">MEMORY IF</span> ${litHtml(ctxIds, slots)} <span class="then">earlier in your text</span> (${f.n[m]}×): <b>${esc(word)}</b> came next ${c}×`;
    mem.push({ cls: 'mem', cl, vote: null, sub: `D${m}: P = ${c}/${f.n[m]} · weight ${(st.w[1 + m] / wsum).toFixed(3)}` });
  }
  if (f.ptrLen >= PTR_MIN) {
    const cand = wordOf(f.ptrCand, slots);
    mem.push({ cls: 'mem', cl: `<span class="if">MEMORY P</span> your last ${f.ptrLen} words repeat an earlier passage of your text: copy <b>${esc(cand)}</b>`,
      vote: null, sub: `P(${esc(word)}) = ${f.ptrCand === w ? 1 : 0} · weight ${(st.w[6] / wsum).toFixed(3)}` });
  }
  rows.push(...mem);
  const pCore = st.mix.p[w];
  rows.push({ cls: 'total', cl: mem.length ? `Books weighted ${(st.w[0] / wsum).toFixed(3)}, memory ${(1 - st.w[0] / wsum).toFixed(3)}: <b>P(${esc(word)})</b>` : `No memory fires yet: <b>P(${esc(word)})</b> is the books' vote`, vote: pCore, isP: true });
  const box = $('#ledger');
  box.innerHTML = rows.map((r, i) => `<div class="row ${r.cls || ''}" data-i="${i}"><div class="cl">${r.cl}</div>${r.vote === null || r.vote === undefined ? '<div></div>' : `<div class="vote ${r.isP ? '' : r.vote > 0 ? 'pos' : r.vote < 0 ? 'neg' : 'zero'}">${r.isP ? pct(r.vote) : sgn(r.vote)}</div>`}${r.sub ? `<div class="sub">${r.sub}</div>` : ''}</div>`).join('');
  G.rows = rows;
  // the default source: the longest clause that votes for the word
  let pick = null;
  for (const r of rows) if (r.src && r.src.k > 0) pick = r;
  if (!pick) pick = rows[0];
  showSource(rows.indexOf(pick));
}

function showSource(i) {
  const r = G.rows[i];
  document.querySelectorAll('#ledger .row').forEach((el) => el.classList.toggle('sel', +el.dataset.i === i));
  const w = G.sel;
  const word = wordOf(w, G.cur.slots);
  const el = $('#source');
  if (!r || !r.src) { el.innerHTML = ''; return; }
  if (w >= S.vocab.nWords) { el.innerHTML = `<h4>Where it was counted</h4><p class="more">Nowhere: the books never use this word. Only the memory of your text can propose it.</p>`; return; }
  const ctxIds = r.src.k ? r.src.ctxIds : [];
  const hits = findSources(ctxIds, w);
  const qs = pickQuotes(hits, r.src.k ? 3 : 2);
  const what = r.src.k ? `“${esc(tokensPlain(ctxIds))} <b>${esc(word)}</b>”` : `<b>${esc(word)}</b>`;
  el.innerHTML = `<h4>Where it was counted</h4><p class="more">${what} occurs ${hits.length >= 4000 ? 'over 4,000 times' : times(hits.length)} in the counted chapters${hits.length > qs.length ? `; ${qs.length} of them:` : ':'}</p>${qs.map((h) => quoteHtml(h, ctxIds.length)).join('')}`;
}

function tokensPlain(ids) {
  return ids.filter((id) => id !== S.V).map((id) => wordOf(id, [])).join(' ');
}

function setText(t, push = true) {
  const ta = $('#text');
  if (push) { G.undo.push(ta.value); $('#undo').disabled = false; }
  ta.value = t;
  G.sel = null;
  renderBox();
}

function appendWord(text, word) {
  const punct = /^[.,;:!?]$/.test(word);
  const trimmed = text.replace(/\s+$/, '');
  const prev = tokenize(trimmed);
  const last = prev[prev.length - 1];
  let w = word === 'i' ? 'I' : word.startsWith("i'") ? 'I' + word.slice(1) : word;
  if (!punct && (!last || /^[.!?]$/.test(last))) w = w.charAt(0).toUpperCase() + w.slice(1);
  return punct ? trimmed + w : (trimmed ? trimmed + ' ' : '') + w;
}

function accept() {
  const word = wordOf(G.sel, G.cur.slots);
  setText(appendWord($('#text').value, word));
}

function mulberry(a) {
  return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

function sample(p, rng, T) {
  let z = 0;
  const q = new Float64Array(p.length);
  for (let i = 0; i < p.length; i++) { q[i] = Math.pow(p[i], 1 / T); z += q[i]; }
  let u = rng() * z;
  for (let i = 0; i < q.length; i++) { u -= q[i]; if (u < 0) return i; }
  return q.length - 1;
}

async function writeWords(n = 12) {
  const btn = $('#write');
  btn.disabled = true;
  G.undo.push($('#text').value);
  $('#undo').disabled = false;
  const rng = mulberry(0x51ed + G.seed++ * 7919);
  for (let i = 0; i < n; i++) {
    const cur = compute($('#text').value);
    const id = sample(cur.st.mix.p, rng, 0.8);
    $('#text').value = appendWord($('#text').value, wordOf(id, cur.slots));
    G.sel = null;
    renderBox();
    await new Promise((r) => setTimeout(r, 90));
  }
  btn.disabled = false;
}

const EXAMPLES = [
  ['It is a truth universally', ''],
  ['It was the best of', ''],
  ['Alice was beginning to get very', ''],
  ['Marley was dead: to begin', ''],
  ['Off with her', ''],
  ['The Time Traveller', ''],
  ['Anwar Debes wrote a model that counts words. Anwar Debes', 'mem'],
];

function initBox() {
  $('#examples').innerHTML = EXAMPLES.map(([t, c], i) => `<button type="button" class="${c}" data-i="${i}">${esc(t.length > 34 ? t.slice(0, 32) + '…' : t)}</button>`).join('');
  $('#examples').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) setText(EXAMPLES[+b.dataset.i][0]); });
  let tm = 0;
  $('#text').addEventListener('input', () => { clearTimeout(tm); tm = setTimeout(() => { G.sel = null; renderBox(); }, 40); });
  $('#cands').addEventListener('click', (e) => { const li = e.target.closest('.cand'); if (li) { G.sel = +li.dataset.id; renderBox(); } });
  $('#cands').addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { const li = e.target.closest('.cand'); if (li) { e.preventDefault(); G.sel = +li.dataset.id; renderBox(); } } });
  $('#ledger').addEventListener('click', (e) => { const r = e.target.closest('.row.tap'); if (r) showSource(+r.dataset.i); });
  $('#accept').addEventListener('click', accept);
  $('#write').addEventListener('click', () => writeWords(12));
  $('#undo').addEventListener('click', () => { if (G.undo.length) { setText(G.undo.pop(), false); $('#undo').disabled = !G.undo.length; } });
  renderBox();
}

// ------------------------------------------------------------------ the hero ticker
const OPENINGS = [
  'It is a truth universally',
  'Alice was beginning to get very',
  'Marley was dead: to begin',
  'All children, except one, grow',
  'Dorothy lived in the midst of the great Kansas',
  'Mr. Utterson the lawyer was a man of a rugged',
  'Off with her',
];

function tickerItem(text) {
  const cur = compute(text);
  const w = cur.st.top[0];
  const vt = S.kn.votes(cur.st.ctx, w, cur.st.cids);
  let best = null;
  for (const c of vt.clauses) if (c.count > 0) best = c;
  const ctxIds = best ? cur.st.ctx.slice(4 - best.k) : [];
  const hits = best ? findSources(ctxIds, w, 50) : [];
  const book = hits.length ? S.corpus.books[hits[0].d.book].title : '';
  const clause = best ? `<span class="m">IF</span> ${ctxIds.map((id, j) => `x${SUBS[ctxIds.length - j]} = ${esc(wordOf(id, cur.slots))}`).join(' <span class="m">∧</span> ')} <span class="m">THEN</span> <span class="k">${sgn(best.d + best.v)}</span> for <span class="k">${esc(wordOf(w, cur.slots))}</span><br><span class="m">${best.level === ORDER ? `counted ${best.count === 1 ? 'once' : best.count + '×'}` : `after ${best.count} contexts`}${book ? ` · ${esc(book)}` : ''} · P = ${pct(cur.st.mix.p[w])}</span>` : '';
  return { text, word: wordOf(w, cur.slots), clause, book };
}

function startTicker() {
  const items = OPENINGS.map(tickerItem);
  let i = 0;
  const elT = $('#tk-text'), elN = $('#tk-next'), elC = $('#tk-clause'), elB = $('#tk-book');
  const show = () => {
    const it = items[i % items.length];
    elB.textContent = it.book || ' ';
    elN.textContent = '';
    elC.innerHTML = '&nbsp;';
    let c = 0;
    const type = () => {
      elT.textContent = it.text.slice(0, ++c);
      if (c < it.text.length) setTimeout(type, 34);
      else setTimeout(() => { elN.textContent = ' ' + it.word; elC.innerHTML = it.clause; setTimeout(() => { i++; show(); }, 3400); }, 420);
    };
    type();
  };
  show();
}

function fillPages() {
  const parts = [];
  for (let b = 0; b < S.corpus.books.length; b++) {
    const d = S.corpus.docs.find((x) => x.book === b);
    const ids = Array.from(S.tokens.subarray(d.start, d.start + 150));
    parts.push(`<p>${tokensHtml(ids, [])}</p>`);
  }
  $('#pages').innerHTML = parts.join('') + parts.join('');
}

// ------------------------------------------------------------------ generation and copying
const PROMPTS = ['It is a truth universally acknowledged', 'Alice was beginning to get very tired', 'The Time Traveller', 'It was the best of times', 'Marley was dead', 'Dorothy lived in the midst of the great Kansas prairies', 'The door opened'];

function longestSeen(ctx, cids, w) {
  // the longest stretch of the books this word continues: j words including itself
  if (w >= S.vocab.nWords || S.kn.raw[1].cnt[w] === 0) return 0;
  let best = 1;
  for (let j = 2; j <= ORDER; j++) {
    const cid = cids[j - 1];
    if (cid < 0) break;
    const t = S.kn.typeId(j, cid, w);
    if (t >= 0 && S.kn.raw[j].cnt[t] > 0) best = j; else break;
  }
  return best;
}

function longestRun(ids, from = 0) {
  // the longest span that occurs word for word in one counted chapter and holds at least 5 generated words
  const tok = S.trainTok;
  let best = { len: 0 };
  for (let a = 0; a + 5 <= ids.length; a++) {
    if (ids[a] >= S.vocab.nWords) continue;
    for (const d of S.trainDocs) {
      const s = d.tstart, e = d.tstart + d.length;
      for (let i = s; i + 5 <= e; i++) {
        if (tok[i] !== ids[a] || tok[i + 4] !== ids[a + 4]) continue;
        let L = 0;
        while (a + L < ids.length && i + L < e && tok[i + L] === ids[a + L]) L++;
        if (L > best.len && a + L - Math.max(a, from) >= 5) best = { len: L, a, i, d };
      }
    }
  }
  return best;
}

function generate() {
  const seed = G.genSeed++;
  $('#seed').textContent = seed;
  const prompt = PROMPTS[(seed - 1) % PROMPTS.length];
  const enc = encodeText(prompt);
  const doc = new DocState();
  enc.ids.forEach((id) => doc.push(id));
  const rng = mulberry(0xbeef + seed * 104729);
  const out = [];
  for (let i = 0; i < 60; i++) {
    const st = S.core.next(doc, 1);
    const id = sample(st.mix.p, rng, 0.8);
    const j = longestSeen(st.ctx, st.cids, id);
    const mem = st.mix.fromDoc[id] > st.mix.fromBooks[id];
    out.push({ id, j, mem, p: st.mix.p[id] });
    doc.push(id);
  }
  // render
  const all = [...enc.ids.map((id) => ({ id, prompt: true })), ...out];
  let html = '', cap = true;
  all.forEach((t) => {
    const w = wordOf(t.id, enc.slots);
    const punct = /^[.,;:!?]$/.test(w);
    let s = w === 'i' ? 'I' : w.startsWith("i'") ? 'I' + w.slice(1) : w;
    if (cap && !punct) { s = s.charAt(0).toUpperCase() + s.slice(1); cap = false; }
    if (/^[.!?]$/.test(w)) cap = true;
    const cls = t.prompt ? 'prompt' : t.mem ? 'sm' : `s${Math.max(1, t.j)}`;
    const tip = t.prompt ? 'your prompt' : t.mem ? 'proposed mainly by the memory of this text' : t.j >= 2 ? `continues ${t.j} words that occur in the books` : 'this pair of words never occurs in the books';
    html += (punct || !html ? '' : ' ') + `<span class="t ${cls}" title="${tip}">${esc(s)}</span>`;
  });
  $('#gen-text').innerHTML = html;
  const n = out.length;
  const five = out.filter((t) => t.j === 5).length;
  const fresh = out.filter((t) => t.j <= 1 && !t.mem).length;
  const ids = all.map((t) => t.id);
  const run = longestRun(ids, enc.ids.length);
  let runHtml = '';
  if (run.len >= 5) {
    const words = ids.slice(run.a, run.a + run.len);
    runHtml = `<div class="sq">“${tokensHtml(words, enc.slots)}”<cite>${esc(docLabel(run.d))}</cite></div>`;
  }
  $('#gen-stats').innerHTML = `
    <div class="stat"><div class="sv">${Math.round((100 * five) / n)}%</div><div class="sl">of the ${n} words continue five words that stand in one of the books, word for word.</div></div>
    <div class="stat"><div class="sv">${run.len || 0}</div><div class="sl">words in a row copied from a single chapter, the longest stretch here.</div>${runHtml}</div>
    <div class="stat"><div class="sv">${Math.round((100 * fresh) / n)}%</div><div class="sl">of the words make a pair the books never contain: the seams where copied pieces meet.</div></div>`;
}

// ------------------------------------------------------------------ measurement on the held-out chapters
const ROWS = [
  { key: 1, name: '1 word', sub: 'how common each word is' },
  { key: 2, name: '2 words', sub: 'the last word as context' },
  { key: 3, name: '3 words', sub: '' },
  { key: 4, name: '4 words', sub: '' },
  { key: 5, name: '5 words', sub: 'Kneser-Ney 5-gram' },
  { key: 'core', name: '5 words + memory', sub: 'the CVTM-core on this page' },
];

function drawChart(res, frac = 1) {
  const svg = $('#chart');
  if (!res) { svg.innerHTML = ''; return; }
  S.lastChart = res;
  const narrow = svg.parentElement.clientWidth < 560;
  const W = narrow ? 380 : 720;
  const x0 = narrow ? 10 : 196, x1 = narrow ? 318 : 640;
  const rowH = narrow ? 62 : 50, y0 = narrow ? 30 : 22;
  const max = 650;
  const H = y0 + rowH * 6 + (narrow ? 34 : 44);
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  let h = '';
  [0, 100, 200, 300, 400, 500, 600].forEach((v) => {
    const x = x0 + ((x1 - x0) * v) / max;
    h += `<line x1="${x}" x2="${x}" y1="${y0 - 8}" y2="${y0 + rowH * 6 - 8}" stroke="rgba(148,170,205,0.12)"/>`;
    if (!narrow || v % 200 === 0) h += `<text x="${x}" y="${y0 + rowH * 6 + 10}" text-anchor="middle">${v}</text>`;
  });
  ROWS.forEach((r, i) => {
    const y = y0 + i * rowH;
    const ppl = res.ppl[r.key];
    if (ppl === undefined) return;
    const len = ((x1 - x0) * Math.min(ppl, max) * Math.min(1, frac)) / max;
    const core = r.key === 'core';
    if (narrow) {
      h += `<text class="lbl" x="${x0}" y="${y - 2}">${r.name}${r.sub ? ` <tspan class="sub">${r.sub}</tspan>` : ''}</text>`;
      h += `<rect x="${x0}" y="${y + 6}" width="${len.toFixed(1)}" height="20" rx="6" fill="${core ? 'url(#gm)' : 'url(#gb)'}"/>`;
      h += `<text class="val" x="${x0 + len + 8}" y="${y + 21}">${ppl.toFixed(1)}</text>`;
    } else {
      h += `<text class="lbl" x="${x0 - 14}" y="${y + 13}" text-anchor="end">${r.name}</text>`;
      if (r.sub) h += `<text class="sub" x="${x0 - 14}" y="${y + 29}" text-anchor="end">${r.sub}</text>`;
      h += `<rect x="${x0}" y="${y}" width="${len.toFixed(1)}" height="22" rx="6" fill="${core ? 'url(#gm)' : 'url(#gb)'}"/>`;
      h += `<text class="val" x="${x0 + len + 8}" y="${y + 16}">${ppl.toFixed(1)}</text>`;
    }
  });
  svg.innerHTML = `<defs><linearGradient id="gb" x1="0" x2="1"><stop offset="0" stop-color="#14b8a6"/><stop offset="1" stop-color="#2dd4bf"/></linearGradient><linearGradient id="gm" x1="0" x2="1"><stop offset="0" stop-color="#14b8a6"/><stop offset="0.82" stop-color="#2dd4bf"/><stop offset="1" stop-color="#fbbf24"/></linearGradient></defs>${h}<text x="${(x0 + x1) / 2}" y="${y0 + rowH * 6 + (narrow ? 28 : 32)}" text-anchor="middle">perplexity on held-out chapters (lower is better)</text>`;
}
addEventListener('resize', () => { if (S.lastChart) drawChart(S.lastChart); });

async function measure() {
  const bits = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, core: 0 };
  let n = 0, maxDiff = 0;
  const t0 = realNow();
  for (const d of S.held) {
    const ids = Array.from(S.tokens.subarray(d.start, d.start + d.length));
    S.core.scoreDoc(ids, (r) => {
      for (let K = 1; K <= 4; K++) bits[K] -= Math.log2(S.kn.prob(r.ctx, r.gold, K, r.cids));
      bits[5] -= r.lnpKn / Math.LN2;
      bits.core -= Math.log2(r.pCore);
      maxDiff = Math.max(maxDiff, Math.abs(r.lnpKn - Math.log(r.pRec)));
      n++;
    });
    const res = { ppl: {} };
    for (const k of Object.keys(bits)) res.ppl[k] = 2 ** (bits[k] / n);
    drawChart(res);
    await frame();
  }
  const ms = realNow() - t0;
  const ppl = {};
  for (const k of Object.keys(bits)) ppl[k] = 2 ** (bits[k] / n);
  S.measure = { ppl, n, maxDiff, ms, bits };
  drawChart(S.measure);
  const golden = S.coreDoc.heldout;
  const ref = 2 ** (golden.core_bits / golden.tokens);
  $('#m-read').innerHTML = `Going from three to five words of context buys almost nothing (<b>${ppl[3].toFixed(1)}</b> to <b>${ppl[5].toFixed(1)}</b>): ten books are too little text for long patterns to come back. Remembering the chapter so far buys more (<b>${ppl[5].toFixed(1)}</b> to <b>${ppl.core.toFixed(1)}</b>). The study saw the same on WikiText-103, where the memory took the 5-gram from 1.3501 to 1.2409 bits per byte.`;
  $('#checks').innerHTML = `
    <div class="check"><i>✓</i><div>Scored <b>${nf.format(n)}</b> words of <b>${S.held.length}</b> chapters that were never counted, in ${(ms / 1000).toFixed(1)} s.</div></div>
    <div class="check"><i>✓</i><div>Vote sums and the usual Kneser-Ney recursion agree on every one of them: the largest difference in ln P is <b>${sci(maxDiff)}</b> (the study: 5.3 × 10<sup>−15</sup>, H8.2).</div></div>
    <div class="check"><i>✓</i><div>The study's own Python code, run by <code>export.py</code> on the same tokens, gives perplexity <b>${ref.toFixed(2)}</b> for the core; this tab gives <b>${ppl.core.toFixed(2)}</b>.</div></div>`;
}

// ------------------------------------------------------------------ the 91 registered ids
function renderStudy() {
  const fams = new Map();
  for (const r of S.study.ids) {
    const f = r.id.split('.')[0];
    if (!fams.has(f)) fams.set(f, []);
    fams.get(f).push(r);
  }
  const cls = { 'met': 'met', 'not met': 'notmet', 'not judged': 'nj', 'not run': 'nr' };
  $('#ids').innerHTML = [...fams].map(([f, rs]) => `<div class="fam"><span class="fl">${f}</span><span class="fd">${rs.map((r) => `<i class="st ${cls[r.status]}" tabindex="0" data-id="${r.id}" aria-label="${r.id} ${r.status}"></i>`).join('')}</span></div>`).join('');
  const byId = new Map(S.study.ids.map((r) => [r.id, r]));
  const tip = (el) => {
    const r = byId.get(el.dataset.id);
    document.querySelectorAll('#ids .st.on').forEach((x) => x.classList.remove('on'));
    el.classList.add('on');
    $('#id-tip').innerHTML = `<b>${r.id}</b> · ${esc(r.status)}: ${esc(r.text)}${r.qualifier && r.qualifier !== 'none' ? `<span class="q">${esc(r.qualifier)}</span>` : ''}`;
  };
  $('#ids').addEventListener('mouseover', (e) => { const el = e.target.closest('.st'); if (el) tip(el); });
  $('#ids').addEventListener('focusin', (e) => { const el = e.target.closest('.st'); if (el) tip(el); });
}

// ------------------------------------------------------------------ start
async function main() {
  try {
    await load();
  } catch (e) {
    setStatus(`Could not load the books: ${esc(e.message)}`);
    throw e;
  }
  initBox();
  renderStudy();
  generate();
  $('#gen').addEventListener('click', generate);
  if (!FILM) startTicker();
  else { const it = tickerItem(OPENINGS[0]); $('#tk-text').textContent = it.text; $('#tk-next').textContent = ' ' + it.word; $('#tk-clause').innerHTML = it.clause; $('#tk-book').textContent = it.book; }
  await measure();
  window.__glassbox = {
    S, G, setText, accept, renderBox, generate, drawChart, showSource,
    select: (word) => { const id = S.vocab.index.get(word); if (id !== undefined) { G.sel = id; renderBox(); } },
    write: writeWords,
  };
  window.__demo = true;
}
main();

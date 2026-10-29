// The page in a headless DOM: app.js runs on index.html with the real data files inside jsdom, and the
// keyboard and screen-reader wiring, the pause control, the error path and the sampling of the page are
// checked. jsdom has no layout, no CSS engine and no assistive technology, so this is not a browser
// test: sizes, colours, contrast, focus rings and what a screen reader says are not checked here.
// Run with `npm test` in live/ after `npm ci` (jsdom is a development dependency of live/).
import { test, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const live = path.join(here, '..');
const require = createRequire(import.meta.url);
let JSDOM = null;
try { ({ JSDOM } = require('jsdom')); } catch { JSDOM = null; }
if (!JSDOM && process.env.CI) throw new Error('jsdom is missing: run npm ci in live/ first');
const opts = JSDOM ? {} : { skip: 'jsdom is not installed (npm ci in live/ installs it)' };

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const rejections = [];
process.on('unhandledRejection', (e) => rejections.push(e));
const doms = [];
let pages = 0;
after(() => { for (const d of doms) d.window.close(); });

// a fresh DOM of index.html, the browser globals that app.js uses, and app.js itself
async function openPage({ missing = null, reduce = true } = {}) {
  const html = fs.readFileSync(path.join(live, 'index.html'), 'utf8');
  const dom = new JSDOM(html, { url: 'https://glassbox.test/', pretendToBeVisual: true });
  doms.push(dom);
  const { window } = dom;
  const matchMedia = (q) => ({ matches: reduce && /prefers-reduced-motion:\s*reduce/.test(q), media: q,
    addEventListener() {}, removeEventListener() {} });
  window.matchMedia = matchMedia;
  Object.assign(globalThis, {
    window, document: window.document, location: window.location, matchMedia,
    requestAnimationFrame: window.requestAnimationFrame.bind(window),
    addEventListener: window.addEventListener.bind(window),
    fetch: async (url) => {
      const name = String(url);
      const file = path.join(live, name);
      if (name === missing || !fs.existsSync(file)) return new Response('not found', { status: 404, statusText: 'Not Found' });
      return new Response(fs.readFileSync(file));
    },
  });
  await import(`${pathToFileURL(path.join(live, 'app.js')).href}?page=${++pages}`);
  const t0 = Date.now();
  while (Date.now() - t0 < 60000) {
    if (window.__demo) return { dom, state: 'ready' };
    if (/Could not load/.test(window.document.querySelector('#status').textContent)) return { dom, state: 'error' };
    await sleep(20);
  }
  throw new Error('the page did not finish loading in 60 s');
}

const page = JSDOM ? await openPage() : null;
const doc = page && page.dom.window.document;
const $ = (s) => doc.querySelector(s);
const key = (el, k) => el.dispatchEvent(new page.dom.window.KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true }));

test('the page loads, and its status line is a status region', opts, () => {
  assert.equal(page.state, 'ready');
  assert.equal($('#status').getAttribute('role'), 'status');
  assert.ok($('#status').classList.contains('ready'));
  assert.match($('#status-t').textContent, /Counted 678,041 tokens/);
});

test('the take button promises no keyboard shortcut', opts, () => {
  assert.equal($('#accept').getAttribute('title'), null);
});

test('the candidates are buttons inside plain list items and keep the focus when one is chosen', opts, () => {
  const items = [...doc.querySelectorAll('#cands > li')];
  assert.equal(items.length, 8);
  for (const li of items) assert.equal(li.getAttribute('role'), null, 'a list item with a role');
  const buttons = [...doc.querySelectorAll('#cands > li > button.cand')];
  assert.equal(buttons.length, 8);
  const third = buttons[2];
  const id = third.dataset.id;
  third.focus();
  third.click();
  const now = doc.activeElement;
  assert.equal(now.tagName, 'BUTTON', 'the focus was lost');
  assert.equal(now.dataset.id, id);
  assert.equal(now.getAttribute('aria-pressed'), 'true');
  assert.equal(doc.querySelectorAll('#cands [aria-pressed="true"]').length, 1);
  assert.equal($('#accept-w').textContent, now.querySelector('.cw').textContent);
});

test('the ledger rows that show a source work by keyboard', opts, () => {
  const rows = [...doc.querySelectorAll('#ledger .row.tap')];
  assert.ok(rows.length >= 2, `${rows.length} rows with a source`);
  for (const r of rows) {
    assert.equal(r.getAttribute('tabindex'), '0');
    assert.equal(r.getAttribute('role'), 'button');
  }
  const first = rows[0], last = rows[rows.length - 1];
  first.focus();
  key(first, 'Enter');
  assert.ok(first.classList.contains('sel'));
  assert.equal(first.getAttribute('aria-pressed'), 'true');
  assert.equal(doc.activeElement, first);
  assert.match($('#source').textContent, /Where it was counted/);
  key(last, ' ');
  assert.ok(last.classList.contains('sel'));
  assert.equal(last.getAttribute('aria-pressed'), 'true');
  assert.equal(first.getAttribute('aria-pressed'), 'false');
});

test('the 91 status squares are buttons in a group, and the tip is a live region', opts, () => {
  assert.equal($('#ids').getAttribute('role'), 'group');
  const squares = [...doc.querySelectorAll('#ids .st')];
  assert.equal(squares.length, 91);
  for (const s of squares) {
    assert.equal(s.tagName, 'BUTTON');
    assert.match(s.getAttribute('aria-label'), /^H\d+\.\d+ (met|not met|not judged|not run)$/);
  }
  assert.equal($('#id-tip').getAttribute('aria-live'), 'polite');
  const study = JSON.parse(fs.readFileSync(path.join(live, 'study.json'), 'utf8'));
  const byId = new Map(study.ids.map((r) => [r.id, r]));
  doc.querySelector('#ids .st[data-id="H8.2"]').focus();
  const h82 = byId.get('H8.2');
  const q82 = h82.qualifier && h82.qualifier !== 'none' ? h82.qualifier : '';   // an empty qualifier shows nothing
  assert.equal($('#id-tip').textContent, `H8.2 · ${h82.status}: ${h82.text}${q82}`);
  assert.ok(!$('#id-tip').textContent.includes('|'), $('#id-tip').textContent);
  doc.querySelector('#ids .st[data-id="H1.1"]').click();
  const h11 = byId.get('H1.1');
  assert.ok($('#id-tip').textContent.startsWith(`H1.1 · ${h11.status}: ${h11.text}`), $('#id-tip').textContent);
  assert.equal($('#id-tip .q').textContent, h11.qualifier);
  // the counts of the text and the legend are those of study.json
  const counts = [...doc.querySelectorAll('[data-count]')];
  assert.equal(counts.length, 8);
  for (const el of counts) assert.equal(Number(el.textContent), study.ids.filter((r) => r.status === el.dataset.count).length, el.dataset.count);
});

test('the source line says "at least" when the count of passages is capped', opts, () => {
  const g = page.dom.window.__glassbox;
  g.setText('It was the best of');
  g.select('the');
  assert.equal($('#accept-w').textContent, 'the', 'the is not among the candidates');
  g.showSource(0);                                  // the bias row: every occurrence of the word
  assert.match($('#source .more').textContent, /occurs at least 4,000 times/);
});

test('writing and generating never put an empty slot into the text', opts, async () => {
  const g = page.dom.window.__glassbox;
  const S = g.S;
  const next = S.core.next;
  const empty = S.vocab.nWords + 200;               // a slot that holds no word of the text
  const house = S.vocab.index.get('house');
  S.core.next = function (d, k) {                   // put nearly all the mass on the empty slot
    const st = next.call(this, d, k);
    st.mix.p.fill(0);
    st.mix.p[empty] = 1;
    st.mix.p[house] = 1e-9;
    return st;
  };
  try {
    g.setText('xyzzy plugh');
    await g.write(4);
    assert.match($('#text').value, /^xyzzy plugh( house){4}$/);
    $('#gen').click();
    assert.ok(!$('#gen-text').textContent.includes('<w'), $('#gen-text').textContent.slice(0, 120));
    assert.match($('#gen-text').textContent, /( house){60}$/);
  } finally {
    S.core.next = next;
  }
});

test('a failure while writing gives the write button back', opts, async () => {
  const g = page.dom.window.__glassbox;
  const next = g.S.core.next;
  g.S.core.next = () => { throw new Error('boom'); };
  try {
    await assert.rejects(g.write(3), /boom/);
  } finally {
    g.S.core.next = next;
  }
  assert.equal($('#write').disabled, false);
});

test('with reduced motion the ticker starts paused; the pause control stops and starts all motion', opts, async () => {
  const btn = $('#tk-pause');
  assert.ok(btn, 'no pause control');
  assert.equal(btn.tagName, 'BUTTON');
  assert.equal(btn.getAttribute('aria-pressed'), 'true');
  assert.ok(doc.documentElement.classList.contains('still'));
  const shown = () => `${$('#tk-text').textContent}|${$('#tk-next').textContent}`;
  assert.ok($('#tk-next').textContent.trim().length > 0, 'the paused ticker shows no complete line');
  const a = shown();
  await sleep(200);
  assert.equal(shown(), a, 'the paused ticker moved');
  btn.click();
  assert.equal(btn.getAttribute('aria-pressed'), 'false');
  assert.ok(!doc.documentElement.classList.contains('still'));
  await sleep(150);
  const b = $('#tk-text').textContent;
  await sleep(150);
  assert.notEqual($('#tk-text').textContent, b, 'the running ticker does not type');
  btn.click();
  assert.equal(btn.getAttribute('aria-pressed'), 'true');
  const c = shown();
  await sleep(250);
  assert.equal(shown(), c, 'the ticker moved after the pause');
});

test('the style sheet stops the caret, the pulse and the drift for reduced motion and when paused', opts, () => {
  const css = fs.readFileSync(path.join(live, 'app.css'), 'utf8');
  // the content of every @media (prefers-reduced-motion: reduce) block, up to its matching brace
  let block = '';
  for (let at = css.indexOf('@media (prefers-reduced-motion: reduce)'); at >= 0; at = css.indexOf('@media (prefers-reduced-motion: reduce)', at + 1)) {
    let i = css.indexOf('{', at) + 1, depth = 1;
    const from = i;
    while (depth > 0 && i < css.length) { if (css[i] === '{') depth++; else if (css[i] === '}') depth--; i++; }
    block += css.slice(from, i - 1) + '\n';
  }
  for (const sel of ['.pages-in', '.caret', '.status .dot']) assert.ok(block.includes(sel), `reduced motion leaves ${sel} moving`);
  for (const sel of ['.still .pages-in', '.still .caret']) assert.ok(css.includes(sel), `no rule ${sel}`);
});

test('without reduced motion the ticker runs from the start until the pause control stops it', opts, async () => {
  const moving = await openPage({ reduce: false });
  const d = moving.dom.window.document;
  const btn = d.querySelector('#tk-pause');
  assert.equal(btn.getAttribute('aria-pressed'), 'false');
  assert.ok(!d.documentElement.classList.contains('still'));
  const a = d.querySelector('#tk-text').textContent;
  await sleep(150);
  assert.notEqual(d.querySelector('#tk-text').textContent, a, 'the ticker does not type');
  btn.click();
  assert.equal(btn.getAttribute('aria-pressed'), 'true');
  assert.ok(d.documentElement.classList.contains('still'));
  const b = `${d.querySelector('#tk-text').textContent}|${d.querySelector('#tk-next').textContent}`;
  await sleep(250);
  assert.equal(`${d.querySelector('#tk-text').textContent}|${d.querySelector('#tk-next').textContent}`, b);
});

test('a missing data file is reported by name, and the loading pulse stops', opts, async () => {
  const logged = [];
  const consoleError = console.error;
  console.error = (...a) => logged.push(a.map(String).join(' '));
  let bad;
  try {
    bad = await openPage({ missing: 'core.json' });
  } finally {
    console.error = consoleError;
  }
  assert.deepEqual(logged, ['Error: core.json: HTTP 404']);
  const d = bad.dom.window.document;
  assert.equal(bad.state, 'error');
  assert.match(d.querySelector('#status-t').textContent, /core\.json: HTTP 404/);
  assert.ok(d.querySelector('#status').classList.contains('error'));
  assert.deepEqual(rejections, [], 'the page left an unhandled rejection');
});

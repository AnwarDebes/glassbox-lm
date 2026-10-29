// The data files of the page against their sources: study.json against the results page of
// tm_language_model (parsed here independently of export.py), and the document-word slots of
// corpus.json and tokens.u16.gz. Run with `npm test` in live/.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import zlib from 'node:zlib';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const live = path.join(here, '..');
const resultsPage = path.join(live, '..', 'studies', 'tm_language_model', 'docs', 'results.md');
const study = JSON.parse(fs.readFileSync(path.join(live, 'study.json'), 'utf8'));
const corpus = JSON.parse(fs.readFileSync(path.join(live, 'corpus.json'), 'utf8'));
const buf = zlib.gunzipSync(fs.readFileSync(path.join(live, 'tokens.u16.gz')));
const tokens = new Uint16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2);

const STATUS = /^(met|not met|not judged|not run)(?: \(.*\))?$/;

const cells = (line) => {
  const s = line.trim();
  return s.length >= 2 && s.startsWith('|') && s.endsWith('|') ? s.slice(1, -1).split('|').map((c) => c.trim()) : null;
};

// every row "| id | claim | status | qualifier | evidence |" whose status cell is a status; the first row of an id counts
function pageIds(text) {
  const out = new Map();
  for (const line of text.split('\n')) {
    const c = cells(line);
    if (!c || c.length !== 5 || !/^H\d+\.\d+$/.test(c[0])) continue;
    const m = STATUS.exec(c[2]);
    if (!m || out.has(c[0])) continue;
    out.set(c[0], { id: c[0], text: c[1], status: m[1], qualifier: c[3] });
  }
  return out;
}

const STATUSES = ['met', 'not met', 'not judged', 'not run'];
const counts = () => STATUSES.map((s) => study.ids.filter((r) => r.status === s).length);

test('study.json holds the ids of the results page with their texts, statuses and qualifiers', () => {
  const text = fs.readFileSync(resultsPage, 'utf8');
  const page = pageIds(text);
  const all = text.split('\n').map(cells).find((c) => c && c[0] === 'all');
  assert.equal(page.size, 91);
  assert.equal(study.ids.length, 91);
  for (const r of study.ids) {
    assert.ok(!r.qualifier.startsWith('|'), `${r.id}: qualifier starts with "|": ${r.qualifier}`);
    const p = page.get(r.id);
    assert.ok(p, `${r.id} is not on the results page`);
    assert.equal(r.text, p.text, `text of ${r.id}`);
    assert.equal(r.status, p.status, `status of ${r.id} (rerun export.py after a change of the results page)`);
    assert.equal(r.qualifier, p.qualifier, `qualifier of ${r.id} (rerun export.py after a change of the results page)`);
  }
  assert.deepEqual([study.ids.length, ...counts()], all.slice(1).map(Number), 'the count table of the results page');
});

test('the counts written in index.html are those of study.json', () => {
  const html = fs.readFileSync(path.join(live, 'index.html'), 'utf8');
  const found = [...html.matchAll(/data-count="([a-z ]+)">(\d+)</g)];
  assert.ok(found.length >= 8, `${found.length} counts in index.html`);
  const want = Object.fromEntries(STATUSES.map((s, i) => [s, counts()[i]]));
  for (const [, status, n] of found) assert.equal(Number(n), want[status], `index.html says ${n} ${status}`);
});

test('every new word of a chapter has its own slot', () => {
  const nWords = corpus.n_words;
  assert.equal(corpus.vocab.length, nWords + corpus.n_slots);
  assert.ok(corpus.vocab.length <= 32768, 'token ids must fit in 15 bits');
  for (const d of corpus.docs) {
    const n = d.new_words.length;
    assert.ok(n <= corpus.n_slots, `${d.label}: ${n} new words, ${corpus.n_slots} slots`);
    // the slot ids of a chapter are first used in the order 0, 1, 2, ... and every one of them is used
    let next = 0;
    for (let i = d.start; i < d.start + d.length; i++) {
      const id = tokens[i];
      assert.ok(id < nWords + n, `${d.label}: id ${id} beyond its ${n} new words`);
      if (id >= nWords) {
        const j = id - nWords;
        assert.ok(j <= next, `${d.label}: slot ${j} used before slot ${next}`);
        if (j === next) next++;
      }
    }
    assert.equal(next, n, `${d.label}: ${n} new words but ${next} distinct slots`);
    if (d.split === 'train') assert.equal(n, 0, `training chapter ${d.label} has a new word`);
  }
});

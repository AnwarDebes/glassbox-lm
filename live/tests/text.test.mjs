// The text side of model.js. The tokenizer and Vocab.encode are checked against export.py: the
// cases and their expected tokens and ids are in tests/golden.json, written by export.py from its own
// tokenize() and encode(). detok, sampling and the number formats are checked against expected
// outputs written by hand below. Run with `npm test` in live/.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as M from '../model.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const live = path.join(here, '..');
const golden = JSON.parse(fs.readFileSync(path.join(here, 'golden.json'), 'utf8'));
const corpus = JSON.parse(fs.readFileSync(path.join(live, 'corpus.json'), 'utf8'));
const vocab = new M.Vocab(corpus.vocab.slice(0, corpus.n_words), corpus.n_slots);

test('tokenize agrees with export.py on the edge cases of golden.json', () => {
  assert.ok(Array.isArray(golden.tokenizer) && golden.tokenizer.length >= 10, 'golden.json holds no tokenizer cases');
  for (const [text, want] of golden.tokenizer) assert.deepEqual(M.tokenize(text), want, JSON.stringify(text));
});

test('every word of the vocabulary is one token of itself', () => {
  for (const w of vocab.words) assert.deepEqual(M.tokenize(w), [w], w);
});

test('Vocab.encode gives the ids of export.py, one slot per new word', () => {
  assert.ok(Array.isArray(golden.encode) && golden.encode.length >= 4, 'golden.json holds no encoding cases');
  for (const c of golden.encode) {
    const { ids, slots } = vocab.encode(M.tokenize(c.text), []);
    assert.deepEqual(ids, c.ids, c.text.slice(0, 40));
    assert.deepEqual(slots, c.new_words, c.text.slice(0, 40));
  }
});

test('Vocab: slots, word texts and the ids a text can hold', () => {
  const v = new M.Vocab(['the', 'cat', '.'], 3);
  const { ids, slots } = v.encode(['the', 'zork', 'cat', 'zork', 'quux', 'a', 'b', 'c'], []);
  // three slots: zork, quux, a; b and c share the last slot with a
  assert.deepEqual(ids, [0, 3, 1, 3, 4, 5, 5, 5]);
  assert.deepEqual(slots, ['zork', 'quux', 'a', 'b', 'c']);
  assert.equal(v.text(1, slots), 'cat');
  assert.equal(v.text(3, slots), 'zork');
  assert.equal(v.text(5, slots), 'a');
  assert.equal(v.text(4, []), '<w1>');
  assert.equal(typeof v.limit, 'function', 'Vocab has no limit()');
  assert.equal(v.limit([]), 3);
  assert.equal(v.limit(['zork']), 4);
  assert.equal(v.limit(slots), 6);
});

test('detok writes tokens back as a sentence', () => {
  assert.equal(M.detok(['it', 'is', 'a', 'truth', ',', 'said', 'i', '.', "i'm", 'here', '!', 'yes', '?']),
    "It is a truth, said I. I'm here! Yes?");
  assert.equal(M.detok(['mr', '.', 'utterson']), 'Mr. Utterson');
  assert.equal(M.detok(['.', 'then']), '. Then');
  assert.equal(M.detok([]), '');
});

test('the seeded generator of the page is unchanged', () => {
  assert.equal(typeof M.mulberry32, 'function', 'model.js exports no mulberry32()');
  // the first draws of the generator that app.js used before it moved to model.js
  const r = M.mulberry32(0x51ed + 7919);
  assert.deepEqual([r(), r(), r()], [0.2584649238269776, 0.2671354026533663, 0.8446857712697238]);
  const s = M.mulberry32(7);
  assert.deepEqual([s(), s(), s()], [0.011704753153026104, 0.06195825757458806, 0.97690763277933]);
});

test('sampling never writes a slot that holds no word of the text', () => {
  assert.equal(typeof M.sample, 'function', 'model.js exports no sample()');
  // ids 0 and 1 are words, 2 to 5 are slots; the text holds one new word, so slot 2 is in use
  const v = new M.Vocab(['a', 'b'], 4);
  const limit = v.limit(['zork']);
  const p = Float64Array.from([0.1, 0.1, 0.2, 0.2, 0.2, 0.2]);
  for (const T of [0.8, 1]) {
    const rng = M.mulberry32(11);
    const n = [0, 0, 0, 0, 0, 0];
    for (let i = 0; i < 6000; i++) n[M.sample(p, rng, T, limit)]++;
    assert.deepEqual(n.slice(3), [0, 0, 0], `T ${T}: empty slots were drawn ${n.slice(3)}`);
    // the draws follow p ** (1 / T) over the ids below the limit
    const q = [0.1, 0.1, 0.2].map((x) => x ** (1 / T));
    const z = q.reduce((a, b) => a + b, 0);
    q.forEach((x, i) => assert.ok(Math.abs(n[i] / 6000 - x / z) < 0.03, `T ${T}, id ${i}: ${n[i] / 6000} vs ${x / z}`));
  }
  // without a limit every id can be drawn
  const rng = M.mulberry32(3);
  const seen = new Set();
  for (let i = 0; i < 2000; i++) seen.add(M.sample(p, rng, 1));
  assert.deepEqual([...seen].sort(), [0, 1, 2, 3, 4, 5]);
});

test('percentages and powers of ten are printed readably', () => {
  assert.equal(typeof M.pct, 'function', 'model.js exports no pct()');
  assert.equal(typeof M.sci, 'function', 'model.js exports no sci()');
  assert.equal(M.pct(0.5), '50.0%');
  assert.equal(M.pct(0.05), '5.00%');
  assert.equal(M.pct(0.001234), '0.123%');
  assert.equal(M.pct(1e-5), '0.001%');
  assert.equal(M.pct(4e-6), '&lt;0.001%');
  assert.equal(M.pct(0), '0%');
  assert.equal(M.sci(0), '0');
  assert.equal(M.sci(4.4e-15), '4.4 × 10<sup>−15</sup>');
  assert.equal(M.sci(9.97e-15), '1.0 × 10<sup>−14</sup>');
  assert.equal(M.sci(1.5e3), '1.5 × 10<sup>3</sup>');
});

/* Plain C clause evaluation for the readable agent's saved TM (benchmark only, not used by any study).
   Build: gcc -O3 -march=native -shared -fPIC -o libtmclause.so tmclause.c
   Both kernels compute the integer class sums of one input: the weights of the firing clauses, summed.
   A clause fires when all its literals are true; evaluation of a clause stops at its first false literal. */
#include <stdint.h>
#include <string.h>

/* Literal lists. Clause c's literals are lits[starts[c] .. starts[c + 1] - 1]; a literal j < f reads x[j],
   a literal j >= f reads NOT x[j - f]. */
void sums_lits(const uint8_t *x, int f, int nc, const int64_t *starts, const int64_t *lits,
               const int64_t *w, int ncls, int64_t *out) {
    memset(out, 0, sizeof(int64_t) * (size_t)ncls);
    for (int c = 0; c < nc; c++) {
        int ok = 1;
        for (int64_t k = starts[c]; k < starts[c + 1]; k++) {
            int64_t j = lits[k];
            if (j < f ? !x[j] : x[j - f]) { ok = 0; break; }
        }
        if (ok) {
            const int64_t *wc = w + (int64_t)c * ncls;
            for (int k = 0; k < ncls; k++) out[k] += wc[k];
        }
    }
}

/* Sparse 64-bit words. xw holds the input bits (feature i is bit i % 64 of word i / 64). Clause c's words are
   widx/pm/nm[ws[c] .. ws[c + 1] - 1]; it fires when (xw & pm) == pm and (xw & nm) == 0 on each of them. */
void sums_words(const uint64_t *xw, int nc, const int64_t *ws, const int32_t *widx, const uint64_t *pm,
                const uint64_t *nm, const int64_t *w, int ncls, int64_t *out) {
    memset(out, 0, sizeof(int64_t) * (size_t)ncls);
    for (int c = 0; c < nc; c++) {
        int ok = 1;
        for (int64_t k = ws[c]; k < ws[c + 1]; k++) {
            uint64_t v = xw[widx[k]];
            if ((v & pm[k]) != pm[k] || (v & nm[k])) { ok = 0; break; }
        }
        if (ok) {
            const int64_t *wc = w + (int64_t)c * ncls;
            for (int k = 0; k < ncls; k++) out[k] += wc[k];
        }
    }
}

/* Pack f input bytes (0/1) into 64-bit words, feature i at bit i % 64 of word i / 64. */
void pack_x(const uint8_t *x, int f, uint64_t *xw) {
    int nw = (f + 63) / 64;
    memset(xw, 0, sizeof(uint64_t) * (size_t)nw);
    for (int i = 0; i < f; i++)
        if (x[i]) xw[i >> 6] |= 1ULL << (i & 63);
}

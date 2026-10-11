"""Read-only PDF text extractor (Python stdlib only; pdftotext/poppler is not installed here).

Used to check quotations against the source. Handles Flate streams, object streams (/ObjStm), the page tree,
and ToUnicode CMaps (CID / Identity-H fonts, as in Elsevier and many Springer PDFs).
Not handled: encrypted PDFs, LZW/ASCII85 filters, and text drawn as images (scanned pages).

    python src/pdftext.py paper.pdf out.txt             # all pages, "=== page N ===" markers
    python src/pdftext.py paper.pdf out.txt --grep "T and s are global"
"""
import re
import sys
import unicodedata
import zlib

OBJ = re.compile(rb"(\d+)\s+(\d+)\s+obj\b(.*?)endobj", re.S)
REF = rb"(\d+)\s+\d+\s+R"


def _stream(body):
    m = re.search(rb"stream\r?\n", body)
    if not m:
        return body, None
    head, raw = body[:m.start()], body[m.end():]
    end = raw.rfind(b"endstream")
    raw = raw[:end] if end >= 0 else raw
    if b"/FlateDecode" in head:
        try:
            raw = zlib.decompressobj().decompress(raw)
        except zlib.error:
            return head, None
    elif b"/Filter" in head:
        return head, None
    return head, raw


def load_objects(data):
    objs = {}
    for m in OBJ.finditer(data):
        objs[int(m.group(1))] = m.group(3)
    for body in list(objs.values()):
        if b"/ObjStm" not in body[:300]:
            continue
        head, raw = _stream(body)
        if raw is None:
            continue
        n = int(re.search(rb"/N\s+(\d+)", head).group(1))
        first = int(re.search(rb"/First\s+(\d+)", head).group(1))
        nums = [int(x) for x in raw[:first].split()]
        offs = [(nums[2 * i], nums[2 * i + 1]) for i in range(n)]
        for i, (num, off) in enumerate(offs):
            end = first + offs[i + 1][1] if i + 1 < n else len(raw)
            objs.setdefault(num, raw[first + off:end])
    return objs


def parse_cmap(raw):
    cmap, width = {}, 2
    cs = re.search(rb"begincodespacerange\s*<([0-9A-Fa-f]+)>", raw)
    if cs:
        width = len(cs.group(1)) // 2
    u16 = lambda h: bytes.fromhex(h.decode()).decode("utf-16-be", "replace")
    for blk in re.findall(rb"beginbfchar(.*?)endbfchar", raw, re.S):
        for src, dst in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]*)>", blk):
            cmap[int(src, 16)] = u16(dst)
    for blk in re.findall(rb"beginbfrange(.*?)endbfrange", raw, re.S):
        for lo, hi, rest in re.findall(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*(<[0-9A-Fa-f]*>|\[.*?\])", blk, re.S):
            lo, hi = int(lo, 16), int(hi, 16)
            if rest.startswith(b"["):
                for k, d in enumerate(re.findall(rb"<([0-9A-Fa-f]*)>", rest)):
                    cmap[lo + k] = u16(d)
            else:
                base = bytes.fromhex(rest[1:-1].decode())
                for k in range(hi - lo + 1):
                    b = base[:-2] + (int.from_bytes(base[-2:], "big") + k).to_bytes(2, "big") if len(base) >= 2 else base
                    cmap[lo + k] = b.decode("utf-16-be", "replace")
    return cmap, width


class Doc:
    def __init__(self, path):
        self.data = open(path, "rb").read()
        self.objs = load_objects(self.data)
        self.cmaps = {}

    def get(self, ref):
        return self.objs.get(int(ref), b"")

    def resolve_dict(self, body, key):
        m = re.search(rb"/" + key + rb"\s*" + REF, body)
        if m:
            return self.get(m.group(1))
        m = re.search(rb"/" + key + rb"\s*<<", body)
        if not m:
            return None
        depth, i = 1, m.end()
        while i < len(body) and depth:
            if body.startswith(b"<<", i):
                depth, i = depth + 1, i + 2
            elif body.startswith(b">>", i):
                depth, i = depth - 1, i + 2
            else:
                i += 1
        return body[m.end():i - 2]

    def pages(self):
        cat = next((b for b in self.objs.values() if re.search(rb"/Type\s*/Catalog", b)), None)
        root = int(re.search(rb"/Pages\s*" + REF, cat).group(1))
        out, stack = [], [root]
        while stack:
            num = stack.pop(0)
            body = self.objs.get(num, b"")
            kids = re.search(rb"/Kids\s*\[(.*?)\]", body, re.S)
            if kids and re.search(rb"/Type\s*/Pages", body):
                stack = [int(k) for k in re.findall(REF, kids.group(1))] + stack
            else:
                out.append(num)
        return out

    def fonts(self, page_num):
        body, node = self.objs[page_num], page_num
        res = None
        while res is None and body:
            res = self.resolve_dict(body, b"Resources")
            if res is None:
                par = re.search(rb"/Parent\s*" + REF, body)
                body = self.get(par.group(1)) if par else b""
        if not res:
            return {}
        fd = self.resolve_dict(res, b"Font") or b""
        fonts = {}
        for name, ref in re.findall(rb"/([^\s/<>\[\]()]+)\s*" + REF, fd):
            ref = int(ref)
            if ref not in self.cmaps:
                f = self.objs.get(ref, b"")
                tu = re.search(rb"/ToUnicode\s*" + REF, f)
                cm = None
                if tu:
                    _, raw = _stream(self.get(tu.group(1)))
                    cm = parse_cmap(raw) if raw else None
                if cm is None and b"/Identity-H" in f:
                    cm = ({}, 2)
                self.cmaps[ref] = cm
            fonts[name] = self.cmaps[ref]
        return fonts

    def page_text(self, page_num):
        body = self.objs[page_num]
        m = re.search(rb"/Contents\s*\[(.*?)\]", body, re.S)
        refs = re.findall(REF, m.group(1)) if m else re.findall(rb"/Contents\s*" + REF, body)
        raw = b"".join((_stream(self.get(r))[1] or b"") + b"\n" for r in refs)
        fonts = self.fonts(page_num)
        out, cur = [], None
        tok = re.compile(rb"/([^\s/<>\[\]()]+)\s+[-\d.]+\s+Tf|\(((?:[^()\\]|\\.)*)\)|<([0-9A-Fa-f\s]*)>|(-?\d*\.?\d+)|(T\*|Td|TD|Tm|ET|')(?![A-Za-z])", re.S)
        for t in tok.finditer(raw):
            if t.group(1) is not None:
                cur = fonts.get(t.group(1))
            elif t.group(2) is not None or t.group(3) is not None:
                if t.group(2) is not None:
                    s = re.sub(rb"\\([0-7]{1,3}|.)", lambda e: bytes([int(e.group(1), 8)]) if e.group(1)[:1].isdigit()
                               else {b"n": b"\n", b"r": b"\r", b"t": b"\t"}.get(e.group(1), e.group(1)), t.group(2), flags=re.S)
                else:
                    h = re.sub(rb"\s", b"", t.group(3))
                    s = bytes.fromhex((h + b"0" * (len(h) % 2)).decode())
                if cur:
                    cmap, w = cur
                    out.append("".join(cmap.get(int.from_bytes(s[i:i + w], "big"), "") for i in range(0, len(s) - w + 1, w)))
                else:
                    out.append(s.decode("latin-1"))
            elif t.group(4) is not None:
                try:
                    if float(t.group(4)) < -200:
                        out.append(" ")
                except ValueError:
                    pass
            else:
                out.append(" " if t.group(5) != b"ET" else "\n")
        return re.sub(r"[ \t]+", " ", unicodedata.normalize("NFKC", "".join(out)))  # NFKC: the fi ligature (U+FB01) -> "fi"


if __name__ == "__main__":
    doc = Doc(sys.argv[1])
    texts = [doc.page_text(p) for p in doc.pages()]
    with open(sys.argv[2], "w") as f:
        for i, t in enumerate(texts, 1):
            f.write(f"\n=== page {i} ===\n{t}\n")
    print(f"{len(texts)} pages, {sum(map(len, texts))} chars -> {sys.argv[2]}")
    if "--grep" in sys.argv:
        pat = sys.argv[sys.argv.index("--grep") + 1]
        flat = [re.sub(r"\s+", " ", t) for t in texts]
        for i, t in enumerate(flat, 1):
            for m in re.finditer(re.escape(pat), t, re.I):
                print(f"page {i}: ...{t[max(0, m.start() - 200):m.end() + 200]}...")

"""Read one string key from a GGUF file's metadata header (pure Python; reads only the header, never the tensors).

Used once, offline, to pin the chat template that llama-server b11742 serves with --jinja (the GGUF's own
tokenizer.chat_template): its sha256 must equal the chat_template that GET /props reports (e4_run.py checks the live
value against pins/server_expected.json). GGUF layout (v2/v3): magic "GGUF", uint32 version, uint64 tensor count,
uint64 key-value count, then the key-value pairs (key: uint64 length + bytes; uint32 type; value).
Run: .../python -I -B gguf_meta.py --gguf <file> --key tokenizer.chat_template [--out <json>]
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _read(f, n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError("truncated GGUF header")
    return b


def _string(f):
    (n,) = struct.unpack("<Q", _read(f, 8))
    return _read(f, n)


def _value(f, t, keep):
    if t in SCALAR:
        fmt = SCALAR[t]
        (v,) = struct.unpack(fmt, _read(f, struct.calcsize(fmt)))
        return v
    if t == 8:
        b = _string(f)
        return b.decode("utf-8") if keep else None
    if t == 9:
        (it,) = struct.unpack("<I", _read(f, 4))
        (cnt,) = struct.unpack("<Q", _read(f, 8))
        if it in SCALAR:
            f.seek(cnt * struct.calcsize(SCALAR[it]), 1)
            return None
        out = [] if keep else None
        for _ in range(cnt):
            v = _value(f, it, keep)
            if keep:
                out.append(v)
        return out
    raise ValueError(f"unknown GGUF value type {t}")


def read_key(path: str, key: str):
    with open(path, "rb") as f:
        if _read(f, 4) != b"GGUF":
            raise ValueError("not a GGUF file")
        (version,) = struct.unpack("<I", _read(f, 4))
        n_tensors, n_kv = struct.unpack("<QQ", _read(f, 16))
        for _ in range(n_kv):
            k = _string(f).decode("utf-8")
            (t,) = struct.unpack("<I", _read(f, 4))
            v = _value(f, t, keep=(k == key))
            if k == key:
                return {"version": version, "n_tensors": n_tensors, "n_kv": n_kv, "value": v}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--key", default="tokenizer.chat_template")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = read_key(a.gguf, a.key)
    if r is None:
        sys.exit(f"key {a.key} not found")
    v = r.pop("value")
    out = {"gguf": a.gguf, "key": a.key, **r, "type": type(v).__name__}
    if isinstance(v, str):
        out.update({"chars": len(v), "sha256_utf8": hashlib.sha256(v.encode("utf-8")).hexdigest()})
    else:
        out["value"] = v
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

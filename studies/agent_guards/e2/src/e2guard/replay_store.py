"""The campaign's paired-replay store, with no third-party imports (so both venvs can use it).

- ReplayStore: append-only JSON lines {key, reply, utc, ms, context}; key = sha256(canonical request body). A stored
  reply is used only for a byte-identical canonical body, so a replayed reply is exactly what the server answered.
- StoreHTTP: the same store for the plain HTTP posts of the A3 optimizer and the D6 rule writer (rev2, review item 16):
  the key is namespaced by the endpoint path, so these keys can never collide with the agent's request keys.
- Store integrity across stages (rev2, review item 6): store_record() gives {bytes, sha256} of the whole file;
  prefix_ok() checks that a later store still starts with exactly those bytes (the store is append-only).
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path


def canonical(body: dict) -> str:
    return json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def body_key(body: dict) -> str:
    return hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()


class ReplayStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.items: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    self.items.setdefault(d["key"], d)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a")

    def get(self, key: str) -> dict | None:
        return self.items.get(key)

    def put(self, key: str, reply: dict, meta: dict) -> None:
        if key in self.items:
            return
        d = {"key": key, "reply": reply, **meta}
        self.items[key] = d
        self._f.write(json.dumps(d, default=str) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def store_record(path: Path | str) -> dict:
    """{bytes, sha256} of the whole store file ({0, sha256 of nothing} when it does not exist)."""
    p = Path(path)
    h = hashlib.sha256()
    n = 0
    if p.exists():
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
                n += len(chunk)
    return {"bytes": n, "sha256": h.hexdigest()}


def prefix_ok(path: Path | str, rec: dict) -> bool:
    """True iff the file's first rec['bytes'] bytes hash to rec['sha256'] (an append-only file kept its past)."""
    p = Path(path)
    want = int(rec["bytes"])
    if want == 0:
        return rec["sha256"] == hashlib.sha256(b"").hexdigest()
    if not p.exists() or p.stat().st_size < want:
        return False
    h = hashlib.sha256()
    left = want
    with p.open("rb") as f:
        while left > 0:
            chunk = f.read(min(1 << 20, left))
            if not chunk:
                return False
            h.update(chunk)
            left -= len(chunk)
    return h.hexdigest() == rec["sha256"]


class StoreHTTP:
    """POST a JSON body to an OpenAI-compatible endpoint through the store. Returns (reply JSON, hit, ms, key)."""

    def __init__(self, store: ReplayStore | None, timeout: float = 600.0) -> None:
        self.store, self.timeout = store, timeout
        self.errors: list[str] = []

    @staticmethod
    def key_of(url: str, body: dict) -> str:
        path = "/" + url.split("://", 1)[-1].split("/", 1)[-1]
        return body_key({"_ns": "http:" + path, "body": body})

    def post(self, url: str, body: dict, context: dict | None = None, use_store: bool = True) -> tuple[dict, bool, float, str]:
        key = self.key_of(url, body)
        hit = self.store.get(key) if (use_store and self.store is not None) else None
        if hit is not None:
            return hit["reply"], True, 0.0, key
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.loads(r.read())
        except Exception as e:
            self.errors.append(f"{type(e).__name__}: {e}"[:300])
            raise
        ms = (time.time() - t0) * 1000
        if use_store and self.store is not None:
            self.store.put(key, out, {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ms": round(ms),
                                      "context": dict(context or {})})
        return out, False, ms, key

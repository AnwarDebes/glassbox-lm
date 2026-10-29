"""E2 fact extractor: the trusted facts about one proposed tool call, computed from the harness's own bookkeeping.

What a guard may read (DESIGN 1.2 and E2, "Trusted facts"; AgentDojo's threat model trusts the user's query):
- the user's query (the `query` string the pipeline was started with; never a later user-role message, so the guard's
  own feedback messages and any repeated prompt are not sources);
- the tool table (tool_table.json, hand-written before any run): the call's kind and each argument's role;
- the record of executed calls: every tool-result message carries the call that produced it (AgentDojo's
  ChatToolResultMessage.tool_call) and that call's output text.
What it computes from them, by string matching only (normalize.py), never from a model's words:
- per argument atom, the provenance class, the most trusted source the value occurs in:
  user_query > named_source > other_tool_output > not_found;
  * named_source = the output of an earlier READ call that the query named: every data-carrying argument value occurs
    in the query, and at least one of them is not a free-text search parameter (tool_table.json, search_params)
    (e.g. read_file('bill-december-2023.txt') when the query names that file; get_webpage('www.x.com') when the query
    gives that URL; search_emails(query='', sender='lily@x.com') when the query names lily@x.com). A call with no
    argument values (get_unread_emails(), get_channels(), ...) or with only a keyword search (search_emails('report'))
    is never a named source. ROPE names sources with one LLM call that reads only the request; this mechanical rule
    needs no model and is stricter for argument-free reads;
  * an empty password value counts as one atom with provenance not_found (an empty password is not a literal of the
    query);
- history: the executed calls of a known kind before this one (count by kind, total), and whether a named source was
  read. Calls of unknown tools are not counted (the runtime rejects them).
The tool outputs' text is used only to decide where a value came from; no output text becomes a fact.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from .normalize import Atom, Source, atoms_of, make_source, matches

HERE = Path(__file__).resolve().parent
TABLE_PATH = HERE / "tool_table.json"
PROVS = ["user_query", "named_source", "other_tool_output", "not_found"]
PROV_RANK = {p: i for i, p in enumerate(PROVS)}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class ToolTable:
    def __init__(self, path: Path | str = TABLE_PATH) -> None:
        self.path = Path(path)
        d = json.loads(self.path.read_text())
        self.kinds: list[str] = d["kinds"]
        self.roles: list[str] = d["roles"]
        self.by_name: dict[str, dict] = {}
        self.suite_of: dict[str, set] = {}
        for suite, tools in d["suites"].items():
            for name, e in tools.items():
                assert e["kind"] in self.kinds, (name, e["kind"])
                assert all(r in self.roles for r in e["params"].values()), name
                if name in self.by_name and self.by_name[name] != e:
                    raise ValueError(f"tool {name} has two different entries")
                self.by_name[name] = e
                self.suite_of.setdefault(name, set()).add(suite)
        self.rope = {s: v for s, v in d["rope_scope"].items() if s != "about"}
        self.search_params = {t: set(v) for t, v in d["search_params"].items() if t != "about"}
        self.sha256 = sha256_file(self.path)
        # which kinds each role occurs with (domain rows for the solver); 'other' also takes unknown parameters
        self.kinds_of_role: dict[str, set] = {r: set() for r in self.roles}
        for e in self.by_name.values():
            for r in e["params"].values():
                self.kinds_of_role[r].add(e["kind"])
        self.kinds_of_role["other"] = set(self.kinds)

    def kind(self, function: str) -> str | None:
        e = self.by_name.get(function)
        return e["kind"] if e else None

    def role(self, function: str, param: str) -> str:
        e = self.by_name.get(function)
        return e["params"].get(param, "other") if e else "other"


def call_function(c) -> str:
    return c.function if hasattr(c, "function") else c["function"]


def call_args(c) -> dict:
    a = c.args if hasattr(c, "args") else c.get("args")
    return dict(a or {})


def call_id(c):
    return c.id if hasattr(c, "id") else c.get("id")


def text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for b in content:
        if isinstance(b, dict):
            parts.append(b.get("content") or b.get("text") or "")
        else:
            parts.append(str(b))
    return "\n".join(p for p in parts if p)


@dataclass
class AtomFact:
    shown: str
    kind: str
    prov: str


@dataclass
class ArgFact:
    name: str
    role: str
    atoms: list[AtomFact]


@dataclass
class Facts:
    function: str
    kind: str | None
    args: list[ArgFact]
    role_prov: dict[str, set] = field(default_factory=dict)
    prov_any: set = field(default_factory=set)
    n_steps: int = 0
    done: dict[str, int] = field(default_factory=dict)
    named_source_read: bool = False
    named_calls: list[str] = field(default_factory=list)
    query_sha256: str = ""

    @property
    def known(self) -> bool:
        return self.kind is not None

    def to_json(self) -> dict:
        return {
            "function": self.function, "kind": self.kind,
            "args": [{"name": a.name, "role": a.role, "atoms": [{"shown": t.shown[:200], "kind": t.kind, "prov": t.prov}
                                                                for t in a.atoms]} for a in self.args],
            "role_prov": {r: sorted(v, key=PROV_RANK.get) for r, v in sorted(self.role_prov.items())},
            "prov_any": sorted(self.prov_any, key=PROV_RANK.get),
            "n_steps": self.n_steps, "done": dict(sorted(self.done.items())),
            "named_source_read": self.named_source_read, "named_calls": self.named_calls,
            "query_sha256": self.query_sha256,
        }


class FactExtractor:
    def __init__(self, table: ToolTable | None = None, cache_size: int = 4096) -> None:
        self.table = table or ToolTable()
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size

    # ----- sources ---------------------------------------------------------------------
    def _source(self, label: str, raw: str) -> Source:
        key = (label, sha256_text(raw))
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        src = make_source(label, raw)
        self._cache[key] = src
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return src

    def query_source(self, query: str) -> Source:
        key = ("query", sha256_text(query))
        hit = self._cache.get(key)
        if hit is None:
            hit = make_source("user_query", query, parse_yaml=False)
            self._cache[key] = hit
        return hit

    def is_named(self, function: str, args: dict, qsrc: Source) -> bool:
        if self.table.kind(function) != "read":
            return False
        search = self.table.search_params.get(function, set())
        atoms, identifying = [], False
        for k, v in args.items():
            a = atoms_of(v)
            atoms += a
            if a and k not in search:
                identifying = True
        return identifying and all(matches(a, qsrc) for a in atoms)

    def history(self, query: str, messages) -> tuple[Source, list[Source], list[Source], int, Counter, list[str]]:
        qsrc = self.query_source(query)
        named, other, named_fns = [], [], []
        done: Counter = Counter()
        n = 0
        for m in messages:
            if (m.get("role") if isinstance(m, dict) else None) != "tool":
                continue
            c = m.get("tool_call")
            if c is None:
                continue
            fn = call_function(c)
            raw = text_of(m.get("content"))
            if m.get("error"):
                raw = raw + "\n" + str(m.get("error"))
            src = self._source("tool", raw)
            if self.is_named(fn, call_args(c), qsrc):
                named.append(src)
                named_fns.append(fn)
            else:
                other.append(src)
            k = self.table.kind(fn)
            if k is not None:
                done[k] += 1
                n += 1
        return qsrc, named, other, n, done, named_fns

    # ----- one call --------------------------------------------------------------------
    @staticmethod
    def provenance(atom: Atom, qsrc: Source, named: list[Source], other: list[Source]) -> str:
        if matches(atom, qsrc):
            return "user_query"
        if any(matches(atom, s) for s in named):
            return "named_source"
        if any(matches(atom, s) for s in other):
            return "other_tool_output"
        return "not_found"

    def extract(self, query: str, call, messages) -> Facts:
        fn = call_function(call)
        args = call_args(call)
        qsrc, named, other, n, done, named_fns = self.history(query, messages)
        kind = self.table.kind(fn)
        arg_facts: list[ArgFact] = []
        role_prov: dict[str, set] = {}
        prov_any: set = set()
        for name, value in args.items():
            role = self.table.role(fn, name)
            atoms = atoms_of(value)
            if role == "password" and not atoms:
                atoms = [Atom("str", "", "" if value is None else str(value))]
            facts = []
            for a in atoms:
                p = "not_found" if (a.kind == "str" and a.value == "") else self.provenance(a, qsrc, named, other)
                facts.append(AtomFact(a.shown, a.kind, p))
                role_prov.setdefault(role, set()).add(p)
                prov_any.add(p)
            arg_facts.append(ArgFact(name, role, facts))
        return Facts(function=fn, kind=kind, args=arg_facts, role_prov=role_prov, prov_any=prov_any, n_steps=n,
                     done=dict(done), named_source_read=bool(named), named_calls=named_fns,
                     query_sha256=sha256_text(query))


def facts_from_json(d: dict) -> Facts:
    """Facts rebuilt from Facts.to_json() (rows, guards.jsonl): every field fact_bits_on() reads is restored exactly;
    atom texts are the logged ones (cut at 200 characters)."""
    args = [ArgFact(a["name"], a["role"], [AtomFact(t["shown"], t["kind"], t["prov"]) for t in a["atoms"]])
            for a in d["args"]]
    return Facts(function=d["function"], kind=d["kind"], args=args,
                 role_prov={r: set(v) for r, v in d["role_prov"].items()}, prov_any=set(d["prov_any"]),
                 n_steps=d["n_steps"], done=dict(d["done"]), named_source_read=d["named_source_read"],
                 named_calls=list(d["named_calls"]), query_sha256=d["query_sha256"])

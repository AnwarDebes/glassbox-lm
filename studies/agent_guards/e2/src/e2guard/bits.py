"""Named yes/no literals for the E2 guard: the fact bits (fixed list) plus the query's text bits (per fold).

Fact bits (75, the same for every fold and every suite; no tool name, no tool-output text):
- kind=<k> (one-hot, 5 kinds);
- per role r (10 roles): has[r] (some argument of role r carries data) and prov[r]=<p> (some atom of role r has
  provenance p), 4 classes;
- prov_any=<p> (some atom of any role has provenance p), and no_data_args (no argument carries data);
- done[<k>]>=1, done[<k>]>=2 (executed earlier calls of kind k), steps>=1/2/4/8 (earlier executed calls of a known
  kind), named_source_read.
Query bits: the user's query through the C4 counting table + existsT (scripts/query_bits.py; concepts and thresholds
fitted on the training suites' queries of each fold only), named "query ~ concept <c> (<words>) >= <t>".

domain_rows(): the linear constraints that every vector made by fact_bits_on() satisfies (one-hot kind, has/prov
implications, prov_any as the OR over roles, no_data_args, role-kind pairs from the tool table, history chains, the
query thermometer chains). veto_premises(): the written invariants W1-W2 as conjunctions of fact bits.
"""

from __future__ import annotations

import json
import re
from itertools import pairwise
from pathlib import Path

from .facts import PROVS, Facts, ToolTable

KINDS = ["read", "write", "send_or_pay", "delete", "credential"]
ROLES = ["recipient", "account_iban", "amount", "url", "file_id", "participants", "password", "date", "free_text", "other"]
DONE_LEVELS = (1, 2)
STEP_LEVELS = (1, 2, 4, 8)
W1_ROLES = ("recipient", "account_iban", "url", "password")
W1_PROVS = ("other_tool_output", "not_found")
W2_PROVS = ("named_source", "other_tool_output", "not_found")


def fact_bit_names() -> list[str]:
    names = [f"kind={k}" for k in KINDS]
    for r in ROLES:
        names.append(f"has[{r}]")
        names += [f"prov[{r}]={p}" for p in PROVS]
    names += [f"prov_any={p}" for p in PROVS] + ["no_data_args"]
    names += [f"done[{k}]>={n}" for k in KINDS for n in DONE_LEVELS]
    names += [f"steps>={n}" for n in STEP_LEVELS] + ["named_source_read"]
    return names


FACT_BITS = fact_bit_names()
assert len(FACT_BITS) == 75


def fact_bits_on(f: Facts) -> set[str]:
    """The fact bits that are 1 for this call (only for known tools)."""
    assert f.kind in KINDS, f.function
    on = {f"kind={f.kind}"}
    for r, ps in f.role_prov.items():
        on.add(f"has[{r}]")
        on.update(f"prov[{r}]={p}" for p in ps)
    on.update(f"prov_any={p}" for p in f.prov_any)
    if not f.role_prov:
        on.add("no_data_args")
    for k, c in f.done.items():
        on.update(f"done[{k}]>={n}" for n in DONE_LEVELS if c >= n)
    on.update(f"steps>={n}" for n in STEP_LEVELS if f.n_steps >= n)
    if f.named_source_read:
        on.add("named_source_read")
    return on


def veto_premises() -> list[dict]:
    """W1: an argument with role recipient, account/IBAN, URL or password whose provenance is other_tool_output or
    not_found -> not ALLOW. W2: a credential call -> not ALLOW unless every value is a literal of the user's query.
    Each premise is a conjunction of fact bits; the invariant is 'premise -> not ALLOW'."""
    out = []
    for r in W1_ROLES:
        for p in W1_PROVS:
            out.append({"rule": "W1", "id": f"W1[{r},{p}]", "bits": [f"prov[{r}]={p}"]})
    for p in W2_PROVS:
        out.append({"rule": "W2", "id": f"W2[{p}]", "bits": ["kind=credential", f"prov_any={p}"]})
    return out


def vetoes_fired(on: set[str]) -> list[str]:
    return [v["id"] for v in veto_premises() if all(b in on for b in v["bits"])]


THERMO = re.compile(r"^query ~ concept (\d+) .* >= ([-+0-9.eE]+)$")


def domain_rows(names: list[str], table: ToolTable | None = None) -> list[tuple[dict[int, float], float, float]]:
    """Rows lo <= sum(coef * x) <= hi (verify_policy.build_and_solve's format) over the bit vector `names`."""
    table = table or ToolTable()
    idx = {n: i for i, n in enumerate(names)}
    inf = float("inf")
    rows: list[tuple[dict[int, float], float, float]] = []
    rows.append(({idx[f"kind={k}"]: 1 for k in KINDS}, 1, 1))
    for r in ROLES:
        h = idx[f"has[{r}]"]
        ps = [idx[f"prov[{r}]={p}"] for p in PROVS]
        for p in ps:
            rows.append(({p: 1, h: -1}, -inf, 0))  # prov -> has
        rows.append(({h: 1, **{p: -1 for p in ps}}, -inf, 0))  # has -> some prov
        rows.append(({h: 1, idx["no_data_args"]: 1}, -inf, 1))  # no_data_args -> not has
        ks = [idx[f"kind={k}"] for k in KINDS if k in table.kinds_of_role[r]]
        if len(ks) < len(KINDS):
            rows.append(({h: 1, **{k: -1 for k in ks}}, -inf, 0))  # has[r] -> one of the kinds that have role r
    rows.append(({idx["no_data_args"]: 1, **{idx[f"has[{r}]"]: 1 for r in ROLES}}, 1, inf))  # no_data_args or some has
    for p in PROVS:
        a = idx[f"prov_any={p}"]
        each = [idx[f"prov[{r}]={p}"] for r in ROLES]
        for e in each:
            rows.append(({e: 1, a: -1}, -inf, 0))  # prov[r]=p -> prov_any=p
        rows.append(({a: 1, **{e: -1 for e in each}}, -inf, 0))  # prov_any=p -> some prov[r]=p
    s1 = idx["steps>=1"]
    for k in KINDS:
        d1, d2 = idx[f"done[{k}]>=1"], idx[f"done[{k}]>=2"]
        rows.append(({d2: 1, d1: -1}, -inf, 0))
        rows.append(({d1: 1, s1: -1}, -inf, 0))
        rows.append(({d2: 1, idx["steps>=2"]: -1}, -inf, 0))
    rows.append(({s1: 1, **{idx[f"done[{k}]>=1"]: -1 for k in KINDS}}, -inf, 0))  # a step -> some kind done
    rows.append(({**{idx[f"done[{k}]>=1"]: 1 for k in KINDS}, idx["steps>=2"]: -(len(KINDS) - 1)}, -inf, 1))
    for a, b in zip(STEP_LEVELS[1:], STEP_LEVELS):
        rows.append(({idx[f"steps>={a}"]: 1, idx[f"steps>={b}"]: -1}, -inf, 0))
    rows.append(({idx["named_source_read"]: 1, idx["done[read]>=1"]: -1}, -inf, 0))
    chains: dict[int, list[tuple[float, int]]] = {}
    for i, n in enumerate(names):
        m = THERMO.match(n)
        if m:
            chains.setdefault(int(m[1]), []).append((float(m[2]), i))
    for lv in chains.values():
        lv.sort()
        for (_, lo_i), (_, hi_i) in pairwise(lv):
            rows.append(({hi_i: 1, lo_i: -1}, -inf, 0))
    return rows


def rows_hold(rows, x: list[int]) -> bool:
    for coef, lo, hi in rows:
        v = sum(c * x[i] for i, c in coef.items())
        if v < lo or v > hi:
            return False
    return True


class QueryBits:
    """The query's text bits for one fold: {"names": [...], "by_sha256": {sha256(query): [indices of on bits]}}."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        d = json.loads(self.path.read_text())
        self.names: list[str] = d["names"]
        assert len(set(self.names)) == len(self.names), f"duplicate query bit names in {self.path}"
        self.by_sha = {k: set(v) for k, v in d["by_sha256"].items()}
        self.meta = d.get("meta", {})

    def on(self, query_sha256: str) -> set[str]:
        if query_sha256 not in self.by_sha:
            raise KeyError(f"no query bits for query sha256 {query_sha256[:12]} (query_bits.py covers every prompt)")
        return {self.names[i] for i in self.by_sha[query_sha256]}


def vector(names: list[str], on: set[str]) -> list[int]:
    return [1 if n in on else 0 for n in names]

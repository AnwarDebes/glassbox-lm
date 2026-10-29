"""A frozen wall clock for AgentDojo v1.2.2's tools (rev2; found by the offline smoke test, 2026-10-21).

AgentDojo stamps a sent e-mail (`email_client.send_email`) and a created or appended file
(`cloud_drive_client.create_file`, `append_to_file`) with `datetime.datetime.now()`, and that text is part of the tool
output the agent reads next. Without a fixed clock, two runs of the same episode send different request bodies after
the first such call, even with no veto: the paired replay of DESIGN 1.7 breaks (every arm's later requests are new),
and E2-1 does not replay E2-0 byte for byte. With the clock frozen, a tool output depends only on the episode itself.

- FIXED = 2026-10-21 00:00:00, a constant (not a counter: upstream caches the ground-truth pass of
  `get_injection_candidates`, which also calls these tools, so a counter would depend on the job order). It is later
  than every date in the v1.2.2 environments, so the one check that reads it (workspace: a file's `last_modified`
  grew after `append_to_file`) keeps its meaning; tests/test_rev2.py checks both.
- Only the module attribute `datetime` of the two tool modules is replaced by a copy of the `datetime` module whose
  `datetime.now()` returns FIXED; every other use (isinstance, arithmetic, parsing, pydantic fields) is unchanged.
- Installed by every entry point that runs AgentDojo episodes (the GPU driver, make_rows.py, offline_e2e.py,
  standin_traces.py). Rows made before rev2 used the real clock; their facts do not depend on it: make_rows.py rerun
  with the clock (2026-10-21 06:12:57-06:14:41Z) reproduced data/rows_gt_v1.2.2.jsonl byte for byte (LOG.md, rev2).
"""

from __future__ import annotations

import datetime as _dt
import types as _types

FIXED = _dt.datetime(2026, 10, 21, 0, 0, 0)  # noqa: DTZ001 (naive, like every date of the v1.2.2 environments)
_NAME = "datetime_frozen_e2"


class _FrozenDateTime(_dt.datetime):
    @classmethod
    def now(cls, tz=None):  # the tools call now() without a zone
        return FIXED if tz is None else FIXED.replace(tzinfo=tz)

    @classmethod
    def today(cls):
        return FIXED


def _frozen_module() -> _types.ModuleType:
    m = _types.ModuleType(_NAME)
    for k in dir(_dt):
        if not k.startswith("__"):
            setattr(m, k, getattr(_dt, k))
    m.datetime = _FrozenDateTime
    return m


def _modules():
    from agentdojo.default_suites.v1.tools import cloud_drive_client, email_client

    return (email_client, cloud_drive_client)


def install() -> _dt.datetime:
    for mod in _modules():
        if getattr(mod.datetime, "__name__", "") != _NAME:
            mod.datetime = _frozen_module()
    return FIXED


def installed() -> bool:
    return all(getattr(mod.datetime, "__name__", "") == _NAME for mod in _modules())

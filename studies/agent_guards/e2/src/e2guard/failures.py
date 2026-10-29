"""Which request failures stop a stage (registration rev3, section 13: failures after E2-0; the decision 3).

- A TRANSPORT error stops the stage at the first one: the driver (or build_d6.py) writes STOPPED.json, exits with
  EXIT_TRANSPORT (3) and no store-chain record is written for the stage. A stopped stage is rerun once in full on the
  same store (e2_check.py --rerun accepts a store that extends the last chain record only for that logged rerun).
  Transport errors are: openai.APIConnectionError and its subclass APITimeoutError, any HTTP status >= 500 (openai's
  InternalServerError and every APIStatusError with status_code >= 500; urllib's HTTPError with code >= 500), and the
  urllib / socket failures of the plain-HTTP clients (the A3 optimizer, the D6 rule writer): URLError without an HTTP
  status (e.g. connection refused), http.client.HTTPException (e.g. RemoteDisconnected), ConnectionError, TimeoutError.
  A request reaches this check only after its retries (the agent: tenacity's 3 tries in the sibling's _request).
- Everything else (a guard or harness exception, an HTTP 4xx such as a BadRequestError) falls under the ERROR RULE
  (utility False, attack success True; report_e2.py) and the stage goes on.
Standard library only (both venvs import it); openai's classes are recognised by name and status code, so this module
needs no openai import.
"""

from __future__ import annotations

import http.client
import socket
import urllib.error

EXIT_TRANSPORT = 3
TRANSPORT_CLASS_NAMES = {"APIConnectionError", "APITimeoutError", "InternalServerError"}


def http_status(e: BaseException) -> int | None:
    """The HTTP status an exception carries (openai: status_code; urllib: code), or None."""
    for attr in ("status_code", "code"):
        v = getattr(e, attr, None)
        if isinstance(v, int) and 100 <= v <= 599:
            return v
    return None


def is_transport(e: BaseException) -> bool:
    """True for the failures that stop a stage (see the module docstring); False for those under the error rule."""
    names = {c.__name__ for c in type(e).__mro__}
    if names & TRANSPORT_CLASS_NAMES:
        return True
    status = http_status(e)
    if isinstance(e, urllib.error.HTTPError) or "APIStatusError" in names:
        return status is not None and status >= 500
    if status is not None:
        return status >= 500
    if isinstance(e, urllib.error.URLError):  # no HTTP status: the connection itself failed (refused, reset, DNS)
        return True
    return isinstance(e, (http.client.HTTPException, ConnectionError, TimeoutError, socket.timeout))


def error_type(e: BaseException) -> str:
    """'<class>' or '<class> <status>' (the open part of an error record: no message text)."""
    s = http_status(e)
    return type(e).__name__ + (f" {s}" if s is not None else "")


class StageStopped(Exception):
    """Raised by the episode loops on the first transport error; the entry point turns it into EXIT_TRANSPORT."""

    def __init__(self, where: dict, cause: BaseException) -> None:
        self.where = dict(where)
        self.cause_type = error_type(cause)
        self.cause_text = f"{type(cause).__name__}: {cause}"[:500]
        super().__init__(f"transport error ({self.cause_type}) at {self.where}: the stage stops")

    def record(self) -> dict:
        return {"rule": "transport error: the stage stops at the first one; no chain record; rerun once in full on "
                        "the same store (registration section 13)",
                "where": self.where, "error_type": self.cause_type, "error": self.cause_text}

"""A silent AgentDojo logger with an explicit log directory.

AgentDojo's TraceLogger (logging.py:153-258) writes one JSON file per run to its delegate's `logdir`. Without an
entered logger, Logger.get() returns a fresh NullLogger that has no `logdir` attribute, and an entered NullLogger has
logdir None, which makes TraceLogger fall back to <site-packages>/../../runs (logging.py:166). Entering QuietLogger
pins the directory and keeps the console quiet (OutputLogger prints every message).
"""

from __future__ import annotations

from agentdojo.logging import Logger


class QuietLogger(Logger):
    def __init__(self, logdir: str) -> None:
        self.logdir = logdir
        self.messages = []
        self.errors: list[str] = []

    def log(self, *args, **kwargs) -> None:
        pass

    def log_error(self, error: str, *args, **kwargs) -> None:
        self.errors.append(error)

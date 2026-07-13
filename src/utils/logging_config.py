"""Logging setup.

Two sinks with independent levels:
  * stderr — carries only warnings/errors by default, so CLI stdout stays clean for answers.
  * per-session file (``logs/<session_id>/session.log``) — captures INFO-level trace records
    (tools, retrieval scores, memory, context summary, backend, timing) plus warnings/errors,
    so any turn can be backtracked. Enabled per CLI session via ``start_session_log``.

The root logger is set to DEBUG and each handler filters to its own level, so adding the
session file handler captures INFO regardless of the (quieter) console level.
"""

import logging
import sys
from pathlib import Path

_CONFIGURED = False
_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: str = "WARNING") -> None:
    """Configure the root logger once. ``level`` controls the STDERR handler only."""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        return
    root.setLevel(logging.DEBUG)  # capture everything; handlers decide what to emit
    root.handlers.clear()
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setLevel(getattr(logging, level.upper(), logging.WARNING))
    stderr.setFormatter(logging.Formatter(_FMT, "%H:%M:%S"))
    stderr.set_name("stderr")
    root.addHandler(stderr)
    _CONFIGURED = True


def start_session_log(session_id: str, base_dir: str = "logs") -> tuple[Path, logging.Handler]:
    """Attach an INFO-level file handler for one CLI session. Returns (dir, handler)."""
    setup_logging()  # ensure the console handler exists
    d = Path(base_dir) / session_id
    d.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(d / "session.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(_FMT, "%H:%M:%S"))
    fh.set_name(f"session:{session_id}")
    logging.getLogger().addHandler(fh)
    logging.getLogger("pipeline").info("session %s started", session_id)
    return d, fh


def stop_session_log(handler: logging.Handler | None) -> None:
    if handler is None:
        return
    logging.getLogger("pipeline").info("session log closed")
    logging.getLogger().removeHandler(handler)
    handler.close()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

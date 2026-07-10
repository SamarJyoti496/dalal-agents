"""
logging_config.py — per-run debug log for the pipeline.

Every agent iteration (tool calls, malformed JSON, validation errors, raw LLM
text) is logged to a timestamped file so a failure like "exhausted 8
iterations" can be diagnosed after the fact instead of just printing a
one-line summary to the console.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from dalal_agents.config import LOG_LEVEL

LOG_DIR: Path = Path(__file__).resolve().parent.parent / "logs"


def setup_logging(run_label: str, level: Optional[int] = None) -> Path:
    """Configure the 'dalal_agents' logger to write to logs/<run_label>_<ts>.log.

    Log verbosity comes from the LOG_LEVEL env var (DEBUG/INFO/WARNING/ERROR,
    default INFO) unless `level` is passed explicitly. Returns the log file
    path so the CLI can tell the user where to look.
    """
    if level is None:
        level = getattr(logging, LOG_LEVEL, logging.INFO)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"{run_label}_{ts}.log"

    logger = logging.getLogger("dalal_agents")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(fh)

    return log_file

"""
Thread-local runtime helpers for network and SQLite resources.

These helpers reduce per-task setup overhead in the parallel pipeline by
reusing one HTTP session / SQLite connection per worker thread.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Optional


_RUNTIME_LOCAL = threading.local()


def _session_cache() -> dict:
    cache = getattr(_RUNTIME_LOCAL, "http_sessions", None)
    if cache is None:
        cache = {}
        _RUNTIME_LOCAL.http_sessions = cache
    return cache


def _sqlite_cache() -> dict:
    cache = getattr(_RUNTIME_LOCAL, "sqlite_conns", None)
    if cache is None:
        cache = {}
        _RUNTIME_LOCAL.sqlite_conns = cache
    return cache


def get_thread_http_session(namespace: str = "default"):
    """Return one requests.Session per thread + namespace."""
    import requests

    cache = _session_cache()
    session = cache.get(namespace)
    if session is None:
        session = requests.Session()
        cache[namespace] = session
    return session


def get_thread_sqlite_connection(
    db_path: str,
    *,
    timeout: float = 30.0,
    row_factory=None,
    enable_wal: bool = False,
    busy_timeout_ms: int = 30000,
) -> sqlite3.Connection:
    """Return one SQLite connection per thread + db path."""
    resolved = str(Path(db_path).resolve())
    cache_key = (resolved, float(timeout), bool(enable_wal))
    cache = _sqlite_cache()
    conn: Optional[sqlite3.Connection] = cache.get(cache_key)

    if conn is None:
        conn = sqlite3.connect(resolved, timeout=timeout)
        conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        if enable_wal:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
        cache[cache_key] = conn

    if row_factory is not None:
        conn.row_factory = row_factory

    return conn

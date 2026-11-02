"""``data/distill.db``: every record, every model's output on it, every score.

One SQLite file, JSON in columns, so the dashboard and the exporter read one place.
The engine input of each record is kept as a pickle beside it, for edits.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.getenv("DISTILL_DB", str(ROOT / "data" / "distill.db")))

_SCHEMA = """
create table if not exists records (
  id text primary key,
  task text not null,            -- interview | assess | study
  track text,
  split text not null,           -- train | val | test | test_in_dist
  source text not null,          -- sim | fork | edit | pair
  question_id integer,
  session_id text,
  doc text not null,             -- the record as JSON
  agent_input blob,              -- pickled engine input, for edits
  created_at real not null
);
create index if not exists records_split on records(split, task);
create table if not exists sessions (
  id text primary key, split text, task text, doc text not null, created_at real not null
);
create table if not exists runs (
  record_id text not null, model text not null, doc text not null, created_at real not null,
  primary key (record_id, model)
);
create table if not exists scores (
  record_id text not null, model text not null, kind text not null, doc text not null,
  created_at real not null,
  primary key (record_id, model, kind)
);
create table if not exists spend (
  at real not null, provider text not null, model text not null, role text not null,
  tokens_in integer, tokens_out integer, usd real
);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=60, isolation_level=None)
    conn.execute("pragma journal_mode=wal")
    conn.executescript(_SCHEMA)
    return conn


def put_record(conn, rec: Dict[str, Any], *, task: str, split: str, source: str,
               track: Optional[str] = None) -> None:
    blob = rec.get("_agent_input")
    doc = {k: v for k, v in rec.items() if k != "_agent_input"}
    doc.update({"task": task, "split": split, "source": source, "track": track})
    conn.execute(
        "insert or replace into records values (?,?,?,?,?,?,?,?,?,?)",
        (doc["id"], task, track, split, source, doc.get("question_id"), doc.get("session_id"),
         json.dumps(doc, default=str), blob, time.time()),
    )


def put_session(conn, out: Dict[str, Any], *, split: str, task: str) -> None:
    doc = {k: v for k, v in out.items() if k != "records"}
    doc["record_ids"] = [r["id"] for r in out.get("records", [])]
    conn.execute("insert or replace into sessions values (?,?,?,?,?)",
                 (out["session_id"], split, task, json.dumps(doc, default=str), time.time()))


def records(conn, *, split: Optional[str] = None, task: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    q, args = "select doc from records where 1=1", []
    if split:
        q += " and split = ?"; args.append(split)
    if task:
        q += " and task = ?"; args.append(task)
    for (doc,) in conn.execute(q + " order by id", args):
        yield json.loads(doc)


def agent_input(conn, record_id: str):
    import pickle

    row = conn.execute("select agent_input from records where id = ?", (record_id,)).fetchone()
    return pickle.loads(row[0]) if row and row[0] else None


def put_run(conn, record_id: str, model: str, doc: Dict[str, Any]) -> None:
    conn.execute("insert or replace into runs values (?,?,?,?)",
                 (record_id, model, json.dumps(doc, default=str), time.time()))


def has_run(conn, record_id: str, model: str) -> bool:
    """A finished call. A call that failed (rate limit, timeout) is run again."""
    row = conn.execute("select doc from runs where record_id=? and model=?",
                       (record_id, model)).fetchone()
    return row is not None and not json.loads(row[0]).get("error")


def put_score(conn, record_id: str, model: str, kind: str, doc: Dict[str, Any]) -> None:
    conn.execute("insert or replace into scores values (?,?,?,?,?)",
                 (record_id, model, kind, json.dumps(doc, default=str), time.time()))


def log_spend(conn, provider: str, model: str, role: str, tokens_in: int, tokens_out: int,
              usd: Optional[float]) -> None:
    conn.execute("insert into spend values (?,?,?,?,?,?,?)",
                 (time.time(), provider, model, role, tokens_in, tokens_out, usd))


def spent(conn, provider: Optional[str] = None) -> float:
    q, args = "select coalesce(sum(usd),0) from spend", []
    if provider:
        q += " where provider = ?"; args.append(provider)
    return float(conn.execute(q, args).fetchone()[0])

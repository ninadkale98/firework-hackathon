"""The questions the dataset runs on, as a local bank the real engine reads.

``fetch`` copies the chosen rows and every entity they reference out of the
production database, read-only, into ``data/bank/``. ``load`` builds the same
``QuestionManager`` the engine uses in production, over those files plus the local
fixtures in ``questions/new/``. Nothing here writes to production.

Run ``python -m distill.bank fetch`` from this folder with ``backend`` on
``PYTHONPATH``.
"""

from __future__ import annotations

import json
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

ROOT = Path(__file__).resolve().parent.parent
BANK_DIR = ROOT / "data" / "bank"
NEW_DIR = ROOT / "questions" / "new"

#: Local fixtures get ids from here up, so they can never collide with a real row.
LOCAL_ID_BASE = 900_000

#: The question list agreed on 2026-11-01 (PLAN.md §10d). Fixtures are named by file.
SPLITS: Dict[str, Dict[str, List[Any]]] = {
    "embedded": {"train": [561, 562, 563, 565, 566, 584, 586], "test": [568, 567]},
    "frontend": {"train": [533, 534, 518, "new-fe-1", "new-fe-2"], "test": [535, "new-fe-3"]},
    "backend": {"train": [611, 426, 589, 588, "new-be-1"], "test": [427, 490, 517]},
    "ai_eng": {"train": [493, 494, 496, 498, 501, 507, 514], "test": [502, 503, 606]},
    "learn": {
        "train": [614, 615, 616, 617, 618, 620, 621, 622, 624, 626, 627, 628],
        "test": [619, 623, 625],
    },
}


def all_ids() -> List[int]:
    return sorted({q for track in SPLITS.values() for ids in track.values() for q in ids
                   if isinstance(q, int)})


def _entity_ids(node: Any) -> Iterable[int]:
    """Every entity id a question references, wherever it sits in the tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("entity_refs", "entities") and isinstance(value, list):
                for ref in value:
                    if isinstance(ref, dict) and isinstance(ref.get("id"), int):
                        yield ref["id"]
                    elif isinstance(ref, int):
                        yield ref
            yield from _entity_ids(value)
    elif isinstance(node, list):
        for item in node:
            yield from _entity_ids(item)


def fetch(ids: Iterable[int]) -> None:
    """Copy rows and their entities out of production, read-only."""
    from dotenv import load_dotenv
    import psycopg2
    from psycopg2.extras import RealDictCursor

    load_dotenv(ROOT.parent.parent / "backend" / ".env")
    conn = psycopg2.connect(
        host=os.environ["PGHOST"], port=os.environ.get("PGPORT", 5432),
        user=os.environ["PGUSER"], password=os.environ["PGPASSWORD"],
        dbname=os.environ["PGDATABASE"], sslmode="require",
    )
    conn.set_session(readonly=True)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    for qid in ids:
        cur.execute("select * from questions where id = %s", (qid,))
        row = cur.fetchone()
        if row is None:
            print(f"missing question {qid}", file=sys.stderr)
            continue
        refs = sorted(set(_entity_ids(row)))
        entities = []
        if refs:
            cur.execute("select * from entities where id = any(%s)", (refs,))
            entities = cur.fetchall()
        (BANK_DIR / f"q{qid}.json").write_text(
            json.dumps({"question": row, "entities": entities}, indent=2, default=str)
        )
        print(f"q{qid}: {len(entities)} entities")


# ── loading ─────────────────────────────────────────────────────────────


class _Db:
    """The slice of the database interface the question storages use."""

    def __init__(self, questions, entities):
        self.questions = {int(q.id): q for q in questions}
        self.entities = {int(e.id): e for e in entities}

    def get_by_id(self, table_name, model_class, id_field, id_value):
        source = self.questions if table_name == "questions" else self.entities
        return source.get(int(id_value))

    def execute_query(self, query, params=None):
        lowered = " ".join(query.lower().split())
        if "from entities where id = any" in lowered:
            wanted = {int(i) for i in params[0]}
            return [e.model_dump() for e in self.entities.values() if int(e.id) in wanted]
        return []


BLOB_DIR = BANK_DIR / "blobs"


def fetch_blobs() -> None:
    """Each entity's stored file, downloaded read-only, so the candidate is shown
    exactly what a real candidate sees (a repo is a zip, not its text)."""
    from dotenv import load_dotenv

    load_dotenv(ROOT.parent.parent / "backend" / ".env")
    from app.config.config import get_config
    from app.storage.file_storage.azure_blob_storage import AzureBlobStorage

    store = AzureBlobStorage()
    container = get_config().QUESTION_ENTITY_BUCKET
    BLOB_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(BANK_DIR.glob("q*.json")):
        for ent in json.loads(path.read_text())["entities"]:
            target = BLOB_DIR / str(ent["id"])
            if target.exists():
                continue
            try:
                target.write_bytes(store.download_file(container, str(ent["id"])).read())
            except Exception as exc:
                print(f"entity {ent['id']} ({ent.get('type')}): {exc!r}"[:200], file=sys.stderr)


def _repo_zip(normalized: str) -> bytes:
    """A local fixture's repo as the zip production stores, rebuilt from the
    ``--- path ---`` sections of its normalized text."""
    import re
    import zipfile

    buf = BytesIO()
    parts = re.split(r"^--- (.+?) ---$", normalized, flags=re.M)
    with zipfile.ZipFile(buf, "w") as zf:
        for path, body in zip(parts[1::2], parts[2::2]):
            zf.writestr(f"repo/{path.strip()}", body.strip("\n") + "\n")
    return buf.getvalue()


class _Blob:
    """Downloaded entity bytes, else the entity's normalized text (a local repo
    fixture is zipped back up first)."""

    def __init__(self, texts: Dict[str, str], repos: Optional[set] = None):
        self.texts = texts
        self.repos = repos or set()

    def get_file_url(self, container, blob_name, expires_in_seconds=None):
        return f"https://local.bank/{blob_name}"

    def download_file(self, container, blob_name):
        path = BLOB_DIR / str(blob_name)
        if path.exists():
            return BytesIO(path.read_bytes())
        if str(blob_name) in self.repos:
            return BytesIO(_repo_zip(self.texts.get(str(blob_name), "")))
        return BytesIO(self.texts.get(str(blob_name), "").encode())


def _localise(fixture: Dict[str, Any], qid: int) -> Dict[str, Any]:
    """A fixture's string entity ids rewritten to numbers the models accept."""
    mapping: Dict[str, int] = {}
    for n, ent in enumerate(fixture.get("entities", []), start=1):
        mapping[str(ent["id"])] = qid * 100 + n

    def swap(node):
        # Any value that is exactly a local entity id ("e1") becomes its number,
        # whether it sits in an ``id`` field or a plain list of ids.
        if isinstance(node, dict):
            return {k: swap(v) for k, v in node.items()}
        if isinstance(node, list):
            return [swap(x) for x in node]
        if isinstance(node, str) and node in mapping:
            return mapping[node]
        return node

    question = swap(fixture["question"])
    question["id"] = qid
    entities = []
    for ent in fixture.get("entities", []):
        e = dict(ent)
        e["id"] = mapping[str(ent["id"])]
        e.setdefault("source", "author")
        e.setdefault("storage_path", f"question-entities/{e['id']}")
        e.setdefault("meta", {})
        entities.append(e)
    return {"question": question, "entities": entities}


def local_ids() -> Dict[str, int]:
    """Fixture name -> the local id it is served under."""
    names = sorted(p.stem for p in NEW_DIR.glob("*.json"))
    return {name: LOCAL_ID_BASE + n for n, name in enumerate(names, start=1)}


def resolve(ref: Any) -> int:
    return ref if isinstance(ref, int) else local_ids()[ref]


def load():
    """A ``QuestionManager`` over the fetched rows and the local fixtures."""
    from app.manager.question_manager import QuestionManager
    from app.models.question import Entity, Question
    from app.storage.database.question_storage import EntityStorage, QuestionStorage

    docs = [json.loads(p.read_text()) for p in sorted(BANK_DIR.glob("q*.json"))]
    for name, qid in local_ids().items():
        docs.append(_localise(json.loads((NEW_DIR / f"{name}.json").read_text()), qid))

    questions, entities, texts, repos = [], [], {}, set()
    for doc in docs:
        questions.append(Question.model_validate(doc["question"]))
        for ent in doc["entities"]:
            entity = Entity.model_validate(ent)
            entities.append(entity)
            texts[str(ent["id"])] = ent.get("normalized_form") or ""
            if ent.get("type") == "repo":
                repos.add(str(ent["id"]))
    db = _Db(questions, entities)
    manager = QuestionManager(
        question_storage=QuestionStorage(db),
        entity_storage=EntityStorage(db),
        blob_storage=_Blob(texts, repos),
    )
    manager._container = "question-entities"
    return manager


if __name__ == "__main__":
    if sys.argv[1:2] == ["fetch"]:
        fetch(all_ids())
        fetch_blobs()
    elif sys.argv[1:2] == ["blobs"]:
        fetch_blobs()
    else:
        manager = load()
        print("loaded", len(manager.questions.db.questions) if hasattr(manager.questions, "db") else "?")

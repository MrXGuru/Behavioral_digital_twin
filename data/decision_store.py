"""Persistence layer for the Behavioral Digital Twin's decision records.

This module defines :class:`DecisionStore`, a stable persistence boundary that loads
and stores :class:`~data.schema.DecisionRecord`s behind a single interface, backed by
either a CSV file or a SQLite database. Keeping persistence behind one interface lets the
synthetic data source be swapped for real data later without touching downstream layers.

Both backends use the canonical column ordering and schema version from
:mod:`data.schema`, so records round-trip identically regardless of backend:

* ``timestamp`` is serialized as ISO-8601 and parsed back to :class:`datetime`,
* ``mood_energy`` is serialized as text/real and parsed back to :class:`float`,
* all other fields are stored as text.

See "Component 2: DecisionStore" in the design document for the contract:

    Responsibilities:
      - Provide a stable persistence boundary so synthetic data can be swapped for real
        data.
      - Return records sorted by timestamp ascending.
"""

from __future__ import annotations

import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from data.schema import COLUMNS, SCHEMA_VERSION, DecisionRecord

#: Supported persistence backends.
CSV_BACKEND = "csv"
SQLITE_BACKEND = "sqlite"
_BACKENDS = (CSV_BACKEND, SQLITE_BACKEND)

#: SQLite table holding decision records.
_TABLE = "decisions"


def _record_to_row(record: DecisionRecord) -> dict[str, str]:
    """Serialize a :class:`DecisionRecord` to a dict of string cells.

    ``timestamp`` is serialized as ISO-8601 and ``mood_energy`` via ``repr`` so the
    float round-trips exactly; everything else is stored as ``str``.
    """
    return {
        "user_id": str(record.user_id),
        "timestamp": record.timestamp.isoformat(),
        "domain": str(record.domain),
        "location": str(record.location),
        "weather": str(record.weather),
        "day_type": str(record.day_type),
        "time_of_day": str(record.time_of_day),
        "mood_energy": repr(float(record.mood_energy)),
        "stress_level": str(getattr(record, "stress_level", "medium")),
        "decision_made": str(record.decision_made),
        "outcome": str(record.outcome),
        "source_mode": str(record.source_mode),
        "domain_category": str(record.domain_category) if getattr(record, "domain_category", None) else "",
        "duration_seconds": str(record.duration_seconds) if getattr(record, "duration_seconds", None) is not None else "",
    }


def _parse_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp string to a tz-aware UTC datetime.

    Normalizes all stored timestamps to UTC so naive and aware datetimes
    can always be sorted together without a TypeError.
    """
    ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        # Treat naive timestamps (old records) as UTC
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _row_to_record(row: dict[str, str]) -> DecisionRecord:
    """Parse a dict of string cells back into a typed :class:`DecisionRecord`.

    Inverse of :func:`_record_to_row`: ``timestamp`` is parsed back to a
    tz-aware UTC :class:`datetime` and ``mood_energy`` back to a ``float``.

    Backward compatibility: rows persisted before ``source_mode`` existed (CSV
    without the column, or SQLite reads where the value is absent/NULL) default
    to ``"synthetic"``.
    """
    source_mode = row.get("source_mode")
    if source_mode is None:
        source_mode = "synthetic"
    return DecisionRecord(
        user_id=row["user_id"],
        timestamp=_parse_timestamp(row["timestamp"]),
        domain=row["domain"],
        location=row["location"],
        weather=row["weather"],
        day_type=row["day_type"],
        time_of_day=row["time_of_day"],
        mood_energy=float(row["mood_energy"]),
        stress_level=row.get("stress_level") or "medium",
        decision_made=row["decision_made"],
        outcome=row["outcome"],
        source_mode=source_mode,
        domain_category=row.get("domain_category") or None,
        duration_seconds=int(row["duration_seconds"]) if row.get("duration_seconds") else None,
    )


class DecisionStore:
    """Persist and load :class:`DecisionRecord`s via a CSV or SQLite backend.

    Args:
        backend: One of ``"csv"`` or ``"sqlite"``.
        path: Filesystem path to the CSV file or SQLite database.

    Raises:
        ValueError: if ``backend`` is not a supported backend.
    """

    def __init__(self, backend: str, path: str) -> None:
        if backend not in _BACKENDS:
            raise ValueError(
                f"invalid backend {backend!r}; must be one of: {', '.join(_BACKENDS)}"
            )
        self.backend = backend
        self.path = Path(path)
        self.schema_version = SCHEMA_VERSION
        # Ensure the parent directory exists so the first append never fails.
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if backend == SQLITE_BACKEND:
            self._init_sqlite()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, records: list[DecisionRecord]) -> None:
        """Append ``records`` to the store, retaining any previously stored records."""
        if not records:
            return
        if self.backend == CSV_BACKEND:
            self._append_csv(records)
        else:
            self._append_sqlite(records)

    def load(
        self,
        user_id: str | None = None,
        since: datetime | None = None,
    ) -> list[DecisionRecord]:
        """Load matching records sorted ascending by timestamp.

        Args:
            user_id: If given, only records for this user are returned.
            since: If given, only records with ``timestamp >= since`` are returned.
                   Naive datetimes are treated as UTC for comparison purposes.

        Returns:
            Matching records sorted ascending by timestamp.
        """
        if self.backend == CSV_BACKEND:
            records = self._load_csv()
        else:
            records = self._load_sqlite()

        if user_id is not None:
            records = [r for r in records if r.user_id == user_id]
        if since is not None:
            # Normalize ``since`` to tz-aware UTC so it compares cleanly with the
            # tz-aware timestamps that ``_row_to_record`` now always produces.
            since_aware = since if since.tzinfo is not None else since.replace(tzinfo=timezone.utc)
            records = [r for r in records if r.timestamp >= since_aware]

        records.sort(key=lambda r: r.timestamp)
        return records

    def count(self, user_id: str | None = None) -> int:
        """Return the number of stored records, optionally filtered by ``user_id``."""
        return len(self.load(user_id=user_id))

    # ------------------------------------------------------------------
    # CSV backend
    # ------------------------------------------------------------------

    def _append_csv(self, records: list[DecisionRecord]) -> None:
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
            if write_header:
                writer.writeheader()
            for record in records:
                writer.writerow(_record_to_row(record))

    def _load_csv(self) -> list[DecisionRecord]:
        if not self.path.exists():
            return []
        with self.path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return [_row_to_record(row) for row in reader]

    # ------------------------------------------------------------------
    # SQLite backend
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self) -> None:
        columns_ddl = ", ".join(f"{col} TEXT" for col in COLUMNS)
        with self._connect() as conn:
            conn.execute(f"CREATE TABLE IF NOT EXISTS {_TABLE} ({columns_ddl})")
            # Backward compatibility: a table created by an older schema version may
            # be missing newer columns. Add any absent columns so existing DBs keep
            # working; pre-existing rows take the column default.
            existing = {
                str(row["name"])
                for row in conn.execute(f"PRAGMA table_info({_TABLE})").fetchall()
            }
            for col in COLUMNS:
                if col not in existing:
                    if col == "source_mode":
                        conn.execute(
                            f"ALTER TABLE {_TABLE} "
                            "ADD COLUMN source_mode TEXT DEFAULT 'synthetic'"
                        )
                    elif col == "stress_level":
                        conn.execute(
                            f"ALTER TABLE {_TABLE} "
                            "ADD COLUMN stress_level TEXT DEFAULT 'medium'"
                        )
                    elif col == "domain_category":
                        conn.execute(
                            f"ALTER TABLE {_TABLE} "
                            "ADD COLUMN domain_category TEXT"
                        )
                    elif col == "duration_seconds":
                        conn.execute(
                            f"ALTER TABLE {_TABLE} "
                            "ADD COLUMN duration_seconds TEXT"
                        )
                    else:
                        conn.execute(f"ALTER TABLE {_TABLE} ADD COLUMN {col} TEXT")
            conn.commit()

    def _append_sqlite(self, records: list[DecisionRecord]) -> None:
        placeholders = ", ".join("?" for _ in COLUMNS)
        column_list = ", ".join(COLUMNS)
        rows = [
            tuple(_record_to_row(record)[col] for col in COLUMNS) for record in records
        ]
        with self._connect() as conn:
            conn.executemany(
                f"INSERT INTO {_TABLE} ({column_list}) VALUES ({placeholders})", rows
            )
            conn.commit()

    def _load_sqlite(self) -> list[DecisionRecord]:
        column_list = ", ".join(COLUMNS)
        with self._connect() as conn:
            cursor = conn.execute(f"SELECT {column_list} FROM {_TABLE}")
            return [_row_to_record(dict(row)) for row in cursor.fetchall()]

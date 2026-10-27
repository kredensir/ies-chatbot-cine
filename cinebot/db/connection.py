"""Conexión y utilidades de acceso a SQLite.

Se usa la librería estándar `sqlite3` (sin ORM) para que quede explícito el
trabajo con SQL: DDL, consultas parametrizadas, transacciones, vistas,
triggers e índices.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from cinebot import config


class Database:
    """Fachada mínima sobre SQLite con consultas parametrizadas."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or config.DB_PATH)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Abre una conexión con claves foráneas activas y filas tipo dict."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        """Crea el esquema completo a partir de database/schema.sql."""
        schema = config.SCHEMA_PATH.read_text(encoding="utf-8")
        with self.connect() as connection:
            connection.executescript(schema)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        with self.connect() as connection:
            row = connection.execute(sql, params).fetchone()
        return row[0] if row else None

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Ejecuta una sentencia de escritura y devuelve el lastrowid."""
        with self.connect() as connection:
            cursor = connection.execute(sql, params)
            connection.commit()
            return int(cursor.lastrowid or 0)

    def execute_many(self, sql: str, rows: Iterable[Sequence[Any]]) -> int:
        """Ejecuta una lista de filas dentro de una única transacción."""
        with self.connect() as connection:
            cursor = connection.executemany(sql, rows)
            connection.commit()
            return int(cursor.rowcount or 0)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Transacción explícita: commit si todo sale bien, rollback si falla."""
        with self.connect() as connection:
            try:
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise

    def table_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for table in (
            "users",
            "items",
            "genres",
            "item_genres",
            "ratings",
            "conversations",
            "messages",
            "recommendations",
            "interactions",
            "model_metrics",
        ):
            counts[table] = int(self.scalar(f"SELECT COUNT(*) FROM {table}") or 0)
        return counts

    def health_check(self) -> dict[str, Any]:
        """Verifica tablas clave y devuelve estadísticas para la interfaz."""
        counts = self.table_counts()
        ok = counts["items"] > 0 and counts["users"] > 0
        return {
            "ok": ok,
            "path": str(self.path),
            "counts": counts,
            "sqlite_version": sqlite3.sqlite_version,
        }

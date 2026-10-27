"""Construcción inicial de la base de datos y verificación del entorno.

El flujo de puesta en marcha es idempotente:
    descargar datos -> catálogo unificado -> base SQLite -> entrenar modelos.

La aplicación puede arrancar con una base ya construida (incluida en el
repositorio) o reconstruirla con `python scripts/build_database.py`.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from cinebot import config
from cinebot.data.catalog import build_catalog, load_processed_catalog
from cinebot.db.connection import Database
from cinebot.db.repository import CatalogRepository, UserRepository


class BootstrapError(RuntimeError):
    """Error de puesta en marcha con instrucciones para el usuario."""


COLORS = [
    "#7C4DFF", "#FF6B6B", "#4ECDC4", "#FFD166", "#06D6A0",
    "#118AB2", "#EF476F", "#8338EC", "#FB5607", "#3A86FF",
]


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


def _seed_genres(db: Database, catalog: pd.DataFrame) -> dict[str, int]:
    names: set[str] = set()
    for value in catalog["genres"].fillna(""):
        for genre in str(value).split("|"):
            if genre.strip():
                names.add(genre.strip())
    with db.transaction() as connection:
        connection.executemany(
            "INSERT OR IGNORE INTO genres (name) VALUES (?)",
            [(name,) for name in sorted(names)],
        )
    rows = db.query("SELECT genre_id, name FROM genres")
    return {str(row["name"]): int(row["genre_id"]) for row in rows}


def _seed_items(db: Database, catalog: pd.DataFrame) -> None:
    genre_ids = _seed_genres(db, catalog)
    item_rows = []
    genre_rows = []
    for row in catalog.itertuples(index=False):
        item_rows.append(
            (
                int(row.item_id),
                int(row.tmdb_id) if pd.notna(row.tmdb_id) else None,
                _text(row.imdb_id),
                int(row.tvmaze_id) if pd.notna(row.tvmaze_id) else None,
                str(row.media_type),
                str(row.title),
                str(row.title_norm),
                int(row.year) if pd.notna(row.year) else None,
                int(row.end_year) if pd.notna(row.end_year) else None,
                _text(row.overview),
                _text(row.keywords),
                int(row.runtime) if pd.notna(row.runtime) else None,
                int(row.seasons) if pd.notna(row.seasons) else None,
                int(row.episodes) if pd.notna(row.episodes) else None,
                float(row.rating_avg),
                int(row.rating_count),
                float(row.bayes_rating),
                float(row.popularity),
                _text(row.source),
                _text(row.source_url),
                _text(row.poster_url),
            )
        )
        for genre in str(row.genres).split("|"):
            genre = genre.strip()
            if genre and genre in genre_ids:
                genre_rows.append((int(row.item_id), genre_ids[genre]))

    with db.transaction() as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO items (
                item_id, tmdb_id, imdb_id, tvmaze_id, media_type, title,
                title_norm, year, end_year, overview, keywords, runtime, seasons,
                episodes, rating_avg, rating_count, bayes_rating, popularity,
                source, source_url, poster_url
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            item_rows,
        )
        connection.executemany(
            "INSERT OR IGNORE INTO item_genres (item_id, genre_id) VALUES (?, ?)",
            genre_rows,
        )
    print(f"[OK] {len(item_rows):,} ítems y {len(genre_rows):,} relaciones de género")


def _seed_users(db: Database, n_demo: int = 610) -> None:
    rows = []
    for user_id in range(1, n_demo + 1):
        rows.append(
            (
                user_id,
                f"Perfil ML {user_id:03d}",
                "movielens",
                1,
                COLORS[(user_id - 1) % len(COLORS)],
            )
        )
    # El usuario local se crea con un id reservado para no chocar con MovieLens.
    rows.append(
        (
            100001,
            config.DEFAULT_USER_NAME,
            "local",
            0,
            "#7C4DFF",
        )
    )
    with db.transaction() as connection:
        connection.executemany(
            """
            INSERT OR IGNORE INTO users (
                user_id, username, source, is_demo, avatar_color
            ) VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
    print(f"[OK] {len(rows)} usuarios (610 históricos + usuario local)")


def _seed_ratings(db: Database, ratings_path: Path | None = None) -> None:
    path = Path(ratings_path or config.RATINGS_CSV)
    if not path.exists():
        raise BootstrapError(
            f"No existe {path}. Ejecute: python scripts/download_data.py"
        )
    ratings = pd.read_csv(path, dtype={"userId": "int64", "movieId": "int64"})
    rows = []
    for row in ratings.itertuples(index=False):
        rated_at = datetime.fromtimestamp(int(row.timestamp), tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        rows.append(
            (int(row.userId), int(row.movieId), float(row.rating), rated_at, "movielens")
        )
    with db.transaction() as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO ratings (user_id, item_id, rating, rated_at, source)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
    print(f"[OK] {len(rows):,} valoraciones históricas importadas")


def build_database(force: bool = False, verbose: bool = True) -> Database:
    """Crea el esquema y carga catálogo, usuarios y valoraciones."""
    config.ensure_directories()
    db = Database(config.DB_PATH)

    if force and config.DB_PATH.exists():
        config.DB_PATH.unlink()
        for suffix in ("-wal", "-shm"):
            Path(str(config.DB_PATH) + suffix).unlink(missing_ok=True)

    if not config.DB_PATH.exists():
        print(f"[..] Creando base de datos en {config.DB_PATH}")
        db.initialize()
    else:
        # Si el archivo existe pero está incompleto, se reconstruye el esquema.
        try:
            has_items = int(db.scalar("SELECT COUNT(*) FROM items") or 0) > 0
        except Exception:
            has_items = False
        if not has_items:
            print("[..] Base de datos vacía: creando esquema")
            db.initialize()

    catalog_repo = CatalogRepository(db)
    user_repo = UserRepository(db)
    stats = catalog_repo.stats()

    if stats["items"] == 0:
        if not config.CATALOG_CSV.exists():
            print("[..] Construyendo catálogo unificado")
            build_catalog(save=True)
        catalog = load_processed_catalog()
        _seed_items(db, catalog)
    else:
        print(f"[OK] Catálogo ya cargado ({stats['items']:,} ítems)")

    n_users = int(db.scalar("SELECT COUNT(*) FROM users") or 0)
    if n_users < 611:
        _seed_users(db)
    else:
        print(f"[OK] Usuarios ya cargados ({n_users})")

    n_ratings = int(
        db.scalar("SELECT COUNT(*) FROM ratings WHERE source = 'movielens'") or 0
    )
    if n_ratings == 0:
        _seed_ratings(db)
    else:
        print(f"[OK] Valoraciones históricas ya cargadas ({n_ratings:,})")

    user_repo.get_or_create_guest()
    db.execute("ANALYZE")
    stats = catalog_repo.stats()
    if verbose:
        print(
            f"[OK] Base lista: {stats['items']:,} ítems "
            f"({stats['movies']:,} películas, {stats['series']} series), "
            f"{stats['ratings']:,} valoraciones, {stats['genres']} géneros"
        )
    return db


def ensure_database() -> Database:
    """Verifica que la base esté lista e intenta crearla si falta."""
    db = Database(config.DB_PATH)
    try:
        health = db.health_check()
    except Exception:
        health = {"ok": False, "counts": {}}
    if not health.get("ok"):
        print("[..] Base de datos no encontrada o incompleta; construyendo...")
        return build_database()
    return db


def ensure_models() -> None:
    """Comprueba que existan los artefactos de los modelos entrenados."""
    missing = [
        name
        for name in ("content", "collaborative", "intent", "clusters")
        if not config.model_path(name).exists()
    ]
    if missing:
        raise BootstrapError(
            "Faltan modelos entrenados: "
            + ", ".join(missing)
            + ". Ejecute: python scripts/train_models.py"
        )


def main() -> int:
    try:
        build_database(force="--force" in sys.argv)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

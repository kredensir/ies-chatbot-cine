"""Descarga URLs de portadas para el catálogo local.

Uso:
    python scripts/fetch_posters.py --movies --limit 500 --popular
    python scripts/fetch_posters.py --movies --update-db --update-catalog

Las series ya obtienen su portada al construir ``series_catalog.csv`` con
``python scripts/build_series_catalog.py``. Este script se ocupa de las
películas: usa TMDB si existe ``TMDB_API_KEY`` y, si no, Letterboxd.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cinebot import config  # noqa: E402
from cinebot.db.connection import Database  # noqa: E402
from cinebot.posters import resolve_movie_poster  # noqa: E402


def _load_cache(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            int(row["item_id"]): row
            for row in csv.DictReader(handle)
            if row.get("item_id") and row.get("poster_url")
        }


def _save_cache(path: Path, rows: dict[int, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = ["item_id", "title", "year", "poster_url", "source_url"]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item_id in sorted(rows):
            writer.writerow(rows[item_id])
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(0.4 + attempt * 0.4)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item_id in sorted(rows):
            writer.writerow(rows[item_id])
    temporary.unlink(missing_ok=True)


def _update_catalog(path: Path, posters: dict[int, dict[str, str]], refresh: bool) -> None:
    if not path.exists():
        return
    catalog = pd.read_csv(path, dtype={"item_id": "int64"})
    current = catalog.set_index("item_id")["poster_url"] if "poster_url" in catalog else None
    for row in catalog.itertuples(index=False):
        poster = posters.get(int(row.item_id), {}).get("poster_url", "")
        if not poster:
            continue
        if not refresh and current is not None:
            existing = current.get(int(row.item_id))
            if isinstance(existing, str) and existing.strip():
                continue
        catalog.loc[catalog["item_id"] == int(row.item_id), "poster_url"] = poster
    catalog.to_csv(path, index=False, encoding="utf-8")


def _update_database(posters: dict[int, dict[str, str]]) -> int:
    db = Database(config.DB_PATH)
    rows = [
        (row["poster_url"], int(item_id))
        for item_id, row in posters.items()
        if row.get("poster_url")
    ]
    with db.transaction() as connection:
        connection.executemany(
            "UPDATE items SET poster_url = ?, updated_at = datetime('now') WHERE item_id = ?",
            rows,
        )
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Obtener portadas para CineBot")
    parser.add_argument("--movies", action="store_true", help="Resolver portadas de películas")
    parser.add_argument("--limit", type=int, default=0, help="Máximo de películas a procesar")
    parser.add_argument("--items", default="", help="IDs separados por coma")
    parser.add_argument("--popular", action="store_true", help="Priorizar las más valoradas")
    parser.add_argument("--workers", type=int, default=6, help="Peticiones simultáneas")
    parser.add_argument("--timeout", type=float, default=6.0, help="Timeout por petición")
    parser.add_argument("--refresh", action="store_true", help="Volver a resolver las existentes")
    parser.add_argument("--update-db", action="store_true", help="Escribir las URLs en SQLite")
    parser.add_argument("--update-catalog", action="store_true", help="Actualizar catalog.csv")
    args = parser.parse_args()

    if not args.movies:
        parser.error("Indica --movies. Las series se resuelven con build_series_catalog.py.")

    config.ensure_directories()
    catalog = pd.read_csv(config.CATALOG_CSV, dtype={"item_id": "int64"})
    movies = catalog[catalog["media_type"] == "movie"].copy()
    if args.popular:
        movies = movies.sort_values(["rating_count", "bayes_rating"], ascending=False)
    else:
        movies = movies.sort_values("item_id")

    cache_path = config.MOVIE_POSTERS_CSV
    cache = _load_cache(cache_path)
    for row in movies.itertuples(index=False):
        existing = str(getattr(row, "poster_url", "") or "").strip()
        if existing and existing.lower() != "nan" and int(row.item_id) not in cache:
            cache[int(row.item_id)] = {
                "item_id": str(int(row.item_id)),
                "title": str(row.title),
                "year": "" if pd.isna(row.year) else str(int(row.year)),
                "poster_url": existing,
                "source_url": str(getattr(row, "source_url", "") or ""),
            }

    wanted_ids = {
        int(value)
        for value in args.items.replace(" ", "").split(",")
        if value.strip().isdigit()
    }
    pending = [
        row
        for row in movies.itertuples(index=False)
        if int(row.item_id) in wanted_ids
        or (not wanted_ids and (args.refresh or int(row.item_id) not in cache))
    ]
    if args.limit:
        pending = pending[: args.limit]

    if not pending:
        print("[OK] No hay películas pendientes de portada")
    else:
        print(f"[..] Resolviendo {len(pending):,} películas con {args.workers} workers")
        started = time.time()
        done = 0
        found = 0
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(
                    resolve_movie_poster,
                    str(row.title),
                    int(row.year) if pd.notna(row.year) else None,
                    int(row.tmdb_id) if pd.notna(row.tmdb_id) else None,
                    args.timeout,
                ): row
                for row in pending
            }
            for future in as_completed(futures):
                row = futures[future]
                done += 1
                try:
                    poster, source = future.result()
                except Exception as exc:  # noqa: BLE001
                    poster, source = "", ""
                    print(f"[WARN] {row.title}: {exc}", file=sys.stderr)
                if poster:
                    found += 1
                    cache[int(row.item_id)] = {
                        "item_id": str(int(row.item_id)),
                        "title": str(row.title),
                        "year": "" if pd.isna(row.year) else str(int(row.year)),
                        "poster_url": poster,
                        "source_url": source,
                    }
                if done % 25 == 0 or done == len(pending):
                    _save_cache(cache_path, cache)
                    elapsed = time.time() - started
                    rate = done / elapsed if elapsed else 0
                    print(
                        f"[..] {done:,}/{len(pending):,} procesadas · "
                        f"{found:,} portadas · {rate:.1f}/s"
                    )
        _save_cache(cache_path, cache)
        elapsed = time.time() - started
        print(f"[OK] {found:,} portadas nuevas en {elapsed:.1f}s")

    if args.update_catalog:
        _update_catalog(config.CATALOG_CSV, cache, args.refresh)
        print(f"[OK] Catálogo actualizado: {config.CATALOG_CSV}")
    if args.update_db:
        updated = _update_database(cache)
        print(f"[OK] Base actualizada: {updated:,} URLs de portada")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

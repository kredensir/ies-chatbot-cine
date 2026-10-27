"""Construye o reconstruye la base SQLite de CineBot.

Uso:
    python scripts/build_database.py
    python scripts/build_database.py --force
    python scripts/build_database.py --force --rebuild-catalog
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cinebot import config  # noqa: E402
from cinebot.bootstrap import build_database  # noqa: E402
from cinebot.data.catalog import build_catalog  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Construye la base SQLite de CineBot")
    parser.add_argument("--force", action="store_true", help="Elimina y recrea la base")
    parser.add_argument(
        "--rebuild-catalog",
        action="store_true",
        help="Vuelve a procesar los CSV crudos antes de cargar la base",
    )
    args = parser.parse_args()

    try:
        if args.rebuild_catalog or not config.CATALOG_CSV.exists():
            print("[..] Procesando catálogo unificado")
            build_catalog(save=True)
        db = build_database(force=args.force)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    health = db.health_check()
    print("\n[OK] Estado de la base")
    for table, count in health["counts"].items():
        print(f"     {table:20s} {count:>9,}")
    print(f"     SQLite {health['sqlite_version']} · {health['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

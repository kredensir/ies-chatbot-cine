"""Entrena todos los modelos de CineBot y genera métricas y figuras.

Uso:
    python scripts/train_models.py
    python scripts/train_models.py --force-db      # reconstruye SQLite antes
    python scripts/train_models.py --quiet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cinebot.ml.training import train_all  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Entrena los modelos de CineBot")
    parser.add_argument("--force-db", action="store_true", help="Reconstruye la base SQLite")
    parser.add_argument("--quiet", action="store_true", help="Menos detalle por época")
    args = parser.parse_args()

    started = time.perf_counter()
    try:
        metrics = train_all(force_db=args.force_db, verbose=not args.quiet)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started
    print(
        f"\n[OK] Entrenamiento completo en {elapsed:.1f}s · "
        f"RMSE={metrics['collaborative']['rmse']:.4f} · "
        f"NDCG@10={metrics['ranking']['hybrid']['ndcg@10']:.4f} · "
        f"accuracy intenciones={metrics['intent']['accuracy']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

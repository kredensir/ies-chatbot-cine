"""Muestra las métricas de los modelos y ejecuta comprobaciones rápidas.

Uso:
    python scripts/evaluate_models.py
    python scripts/evaluate_models.py --live
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from cinebot import config  # noqa: E402
from cinebot.bootstrap import ensure_database, ensure_models  # noqa: E402


def _print_table(title: str, rows: list[tuple[str, object]]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for label, value in rows:
        if isinstance(value, float):
            value = f"{value:.4f}"
        print(f"  {label:<32} {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evalúa y resume los modelos de CineBot")
    parser.add_argument("--live", action="store_true", help="Ejecuta consultas de comprobación")
    args = parser.parse_args()

    try:
        ensure_models()
        db = ensure_database()
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    metadata_path = config.model_path("metadata")
    if not metadata_path.exists():
        print("[ERROR] Falta models/metadata.json. Ejecute scripts/train_models.py", file=sys.stderr)
        return 1
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    catalog = metadata.get("catalog", {})
    collab = metadata.get("collaborative", {})
    baseline = metadata.get("baseline", {})
    ranking = metadata.get("ranking", {})
    intent = metadata.get("intent", {})
    clusters = metadata.get("clusters", {})

    _print_table(
        "Datos",
        [
            ("Ítems", f"{catalog.get('items', 0):,}"),
            ("Películas", f"{catalog.get('movies', 0):,}"),
            ("Series", catalog.get("series", 0)),
            ("Valoraciones MovieLens", f"{catalog.get('movielens_ratings', 0):,}"),
            ("Usuarios históricos", catalog.get("users", 0)),
            ("Entrenado", metadata.get("trained_at", "")),
        ],
    )
    _print_table(
        "Predicción de notas",
        [
            ("RMSE baseline", baseline.get("rmse", 0.0)),
            ("RMSE FunkSVD", collab.get("rmse", 0.0)),
            ("MAE FunkSVD", collab.get("mae", 0.0)),
            ("R² FunkSVD", collab.get("r2", 0.0)),
            ("Mejora vs baseline (%)", (baseline.get("rmse", 1) - collab.get("rmse", 0)) / max(baseline.get("rmse", 1), 1e-9) * 100),
        ],
    )
    _print_table(
        "Ranking top-10 (test)",
        [
            ("Precision@10 híbrido", ranking.get("hybrid", {}).get("precision@10", 0.0)),
            ("Recall@10 híbrido", ranking.get("hybrid", {}).get("recall@10", 0.0)),
            ("NDCG@10 híbrido", ranking.get("hybrid", {}).get("ndcg@10", 0.0)),
            ("Hit Rate@10 híbrido", ranking.get("hybrid", {}).get("hit_rate@10", 0.0)),
            ("Cobertura", ranking.get("hybrid", {}).get("coverage", 0.0)),
            ("Diversidad", ranking.get("hybrid", {}).get("diversity", 0.0)),
        ],
    )
    _print_table(
        "Chatbot y segmentación",
        [
            ("Accuracy intenciones", intent.get("accuracy", 0.0)),
            ("F1 macro", intent.get("macro_f1", 0.0)),
            ("Clusters K", clusters.get("best_k", 0)),
            ("Silueta", clusters.get("silhouette", 0.0)),
            ("Varianza PCA", clusters.get("explained_variance", 0.0)),
        ],
    )

    if args.live:
        from cinebot.chatbot import ChatEngine  # noqa: PLC0415
        from cinebot.ml import ContentModel, SVDModel  # noqa: PLC0415

        print("\nComprobaciones en vivo")
        print("-" * 23)
        content = ContentModel.load()
        svd = SVDModel.load()
        print(f"  Vocabulario TF-IDF: {content.vocabulary_size:,} términos")
        print(f"  Factores SVD: {svd.P.shape[1]} · usuarios {len(svd.user_ids)} · ítems {len(svd.item_ids)}")
        similar = content.similar_items(79132, limit=3)
        print(f"  Parecidos a Inception: {[item_id for item_id, _ in similar]}")
        engine = ChatEngine(db=db, recommender=None, nlu=None)
        for text in ("hola", "series de misterio cortas", "de qué trata Parasite", "top 5 películas"):
            response = engine.handle(text)
            print(f"  «{text}» -> {response.intent} ({response.confidence:.2f}) · {len(response.recommendations)} recomendaciones")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

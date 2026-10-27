"""Orquestador de entrenamiento y evaluación de todos los modelos de CineBot.

Ejecuta:
1. Carga de datos desde SQLite y del catálogo procesado.
2. Partición train / validación / test por usuario.
3. Entrenamiento del modelo de contenido (TF-IDF + vecinos).
4. Entrenamiento del filtrado colaborativo (FunkSVD + sesgos).
5. Ajuste de pesos del recomendador híbrido con validación.
6. Evaluación final en test: RMSE/MAE, Precision@10, Recall@10, NDCG@10,
   cobertura y diversidad.
7. Entrenamiento del clasificador de intenciones y de K-Means de usuarios.
8. Persistencia de artefactos, métricas en SQLite, figuras y reporte.
"""

from __future__ import annotations

import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cinebot import config
from cinebot.bootstrap import build_database, ensure_database
from cinebot.data.catalog import load_processed_catalog
from cinebot.db.repository import MetricsRepository, UserRepository
from cinebot.ml.clustering import UserClustering
from cinebot.ml.collaborative import SVDModel, split_ratings
from cinebot.ml.content import ContentModel
from cinebot.ml.evaluation import (
    cf_score_function,
    content_score_function,
    evaluate_ranking,
    hybrid_score_function,
    tune_hybrid_weights,
)
from cinebot.ml.intent import IntentClassifier


def _plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
        }
    )


def _rmse(real: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(real) - np.asarray(predicted)) ** 2)))


def _mae(real: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(real) - np.asarray(predicted))))


def _r2(real: np.ndarray, predicted: np.ndarray) -> float:
    real = np.asarray(real, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = float(np.sum((real - real.mean()) ** 2))
    if denominator == 0:
        return 0.0
    return float(1.0 - np.sum((real - predicted) ** 2) / denominator)


def _baseline_metrics(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, float]:
    mean_rating = float(train["rating"].mean())
    predictions = np.full(len(test), mean_rating)
    real = test["rating"].to_numpy(dtype=float)
    return {
        "rmse": _rmse(real, predictions),
        "mae": _mae(real, predictions),
        "r2": _r2(real, predictions),
        "n": float(len(test)),
    }


def _svd_metrics(model: SVDModel, test: pd.DataFrame) -> dict[str, float]:
    mask = test["userId"].isin(model.user_index) & test["movieId"].isin(model.item_index)
    data = test.loc[mask]
    if data.empty:
        return {"rmse": 0.0, "mae": 0.0, "r2": 0.0, "n": 0.0}
    users = data["userId"].map(model.user_index).to_numpy(dtype=np.int64)
    items = data["movieId"].map(model.item_index).to_numpy(dtype=np.int64)
    predictions = model.mu + model.bu[users] + model.bi[items] + np.einsum(
        "ij,ij->i", model.P[users], model.Q[items]
    )
    predictions = np.clip(predictions, 0.5, 5.0)
    real = data["rating"].to_numpy(dtype=float)
    return {
        "rmse": _rmse(real, predictions),
        "mae": _mae(real, predictions),
        "r2": _r2(real, predictions),
        "n": float(len(data)),
    }


def _popularity_score_function(catalog: pd.DataFrame):
    popularity = {
        int(row.item_id): float(row.popularity or 0)
        for row in catalog[["item_id", "popularity"]].itertuples(index=False)
    }
    maximum = max(1e-9, max(popularity.values()))

    def score(user_id: int, candidates: list[int], ratings: dict[int, float]) -> np.ndarray:
        return np.asarray(
            [popularity.get(int(item), 0.0) / maximum for item in candidates],
            dtype=np.float32,
        )

    return score


def _save_figures(
    ratings: pd.DataFrame,
    catalog: pd.DataFrame,
    model: SVDModel,
    ranking: dict[str, dict[str, float]],
    clusters: UserClustering,
    intent: IntentClassifier,
) -> list[str]:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    _plot_style()

    # 1. Distribución de valoraciones
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = ratings["rating"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color="#7C4DFF")
    ax.set_title("Distribución de valoraciones de MovieLens")
    ax.set_xlabel("Nota")
    ax.set_ylabel("Cantidad")
    path = config.FIGURES_DIR / "ratings_distribution.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path))

    # 2. Curva de aprendizaje del SVD
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = [row["epoch"] for row in model.history]
    ax.plot(epochs, [row["train_rmse"] for row in model.history], label="Train", color="#4ECDC4")
    ax.plot(epochs, [row["val_rmse"] for row in model.history], label="Validación", color="#FF6B6B")
    ax.axvline(model.best_epoch, linestyle="--", color="#FFD166", label=f"Mejor época ({model.best_epoch})")
    ax.set_title("FunkSVD: RMSE por época")
    ax.set_xlabel("Época")
    ax.set_ylabel("RMSE")
    ax.legend()
    path = config.FIGURES_DIR / "svd_learning_curve.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path))

    # 3. Comparación de modelos de ranking
    models = list(ranking.keys())
    metrics = ["precision@10", "ndcg@10", "recall@10"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / max(1, len(metrics))
    x = np.arange(len(models))
    colors = ["#7C4DFF", "#4ECDC4", "#FF6B6B"]
    for index, metric in enumerate(metrics):
        values = [ranking[name].get(metric, 0.0) for name in models]
        ax.bar(x + index * width, values, width, label=metric, color=colors[index % len(colors)])
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels([name.replace("_", "\n") for name in models])
    ax.set_title("Comparación de modelos de recomendación (test)")
    ax.set_ylabel("Valor")
    ax.legend()
    path = config.FIGURES_DIR / "model_comparison.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path))

    # 4. Géneros más frecuentes
    genre_counts = (
        catalog["genres"].fillna("").str.split("|").explode().loc[lambda s: s != ""].value_counts().head(15)
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(genre_counts.index[::-1], genre_counts.values[::-1], color="#118AB2")
    ax.set_title("Géneros con más títulos en el catálogo")
    ax.set_xlabel("Cantidad de títulos")
    path = config.FIGURES_DIR / "genre_distribution.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path))

    # 5. Clusters de usuarios en 2D
    if clusters.coordinates is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = ax.scatter(
            clusters.coordinates[:, 0],
            clusters.coordinates[:, 1],
            c=clusters.user_labels,
            cmap="tab10",
            s=18,
            alpha=0.75,
        )
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=plt.cm.tab10(i / 10), markersize=7,
                       label=clusters.cluster_names.get(i, f"Grupo {i}"))
            for i in range(clusters.kmeans.n_clusters)
        ]
        ax.legend(handles=handles, fontsize=8, loc="best")
        ax.set_title("Segmentación de usuarios con K-Means (PCA 2D)")
        ax.set_xlabel("Componente principal 1")
        ax.set_ylabel("Componente principal 2")
        path = config.FIGURES_DIR / "user_clusters.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))

    # 6. Matriz de confusión del clasificador de intenciones
    if intent.confusion:
        fig, ax = plt.subplots(figsize=(10, 8))
        matrix = np.asarray(intent.confusion)
        im = ax.imshow(matrix, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(intent.labels)))
        ax.set_yticks(range(len(intent.labels)))
        ax.set_xticklabels(intent.labels, rotation=90, fontsize=7)
        ax.set_yticklabels(intent.labels, fontsize=7)
        ax.set_title("Matriz de confusión: intenciones del chatbot")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        path = config.FIGURES_DIR / "intent_confusion.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))

    # 7. Composición del catálogo
    fig, ax = plt.subplots(figsize=(5, 4))
    sizes = catalog["media_type"].value_counts()
    ax.pie(sizes.values, labels=[f"{k} ({v:,})" for k, v in sizes.items()],
           colors=["#7C4DFF", "#FFD166"], autopct="%1.1f%%", startangle=90)
    ax.set_title("Composición del catálogo")
    path = config.FIGURES_DIR / "catalog_composition.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    paths.append(str(path))

    return paths


def _write_report(
    metrics: dict[str, Any],
    weights: tuple[float, float, float, float],
    tuning: list[dict[str, float]],
    figure_paths: list[str],
) -> Path:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = config.REPORTS_DIR / "evaluacion_modelos.md"
    collab = metrics["collaborative"]
    baseline = metrics["baseline"]
    ranking = metrics["ranking"]
    intent = metrics["intent"]
    clusters = metrics["clusters"]

    def table(rows: list[tuple[str, str]]) -> str:
        body = "\n".join(f"| {a} | {b} |" for a, b in rows)
        return "| Métrica | Valor |\n|---|---|\n" + body

    lines = [
        "# Evaluación de modelos — CineBot",
        "",
        f"Generado: {metrics['trained_at']}",
        "",
        "## 1. Datos",
        "",
        f"- Ítems del catálogo: **{metrics['catalog']['items']:,}** "
        f"({metrics['catalog']['movies']:,} películas y {metrics['catalog']['series']} series).",
        f"- Valoraciones MovieLens: **{metrics['catalog']['movielens_ratings']:,}** "
        f"de {metrics['catalog']['users']} usuarios.",
        f"- Partición: train={metrics['splits']['train']:,}, validación={metrics['splits']['validation']:,}, "
        f"test={metrics['splits']['test']:,}.",
        "",
        "## 2. Predicción de notas",
        "",
        table(
            [
                ("RMSE baseline (media global)", f"{baseline['rmse']:.4f}"),
                ("RMSE FunkSVD", f"{collab['rmse']:.4f}"),
                ("MAE FunkSVD", f"{collab['mae']:.4f}"),
                ("R² FunkSVD", f"{collab['r2']:.4f}"),
                ("Mejora de RMSE vs baseline", f"{(baseline['rmse'] - collab['rmse']) / baseline['rmse'] * 100:.2f}%"),
                ("Factores latentes", str(metrics['svd_params']['n_factors'])),
                ("Mejor época", str(metrics['svd_params']['best_epoch'])),
                ("Valoraciones de test evaluadas", f"{int(collab['n']):,}"),
            ]
        ),
        "",
        "## 3. Ranking top-10 (test)",
        "",
        table(
            [
                ("Usuarios evaluados", f"{int(ranking['hybrid']['evaluated_users']):,}"),
                ("Precision@10 popularidad (baseline)", f"{ranking['popularidad']['precision@10']:.4f}"),
                ("Precision@10 contenido", f"{ranking['content']['precision@10']:.4f}"),
                ("Precision@10 colaborativo", f"{ranking['collaborative']['precision@10']:.4f}"),
                ("Precision@10 híbrido", f"{ranking['hybrid']['precision@10']:.4f}"),
                ("Recall@10 híbrido", f"{ranking['hybrid']['recall@10']:.4f}"),
                ("NDCG@10 híbrido", f"{ranking['hybrid']['ndcg@10']:.4f}"),
                ("Hit Rate@10 híbrido", f"{ranking['hybrid']['hit_rate@10']:.4f}"),
                ("Cobertura del catálogo", f"{ranking['hybrid']['coverage'] * 100:.2f}%"),
                ("Diversidad media (Jaccard)", f"{ranking['hybrid']['diversity']:.4f}"),
            ]
        ),
        "",
        "Pesos elegidos por validación: "
        f"contenido={weights[0]:.2f}, colaborativo={weights[1]:.2f}, "
        f"calidad={weights[2]:.2f}, popularidad={weights[3]:.2f}.",
        "",
        "## 4. Clasificador de intenciones",
        "",
        table(
            [
                ("Accuracy", f"{intent['accuracy']:.4f}"),
                ("F1 macro", f"{intent['macro_f1']:.4f}"),
                ("F1 ponderado", f"{intent['weighted_f1']:.4f}"),
                ("Validación cruzada (media)", f"{intent['cv_accuracy_mean']:.4f} ± {intent['cv_accuracy_std']:.4f}"),
                ("Ejemplos etiquetados", f"{int(intent['n_examples'])}"),
                ("Intenciones", f"{int(intent['n_intents'])}"),
            ]
        ),
        "",
        "## 5. Segmentación K-Means",
        "",
        table(
            [
                ("K elegido", str(clusters['best_k'])),
                ("Silueta", f"{clusters['silhouette']:.4f}"),
                ("Usuarios agrupados", f"{int(clusters['n_users'])}"),
                ("Varianza explicada por PCA", f"{clusters['explained_variance'] * 100:.1f}%"),
            ]
        ),
        "",
        "## 6. Figuras",
        "",
    ]
    for path in figure_paths:
        lines.append(f"### {Path(path).stem.replace('_', ' ').title()}")
        lines.append("")
        lines.append(f"![{Path(path).stem}](figures/{Path(path).name})")
        lines.append("")
    lines.append("## 7. Búsqueda de pesos del híbrido")
    lines.append("")
    lines.append("| Contenido | Colaborativo | Calidad | Popularidad | NDCG@10 | Precision@10 |")
    lines.append("|---|---|---|---|---|---|")
    for row in tuning:
        lines.append(
            f"| {row['content']:.2f} | {row['collaborative']:.2f} | {row['quality']:.2f} | "
            f"{row['popularity']:.2f} | {row['ndcg@10']:.4f} | {row['precision@10']:.4f} |"
        )
    lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def train_all(force_db: bool = False, verbose: bool = True) -> dict[str, Any]:
    config.ensure_directories()
    _plot_style()
    if force_db:
        db = build_database(force=True)
    else:
        db = ensure_database()

    catalog = load_processed_catalog()
    ratings = pd.DataFrame(
        db.query(
            "SELECT user_id AS userId, item_id AS movieId, rating FROM ratings WHERE source = 'movielens'"
        )
    )
    if ratings.empty:
        raise RuntimeError("No hay valoraciones de MovieLens en la base. Reconstruya la base.")
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    print(f"[..] Catálogo: {len(catalog):,} ítems · valoraciones: {len(ratings):,}")
    train, validation, test = split_ratings(ratings)
    print(
        f"[OK] Partición: train={len(train):,} · validación={len(validation):,} · test={len(test):,}"
    )

    # ------------------------------------------------------------------
    # 1. Modelo de contenido
    # ------------------------------------------------------------------
    print("[..] Entrenando modelo de contenido (TF-IDF + vecinos)")
    content = ContentModel.fit(catalog)
    content.save()
    print(
        f"[OK] TF-IDF: {content.vocabulary_size:,} términos · matriz {content.matrix.shape}"
    )

    # ------------------------------------------------------------------
    # 2. Filtrado colaborativo
    # ------------------------------------------------------------------
    print("[..] Entrenando FunkSVD (esto puede tardar un poco)")
    collaborative = SVDModel.fit(train, validation, verbose=verbose)
    collaborative.save()
    collab_metrics = _svd_metrics(collaborative, test)
    baseline_metrics = _baseline_metrics(train, test)
    print(
        f"[OK] RMSE test={collab_metrics['rmse']:.4f} "
        f"(baseline {baseline_metrics['rmse']:.4f}) · MAE={collab_metrics['mae']:.4f}"
    )

    # ------------------------------------------------------------------
    # 3. Ajuste de pesos del híbrido con validación
    # ------------------------------------------------------------------
    item_ids = sorted(int(x) for x in collaborative.item_ids)
    print(f"[..] Ajustando pesos del híbrido con {len(validation):,} valoraciones de validación")
    best_weights, tuning = tune_hybrid_weights(
        validation, train, item_ids, catalog, content, collaborative,
        k=10, max_users=120,
    )
    print(
        f"[OK] Pesos: contenido={best_weights[0]:.2f}, colaborativo={best_weights[1]:.2f}, "
        f"calidad={best_weights[2]:.2f}, popularidad={best_weights[3]:.2f}"
    )

    # ------------------------------------------------------------------
    # 4. Evaluación de ranking en test
    # ------------------------------------------------------------------
    print("[..] Evaluando ranking en test")
    ranking: dict[str, dict[str, float]] = {}
    ranking["popularidad"] = evaluate_ranking(
        _popularity_score_function(catalog), test, train, item_ids, catalog, k=10
    )
    ranking["contenido"] = evaluate_ranking(
        content_score_function(content), test, train, item_ids, catalog, k=10
    )
    ranking["colaborativo"] = evaluate_ranking(
        cf_score_function(collaborative), test, train, item_ids, catalog, k=10
    )
    ranking["híbrido"] = evaluate_ranking(
        hybrid_score_function(content, collaborative, catalog, best_weights),
        test, train, item_ids, catalog, k=10,
    )
    for name, values in ranking.items():
        print(
            f"  {name:14s} P@10={values['precision@10']:.4f} "
            f"R@10={values['recall@10']:.4f} NDCG@10={values['ndcg@10']:.4f}"
        )

    # ------------------------------------------------------------------
    # 5. Clasificador de intenciones
    # ------------------------------------------------------------------
    print("[..] Entrenando clasificador de intenciones")
    intent = IntentClassifier.fit()
    intent.save()
    print(
        f"[OK] Intenciones: accuracy={intent.metrics['accuracy']:.4f} "
        f"F1 macro={intent.metrics['macro_f1']:.4f}"
    )

    # ------------------------------------------------------------------
    # 6. Segmentación de usuarios
    # ------------------------------------------------------------------
    print("[..] Entrenando K-Means de usuarios")
    clusters = UserClustering.fit(ratings, catalog)
    clusters.save()
    print(
        f"[OK] K-Means: k={clusters.evaluation['best_k']} "
        f"silueta={clusters.evaluation['silhouette']:.4f}"
    )

    # Persistir clusters en la tabla users para los perfiles históricos.
    user_repo = UserRepository(db)
    for user_id in clusters.user_ids:
        profile = clusters.profile_for_user(user_id)
        if profile:
            user_repo.update_cluster(user_id, *profile)

    # ------------------------------------------------------------------
    # 7. Métricas en SQLite
    # ------------------------------------------------------------------
    metrics_repo = MetricsRepository(db)
    run_id = metrics_repo.start_run(
        "hybrid",
        params={
            "weights": list(best_weights),
            "svd": collaborative.params,
            "content_vocab": content.vocabulary_size,
        },
        notes="Entrenamiento completo de CineBot",
    )
    metrics_repo.finish_run(run_id, n_users=len(collaborative.user_ids), n_items=len(catalog), n_ratings=len(train))
    metrics_repo.save_metrics(
        "collaborative",
        {"rmse": collab_metrics["rmse"], "mae": collab_metrics["mae"], "r2": collab_metrics["r2"]},
        run_id,
    )
    metrics_repo.save_metrics(
        "baseline",
        {"rmse": baseline_metrics["rmse"], "mae": baseline_metrics["mae"], "r2": baseline_metrics["r2"]},
        run_id,
    )
    for name, values in ranking.items():
        metrics_repo.save_metrics(name, values, run_id)
    metrics_repo.save_metrics("intent", intent.metrics, run_id)
    metrics_repo.save_metrics(
        "clusters",
        {
            "best_k": clusters.evaluation["best_k"],
            "silhouette": clusters.evaluation["silhouette"],
            "explained_variance": clusters.evaluation["explained_variance"],
            "n_users": clusters.evaluation["n_users"],
        },
        run_id,
    )

    # ------------------------------------------------------------------
    # 8. Figuras y reporte
    # ------------------------------------------------------------------
    print("[..] Generando figuras y reporte")
    figure_paths = _save_figures(ratings, catalog, collaborative, ranking, clusters, intent)

    catalog_stats = {
        "items": len(catalog),
        "movies": int((catalog["media_type"] == "movie").sum()),
        "series": int((catalog["media_type"] == "series").sum()),
        "movielens_ratings": len(ratings),
        "users": int(ratings["userId"].nunique()),
    }
    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    metrics: dict[str, Any] = {
        "version": "1.0.0",
        "trained_at": trained_at,
        "python": platform.python_version(),
        "catalog": catalog_stats,
        "splits": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "baseline": baseline_metrics,
        "collaborative": collab_metrics,
        "svd_params": collaborative.params,
        "content": {
            "vocabulary_size": content.vocabulary_size,
            "items": content.n_features,
            "top_k_neighbors": content.top_k,
        },
        "ranking": {
            "popularidad": ranking["popularidad"],
            "content": ranking["contenido"],
            "collaborative": ranking["colaborativo"],
            "hybrid": ranking["híbrido"],
            "test": ranking["híbrido"],
            "content_test": ranking["contenido"],
            "collaborative_test": ranking["colaborativo"],
        },
        "hybrid_weights": {
            "content": best_weights[0],
            "collaborative": best_weights[1],
            "quality": best_weights[2],
            "popularity": best_weights[3],
        },
        "cold_start_weights": {
            "content": 0.55,
            "collaborative": 0.0,
            "quality": 0.30,
            "popularity": 0.15,
        },
        "intent": intent.metrics,
        "clusters": clusters.evaluation,
        "clusters_names": clusters.cluster_names,
        "clusters_descriptions": clusters.cluster_descriptions,
        "figures": [Path(p).relative_to(config.BASE_DIR).as_posix() for p in figure_paths],
    }
    metadata_path = config.model_path("metadata")
    metadata_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = _write_report(metrics, best_weights, tuning, figure_paths)
    print(f"[OK] Reporte: {report_path}")
    print(f"[OK] Metadatos: {metadata_path}")
    return metrics

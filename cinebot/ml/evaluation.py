"""Evaluación de modelos de recomendación.

Se usan dos familias de métricas:

1. Predicción de notas (regresión): RMSE, MAE y R².
2. Ranking top-K (recuperación de ítems relevantes): Precision@K, Recall@K,
   NDCG@K, Hit Rate@K, cobertura del catálogo y diversidad.

El protocolo de ranking deja fuera del entrenamiento una parte de las
valoraciones de cada usuario y considera "relevantes" las notas >= 4.
"""

from __future__ import annotations

import itertools
from typing import Callable

import numpy as np
import pandas as pd

from cinebot.ml.collaborative import SVDModel
from cinebot.ml.content import ContentModel

WEIGHT_GRID = [
    (0.50, 0.30, 0.15, 0.05),
    (0.45, 0.35, 0.15, 0.05),
    (0.40, 0.40, 0.15, 0.05),
    (0.35, 0.45, 0.15, 0.05),
    (0.55, 0.25, 0.15, 0.05),
    (0.50, 0.40, 0.10, 0.00),
    (0.40, 0.50, 0.10, 0.00),
    (0.60, 0.20, 0.15, 0.05),
    (0.45, 0.30, 0.20, 0.05),
    (0.35, 0.35, 0.25, 0.05),
    (0.50, 0.30, 0.10, 0.10),
    (0.40, 0.30, 0.20, 0.10),
]

COLD_START_WEIGHTS = (0.55, 0.0, 0.30, 0.15)


def ranking_metrics(recommended: list[int], relevant: set[int], k: int = 10) -> dict[str, float]:
    top = recommended[:k]
    hits = [1 if item in relevant else 0 for item in top]
    n_hits = sum(hits)
    precision = n_hits / k if k else 0.0
    recall = n_hits / len(relevant) if relevant else 0.0
    dcg = sum(1.0 / np.log2(index + 2) for index, hit in enumerate(hits) if hit)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(index + 2) for index in range(ideal_hits))
    ndcg = dcg / idcg if idcg else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "ndcg": float(ndcg),
        "hit": float(1.0 if n_hits else 0.0),
    }


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def evaluate_ranking(
    score_fn: Callable[[int, list[int], dict[int, float]], np.ndarray],
    test: pd.DataFrame,
    train: pd.DataFrame,
    item_ids: list[int],
    catalog: pd.DataFrame,
    k: int = 10,
    max_users: int | None = None,
    relevance_threshold: float = 4.0,
) -> dict[str, float]:
    """Evalúa un modelo de ranking sobre el conjunto de prueba."""
    train_items: dict[int, set[int]] = {}
    train_ratings: dict[int, dict[int, float]] = {}
    for user_id, group in train.groupby("userId"):
        train_items[int(user_id)] = set(int(x) for x in group["movieId"])
        train_ratings[int(user_id)] = {
            int(row.movieId): float(row.rating) for row in group.itertuples(index=False)
        }

    relevant: dict[int, set[int]] = {}
    for user_id, group in test.groupby("userId"):
        items = set(
            int(x)
            for x in group.loc[group["rating"] >= relevance_threshold, "movieId"]
        )
        if items:
            relevant[int(user_id)] = items

    users = [u for u in relevant if u in train_items]
    if max_users:
        rng = np.random.default_rng(42)
        users = list(rng.choice(users, size=min(max_users, len(users)), replace=False))
        users = [int(u) for u in users]

    genre_by_item = {
        int(row.item_id): set(str(row.genres).split("|"))
        for row in catalog[["item_id", "genres"]].itertuples(index=False)
    }
    item_set = set(item_ids)
    metrics: list[dict[str, float]] = []
    all_recommended: set[int] = set()
    diversities: list[float] = []

    for user_id in users:
        seen = train_items[user_id]
        candidates = [item for item in item_ids if item not in seen]
        if not candidates:
            continue
        scores = score_fn(user_id, candidates, train_ratings.get(user_id, {}))
        top_k = min(k, len(candidates))
        top_idx = np.argpartition(-scores, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        recommended = [candidates[int(i)] for i in top_idx]
        metrics.append(ranking_metrics(recommended, relevant[user_id], k))
        all_recommended.update(recommended)

        genres = [genre_by_item.get(item, set()) for item in recommended]
        pairs = list(itertools.combinations(genres, 2))
        if pairs:
            diversities.append(float(np.mean([_jaccard(a, b) for a, b in pairs])))

    if not metrics:
        return {
            f"precision@{k}": 0.0,
            f"recall@{k}": 0.0,
            f"ndcg@{k}": 0.0,
            f"hit_rate@{k}": 0.0,
            "coverage": 0.0,
            "diversity": 0.0,
            "evaluated_users": 0.0,
        }
    return {
        f"precision@{k}": float(np.mean([m["precision"] for m in metrics])),
        f"recall@{k}": float(np.mean([m["recall"] for m in metrics])),
        f"ndcg@{k}": float(np.mean([m["ndcg"] for m in metrics])),
        f"hit_rate@{k}": float(np.mean([m["hit"] for m in metrics])),
        "coverage": len(all_recommended) / max(1, len(item_set)),
        "diversity": float(np.mean(diversities)) if diversities else 0.0,
        "evaluated_users": float(len(metrics)),
    }


def content_score_function(content: ContentModel) -> Callable:
    def score(user_id: int, candidates: list[int], ratings: dict[int, float]) -> np.ndarray:
        scores = content.score_candidates(candidates, ratings=ratings)
        return np.asarray([scores.get(int(item), 0.0) for item in candidates], dtype=np.float32)

    return score


def cf_score_function(model: SVDModel) -> Callable:
    def score(user_id: int, candidates: list[int], ratings: dict[int, float]) -> np.ndarray:
        predictions = model.predict_user(user_id, candidates, ratings)
        return np.asarray(
            [(predictions.get(int(item), 3.0) - 0.5) / 4.5 for item in candidates],
            dtype=np.float32,
        )

    return score


def hybrid_score_function(
    content: ContentModel,
    model: SVDModel,
    catalog: pd.DataFrame,
    weights: tuple[float, float, float, float],
) -> Callable:
    w_content, w_cf, w_quality, w_pop = weights
    quality = {
        int(row.item_id): float(row.bayes_rating) / 5.0
        for row in catalog[["item_id", "bayes_rating"]].itertuples(index=False)
    }
    max_pop = max(
        1e-9,
        float(catalog["popularity"].fillna(0).max()),
    )
    popularity = {
        int(row.item_id): float(row.popularity or 0) / max_pop
        for row in catalog[["item_id", "popularity"]].itertuples(index=False)
    }

    def score(user_id: int, candidates: list[int], ratings: dict[int, float]) -> np.ndarray:
        content_scores = content.score_candidates(candidates, ratings=ratings)
        cf = model.predict_user(user_id, candidates, ratings)
        values = []
        for item in candidates:
            c = content_scores.get(int(item), 0.0)
            f = (cf.get(int(item), 3.0) - 0.5) / 4.5
            q = quality.get(int(item), 0.5)
            p = popularity.get(int(item), 0.0)
            values.append(w_content * c + w_cf * f + w_quality * q + w_pop * p)
        return np.asarray(values, dtype=np.float32)

    return score


def tune_hybrid_weights(
    validation: pd.DataFrame,
    train: pd.DataFrame,
    item_ids: list[int],
    catalog: pd.DataFrame,
    content: ContentModel,
    model: SVDModel,
    k: int = 10,
    max_users: int = 120,
) -> tuple[tuple[float, float, float, float], list[dict[str, float]]]:
    """Búsqueda en grilla de pesos del híbrido sobre validación.

    Para no recalcular los componentes en cada combinación, se precomputan
    content/CF/calidad/popularidad por usuario y luego se combinan.
    """
    train_items: dict[int, set[int]] = {}
    train_ratings: dict[int, dict[int, float]] = {}
    for user_id, group in train.groupby("userId"):
        train_items[int(user_id)] = set(int(x) for x in group["movieId"])
        train_ratings[int(user_id)] = {
            int(row.movieId): float(row.rating) for row in group.itertuples(index=False)
        }
    relevant: dict[int, set[int]] = {}
    for user_id, group in validation.groupby("userId"):
        items = set(int(x) for x in group.loc[group["rating"] >= 4.0, "movieId"])
        if items:
            relevant[int(user_id)] = items
    users = [u for u in relevant if u in train_items]
    rng = np.random.default_rng(42)
    if max_users and len(users) > max_users:
        users = [int(u) for u in rng.choice(users, size=max_users, replace=False)]

    quality = {
        int(row.item_id): float(row.bayes_rating) / 5.0
        for row in catalog[["item_id", "bayes_rating"]].itertuples(index=False)
    }
    max_pop = max(1e-9, float(catalog["popularity"].fillna(0).max()))
    popularity = {
        int(row.item_id): float(row.popularity or 0) / max_pop
        for row in catalog[["item_id", "popularity"]].itertuples(index=False)
    }

    cached: list[tuple[set[int], list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for user_id in users:
        seen = train_items[user_id]
        candidates = [item for item in item_ids if item not in seen]
        content_scores = content.score_candidates(candidates, ratings=train_ratings.get(user_id, {}))
        cf_predictions = model.predict_user(user_id, candidates, train_ratings.get(user_id, {}))
        c_array = np.asarray([content_scores.get(i, 0.0) for i in candidates], dtype=np.float32)
        f_array = np.asarray([(cf_predictions.get(i, 3.0) - 0.5) / 4.5 for i in candidates], dtype=np.float32)
        q_array = np.asarray([quality.get(i, 0.5) for i in candidates], dtype=np.float32)
        p_array = np.asarray([popularity.get(i, 0.0) for i in candidates], dtype=np.float32)
        cached.append((relevant[user_id], candidates, c_array, f_array, q_array, p_array))

    results: list[dict[str, float]] = []
    for w_content, w_cf, w_quality, w_pop in WEIGHT_GRID:
        ndcgs = []
        precisions = []
        for truth, candidates, c_array, f_array, q_array, p_array in cached:
            scores = w_content * c_array + w_cf * f_array + w_quality * q_array + w_pop * p_array
            top_idx = np.argpartition(-scores, k - 1)[:k]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            recommended = [candidates[int(i)] for i in top_idx]
            metrics = ranking_metrics(recommended, truth, k)
            ndcgs.append(metrics["ndcg"])
            precisions.append(metrics["precision"])
        results.append(
            {
                "content": w_content,
                "collaborative": w_cf,
                "quality": w_quality,
                "popularity": w_pop,
                f"ndcg@{k}": float(np.mean(ndcgs)),
                f"precision@{k}": float(np.mean(precisions)),
            }
        )
    best = max(results, key=lambda row: row[f"ndcg@{k}"])
    best_weights = (
        float(best["content"]),
        float(best["collaborative"]),
        float(best["quality"]),
        float(best["popularity"]),
    )
    return best_weights, results

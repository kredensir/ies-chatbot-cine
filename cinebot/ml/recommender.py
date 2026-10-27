"""Recomendador híbrido: contenido + colaborativo + calidad + popularidad.

El puntaje final es una combinación lineal:

    score = w_c * contenido + w_f * colaborativo + w_q * calidad + w_p * popularidad

donde los pesos se ajustan con el conjunto de validación (ver evaluation.py).
Además se aplica:
- filtrado por tipo, género, año, nota y cantidad de temporadas,
- exclusión de ítems ya vistos,
- penalización de diversidad (MMR aproximado con Jaccard de géneros),
- explicaciones en lenguaje natural,
- manejo de cold start y de series sin factores colaborativos.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cinebot import config
from cinebot.data.catalog import load_processed_catalog
from cinebot.db.connection import Database
from cinebot.db.repository import CatalogRepository, UserRepository
from cinebot.domain import MediaItem, QueryFilters, Recommendation, UserProfile
from cinebot.ml.clustering import UserClustering
from cinebot.ml.collaborative import SVDModel
from cinebot.ml.content import ContentModel


@dataclass
class RecommendationBatch:
    recommendations: list[Recommendation]
    meta: dict[str, Any]


class HybridRecommender:
    def __init__(
        self,
        db: Database,
        content: ContentModel | None = None,
        collaborative: SVDModel | None = None,
        clusters: UserClustering | None = None,
        catalog: pd.DataFrame | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.db = db
        self.catalog_repo = CatalogRepository(db)
        self.user_repo = UserRepository(db)
        self.content = content or ContentModel.load()
        self.collaborative = collaborative or SVDModel.load()
        self.clusters = clusters or UserClustering.load()
        self.catalog = catalog if catalog is not None else self._load_catalog()
        self.catalog_index = {
            int(row.item_id): row for row in self.catalog.itertuples(index=False)
        }
        self.genres_by_item = {
            int(row.item_id): [
                g for g in str(row.genres).split("|")
                if g and g.lower() not in {"nan", "none", ""}
            ]
            for row in self.catalog.itertuples(index=False)
        }
        self.max_popularity = max(
            1e-9, float(self.catalog["popularity"].fillna(0).max())
        )
        self.metadata = metadata if metadata is not None else self._load_metadata()
        weights = self.metadata.get("hybrid_weights", {})
        self.weights = (
            float(weights.get("content", 0.50)),
            float(weights.get("collaborative", 0.30)),
            float(weights.get("quality", 0.15)),
            float(weights.get("popularity", 0.05)),
        )
        cold = self.metadata.get("cold_start_weights", {})
        self.cold_weights = (
            float(cold.get("content", 0.55)),
            float(cold.get("collaborative", 0.0)),
            float(cold.get("quality", 0.30)),
            float(cold.get("popularity", 0.15)),
        )

    # ------------------------------------------------------------------
    # Carga auxiliar
    # ------------------------------------------------------------------
    def _load_catalog(self) -> pd.DataFrame:
        try:
            return load_processed_catalog()
        except Exception:
            rows = self.catalog_repo.dataset()
            return pd.DataFrame(rows)

    def _load_metadata(self) -> dict[str, Any]:
        path = config.model_path("metadata")
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: int,
        filters: QueryFilters | None = None,
        limit: int | None = None,
        exclude_ids: set[int] | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> RecommendationBatch:
        filters = copy.deepcopy(filters or QueryFilters())
        limit = max(1, min(limit or filters.count or config.DEFAULT_RECOMMENDATIONS, config.MAX_RECOMMENDATIONS))
        filters.count = limit
        profile = self.user_repo.profile(user_id)
        if profile is None:
            raise ValueError(f"Usuario {user_id} inexistente")

        ratings_rows = self.user_repo.ratings(user_id)
        all_ratings = {int(row["item_id"]): float(row["rating"]) for row in ratings_rows}
        app_ratings = {
            int(row["item_id"]): float(row["rating"])
            for row in ratings_rows
            if str(row.get("source")) == "app"
        }
        seen = self.user_repo.seen_item_ids(user_id) if filters.exclude_seen else set()
        excluded = set(exclude_ids or set()) | (seen if filters.exclude_seen else set())

        candidates, relaxed = self._candidates_with_fallback(filters, user_id)
        if excluded:
            candidates = [item for item in candidates if item.item_id not in excluded]
        if filters.similar_to:
            candidates = self._add_similar(filters.similar_to, candidates, filters.media_type)
        if not candidates:
            return RecommendationBatch(
                [],
                {
                    "cold_start": profile.is_cold_start,
                    "relaxed": relaxed,
                    "reason": "sin_candidatos",
                },
            )

        candidate_ids = [item.item_id for item in candidates]
        content_scores = self.content.score_candidates(
            candidate_ids,
            ratings=all_ratings,
            genres=filters.genres,
            moods=filters.moods,
            keywords=filters.keywords,
            similar_to=filters.similar_to,
        )

        cf_predictions: dict[int, float] = {}
        cf_used = False
        if not profile.is_cold_start and not filters.similar_to:
            cf_predictions, cf_used = self._collaborative_predictions(
                user_id, candidate_ids, all_ratings, app_ratings, profile
            )
        elif not profile.is_cold_start and filters.similar_to:
            cf_predictions, cf_used = self._collaborative_predictions(
                user_id, candidate_ids, all_ratings, app_ratings, profile
            )

        selected: list[tuple[MediaItem, float, float, float, float]] = []
        remaining = list(candidates)
        diversity_lambda = 0.04 if filters.similar_to else 0.14
        for _ in range(min(limit, len(remaining))):
            best_item = None
            best_values = None
            best_adjusted = -1e9
            for item in remaining:
                content_score = content_scores.get(item.item_id, 0.0)
                cf_score = 0.0
                has_cf = cf_used and item.item_id in cf_predictions and item.media_type == "movie"
                if has_cf:
                    cf_score = (cf_predictions[item.item_id] - 0.5) / 4.5
                quality = self._quality(item)
                popularity = min(1.0, max(0.0, item.popularity / self.max_popularity))
                w_c, w_f, w_q, w_p = (
                    self.cold_weights if profile.is_cold_start or not cf_used else self.weights
                )
                if not has_cf:
                    # Las series no tienen factores colaborativos: se reparte
                    # el peso del CF entre contenido y calidad.
                    extra = w_f
                    w_f = 0.0
                    w_c += extra * 0.6
                    w_q += extra * 0.4
                denominator = max(1e-9, w_c + w_f + w_q + w_p)
                score = (
                    w_c * content_score
                    + w_f * cf_score
                    + w_q * quality
                    + w_p * popularity
                ) / denominator

                if filters.genres:
                    overlap = len(set(item.genres) & set(filters.genres)) / len(filters.genres)
                    score += 0.07 * overlap
                if filters.similar_to:
                    similarity = self.content.similarity_between(filters.similar_to, item.item_id)
                    score += 0.08 * similarity
                if selected:
                    max_overlap = max(
                        _jaccard(set(item.genres), set(chosen.genres)) for chosen, *_ in selected
                    )
                    adjusted = score - diversity_lambda * max_overlap
                else:
                    adjusted = score
                if adjusted > best_adjusted:
                    best_adjusted = adjusted
                    best_item = item
                    best_values = (score, content_score, cf_score, quality, popularity)

            if best_item is None or best_values is None:
                break
            selected.append((best_item, *best_values))
            remaining = [item for item in remaining if item.item_id != best_item.item_id]

        recommendations: list[Recommendation] = []
        for rank, (item, score, content_score, cf_score, quality, popularity) in enumerate(selected, start=1):
            cf_prediction = cf_predictions.get(item.item_id)
            if cf_prediction is not None and item.media_type == "movie":
                predicted = 0.65 * cf_prediction + 0.35 * (quality * 5.0)
            else:
                predicted = 0.75 * (quality * 5.0) + 0.25 * (score * 5.0)
            predicted = float(min(5.0, max(0.5, predicted)))
            reason = self._explain(
                item=item,
                filters=filters,
                profile=profile,
                ratings=all_ratings,
                content_score=content_score,
                cf_score=cf_score,
                cf_used=cf_used,
            )
            recommendations.append(
                Recommendation(
                    item=item,
                    score=float(score),
                    content_score=float(content_score),
                    cf_score=float(cf_score),
                    quality_score=float(quality),
                    popularity_score=float(popularity),
                    predicted_rating=predicted,
                    reason=reason,
                    rank=rank,
                )
            )

        meta = {
            "cold_start": profile.is_cold_start,
            "cf_used": cf_used,
            "relaxed": relaxed,
            "weights": {
                "content": self.cold_weights[0] if profile.is_cold_start else self.weights[0],
                "collaborative": 0.0 if profile.is_cold_start else self.weights[1],
                "quality": self.weights[2],
                "popularity": self.weights[3],
            },
            "candidates": len(candidates),
            "profile_genres": [g.genre for g in profile.top_genres[:3]],
            "filters": {
                "media_type": filters.media_type,
                "genres": filters.genres,
                "moods": filters.moods,
                "year_min": filters.year_min,
                "year_max": filters.year_max,
                "min_rating": filters.min_rating,
                "similar_to": filters.similar_to,
                "max_seasons": filters.max_seasons,
            },
        }
        if extra_meta:
            meta.update(extra_meta)
        return RecommendationBatch(recommendations, meta)

    # ------------------------------------------------------------------
    # Candidatos
    # ------------------------------------------------------------------
    def _candidates_with_fallback(
        self, filters: QueryFilters, user_id: int
    ) -> tuple[list[MediaItem], list[str]]:
        attempts: list[tuple[str, QueryFilters]] = [("original", copy.deepcopy(filters))]
        loose = copy.deepcopy(filters)
        loose.min_rating = None
        attempts.append(("sin_nota_minima", copy.deepcopy(loose)))
        loose.year_min = None
        loose.year_max = None
        loose.max_seasons = None
        attempts.append(("sin_filtros_de_epoca", copy.deepcopy(loose)))
        loose.genres = []
        loose.moods = []
        attempts.append(("sin_genero", copy.deepcopy(loose)))
        loose.media_type = None
        attempts.append(("sin_tipo", copy.deepcopy(loose)))

        for label, attempt in attempts:
            items = self.catalog_repo.candidates(attempt, user_id=user_id)
            if len(items) >= max(5, filters.count):
                return items, [] if label == "original" else [label]
        return items, ["sin_tipo"]

    def _add_similar(
        self, similar_to: int, candidates: list[MediaItem], media_type: str | None = None
    ) -> list[MediaItem]:
        existing = {item.item_id for item in candidates}
        extra = []
        for item_id, _ in self.content.similar_items(similar_to, limit=30):
            if item_id not in existing:
                item = self.catalog_repo.get_item(item_id)
                if item and (not media_type or item.media_type == media_type):
                    extra.append(item)
        return extra + candidates

    def _collaborative_predictions(
        self,
        user_id: int,
        candidate_ids: list[int],
        all_ratings: dict[int, float],
        app_ratings: dict[int, float],
        profile: UserProfile,
    ) -> tuple[dict[int, float], bool]:
        if profile.n_ratings < 2:
            return {}, False
        if self.collaborative.knows_user(user_id):
            return self.collaborative.predict_user(
                user_id, candidate_ids, app_ratings or None
            ), True
        factor = self.collaborative.fold_in_user(all_ratings)
        if factor is None or factor.n_items < 2:
            return {}, False
        return self.collaborative.predict_with_factor(factor, candidate_ids), True

    # ------------------------------------------------------------------
    # Calidad, perfiles y explicaciones
    # ------------------------------------------------------------------
    def _quality(self, item: MediaItem) -> float:
        if item.media_type == "series":
            return min(1.0, max(0.0, item.rating_avg / 5.0))
        return min(1.0, max(0.0, item.bayes_rating / 5.0))

    def _explain(
        self,
        item: MediaItem,
        filters: QueryFilters,
        profile: UserProfile,
        ratings: dict[int, float],
        content_score: float,
        cf_score: float,
        cf_used: bool,
    ) -> str:
        parts: list[str] = []
        if filters.similar_to:
            source = self.catalog_index.get(int(filters.similar_to))
            if source is not None:
                title = str(source.title)
                similarity = self.content.similarity_between(filters.similar_to, item.item_id)
                if similarity >= 0.15:
                    parts.append(f"Muy parecida a {title}")
        if filters.genres:
            matched = [g for g in filters.genres if g in item.genres]
            if matched:
                parts.append(f"Coincide con {', '.join(matched[:2])}")
        if profile.top_genres:
            favorite = next(
                (g.genre for g in profile.top_genres if g.genre in item.genres), None
            )
            if favorite and not parts:
                parts.append(f"Afinidad con {favorite}, uno de tus géneros favoritos")
        if not filters.similar_to and ratings:
            source = self.content.best_explanation_source(item.item_id, ratings.keys())
            if source and len(parts) < 2:
                source_item = self.catalog_index.get(source[0])
                if source_item is not None:
                    parts.append(f"Porque te gustó {source_item.title}")
        if cf_used and cf_score >= 0.62 and len(parts) < 2:
            parts.append("Usuarios con gustos similares la valoraron muy bien")
        if profile.is_cold_start and len(parts) < 2:
            parts.append("Buena valoración y popularidad para empezar")
        if not parts:
            if item.media_type == "series" and item.seasons:
                parts.append(f"Serie de {item.seasons} temporadas con {item.rating_avg:.1f}/5")
            else:
                parts.append(f"Valoración media de {item.bayes_rating:.2f}/5")
        return " · ".join(parts[:2])

    def cluster_for(self, user_id: int, profile: UserProfile) -> tuple[int, str, str] | None:
        existing = self.clusters.profile_for_user(user_id)
        if existing:
            return existing
        ratings = {int(row["item_id"]): float(row["rating"]) for row in self.user_repo.ratings(user_id)}
        predicted = self.clusters.predict_profile(ratings, self.genres_by_item)
        if predicted:
            cluster_id, name, description = predicted
            self.user_repo.update_cluster(user_id, cluster_id, name, description)
            return predicted
        return None

    # ------------------------------------------------------------------
    # Similares / top
    # ------------------------------------------------------------------
    def similar_to(
        self, user_id: int, item_id: int, limit: int = 5, media_type: str | None = None
    ) -> RecommendationBatch:
        item = self.catalog_repo.get_item(item_id)
        if item is None:
            return RecommendationBatch([], {"reason": "item_inexistente"})
        filters = QueryFilters(
            media_type=media_type or item.media_type,
            similar_to=item_id,
            count=limit,
            exclude_seen=True,
        )
        return self.recommend(user_id, filters, limit=limit)

    def top_items(
        self, media_type: str | None = None, genre: str | None = None, limit: int = 10
    ) -> list[MediaItem]:
        if genre:
            return self.catalog_repo.by_genre([genre], media_type=media_type, limit=limit)
        return self.catalog_repo.top_rated(media_type=media_type, limit=limit)

    def refresh_models(self) -> None:
        self.content = ContentModel.load()
        self.collaborative = SVDModel.load()
        self.clusters = UserClustering.load()
        self.metadata = self._load_metadata()


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)

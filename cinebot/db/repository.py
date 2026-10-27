"""Repositorios de acceso a datos.

Cada repositorio encapsula las consultas SQL de un agregado del dominio:
catálogo, usuarios/valoraciones, conversaciones y métricas de modelos.
Todas las consultas usan parámetros (?) para prevenir inyección SQL.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Iterable

from cinebot import config
from cinebot.db.connection import Database
from cinebot.domain import (
    GenreTaste,
    MediaItem,
    QueryFilters,
    Recommendation,
    UserProfile,
)
from cinebot.text_utils import normalize_text, normalize_title


class CatalogRepository:
    """Consultas sobre `items`, `genres` y la vista `v_catalog`."""

    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Lectura de ítems
    # ------------------------------------------------------------------
    def get_item(self, item_id: int) -> MediaItem | None:
        row = self.db.query_one("SELECT * FROM v_catalog WHERE item_id = ?", (item_id,))
        return MediaItem.from_row(row) if row else None

    def get_items(self, item_ids: Iterable[int]) -> list[MediaItem]:
        ids = list(dict.fromkeys(int(i) for i in item_ids))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = self.db.query(
            f"SELECT * FROM v_catalog WHERE item_id IN ({placeholders})", ids
        )
        by_id = {int(row["item_id"]): MediaItem.from_row(row) for row in rows}
        return [by_id[i] for i in ids if i in by_id]

    def all_titles(self) -> list[dict[str, Any]]:
        return self.db.query(
            "SELECT item_id, title, title_norm, media_type, year FROM items"
        )

    def genres(self) -> list[str]:
        rows = self.db.query("SELECT name FROM genres ORDER BY name")
        return [str(row["name"]) for row in rows]

    def stats(self) -> dict[str, Any]:
        by_type = {
            row["media_type"]: int(row["total"])
            for row in self.db.query(
                "SELECT media_type, COUNT(*) AS total FROM items GROUP BY media_type"
            )
        }
        ratings = self.db.query_one(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN source = 'movielens' THEN 1 ELSE 0 END) AS movielens,
                SUM(CASE WHEN source = 'app' THEN 1 ELSE 0 END) AS app
            FROM ratings
            """
        ) or {}
        return {
            "items": sum(by_type.values()),
            "movies": by_type.get("movie", 0),
            "series": by_type.get("series", 0),
            "ratings": int(ratings.get("total") or 0),
            "movielens_ratings": int(ratings.get("movielens") or 0),
            "app_ratings": int(ratings.get("app") or 0),
            "genres": int(self.db.scalar("SELECT COUNT(*) FROM genres") or 0),
        }

    # ------------------------------------------------------------------
    # Búsquedas
    # ------------------------------------------------------------------
    def search_items(
        self, text: str, media_type: str | None = None, limit: int = 10
    ) -> list[MediaItem]:
        """Búsqueda de texto completo con FTS5 (con respaldo LIKE)."""
        tokens = [t for t in normalize_text(text).split() if len(t) > 1]
        if not tokens:
            return []
        match_query = " AND ".join(f'"{token}"*' for token in tokens)
        params: list[Any] = [match_query]
        type_clause = ""
        if media_type in ("movie", "series"):
            type_clause = " AND v.media_type = ?"
            params.append(media_type)
        params.append(limit)
        try:
            rows = self.db.query(
                f"""
                SELECT v.*
                FROM items_fts f
                JOIN v_catalog v ON v.item_id = f.rowid
                WHERE items_fts MATCH ? {type_clause}
                ORDER BY bm25(items_fts, 8.0, 10.0, 1.0, 2.0), v.bayes_rating DESC
                LIMIT ?
                """,
                params,
            )
        except Exception:
            # Respaldo si FTS5 no está disponible en la compilación de SQLite.
            like = f"%{normalize_text(text)}%"
            type_clause = " AND media_type = ?" if media_type in ("movie", "series") else ""
            like_params: list[Any] = [like]
            if type_clause:
                like_params.append(media_type)
            like_params.append(limit)
            rows = self.db.query(
                f"SELECT * FROM v_catalog WHERE title_norm LIKE ? {type_clause} LIMIT ?",
                like_params,
            )
        return [MediaItem.from_row(row) for row in rows]

    def find_title(
        self, query: str, media_type: str | None = None, limit: int = 5
    ) -> list[MediaItem]:
        norm = normalize_title(query)
        if not norm:
            return []
        type_clause = " AND media_type = ?" if media_type in ("movie", "series") else ""
        query_params: list[Any] = [f"%{norm}%"]
        if type_clause:
            query_params.append(media_type)
        query_params.extend([norm, limit])
        rows = self.db.query(
            f"""
            SELECT * FROM v_catalog
            WHERE title_norm LIKE ? {type_clause}
            ORDER BY CASE WHEN title_norm = ? THEN 0 ELSE 1 END, bayes_rating DESC
            LIMIT ?
            """,
            query_params,
        )
        return [MediaItem.from_row(row) for row in rows]

    def top_rated(self, media_type: str | None = None, limit: int = 10) -> list[MediaItem]:
        type_clause = " WHERE media_type = ?" if media_type in ("movie", "series") else ""
        params: list[Any] = [media_type] if type_clause else []
        params.append(limit)
        rows = self.db.query(
            f"""
            SELECT * FROM v_catalog
            {type_clause}
            ORDER BY bayes_rating DESC, rating_count DESC
            LIMIT ?
            """,
            params,
        )
        return [MediaItem.from_row(row) for row in rows]

    def by_genre(
        self,
        genres: list[str],
        media_type: str | None = None,
        limit: int = 20,
        min_year: int | None = None,
    ) -> list[MediaItem]:
        if not genres:
            return []
        placeholders = ",".join("?" for _ in genres)
        clauses = [
            f"""EXISTS (
                SELECT 1 FROM item_genres ig
                JOIN genres g ON g.genre_id = ig.genre_id
                WHERE ig.item_id = v.item_id AND g.name IN ({placeholders})
            )"""
        ]
        params: list[Any] = list(genres)
        if media_type in ("movie", "series"):
            clauses.append("v.media_type = ?")
            params.append(media_type)
        if min_year:
            clauses.append("v.year >= ?")
            params.append(min_year)
        params.append(limit)
        rows = self.db.query(
            f"""
            SELECT v.* FROM v_catalog v
            WHERE {' AND '.join(clauses)}
            ORDER BY v.bayes_rating DESC, v.popularity DESC
            LIMIT ?
            """,
            params,
        )
        return [MediaItem.from_row(row) for row in rows]

    # ------------------------------------------------------------------
    # Candidatos para el recomendador
    # ------------------------------------------------------------------
    def candidates(
        self, filters: QueryFilters, user_id: int | None = None, hard_limit: int = 12000
    ) -> list[MediaItem]:
        """Filtra el catálogo con SQL antes de puntuar en memoria.

        Se demuestra el uso combinado de WHERE dinámico, EXISTS, subconsultas
        y parámetros para evitar inyección SQL.
        """
        clauses: list[str] = ["1 = 1"]
        params: list[Any] = []

        if filters.media_type in ("movie", "series"):
            clauses.append("v.media_type = ?")
            params.append(filters.media_type)
        if filters.year_min:
            clauses.append("v.year >= ?")
            params.append(filters.year_min)
        if filters.year_max:
            clauses.append("v.year <= ?")
            params.append(filters.year_max)
        if filters.min_rating:
            clauses.append("v.bayes_rating >= ?")
            params.append(filters.min_rating)
        if filters.max_seasons:
            clauses.append("v.seasons IS NOT NULL AND v.seasons <= ?")
            params.append(filters.max_seasons)
        if filters.genres:
            placeholders = ",".join("?" for _ in filters.genres)
            clauses.append(
                f"""EXISTS (
                    SELECT 1 FROM item_genres ig
                    JOIN genres g ON g.genre_id = ig.genre_id
                    WHERE ig.item_id = v.item_id AND g.name IN ({placeholders})
                )"""
            )
            params.extend(filters.genres)
        if filters.exclude_seen and user_id:
            clauses.append(
                """v.item_id NOT IN (
                    SELECT item_id FROM ratings WHERE user_id = ?
                    UNION
                    SELECT item_id FROM feedback
                    WHERE user_id = ? AND action IN ('seen', 'not_interested')
                )"""
            )
            params.extend([user_id, user_id])

        params.append(hard_limit)
        rows = self.db.query(
            f"""
            SELECT v.* FROM v_catalog v
            WHERE {' AND '.join(clauses)}
            ORDER BY v.bayes_rating DESC, v.popularity DESC
            LIMIT ?
            """,
            params,
        )
        return [MediaItem.from_row(row) for row in rows]

    def dataset(self) -> list[dict[str, Any]]:
        """Catálogo completo como filas planas (para modelos y exportaciones)."""
        return self.db.query("SELECT * FROM v_catalog ORDER BY item_id")

    def set_poster_url(self, item_id: int, poster_url: str, overwrite: bool = False) -> None:
        """Guarda una URL de portada; por defecto respeta una ya existente."""
        if not poster_url:
            return
        if overwrite:
            self.db.execute(
                "UPDATE items SET poster_url = ?, updated_at = datetime('now') WHERE item_id = ?",
                (poster_url, int(item_id)),
            )
            return
        self.db.execute(
            """
            UPDATE items
            SET poster_url = ?, updated_at = datetime('now')
            WHERE item_id = ? AND (poster_url IS NULL OR poster_url = '')
            """,
            (poster_url, int(item_id)),
        )


class UserRepository:
    """Usuarios, valoraciones, feedback y perfil de gustos."""

    def __init__(self, db: Database) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Usuarios
    # ------------------------------------------------------------------
    def get_user(self, user_id: int) -> dict[str, Any] | None:
        return self.db.query_one("SELECT * FROM users WHERE user_id = ?", (user_id,))

    def get_or_create_user(
        self,
        username: str,
        source: str = "local",
        is_demo: bool = False,
        avatar_color: str = "#7C4DFF",
    ) -> int:
        row = self.db.query_one(
            "SELECT user_id FROM users WHERE username = ? AND source = ?",
            (username, source),
        )
        if row:
            return int(row["user_id"])
        max_id = int(self.db.scalar("SELECT COALESCE(MAX(user_id), 0) FROM users") or 0)
        user_id = max(100001, max_id + 1) if source == "local" else max_id + 1
        self.db.execute(
            """
            INSERT INTO users (user_id, username, source, is_demo, avatar_color)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, username, source, int(is_demo), avatar_color),
        )
        return user_id

    def get_or_create_guest(self) -> int:
        return self.get_or_create_user(config.DEFAULT_USER_NAME, source="local")

    def list_demo_users(self, limit: int = 12) -> list[UserProfile]:
        rows = self.db.query(
            """
            SELECT * FROM v_user_stats
            WHERE is_demo = 1 AND n_ratings > 0
            ORDER BY n_ratings DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [self._profile_from_row(row) for row in rows]

    def list_local_users(self) -> list[UserProfile]:
        rows = self.db.query(
            "SELECT * FROM v_user_stats WHERE source = 'local' ORDER BY user_id"
        )
        return [self._profile_from_row(row) for row in rows]

    def profile(self, user_id: int) -> UserProfile | None:
        row = self.db.query_one("SELECT * FROM v_user_stats WHERE user_id = ?", (user_id,))
        if not row:
            return None
        profile = self._profile_from_row(row)
        user = self.get_user(user_id) or {}
        profile.cluster_name = str(user.get("cluster_name") or "")
        profile.cluster_description = str(user.get("cluster_description") or "")
        return profile

    def _profile_from_row(self, row: dict[str, Any]) -> UserProfile:
        tastes = self.genre_taste(int(row["user_id"]), limit=6)
        return UserProfile(
            user_id=int(row["user_id"]),
            username=str(row["username"]),
            source=str(row["source"]),
            is_demo=bool(row["is_demo"]),
            n_ratings=int(row["n_ratings"] or 0),
            avg_rating=float(row["avg_rating"] or 0),
            n_likes=int(row["n_likes"] or 0),
            last_rating_at=row.get("last_rating_at"),
            top_genres=tastes,
        )

    # ------------------------------------------------------------------
    # Gustos y valoraciones
    # ------------------------------------------------------------------
    def genre_taste(self, user_id: int, limit: int = 6) -> list[GenreTaste]:
        rows = self.db.query(
            """
            SELECT genre, n_ratings, avg_rating, affinity
            FROM v_user_genre_taste
            WHERE user_id = ? AND n_ratings >= 1
            ORDER BY affinity DESC, n_ratings DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        return [
            GenreTaste(
                genre=str(row["genre"]),
                n_ratings=int(row["n_ratings"]),
                avg_rating=float(row["avg_rating"]),
                affinity=float(row["affinity"]),
            )
            for row in rows
        ]

    def ratings(self, user_id: int) -> list[dict[str, Any]]:
        """Valoraciones efectivas (app preferida sobre MovieLens)."""
        return self.db.query(
            """
            WITH effective AS (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY user_id, item_id
                    ORDER BY CASE source WHEN 'app' THEN 0 ELSE 1 END
                ) AS rn
                FROM ratings WHERE user_id = ?
            )
            SELECT e.item_id, e.rating, e.rated_at, e.source,
                   i.title, i.media_type, i.year, i.genres
            FROM effective e
            JOIN v_catalog i ON i.item_id = e.item_id
            WHERE e.rn = 1
            ORDER BY e.rated_at DESC
            """,
            (user_id,),
        )

    def rating_for(self, user_id: int, item_id: int) -> dict[str, Any] | None:
        return self.db.query_one(
            """
            SELECT rating, source, rated_at FROM ratings
            WHERE user_id = ? AND item_id = ?
            ORDER BY CASE source WHEN 'app' THEN 0 ELSE 1 END
            LIMIT 1
            """,
            (user_id, item_id),
        )

    def upsert_rating(
        self, user_id: int, item_id: int, rating: float, source: str = "app"
    ) -> None:
        rating = max(0.5, min(5.0, float(rating)))
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO ratings (user_id, item_id, rating, rated_at, source)
                VALUES (?, ?, ?, datetime('now'), ?)
                ON CONFLICT(user_id, item_id, source)
                DO UPDATE SET rating = excluded.rating, rated_at = datetime('now')
                """,
                (user_id, item_id, rating, source),
            )
            connection.execute(
                """
                INSERT INTO interactions (user_id, item_id, event_type, value)
                VALUES (?, ?, 'rate', ?)
                """,
                (user_id, item_id, rating),
            )

    def delete_rating(self, user_id: int, item_id: int) -> None:
        self.db.execute(
            "DELETE FROM ratings WHERE user_id = ? AND item_id = ? AND source = 'app'",
            (user_id, item_id),
        )

    def reset_ratings(self, user_id: int) -> int:
        return self.db.execute(
            "DELETE FROM ratings WHERE user_id = ? AND source = 'app'", (user_id,)
        )

    def seen_item_ids(self, user_id: int) -> set[int]:
        rows = self.db.query(
            """
            SELECT item_id FROM ratings WHERE user_id = ?
            UNION
            SELECT item_id FROM feedback
            WHERE user_id = ? AND action IN ('seen', 'not_interested')
            """,
            (user_id, user_id),
        )
        return {int(row["item_id"]) for row in rows}

    def add_feedback(self, user_id: int, item_id: int, action: str) -> None:
        if action not in {"like", "dislike", "seen", "not_interested"}:
            raise ValueError(f"Acción de feedback inválida: {action}")
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO feedback (user_id, item_id, action)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, item_id, action) DO UPDATE SET
                    created_at = datetime('now')
                """,
                (user_id, item_id, action),
            )
            connection.execute(
                """
                INSERT INTO interactions (user_id, item_id, event_type)
                VALUES (?, ?, ?)
                """,
                (user_id, item_id, action),
            )

    def log_interaction(
        self,
        user_id: int,
        item_id: int | None,
        event_type: str,
        value: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.db.execute(
            """
            INSERT INTO interactions (user_id, item_id, event_type, value, metadata_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, item_id, event_type, value, json.dumps(metadata or {}, ensure_ascii=False)),
        )

    def update_cluster(
        self, user_id: int, cluster_id: int, name: str, description: str
    ) -> None:
        self.db.execute(
            """
            UPDATE users
            SET cluster_id = ?, cluster_name = ?, cluster_description = ?
            WHERE user_id = ?
            """,
            (cluster_id, name, description, user_id),
        )


class ChatRepository:
    """Persistencia de conversaciones, mensajes y recomendaciones mostradas."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def create_conversation(
        self, user_id: int, title: str = "Nueva conversación", model_version: str = "1.0.0"
    ) -> int:
        return self.db.execute(
            """
            INSERT INTO conversations (user_id, title, model_version)
            VALUES (?, ?, ?)
            """,
            (user_id, title, model_version),
        )

    def end_conversation(self, conversation_id: int) -> None:
        self.db.execute(
            "UPDATE conversations SET ended_at = datetime('now') WHERE conversation_id = ?",
            (conversation_id,),
        )

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        intent: str | None = None,
        confidence: float | None = None,
        entities: dict[str, Any] | None = None,
    ) -> int:
        return self.db.execute(
            """
            INSERT INTO messages (conversation_id, role, content, intent, confidence, entities_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                role,
                content,
                intent,
                confidence,
                json.dumps(entities or {}, ensure_ascii=False),
            ),
        )

    def add_recommendations(
        self, message_id: int, user_id: int, recommendations: list[Recommendation]
    ) -> None:
        if not recommendations:
            return
        rows = [
            (
                message_id,
                user_id,
                rec.item.item_id,
                rec.rank,
                rec.score,
                rec.content_score,
                rec.cf_score,
                rec.quality_score,
                rec.reason,
            )
            for rec in recommendations
        ]
        self.db.execute_many(
            """
            INSERT INTO recommendations (
                message_id, user_id, item_id, rank, score,
                content_score, cf_score, quality_score, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def messages(self, conversation_id: int, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.db.query(
            """
            SELECT * FROM (
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY message_id DESC
                LIMIT ?
            ) ORDER BY message_id ASC
            """,
            (conversation_id, limit),
        )
        return rows

    def conversations(self, user_id: int, limit: int = 10) -> list[dict[str, Any]]:
        return self.db.query(
            """
            SELECT c.*, COUNT(m.message_id) AS n_messages
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.conversation_id
            WHERE c.user_id = ?
            GROUP BY c.conversation_id
            ORDER BY c.started_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )

    def mark_recommendation_clicked(self, recommendation_id: int) -> None:
        self.db.execute(
            "UPDATE recommendations SET clicked = 1 WHERE recommendation_id = ?",
            (recommendation_id,),
        )

    def recent_recommendations(self, user_id: int, limit: int = 10) -> list[dict[str, Any]]:
        return self.db.query(
            """
            SELECT rec.*, i.title, i.media_type, i.year
            FROM recommendations rec
            JOIN items i ON i.item_id = rec.item_id
            WHERE rec.user_id = ?
            ORDER BY rec.shown_at DESC, rec.rank ASC
            LIMIT ?
            """,
            (user_id, limit),
        )

    def recommendation_click_rate(self) -> float:
        row = self.db.query_one(
            "SELECT COUNT(*) AS total, COALESCE(SUM(clicked), 0) AS clicks FROM recommendations"
        )
        total = int(row["total"] or 0) if row else 0
        clicks = int(row["clicks"] or 0) if row else 0
        return (clicks / total) if total else 0.0


class MetricsRepository:
    """Registro de corridas de entrenamiento y métricas de evaluación."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def start_run(
        self,
        model_name: str,
        params: dict[str, Any] | None = None,
        notes: str = "",
    ) -> int:
        return self.db.execute(
            """
            INSERT INTO training_runs (model_name, params_json, notes)
            VALUES (?, ?, ?)
            """,
            (model_name, json.dumps(params or {}, ensure_ascii=False), notes),
        )

    def finish_run(
        self,
        run_id: int,
        n_users: int | None = None,
        n_items: int | None = None,
        n_ratings: int | None = None,
    ) -> None:
        self.db.execute(
            """
            UPDATE training_runs
            SET finished_at = datetime('now'), n_users = ?, n_items = ?, n_ratings = ?
            WHERE run_id = ?
            """,
            (n_users, n_items, n_ratings, run_id),
        )

    def save_metric(
        self,
        model_name: str,
        metric_name: str,
        value: float,
        run_id: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.db.execute(
            """
            INSERT INTO model_metrics (run_id, model_name, metric_name, value, params_json)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(model_name, metric_name, run_id) DO UPDATE SET
                value = excluded.value,
                params_json = excluded.params_json,
                evaluated_at = datetime('now')
            """,
            (run_id, model_name, metric_name, float(value), json.dumps(params or {}, ensure_ascii=False)),
        )

    def save_metrics(
        self,
        model_name: str,
        metrics: dict[str, float],
        run_id: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.save_metric(model_name, name, float(value), run_id, params)

    def latest_metrics(self, model_name: str | None = None) -> list[dict[str, Any]]:
        clause = "WHERE model_name = ?" if model_name else ""
        params: list[Any] = [model_name] if model_name else []
        return self.db.query(
            f"""
            SELECT * FROM model_metrics {clause}
            ORDER BY evaluated_at DESC, metric_id DESC
            """,
            params,
        )

    def metric_map(self, model_name: str) -> dict[str, float]:
        rows = self.db.query(
            """
            SELECT metric_name, value FROM model_metrics
            WHERE model_name = ?
              AND evaluated_at = (
                  SELECT MAX(evaluated_at) FROM model_metrics WHERE model_name = ?
              )
            """,
            (model_name, model_name),
        )
        return {str(row["metric_name"]): float(row["value"]) for row in rows}

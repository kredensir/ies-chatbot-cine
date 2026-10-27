"""Modelos de dominio compartidos por la base de datos, el motor y la UI.

Se usan dataclasses para mantener un contrato claro entre capas y evitar
diccionarios "mágicos" dispersos por el código (programación orientada a objetos).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _as_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None and str(value) != "" else None
    except (TypeError, ValueError):
        return None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return "" if text.lower() == "nan" else text


@dataclass(slots=True)
class MediaItem:
    item_id: int
    media_type: str
    title: str
    year: int | None
    genres: list[str]
    overview: str = ""
    runtime: int | None = None
    seasons: int | None = None
    episodes: int | None = None
    rating_avg: float = 0.0
    rating_count: int = 0
    bayes_rating: float = 0.0
    popularity: float = 0.0
    source: str = ""
    source_url: str = ""
    poster_url: str = ""
    tmdb_id: int | None = None
    tvmaze_id: int | None = None
    local_rating_avg: float = 0.0
    local_rating_count: int = 0

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "MediaItem":
        genres_value = row.get("genres") or row.get("genres_text") or ""
        if isinstance(genres_value, str):
            genres = [g for g in genres_value.split("|") if g]
        elif isinstance(genres_value, (list, tuple)):
            genres = [str(g) for g in genres_value]
        else:
            genres = []
        return cls(
            item_id=int(row["item_id"]),
            media_type=str(row.get("media_type", "movie")),
            title=str(row.get("title", "")),
            year=_as_int(row.get("year")),
            genres=genres,
            overview=_as_text(row.get("overview")),
            runtime=_as_int(row.get("runtime")),
            seasons=_as_int(row.get("seasons")),
            episodes=_as_int(row.get("episodes")),
            rating_avg=_as_float(row.get("rating_avg")),
            rating_count=int(row.get("rating_count") or 0),
            bayes_rating=_as_float(row.get("bayes_rating")),
            popularity=_as_float(row.get("popularity")),
            source=_as_text(row.get("source")),
            source_url=_as_text(row.get("source_url")),
            poster_url=_as_text(row.get("poster_url")),
            tmdb_id=_as_int(row.get("tmdb_id")),
            tvmaze_id=_as_int(row.get("tvmaze_id")),
            local_rating_avg=_as_float(row.get("local_rating_avg")),
            local_rating_count=int(row.get("local_rating_count") or 0),
        )

    @property
    def type_label(self) -> str:
        return "Película" if self.media_type == "movie" else "Serie"

    @property
    def single_label(self) -> str:
        return "película" if self.media_type == "movie" else "serie"

    @property
    def year_label(self) -> str:
        if not self.year:
            return "s/f"
        return str(self.year)

    @property
    def rating_label(self) -> str:
        if self.media_type == "series" and self.rating_avg:
            return f"{self.rating_avg:.1f}/5"
        return f"{self.bayes_rating:.2f}/5"

    @property
    def duration_label(self) -> str:
        if self.media_type == "series":
            parts = []
            if self.seasons:
                parts.append(f"{self.seasons} temp.")
            if self.episodes:
                parts.append(f"{self.episodes} ep.")
            if self.runtime:
                parts.append(f"~{self.runtime} min/ep.")
            return " · ".join(parts) or "Serie"
        if self.runtime:
            return f"{self.runtime} min"
        return "Película"

    @property
    def external_url(self) -> str:
        if self.source_url:
            return self.source_url
        if self.tmdb_id:
            return f"https://www.themoviedb.org/movie/{self.tmdb_id}"
        if self.tvmaze_id:
            return f"https://www.tvmaze.com/shows/{self.tvmaze_id}"
        return ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "media_type": self.media_type,
            "title": self.title,
            "year": self.year,
            "genres": self.genres,
            "overview": self.overview,
            "rating_avg": self.rating_avg,
            "bayes_rating": self.bayes_rating,
            "rating_count": self.rating_count,
            "source_url": self.external_url,
        }


@dataclass(slots=True)
class GenreTaste:
    genre: str
    n_ratings: int
    avg_rating: float
    affinity: float

    @property
    def label(self) -> str:
        return f"{self.genre} ({self.avg_rating:.1f}★)"


@dataclass(slots=True)
class UserProfile:
    user_id: int
    username: str
    source: str = "local"
    is_demo: bool = False
    n_ratings: int = 0
    avg_rating: float = 0.0
    n_likes: int = 0
    last_rating_at: str | None = None
    top_genres: list[GenreTaste] = field(default_factory=list)
    cluster_name: str = ""
    cluster_description: str = ""

    @property
    def is_cold_start(self) -> bool:
        return self.n_ratings < 3

    @property
    def taste_summary(self) -> str:
        if not self.top_genres:
            return "Todavía no tengo suficientes valoraciones tuyas."
        names = ", ".join(g.genre for g in self.top_genres[:3])
        return f"Tus géneros con más afinidad: {names}."


@dataclass(slots=True)
class QueryFilters:
    media_type: str | None = None
    genres: list[str] = field(default_factory=list)
    moods: list[str] = field(default_factory=list)
    year_min: int | None = None
    year_max: int | None = None
    min_rating: float | None = None
    count: int = 5
    exclude_seen: bool = True
    similar_to: int | None = None
    keywords: list[str] = field(default_factory=list)
    max_seasons: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Representación serializable de los filtros (para metadatos y JSON)."""
        return {
            "media_type": self.media_type,
            "genres": list(self.genres),
            "moods": list(self.moods),
            "year_min": self.year_min,
            "year_max": self.year_max,
            "min_rating": self.min_rating,
            "count": self.count,
            "exclude_seen": self.exclude_seen,
            "similar_to": self.similar_to,
            "keywords": list(self.keywords),
            "max_seasons": self.max_seasons,
        }

    def is_empty(self) -> bool:
        return not (
            self.media_type
            or self.genres
            or self.moods
            or self.year_min
            or self.year_max
            or self.min_rating
            or self.similar_to
            or self.keywords
            or self.max_seasons
        )

    def merge(self, other: "QueryFilters") -> "QueryFilters":
        """Combina dos filtros; `other` tiene prioridad en los campos escalares."""
        return QueryFilters(
            media_type=other.media_type or self.media_type,
            genres=list(dict.fromkeys(self.genres + other.genres)),
            moods=list(dict.fromkeys(self.moods + other.moods)),
            year_min=other.year_min if other.year_min is not None else self.year_min,
            year_max=other.year_max if other.year_max is not None else self.year_max,
            min_rating=other.min_rating if other.min_rating is not None else self.min_rating,
            count=other.count or self.count,
            exclude_seen=other.exclude_seen if other is not None else self.exclude_seen,
            similar_to=other.similar_to if other.similar_to is not None else self.similar_to,
            keywords=list(dict.fromkeys(self.keywords + other.keywords)),
            max_seasons=other.max_seasons if other.max_seasons is not None else self.max_seasons,
        )


@dataclass(slots=True)
class NLUResult:
    raw_text: str
    intent: str
    confidence: float
    filters: QueryFilters = field(default_factory=QueryFilters)
    title_query: str | None = None
    matched_item_id: int | None = None
    rating_value: float | None = None
    entities: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Recommendation:
    item: MediaItem
    score: float
    content_score: float = 0.0
    cf_score: float = 0.0
    quality_score: float = 0.0
    popularity_score: float = 0.0
    predicted_rating: float = 0.0
    reason: str = ""
    rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item.item_id,
            "title": self.item.title,
            "media_type": self.item.media_type,
            "year": self.item.year,
            "genres": self.item.genres,
            "score": round(self.score, 4),
            "predicted_rating": round(self.predicted_rating, 2),
            "reason": self.reason,
            "rank": self.rank,
        }


@dataclass(slots=True)
class ChatResponse:
    text: str
    recommendations: list[Recommendation] = field(default_factory=list)
    intent: str = "desconocido"
    confidence: float = 0.0
    followups: list[str] = field(default_factory=list)
    filters: QueryFilters = field(default_factory=QueryFilters)
    meta: dict[str, Any] = field(default_factory=dict)
    conversation_id: int | None = None
    message_id: int | None = None

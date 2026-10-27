"""Ingestión, limpieza y unificación del catálogo de películas y series.

Conceptos de bases de datos / data wrangling aplicados:
- lectura de fuentes heterogéneas (CSV de MovieLens y CSV curado de TVmaze),
- tipado explícito y tratamiento de valores nulos,
- eliminación de duplicados,
- normalización de claves y textos,
- construcción de una clave primaria común (`item_id`) para el modelo relacional,
- agregación de valoraciones para obtener métricas por ítem,
- separación entre datos crudos (`data/raw`) y procesados (`data/processed`).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

from cinebot import config
from cinebot.text_utils import clean_movie_title, extract_year, normalize_text, normalize_title

MOVIE_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]
EXTRA_GENRES = ["Adult", "Food", "History", "Legal", "Medical", "Sports"]
ALL_GENRES = MOVIE_GENRES + EXTRA_GENRES


def _split_genres(value: str) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [g.strip() for g in value.split("|") if g.strip() and g.strip() != "(no genres listed)"]


def load_movielens(
    movies_path: Path | None = None,
    ratings_path: Path | None = None,
    tags_path: Path | None = None,
    links_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga y limpia películas/valoraciones de MovieLens.

    Devuelve (películas, valoraciones). Si faltan archivos lanza un error
    explicativo con el comando que debe ejecutarse.
    """
    movies_path = Path(movies_path or config.MOVIES_CSV)
    ratings_path = Path(ratings_path or config.RATINGS_CSV)
    tags_path = Path(tags_path or config.TAGS_CSV)
    links_path = Path(links_path or config.LINKS_CSV)

    required = [movies_path, ratings_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan datos de MovieLens: "
            + ", ".join(missing)
            + ". Ejecute: python scripts/download_data.py"
        )

    movies = pd.read_csv(movies_path, dtype={"movieId": "int64", "title": "string", "genres": "string"})
    ratings = pd.read_csv(
        ratings_path,
        dtype={"userId": "int64", "movieId": "int64", "rating": "float64", "timestamp": "int64"},
    )

    # --- Calidad de datos: duplicados y nulos ---
    movies = movies.drop_duplicates(subset=["movieId"], keep="first")
    ratings = ratings.drop_duplicates(subset=["userId", "movieId"], keep="last")
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
    ratings = ratings[(ratings["rating"] >= 0.5) & (ratings["rating"] <= 5.0)]
    movies["title"] = movies["title"].fillna("").astype(str).str.strip()
    movies["genres"] = movies["genres"].fillna("(no genres listed)").astype(str)
    movies = movies[movies["title"] != ""]

    # --- Métricas agregadas por película ---
    stats = (
        ratings.groupby("movieId")["rating"]
        .agg(rating_avg="mean", rating_count="count")
        .reset_index()
    )
    movies = movies.merge(stats, on="movieId", how="left")
    movies["rating_avg"] = movies["rating_avg"].fillna(0.0).round(3)
    movies["rating_count"] = movies["rating_count"].fillna(0).astype(int)

    # Promedio bayesiano: evita que una película con una sola nota de 5
    # aparezca por encima de clásicos con cientos de valoraciones.
    global_mean = float(ratings["rating"].mean()) if len(ratings) else 3.5
    prior_weight = 25.0
    movies["bayes_rating"] = (
        (movies["rating_count"] * movies["rating_avg"] + prior_weight * global_mean)
        / (movies["rating_count"] + prior_weight)
    ).round(3)

    # --- Tags: palabras clave reales aportadas por los usuarios ---
    tag_keywords: dict[int, str] = {}
    if tags_path.exists():
        tags = pd.read_csv(tags_path, dtype={"movieId": "int64", "tag": "string"})
        tags = tags.dropna(subset=["tag"])
        tags["tag"] = tags["tag"].astype(str).str.strip()
        tags = tags[(tags["tag"] != "") & (tags["tag"].str.len() <= 60)]
        grouped = tags.groupby("movieId")["tag"].apply(lambda values: " | ".join(dict.fromkeys(values)))
        tag_keywords = grouped.to_dict()

    # --- Enlaces a TMDB / IMDb (sin consumir la API) ---
    tmdb_ids: dict[int, int | None] = {}
    imdb_ids: dict[int, str] = {}
    if links_path.exists():
        links = pd.read_csv(links_path, dtype={"movieId": "int64"})
        for row in links.itertuples(index=False):
            tmdb_ids[int(row.movieId)] = int(row.tmdbId) if pd.notna(row.tmdbId) else None
            imdb_value = str(row.imdbId) if pd.notna(row.imdbId) else ""
            imdb_ids[int(row.movieId)] = imdb_value.zfill(7) if imdb_value.isdigit() else imdb_value

    poster_urls: dict[int, str] = {}
    if config.MOVIE_POSTERS_CSV.exists():
        posters = pd.read_csv(
            config.MOVIE_POSTERS_CSV,
            dtype={"item_id": "int64", "poster_url": "string"},
        ).dropna(subset=["item_id", "poster_url"])
        poster_urls = {
            int(row.item_id): str(row.poster_url).strip()
            for row in posters.itertuples(index=False)
            if str(row.poster_url).strip()
        }

    rows = []
    for row in movies.itertuples(index=False):
        movie_id = int(row.movieId)
        title = clean_movie_title(row.title)
        year = extract_year(row.title)
        genres = _split_genres(row.genres)
        genre_text = " ".join(g.lower() for g in genres)
        decade = f"{(year // 10) * 10}s" if year else ""
        tags_text = tag_keywords.get(movie_id, "")
        keywords = " ".join(
            part
            for part in [
                normalize_text(title),
                " ".join(g.lower() for g in genres),
                decade,
                normalize_text(tags_text),
                "pelicula movie film cine",
            ]
            if part
        )
        tmdb_id = tmdb_ids.get(movie_id)
        rows.append(
            {
                "item_id": movie_id,
                "tmdb_id": tmdb_id,
                "imdb_id": imdb_ids.get(movie_id, ""),
                "tvmaze_id": None,
                "media_type": "movie",
                "title": title,
                "title_norm": normalize_title(title),
                "year": year,
                "end_year": None,
                "genres": "|".join(genres),
                "overview": "",
                "keywords": keywords[:1500],
                "runtime": None,
                "seasons": None,
                "episodes": None,
                "rating_avg": float(row.rating_avg),
                "rating_count": int(row.rating_count),
                "bayes_rating": float(row.bayes_rating),
                "popularity": math.log1p(int(row.rating_count)),
                "source": "MovieLens",
                "source_url": (
                    f"https://www.themoviedb.org/movie/{tmdb_id}" if tmdb_id else ""
                ),
                "poster_url": poster_urls.get(movie_id, ""),
            }
        )
    movies_df = pd.DataFrame(rows)
    return movies_df, ratings


def load_series(path: Path | None = None) -> pd.DataFrame:
    """Carga el catálogo curado de series construido desde TVmaze."""
    path = Path(path or config.SERIES_CSV)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el catálogo de series: {path}. "
            "Ejecute: python scripts/build_series_catalog.py"
        )
    series = pd.read_csv(path, dtype={"series_id": "int64", "tvmaze_id": "int64"})
    series = series.drop_duplicates(subset=["series_id"], keep="first")

    rows = []
    for row in series.itertuples(index=False):
        title = str(row.title).strip()
        genres = _split_genres(row.genres)
        overview = str(row.overview).strip() if pd.notna(row.overview) else ""
        year = int(row.start_year) if pd.notna(row.start_year) and int(row.start_year) > 0 else None
        end_year = int(row.end_year) if pd.notna(row.end_year) and int(row.end_year) > 0 else None
        source_rating = float(row.rating) if pd.notna(row.rating) else 0.0
        rating = round(source_rating / 2.0, 3) if source_rating > 0 else 3.6
        weight = int(row.weight) if pd.notna(row.weight) else 0
        poster_url = str(getattr(row, "poster_url", "") or "").strip()
        keywords = " ".join(
            part
            for part in [
                str(row.keywords) if pd.notna(row.keywords) else "",
                normalize_text(title),
                " ".join(g.lower() for g in genres),
                f"{(year // 10) * 10}s" if year else "",
                "serie series tv television",
            ]
            if part
        )
        rows.append(
            {
                "item_id": int(row.series_id),
                "tmdb_id": None,
                "imdb_id": "",
                "tvmaze_id": int(row.tvmaze_id) if pd.notna(row.tvmaze_id) else None,
                "media_type": "series",
                "title": title,
                "title_norm": normalize_title(title),
                "year": year,
                "end_year": end_year,
                "genres": "|".join(genres),
                "overview": overview,
                "keywords": keywords[:1500],
                "runtime": int(row.runtime) if pd.notna(row.runtime) and row.runtime else None,
                "seasons": int(row.seasons) if pd.notna(row.seasons) else None,
                "episodes": int(row.episodes) if pd.notna(row.episodes) and row.episodes else None,
                "rating_avg": rating,
                "rating_count": 0,
                "bayes_rating": rating,
                "popularity": weight / 100.0,
                "source": "TVmaze",
                "source_url": f"https://www.tvmaze.com/shows/{int(row.tvmaze_id)}",
                "poster_url": poster_url if poster_url.lower() != "nan" else "",
            }
        )
    return pd.DataFrame(rows)


def build_catalog(save: bool = True) -> pd.DataFrame:
    """Construye el catálogo unificado de películas y series."""
    movies, _ = load_movielens()
    series = load_series()
    catalog = pd.concat([movies, series], ignore_index=True)

    # Validaciones de integridad del catálogo.
    if catalog["item_id"].duplicated().any():
        duplicated = catalog.loc[catalog["item_id"].duplicated(), "item_id"].tolist()
        raise ValueError(f"item_id duplicado en el catálogo: {duplicated[:10]}")
    catalog["media_type"] = catalog["media_type"].astype("string")
    catalog["year"] = pd.to_numeric(catalog["year"], errors="coerce").astype("Int64")
    catalog["rating_avg"] = pd.to_numeric(catalog["rating_avg"], errors="coerce").fillna(0.0)
    catalog["bayes_rating"] = pd.to_numeric(catalog["bayes_rating"], errors="coerce").fillna(0.0)
    catalog["popularity"] = pd.to_numeric(catalog["popularity"], errors="coerce").fillna(0.0)
    catalog = catalog.sort_values(["media_type", "item_id"]).reset_index(drop=True)

    if save:
        config.ensure_directories()
        catalog.to_csv(config.CATALOG_CSV, index=False, encoding="utf-8")
    return catalog


def load_processed_catalog() -> pd.DataFrame:
    """Carga el catálogo procesado; lo construye si aún no existe."""
    if config.CATALOG_CSV.exists():
        return pd.read_csv(config.CATALOG_CSV)
    return build_catalog(save=True)


def main() -> int:
    catalog = build_catalog(save=True)
    movies = catalog[catalog["media_type"] == "movie"]
    series = catalog[catalog["media_type"] == "series"]
    print(f"[OK] Catálogo: {len(catalog):,} ítems")
    print(f"     Películas: {len(movies):,}")
    print(f"     Series:    {len(series):,}")
    print(f"     Géneros:   {catalog['genres'].str.get_dummies(sep='|').shape[1]}")
    print(f"     Archivo:   {config.CATALOG_CSV}")
    print(catalog.head(5)[["item_id", "media_type", "title", "year", "rating_avg"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

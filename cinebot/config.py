"""Configuración central de CineBot.

Todas las rutas se calculan a partir de la ubicación del paquete para que la
aplicación funcione sin importar desde qué directorio se ejecute.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
CURATED_DIR = DATA_DIR / "curated"
PROCESSED_DIR = DATA_DIR / "processed"
DB_DIR = BASE_DIR / "database"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ASSETS_DIR = BASE_DIR / "assets"
DOCS_DIR = BASE_DIR / "docs"

DB_PATH = Path(os.environ.get("CINEBOT_DB", DB_DIR / "cinebot.db"))
SCHEMA_PATH = DB_DIR / "schema.sql"
MODELS_PATH = Path(os.environ.get("CINEBOT_MODELS", MODELS_DIR))

MOVIELENS_DIR = RAW_DIR / "ml-latest-small"
MOVIES_CSV = MOVIELENS_DIR / "movies.csv"
RATINGS_CSV = MOVIELENS_DIR / "ratings.csv"
TAGS_CSV = MOVIELENS_DIR / "tags.csv"
LINKS_CSV = MOVIELENS_DIR / "links.csv"
SERIES_CSV = CURATED_DIR / "series_catalog.csv"
MOVIE_POSTERS_CSV = CURATED_DIR / "movie_posters.csv"
CATALOG_CSV = PROCESSED_DIR / "catalog.csv"

MODEL_FILES = {
    "content": "content_model.joblib",
    "collaborative": "collaborative_model.joblib",
    "intent": "intent_model.joblib",
    "clusters": "user_clusters.joblib",
    "metadata": "metadata.json",
}

RANDOM_STATE = 42

# Partición de valoraciones: 70 % entrenamiento, 15 % validación, 15 % prueba.
TRAIN_FRACTION = 0.70
VALID_FRACTION = 0.15
TEST_FRACTION = 0.15

# Filtrado colaborativo (factorización de matrices con SGD).
SVD_FACTORS = 24
SVD_EPOCHS = 35
SVD_LR = 0.008
SVD_REG = 0.08
SVD_PATIENCE = 6

# Modelo híbrido: pesos iniciales; luego se ajustan con el conjunto de validación.
HYBRID_GRID = {
    "content": [0.30, 0.40, 0.50, 0.60, 0.70],
    "collaborative": [0.10, 0.20, 0.30, 0.40],
    "quality": [0.05, 0.10, 0.15, 0.20],
    "popularity": [0.00, 0.05, 0.10],
}

DEFAULT_RECOMMENDATIONS = 5
MAX_RECOMMENDATIONS = 10
MIN_RATINGS_FOR_PROFILE = 3

APP_NAME = "CineBot"
APP_TAGLINE = "Tu copiloto cinéfilo con IA"
DEFAULT_USER_NAME = "Invitado"


def ensure_directories() -> None:
    """Crea las carpetas de salida que necesita la aplicación."""
    for folder in (
        DATA_DIR,
        RAW_DIR,
        CURATED_DIR,
        PROCESSED_DIR,
        DB_DIR,
        MODELS_PATH,
        REPORTS_DIR,
        FIGURES_DIR,
        ASSETS_DIR,
        DOCS_DIR,
    ):
        Path(folder).mkdir(parents=True, exist_ok=True)


def model_path(name: str) -> Path:
    """Devuelve la ruta de un artefacto del modelo por su clave lógica."""
    return MODELS_PATH / MODEL_FILES[name]

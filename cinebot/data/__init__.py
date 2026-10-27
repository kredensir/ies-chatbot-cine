"""Capa de datos de CineBot."""

from cinebot.data.catalog import (
    ALL_GENRES,
    build_catalog,
    load_movielens,
    load_processed_catalog,
    load_series,
)

__all__ = [
    "ALL_GENRES",
    "build_catalog",
    "load_movielens",
    "load_processed_catalog",
    "load_series",
]

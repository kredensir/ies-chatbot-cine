"""Utilidades de normalización de texto compartidas por datos, NLU y modelos."""

from __future__ import annotations

import re
import unicodedata

_ARTICLE_RE = re.compile(r"^(the|a|an|el|la|los|las|un|una)\s+")
_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_SPACES_RE = re.compile(r"\s+")


def strip_accents(value: str) -> str:
    """Elimina tildes y diacríticos manteniendo las letras base."""
    normalized = unicodedata.normalize("NFKD", str(value))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(value: str) -> str:
    """Minúsculas, sin acentos, sin signos y con espacios colapsados."""
    text = strip_accents(value or "").lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    return _SPACES_RE.sub(" ", text).strip()


def normalize_title(value: str, drop_articles: bool = True) -> str:
    """Normaliza un título para búsquedas y comparaciones difusas.

    MovieLens usa el formato "Godfather, The (1972)"; se elimina el artículo
    final y el año para que la búsqueda funcione escribiendo "el padrino" o
    simplemente "godfather".
    """
    text = normalize_text(value)
    # Algunos títulos de MovieLens incluyen el año entre paréntesis al final.
    text = re.sub(r"\(\s*\d{4}\s*\)", "", text).strip()
    text = re.sub(r"\s+(19|20)\d{2}\s*$", "", text).strip()
    # El artículo final ("Godfather, The") se elimina tras quitar el año.
    text = re.sub(r"\s+(the|a|an)$", "", text).strip()
    text = _YEAR_RE.sub("", text).strip()
    if drop_articles:
        text = _ARTICLE_RE.sub("", text)
    return text


def extract_year(title: str) -> int | None:
    """Extrae el año entre paréntesis del final del título de MovieLens."""
    match = _YEAR_RE.search(str(title))
    return int(match.group(1)) if match else None


def clean_movie_title(title: str) -> str:
    """Quita el año entre paréntesis del título para mostrarlo en la interfaz."""
    return re.sub(r"\s*\(\s*\d{4}\s*\)\s*$", "", str(title)).strip()


def title_tokens(value: str) -> list[str]:
    return [token for token in normalize_title(value).split() if len(token) > 1]


def truncate(value: str, length: int = 180) -> str:
    value = (value or "").strip()
    if len(value) <= length:
        return value
    return value[: length - 1].rstrip() + "…"

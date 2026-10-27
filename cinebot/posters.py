"""Resolución de portadas para películas y series.

Las series usan las imágenes públicas de TVmaze. Las películas usan la API
oficial de TMDB cuando existe ``TMDB_API_KEY`` y, si no hay clave, la ficha
pública de Letterboxd. Solo se guardan URLs: la imagen la carga Flet y, si no
está disponible, la interfaz dibuja una portada local.
"""

from __future__ import annotations

import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable
from urllib.parse import quote

import requests

from cinebot.db.connection import Database
from cinebot.db.repository import CatalogRepository
from cinebot.domain import MediaItem

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}
LETTERBOXD_BASE = "https://letterboxd.com"
LETTERBOXD_POSTER_RE = re.compile(
    r'https://a\.ltrbxd\.com/resized/(?:film-poster|sm/upload)/[^"\']+-0-600-0-900-crop\.jpg[^"\']*'
)
LETTERBOXD_FILM_RE = re.compile(r'href="(/film/[a-z0-9\-]+/)"')
LETTERBOXD_NAME_RE = re.compile(r'data-item-name="([^"]+)"')
YEAR_RE = re.compile(r"\((\d{4})\)")
ARTICLE_RE = re.compile(r"^(.*),\s*(The|A|An)$", flags=re.IGNORECASE)
LEADING_ARTICLE_RE = re.compile(r"^(The|A|An)\s+(.*)$", flags=re.IGNORECASE)
PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")


def _normalized_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    return "".join(char for char in text if not unicodedata.combining(char)).lower()


def slugify(value: str) -> str:
    """Convierte un título en el slug usado por Letterboxd."""
    text = _normalized_text(value).replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def title_variants(title: str) -> list[str]:
    """Genera variantes para títulos como ``Shining, The`` o ``Departures (Okuribito)``."""
    base = re.sub(r"\s*\(\s*\d{4}\s*\)\s*$", "", title).strip()
    variants = [base]

    article = ARTICLE_RE.match(base)
    if article:
        variants.insert(0, f"{article.group(2)} {article.group(1)}")

    without_parentheses = PAREN_RE.sub("", base).strip()
    if without_parentheses and without_parentheses != base:
        variants.append(without_parentheses)
        article = ARTICLE_RE.match(without_parentheses)
        if article:
            variants.insert(0, f"{article.group(2)} {article.group(1)}")

    for value in list(variants):
        article = LEADING_ARTICLE_RE.match(value)
        if article:
            variants.append(article.group(2))

    return list(dict.fromkeys(value for value in variants if value))


def parse_letterboxd_poster(html: str) -> str:
    match = LETTERBOXD_POSTER_RE.search(html)
    return match.group(0) if match else ""


def parse_letterboxd_year(html: str) -> int | None:
    match = LETTERBOXD_NAME_RE.search(html)
    if not match:
        return None
    year_match = YEAR_RE.search(match.group(1))
    return int(year_match.group(1)) if year_match else None


def _year_matches(expected: int | None, actual: int | None) -> bool:
    if expected is None or actual is None:
        return True
    return abs(expected - actual) <= 1


def _get(url: str, timeout: float, retries: int = 1) -> requests.Response | None:
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=(3.0, timeout))
        except requests.RequestException:
            return None
        if response.status_code == 200:
            return response
        if response.status_code in (429, 500, 502, 503, 504) and attempt < retries:
            time.sleep(0.8 + attempt * 0.7)
            continue
        return None
    return None


def resolve_tvmaze(tvmaze_id: int | None, timeout: float = 6.0) -> tuple[str, str]:
    if not tvmaze_id:
        return "", ""
    response = _get(f"https://api.tvmaze.com/shows/{int(tvmaze_id)}", timeout)
    if response is None:
        return "", ""
    try:
        data = response.json()
    except ValueError:
        return "", ""
    image = data.get("image") or {}
    poster = image.get("original") or image.get("medium") or ""
    source = data.get("url") or f"https://www.tvmaze.com/shows/{int(tvmaze_id)}"
    return str(poster), str(source)


def resolve_tmdb(
    tmdb_id: int | None,
    api_key: str | None = None,
    timeout: float = 6.0,
) -> tuple[str, str]:
    key = (api_key or os.environ.get("TMDB_API_KEY", "")).strip()
    if not key or not tmdb_id:
        return "", ""
    response = _get(
        f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={quote(key)}&language=es-ES",
        timeout,
    )
    if response is None:
        return "", ""
    try:
        data = response.json()
    except ValueError:
        return "", ""
    path = data.get("poster_path")
    if not path:
        return "", ""
    return f"https://image.tmdb.org/t/p/w500{path}", f"https://www.themoviedb.org/movie/{int(tmdb_id)}"


def resolve_letterboxd(
    title: str,
    year: int | None,
    timeout: float = 6.0,
) -> tuple[str, str]:
    for variant in title_variants(title):
        slug = slugify(variant)
        if not slug:
            continue
        source = f"{LETTERBOXD_BASE}/film/{slug}/"
        response = _get(source, timeout)
        if response is None:
            continue
        poster = parse_letterboxd_poster(response.text)
        if poster and _year_matches(year, parse_letterboxd_year(response.text)):
            return poster, source

    query = quote(f"{title} {year or ''}".strip())
    response = _get(f"{LETTERBOXD_BASE}/search/films/{query}/", timeout)
    if response is None:
        return "", ""
    seen: set[str] = set()
    for href in LETTERBOXD_FILM_RE.findall(response.text):
        if href in seen:
            continue
        seen.add(href)
        source = LETTERBOXD_BASE + href
        page = _get(source, timeout)
        if page is None:
            continue
        poster = parse_letterboxd_poster(page.text)
        if poster and _year_matches(year, parse_letterboxd_year(page.text)):
            return poster, source
        if len(seen) >= 3:
            break
    return "", ""


def resolve_movie_poster(
    title: str,
    year: int | None,
    tmdb_id: int | None = None,
    timeout: float = 6.0,
) -> tuple[str, str]:
    poster, source = resolve_tmdb(tmdb_id, timeout=timeout)
    if poster:
        return poster, source
    return resolve_letterboxd(title, year, timeout=timeout)


class PosterResolver:
    """Completa las URLs de portada y las cachea en SQLite."""

    def __init__(
        self,
        db: Database,
        repository: CatalogRepository | None = None,
        max_workers: int = 5,
        timeout: float = 6.0,
    ) -> None:
        self.db = db
        self.repository = repository or CatalogRepository(db)
        self.max_workers = max(1, max_workers)
        self.timeout = timeout
        self._cache: dict[int, str] = {}

    def _resolve(self, item: MediaItem) -> str:
        if item.media_type == "series":
            poster, _ = resolve_tvmaze(item.tvmaze_id, timeout=self.timeout)
            return poster
        poster, _ = resolve_movie_poster(
            item.title,
            item.year,
            tmdb_id=item.tmdb_id,
            timeout=self.timeout,
        )
        return poster

    def enrich(self, items: Iterable[MediaItem]) -> None:
        pending: list[MediaItem] = []
        for item in items:
            if item.poster_url:
                continue
            cached = self._cache.get(item.item_id)
            if cached:
                item.poster_url = cached
                continue
            pending.append(item)
        if not pending:
            return

        workers = min(self.max_workers, len(pending))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._resolve, item): item for item in pending}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    poster = future.result()
                except Exception:  # noqa: BLE001
                    poster = ""
                if not poster:
                    continue
                item.poster_url = poster
                self._cache[item.item_id] = poster
                try:
                    self.repository.set_poster_url(item.item_id, poster)
                except Exception:  # noqa: BLE001
                    pass

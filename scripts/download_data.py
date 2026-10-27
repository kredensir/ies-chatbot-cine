"""Descarga y verificación de las fuentes de datos de CineBot.

Fuentes
-------
1. MovieLens *latest-small* (GroupLens, Universidad de Minnesota)
   - 9.742 películas, 100.836 valoraciones reales de 610 usuarios.
   - Licencia de uso académico: https://files.grouplens.org/datasets/movielens/
   - Se eligió MovieLens porque permite entrenar filtrado colaborativo con
     valoraciones explícitas reales y es totalmente reproducible (no requiere
     una clave de API, a diferencia de TMDB).

2. TVmaze (https://www.tvmaze.com/api)
   - Metadatos públicos de series (géneros, año, temporadas, rating promedio).
   - Se usa para enriquecer el catálogo de series del proyecto.

3. TMDB (opcional, requiere TMDB_API_KEY)
   - El archivo links.csv de MovieLens ya incluye el tmdbId de cada película,
     por lo que la app puede enlazar a la ficha oficial de TMDB sin API.
   - scripts/enrich_tmdb.py permite añadir sinopsis/pósters si el docente
     provee una clave.

Uso:
    python scripts/download_data.py            # solo verifica/descarga MovieLens
    python scripts/download_data.py --series   # además descarga TVmaze
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

try:  # requests maneja mejor los certificados en Windows
    import requests
except ImportError:  # pragma: no cover - requests es una dependencia declarada
    requests = None

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
ML_DIR = RAW_DIR / "ml-latest-small"
ML_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
TVMAZE_URL = "https://api.tvmaze.com/shows?page={page}"
EXPECTED = ("movies.csv", "ratings.csv", "tags.csv", "links.csv")


def _download(url: str, destination: Path, timeout: int = 120) -> None:
    if requests is not None:
        response = requests.get(
            url, timeout=timeout, headers={"User-Agent": "CineBot-Academico/1.0"}
        )
        response.raise_for_status()
        destination.write_bytes(response.content)
        return
    request = Request(url, headers={"User-Agent": "CineBot-Academico/1.0"})
    with urlopen(request, timeout=timeout) as response:
        destination.write_bytes(response.read())


def _get_json(url: str, timeout: int = 60):
    if requests is not None:
        response = requests.get(
            url, timeout=timeout, headers={"User-Agent": "CineBot-Academico/1.0"}
        )
        response.raise_for_status()
        return response.json()
    request = Request(url, headers={"User-Agent": "CineBot-Academico/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ensure_movielens(force: bool = False) -> Path:
    """Descarga y descomprime MovieLens si aún no está disponible."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if all((ML_DIR / name).exists() for name in EXPECTED) and not force:
        print(f"[OK] MovieLens ya está en {ML_DIR}")
        return ML_DIR

    archive = RAW_DIR / "ml-latest-small.zip"
    print(f"[..] Descargando MovieLens desde {ML_URL}")
    _download(ML_URL, archive)
    print(f"[OK] Descargado ({archive.stat().st_size / 1024:.0f} KB)")

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(RAW_DIR)
    archive.unlink(missing_ok=True)

    missing = [name for name in EXPECTED if not (ML_DIR / name).exists()]
    if missing:
        raise RuntimeError(f"Faltan archivos de MovieLens: {missing}")
    print(f"[OK] MovieLens listo en {ML_DIR}")
    return ML_DIR


def download_tvmaze(pages: int = 4, force: bool = False) -> Path:
    """Descarga un catálogo amplio de series desde TVmaze (sin API key).

    TVmaze devuelve 250 series por página. Se guardan todas las páginas y
    luego scripts/build_series_catalog.py selecciona las más populares.
    """
    destination = RAW_DIR / "tvmaze_shows_raw.json"
    if destination.exists() and not force:
        print(f"[OK] TVmaze ya está en {destination}")
        return destination

    shows: list[dict] = []
    for page in range(pages):
        url = TVMAZE_URL.format(page=page)
        print(f"[..] TVmaze página {page}: {url}")
        page_data = _get_json(url)
        if not isinstance(page_data, list):
            break
        shows.extend(page_data)
        time.sleep(0.25)  # cortesía con la API pública

    destination.write_text(
        json.dumps(shows, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"[OK] {len(shows)} series guardadas en {destination}")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Descarga de datos de CineBot")
    parser.add_argument("--series", action="store_true", help="descargar TVmaze")
    parser.add_argument("--force", action="store_true", help="forzar descarga")
    args = parser.parse_args(argv)

    try:
        ensure_movielens(force=args.force)
        if args.series:
            download_tvmaze(force=args.force)
    except Exception as exc:  # noqa: BLE001 - mensaje amigable para el usuario
        print(f"[ERROR] {exc}", file=sys.stderr)
        print(
            "Verifique su conexión a Internet o copie manualmente los CSV de "
            "MovieLens en data/raw/ml-latest-small/.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

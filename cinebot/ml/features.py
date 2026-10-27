"""Ingeniería de características de contenido.

Cada ítem se representa como un documento de texto que combina:
- título (repetido para darle más peso),
- géneros canónicos y sus equivalentes en español,
- palabras clave de MovieLens / sinopsis de series,
- década de estreno,
- tipo (película o serie),
- etiquetas de "ánimo" derivadas de los géneros.

Luego se aplica TF-IDF (Term Frequency - Inverse Document Frequency) y se
normalizan los vectores para poder usar similitud coseno.
"""

from __future__ import annotations

import re

import pandas as pd

# Lista de palabras vacías en español e inglés (artículos, preposiciones,
# pronombres y verbos muy frecuentes). Evita que el modelo se concentre en
# palabras sin poder discriminativo.
STOPWORDS = {
    # español
    "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con",
    "contra", "cual", "cuando", "de", "del", "desde", "donde", "dos", "el",
    "ella", "ellas", "ellos", "en", "entre", "era", "es", "esa", "esas", "ese",
    "eso", "esos", "esta", "estas", "este", "esto", "estos", "fue", "ha", "hacia",
    "hasta", "la", "las", "le", "les", "lo", "los", "más", "me", "mi", "mis",
    "mucho", "muy", "no", "nos", "o", "otra", "otro", "para", "pero", "poco",
    "por", "porque", "que", "quien", "se", "ser", "si", "sin", "sobre", "su",
    "sus", "también", "te", "tiene", "todo", "todos", "tu", "tus", "un", "una",
    "uno", "unos", "y", "ya", "yo", "tras", "durante", "mientras", "asi",
    "donde", "cada", "vez", "hace", "hacen", "ser", "estar", "tiene", "tienen",
    # variantes sin tilde que produce el preprocesamiento de scikit-learn
    "mas", "tambien", "aqui", "alli", "solo", "unico", "unica",
    # inglés
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "with",
    "from", "by", "is", "are", "was", "were", "be", "been", "being", "as", "it",
    "its", "this", "that", "these", "those", "he", "she", "they", "his", "her",
    "their", "him", "them", "we", "you", "i", "me", "my", "our", "your", "who",
    "what", "when", "where", "why", "how", "not", "no", "do", "does", "did",
    "has", "have", "had", "but", "if", "then", "than", "so", "too", "very",
    "can", "could", "will", "would", "just", "about", "into", "over", "after",
}

GENRE_ES = {
    "Action": "accion adrenalina",
    "Adventure": "aventura viaje",
    "Animation": "animacion dibujos animados",
    "Children": "familiar niños infancia",
    "Comedy": "comedia humor risa divertida",
    "Crime": "crimen policial mafia delito",
    "Documentary": "documental realidad",
    "Drama": "drama emotivo serio",
    "Fantasy": "fantasia magia",
    "Film-Noir": "cine negro clasico",
    "Horror": "terror miedo sustos",
    "IMAX": "imax espectacular",
    "Musical": "musical musica canciones",
    "Mystery": "misterio enigma investigacion",
    "Romance": "romance amor romantico",
    "Sci-Fi": "ciencia ficcion futurista espacio",
    "Thriller": "suspenso tension thriller",
    "War": "guerra belico militar",
    "Western": "western vaqueros oeste",
    "Adult": "adultos maduro",
    "Food": "cocina gastronomia restaurante",
    "History": "historia historico epoca",
    "Legal": "abogados justicia legal",
    "Medical": "medicina hospital medico",
    "Sports": "deporte futbol equipo",
}

MOOD_TAGS = {
    "Comedy": "animo_divertido reir humor ligera",
    "Horror": "animo_miedo terror tensa oscura",
    "Thriller": "animo_tenso suspenso adrenalina",
    "Action": "animo_intenso accion adrenalina",
    "Adventure": "animo_aventura viaje epico",
    "Romance": "animo_romantico amor emotivo",
    "Drama": "animo_profundo emotivo reflexivo",
    "Sci-Fi": "animo_futurista pensar ciencia",
    "Mystery": "animo_intriga misterio pensar",
    "Fantasy": "animo_fantastico magia evasion",
    "Animation": "animo_ligero familiar animacion",
    "Children": "animo_familiar niños ligera",
    "Documentary": "animo_aprender realidad",
    "Musical": "animo_musical alegre musica",
    "War": "animo_epico guerra historico",
    "Western": "animo_clasico western aventura",
    "Crime": "animo_intriga crimen policial",
    "History": "animo_epico historia pasado",
    "Sports": "animo_motivador deporte equipo",
    "Food": "animo_calido cocina familia",
    "Legal": "animo_intriga abogados justicia",
    "Medical": "animo_emocional hospital",
    "Adult": "animo_oscuro adulto",
    "Film-Noir": "animo_clasico cine negro",
    "IMAX": "animo_espectaculo visual",
}

_TOKEN_RE = re.compile(r"[^\wáéíóúñü]+", re.IGNORECASE)


def _clean(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).replace("|", " ").strip()


def item_document(row: pd.Series | dict) -> str:
    """Construye el documento textual de un ítem para TF-IDF."""
    data = row if isinstance(row, dict) else row.to_dict()
    title = _clean(data.get("title"))
    genres = [g for g in _clean(data.get("genres")).split("|") if g]
    overview = _clean(data.get("overview"))
    keywords = _clean(data.get("keywords"))
    media_type = _clean(data.get("media_type")) or "movie"
    year = data.get("year")
    decade = ""
    try:
        if year and not pd.isna(year):
            decade = f"decada_{(int(year) // 10) * 10}"
    except (TypeError, ValueError):
        decade = ""

    genre_tokens: list[str] = []
    for genre in genres:
        slug = _TOKEN_RE.sub("_", genre.lower()).strip("_")
        genre_tokens.append(f"genero_{slug}")
        genre_tokens.append(GENRE_ES.get(genre, ""))
        genre_tokens.append(MOOD_TAGS.get(genre, ""))

    parts = [
        f"tipo_{media_type}",
        (title + " ") * 3,
        " ".join(genre_tokens) + " ",
        (keywords + " ") * 2,
        overview,
        decade,
        "serie television tv show" if media_type == "series" else "pelicula cine movie film",
    ]
    document = " ".join(part for part in parts if part)
    document = re.sub(r"\s+", " ", document)
    return document.strip()


def catalog_documents(catalog: pd.DataFrame) -> list[str]:
    return [item_document(row) for _, row in catalog.iterrows()]

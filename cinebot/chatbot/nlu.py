"""Comprensión de lenguaje natural (NLU) para CineBot.

El módulo combina dos enfoques complementarios:

1. Un clasificador supervisado (TF-IDF + regresión logística) entrenado con
   ejemplos en español para reconocer la *intención* del mensaje.
2. Reglas deterministas y diccionarios para extraer *entidades*: géneros,
   estados de ánimo, años, cantidad de resultados, notas y títulos.

Esta separación es deliberada: el modelo aprende patrones generales y las
reglas garantizan precisión en datos concretos (por ejemplo, el título de una
película), sin depender de un LLM externo.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any, Iterable

from cinebot.domain import NLUResult, QueryFilters
from cinebot.ml.intent import IntentClassifier
from cinebot.text_utils import normalize_text, normalize_title

# ---------------------------------------------------------------------------
# Diccionarios de entidades
# ---------------------------------------------------------------------------

GENRE_TERMS: dict[str, list[str]] = {
    "Action": [
        "accion", "action", "pelea", "peleas", "explosiones", "artes marciales",
        "superheroes", "superheroe", "superheroina", "adrenalina",
    ],
    "Adventure": ["aventura", "adventure", "viaje", "exploracion", "epico", "epica"],
    "Animation": ["animacion", "animation", "animada", "animado", "anime", "dibujos animados"],
    "Children": ["infantil", "ninos", "ninas", "children", "familiar", "familia", "family"],
    "Comedy": ["comedia", "comedy", "humor", "risa", "divertida", "divertido", "reir"],
    "Crime": ["crimen", "crime", "policial", "policiaca", "mafia", "delito", "criminal"],
    "Documentary": ["documental", "documentary", "documentales"],
    "Drama": ["drama", "dramatica", "dramatico", "dramas"],
    "Fantasy": ["fantasia", "fantasy", "magia", "magico", "magica"],
    "Film-Noir": ["cine negro", "film noir", "film-noir", "noir"],
    "Horror": ["terror", "horror", "miedo", "sustos", "escalofriante", "pesadilla"],
    "IMAX": ["imax"],
    "Musical": ["musical", "musicales", "musica", "canciones"],
    "Mystery": ["misterio", "mystery", "enigma", "investigacion", "intriga"],
    "Romance": ["romance", "romantica", "romantico", "amor", "enamorarse"],
    "Sci-Fi": [
        "ciencia ficcion", "sci fi", "sci-fi", "scifi", "espacial", "espacio",
        "futurista", "futuro", "robots", "alienigenas",
    ],
    "Thriller": ["suspenso", "thriller", "tension", "tenso", "tensa", "psicologico"],
    "War": ["guerra", "war", "belico", "belica", "militar", "soldados"],
    "Western": ["western", "vaqueros", "oeste", "cowboys"],
    "History": ["historia", "history", "historico", "historica", "epoca"],
    "Legal": ["abogados", "legal", "juicio", "tribunal", "justicia"],
    "Medical": ["medicina", "medico", "medica", "hospital", "doctores"],
    "Sports": ["deportes", "deporte", "futbol", "sports", "equipo"],
    "Food": ["cocina", "gastronomia", "food", "restaurante", "chef"],
    "Adult": ["adultos", "adulto"],
}

MOOD_TERMS: dict[str, list[str]] = {
    "animo_divertido": [
        "reir", "reirme", "risa", "divertido", "divertida", "humor", "feel good",
        "ligero", "ligera", "alegre", "buen rollo",
    ],
    "animo_emotivo": [
        "llorar", "triste", "emotivo", "emotiva", "conmovedor", "conmovedora",
        "sensibilidad", "sensiblero", "moco tendido", "desgarrador",
    ],
    "animo_miedo": ["miedo", "terror", "asustar", "asustarme", "susto", "escalofrio"],
    "animo_tenso": ["tension", "tenso", "tensa", "intenso", "intensa", "adrenalina", "suspenso"],
    "animo_relajante": [
        "relajante", "relajarme", "tranquilo", "tranquila", "ligero", "descansar",
        "dormir", "domingo lluvioso", "sin pensar",
    ],
    "animo_reflexivo": ["pensar", "reflexionar", "profundo", "profunda", "filosofico", "mente"],
    "animo_romantico": ["romantico", "romantica", "amor", "pareja", "cita", "enamorar"],
    "animo_epico": ["epico", "epica", "aventura", "grandioso", "grandiosa", "fantasia"],
    "animo_familiar": ["familia", "familiar", "ninos", "ninas", "infantil"],
    "animo_clasico": ["nostalgia", "nostalgico", "clasico", "clasica", "retro", "antiguo", "antigua"],
    "animo_sorprendente": ["sorprender", "sorprenderme", "inesperado", "inesperada", "original", "giro"],
    "animo_aprender": ["aprender", "documental", "realidad", "curiosidad"],
    "animo_musical": ["musical", "musica", "canciones", "cantar"],
}

MOOD_LABELS = {
    "animo_divertido": "divertida",
    "animo_emotivo": "emotiva",
    "animo_miedo": "de miedo",
    "animo_tenso": "intensa",
    "animo_relajante": "relajante",
    "animo_reflexivo": "para pensar",
    "animo_romantico": "romántica",
    "animo_epico": "épica",
    "animo_familiar": "familiar",
    "animo_clasico": "clásica",
    "animo_sorprendente": "sorprendente",
    "animo_aprender": "para aprender",
    "animo_musical": "musical",
}

# Los artículos "un"/"una" no se cuentan como cantidad: en español
# "recomiéndame una película" significa "una película cualquiera", no
# exactamente una. Para pedir un solo resultado se usan las fórmulas de
# ONE_ONLY_PATTERNS ("solo una", "una sola").
COUNT_WORDS = {
    "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
    "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
    "once": 10, "doce": 10, "un par": 2, "algunas": 4, "algunos": 4,
    "varias": 6, "varios": 6, "muchas": 8, "muchos": 8,
}

ONE_ONLY_PATTERNS = (
    re.compile(r"\b(?:solo|solamente|unicamente|nada mas)\s+(?:un|una)\b"),
    re.compile(r"\b(?:un|una)\s+(?:solo|sola|unico|unica)\b"),
)

# Alias en español/coloquiales de títulos del catálogo. Se resuelven contra
# los títulos reales al construir el matcher; si un alias no existe se ignora.
TITLE_ALIASES: dict[str, str] = {
    "interstellar": "Interstellar",
    "interestelar": "Interstellar",
    "origen": "Inception",
    "el origen": "Inception",
    "inception": "Inception",
    "el padrino": "Godfather, The",
    "el padrino ii": "Godfather: Part II, The",
    "el padrino parte ii": "Godfather: Part II, The",
    "el padrino iii": "Godfather: Part III, The",
    "matrix": "Matrix, The",
    "la matrix": "Matrix, The",
    "el senor de los anillos": "Lord of the Rings: The Fellowship of the Ring, The",
    "la comunidad del anillo": "Lord of the Rings: The Fellowship of the Ring, The",
    "las dos torres": "Lord of the Rings: The Two Towers, The",
    "el retorno del rey": "Lord of the Rings: The Return of the King, The",
    "juego de tronos": "Game of Thrones",
    "la casa de papel": "La Casa de Papel",
    "casa de papel": "La Casa de Papel",
    "merlina": "Wednesday",
    "los simpson": "The Simpsons",
    "los simpsons": "The Simpsons",
    "el juego del calamar": "Squid Game",
    "los soprano": "Sopranos, The",
    "el club de la lucha": "Fight Club",
    "el resplandor": "Shining, The",
    "la naranja mecanica": "Clockwork Orange, A",
    "el viaje de chihiro": "Spirited Away",
    "mi vecino totoro": "My Neighbor Totoro",
    "la princesa mononoke": "Princess Mononoke",
    "volver al futuro": "Back to the Future",
    "regreso al futuro": "Back to the Future",
    "tiburon": "Jaws",
    "psicosis": "Psycho",
    "buenos muchachos": "Goodfellas",
    "ciudad de dios": "City of God",
    "el sexto sentido": "Sixth Sense, The",
    "el show de truman": "Truman Show, The",
    "la lista de schindler": "Schindler's List",
    "salvar al soldado ryan": "Saving Private Ryan",
    "el lobo de wall street": "Wolf of Wall Street, The",
    "el gran pez": "Big Fish",
    "pesadilla antes de navidad": "Nightmare Before Christmas, The",
    "el extraño mundo de jack": "Nightmare Before Christmas, The",
    "el libro de la selva": "Jungle Book, The",
    "blanca nieves": "Snow White and the Seven Dwarfs",
    "la bella y la bestia": "Beauty and the Beast",
    "el rey leon": "Lion King, The",
    "buscando a nemo": "Finding Nemo",
    "los increibles": "Incredibles, The",
    "monsters inc": "Monsters, Inc.",
    "historia de juguete": "Toy Story",
    "el diablo viste a la moda": "Devil Wears Prada, The",
    "que paso ayer": "Hangover, The",
    "resacon en las vegas": "Hangover, The",
    "la gran estafa": "Ocean's Eleven",
    "el codigo da vinci": "Da Vinci Code, The",
    "el ilusionista": "Illusionist, The",
    "el prestigio": "Prestige, The",
    "el truco final": "Prestige, The",
    "la isla siniestra": "Shutter Island",
    "gravedad": "Gravity",
    "el marciano": "Martian, The",
    "marte": "Martian, The",
    "sin limites": "Limitless",
    "el profesional": "Leon: The Professional",
    "leon el profesional": "Leon: The Professional",
    "la vida es bella": "Life Is Beautiful",
    "el pianista": "Pianist, The",
    "la milla verde": "Green Mile, The",
    "suenos de libertad": "Shawshank Redemption, The",
    "cadena perpetua": "Shawshank Redemption, The",
    "el precio del manana": "In Time",
    "una noche en el museo": "Night at the Museum",
    "hombres de negro": "Men in Black",
    "piratas del caribe": "Pirates of the Caribbean: The Curse of the Black Pearl",
    "rapidos y furiosos": "Fast and the Furious, The",
    "mi pobre angelito": "Home Alone",
    "solo en casa": "Home Alone",
    "el dia de la marmota": "Groundhog Day",
    "el gran hotel budapest": "Grand Budapest Hotel, The",
    "la la land": "La La Land",
    "whiplash": "Whiplash",
    "parasitos": "Parasite",
    "parasite": "Parasite",
    "roma": "Roma",
    "el irlandes": "Irishman, The",
    "habia una vez en hollywood": "Once Upon a Time in... Hollywood",
    "django sin cadenas": "Django Unchained",
    "malditos bastardos": "Inglourious Basterds",
    "bastardos sin gloria": "Inglourious Basterds",
    "reservoir dogs": "Reservoir Dogs",
    "perros de la calle": "Reservoir Dogs",
    "tiempos violentos": "Pulp Fiction",
    "pulp fiction": "Pulp Fiction",
    "el bueno el malo y el feo": "Good, the Bad and the Ugly, The",
    "la guerra de las galaxias": "Star Wars: Episode IV - A New Hope",
    "star wars": "Star Wars: Episode IV - A New Hope",
    "el imperio contraataca": "Star Wars: Episode V - The Empire Strikes Back",
    "el regreso del jedi": "Star Wars: Episode VI - Return of the Jedi",
    "el senor de los anillos": "Lord of the Rings: The Fellowship of the Ring, The",
    "harry potter": "Harry Potter and the Sorcerer's Stone",
    "la piedra filosofal": "Harry Potter and the Sorcerer's Stone",
    "el prisionero de azkaban": "Harry Potter and the Prisoner of Azkaban",
    "el señor de los anillos": "Lord of the Rings: The Fellowship of the Ring, The",
    "los juegos del hambre": "Hunger Games, The",
    "crepusculo": "Twilight",
    "luna nueva": "Twilight Saga: New Moon, The",
    "amanecer": "Twilight Saga: Breaking Dawn - Part 1, The",
    "culpa mia": "My Fault",
}

INFO_MARKERS = (
    "de que trata", "de que va", "sinopsis", "cuentame", "hablame", "informacion",
    "que es", "ficha", "resumen", "detalles", "argumento", "de que se trata",
    "quien actua", "quienes actuan", "ano de estreno", "que ano", "cuanto dura",
    "cuantas temporadas", "cuantos episodios", "que genero", "donde ver",
)
RATING_MARKERS = (
    "califica", "calificar", "califico", "puntua", "puntuar", "puntuo", "valora", "valorar", "valoro",
    "le doy", "le pongo", "pongo", "mi nota", "mi puntuacion", "puntaje", "estrellas",
    "puntos", "me gusto", "me encanto", "me fascino", "no me gusto", "odio",
    "me parecio", "le puse",
)
SIMILAR_MARKERS = (
    "parecid", "similar", "algo como", "tipo ", "estilo de", "estilo ", "en la linea de",
    "mas como", "otra como", "del estilo",
)
TOP_MARKERS = (
    "top", "mejores", "mejor valoradas", "mejor puntuadas", "ranking", "mas populares",
    "lo mejor", "aclamadas", "aclamados", "mejor nota",
)
HISTORY_MARKERS = (
    "historial", "mis valoraciones", "mis notas", "mis gustos", "mi perfil",
    "que he visto", "que he valorado", "que he calificado", "mis peliculas",
    "mis series", "mi actividad", "cuantas he valorado", "mis favoritas",
)
RESET_MARKERS = (
    "borra mis", "borrar mis", "elimina mis", "reinicia mi", "reiniciar mi",
    "reset", "empezar de cero", "desde cero", "limpia mi", "olvida mis",
    "elimina todo", "borra mi historial", "borra mis calificaciones",
)
OTHER_MARKERS = (
    "otra", "otras", "dame mas", "mas opciones", "siguiente", "muestrame mas",
    "recomiendame mas", "otros resultados", "otra opcion", "no me convence",
)
REJECT_MARKERS = (
    "no me gusta", "no me gusto", "no me interesa", "esa no", "no quiero esa",
    "no me llama", "no me convence", "pesima", "mala recomendacion", "descartala",
    "no era eso", "no es lo que busco", "nada que ver", "no gracias",
)
GREETING_MARKERS = ("hola", "buenas", "buen dia", "que tal", "hey", "saludos", "que onda")
BYE_MARKERS = ("adios", "chau", "chao", "hasta luego", "nos vemos", "hasta pronto", "me voy", "bye")
THANKS_MARKERS = ("gracias", "te agradezco", "muy amable", "mil gracias", "gracias totales")
HELP_MARKERS = (
    "ayuda", "que puedes hacer", "como funciona", "como te uso", "comandos",
    "opciones", "instrucciones", "que te puedo preguntar", "para que sirves",
    "funciones", "guia", "tutorial", "ejemplos",
)

_STOP_TITLE_WORDS = {
    "una", "uno", "unos", "unas", "amor", "vida", "muerte", "fuego", "agua",
    "tierra", "cielo", "noche", "dia", "casa", "mundo", "guerra", "juego",
}


@dataclass(slots=True)
class TitleMatch:
    item_id: int
    title: str
    score: float
    media_type: str
    year: int | None
    matched_text: str


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms)


def _term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])")


class TitleMatcher:
    """Busca títulos del catálogo dentro de una frase libre.

    Combina coincidencia exacta, alias y similitud difusa. Se priorizan los
    títulos más largos para evitar que "Alien" gane sobre "Alien: Resurrection".
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.entries: list[tuple[str, str, dict[str, Any]]] = []
        by_norm: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            title = str(row.get("title") or "")
            norm = str(row.get("title_norm") or normalize_title(title))
            if not norm or len(norm) < 3 or norm in _STOP_TITLE_WORDS:
                continue
            self.entries.append((norm, title, row))
            by_norm.setdefault(norm, []).append(row)
        self.entries.sort(key=lambda item: len(item[0]), reverse=True)
        self._patterns = [(norm, _term_pattern(norm), row) for norm, _title, row in self.entries]
        self._norm_index = list(by_norm)
        self._aliases: list[tuple[str, re.Pattern[str], list[dict[str, Any]]]] = []
        for alias, target in TITLE_ALIASES.items():
            alias_norm = normalize_text(alias)
            target_norm = normalize_title(target)
            candidates = by_norm.get(target_norm)
            if not candidates:
                candidates = [
                    row for norm, _title, row in self.entries
                    if target_norm and target_norm in norm
                ][:3]
            if candidates:
                self._aliases.append((alias_norm, _term_pattern(alias_norm), candidates))
        self._aliases.sort(key=lambda item: len(item[0]), reverse=True)

    def find_in_text(self, text: str, media_type: str | None = None) -> TitleMatch | None:
        norm = normalize_text(text)
        if not norm:
            return None
        matches: list[TitleMatch] = []
        for alias, pattern, candidates in self._aliases:
            if pattern.search(norm):
                for row in candidates[:1]:
                    matches.append(self._match(row, 1.0, alias))
        for title_norm, pattern, row in self._patterns:
            if pattern.search(norm):
                matches.append(self._match(row, 0.99 if len(title_norm) > 5 else 0.92, title_norm))
                if len(matches) >= 8:
                    break
        if not matches:
            fuzzy = self._fuzzy_match(norm)
            if fuzzy:
                matches.append(fuzzy)
        if not matches:
            return None
        if media_type:
            matches.sort(key=lambda m: (m.media_type == media_type, m.score, len(m.title)), reverse=True)
        else:
            matches.sort(key=lambda m: (m.score, len(m.matched_text), len(m.title)), reverse=True)
        return matches[0]

    def _match(self, row: dict[str, Any], score: float, matched_text: str) -> TitleMatch:
        year = row.get("year")
        return TitleMatch(
            item_id=int(row["item_id"]),
            title=str(row.get("title") or ""),
            score=float(score),
            media_type=str(row.get("media_type") or "movie"),
            year=int(year) if year not in (None, "") and str(year) != "nan" else None,
            matched_text=matched_text,
        )

    def _fuzzy_match(self, norm: str) -> TitleMatch | None:
        # Se prueban frases después de "como", "parecida a", "tipo", etc.
        phrase = norm
        for marker in ("parecida a ", "parecido a ", "similar a ", "algo como ", "tipo ", "estilo "):
            if marker in norm:
                phrase = norm.split(marker, 1)[1]
                break
        else:
            words = norm.split()
            phrase = " ".join(words[:6])
        phrase = phrase.strip(" .!?")
        if len(phrase) < 3:
            return None
        candidates = difflib.get_close_matches(phrase, self._norm_index, n=1, cutoff=0.86)
        if not candidates:
            return None
        norm_title = candidates[0]
        for entry_norm, _title, row in self.entries:
            if entry_norm == norm_title:
                ratio = difflib.SequenceMatcher(None, phrase, norm_title).ratio()
                return self._match(row, float(ratio), phrase)
        return None


class CineBotNLU:
    """Fachada de comprensión: intención + entidades + filtros."""

    def __init__(
        self,
        catalog_rows: list[dict[str, Any]],
        classifier: IntentClassifier | None = None,
    ) -> None:
        self.classifier = classifier or IntentClassifier.load()
        self.title_matcher = TitleMatcher(catalog_rows)

    def parse(self, text: str, context: dict[str, Any] | None = None) -> NLUResult:
        raw = (text or "").strip()
        norm = normalize_text(raw)
        context = context or {}
        title = self.title_matcher.find_in_text(raw)
        media_type = _extract_media_type(norm)
        genres = _extract_genres(norm)
        moods = _extract_moods(norm)
        year_min, year_max = _extract_years(norm)
        count = _extract_count(norm)
        max_seasons = _extract_seasons(norm)
        rating = _extract_rating(norm)

        intent, confidence, top = self.classifier.predict(raw)
        intent, confidence = _refine_intent(
            norm, intent, confidence, title, media_type, genres, moods, rating
        )

        if count is None:
            count = 10 if intent == "top_valoradas" else 5
        if intent in {"recomendar_pelicula", "recomendar_serie", "recomendar_genero", "recomendar_animo"}:
            count = count or 5
        if title and intent == "similar":
            count = count or 5
        count = max(1, min(10, int(count)))

        filters = QueryFilters(
            media_type=media_type,
            genres=genres,
            moods=moods,
            year_min=year_min,
            year_max=year_max,
            min_rating=4.0 if _contains_any(norm, TOP_MARKERS) else None,
            count=count,
            exclude_seen=True,
            similar_to=title.item_id if (title and intent == "similar") else None,
            max_seasons=max_seasons,
        )
        if intent == "informacion" and title:
            filters.media_type = title.media_type

        entities: dict[str, Any] = {
            "genres": genres,
            "moods": moods,
            "media_type": media_type,
            "year_min": year_min,
            "year_max": year_max,
            "count": count,
            "max_seasons": max_seasons,
        }
        if title:
            entities.update(
                {
                    "title": title.title,
                    "title_item_id": title.item_id,
                    "title_media_type": title.media_type,
                }
            )
        if rating is not None:
            entities["rating"] = rating

        return NLUResult(
            raw_text=raw,
            intent=intent,
            confidence=float(confidence),
            filters=filters,
            title_query=title.title if title else None,
            matched_item_id=title.item_id if title else None,
            rating_value=rating,
            entities=entities,
        )


# ---------------------------------------------------------------------------
# Extracción de entidades
# ---------------------------------------------------------------------------

def _extract_media_type(norm: str) -> str | None:
    series_terms = ("serie", "series", "tv", "television", "maraton", "maratonear", "temporada", "temporadas", "episodios")
    movie_terms = ("pelicula", "peliculas", "peli", "pelis", "cine", "film", "films", "movie", "movies")
    has_series = any(_term_pattern(term).search(norm) for term in series_terms)
    has_movie = any(_term_pattern(term).search(norm) for term in movie_terms)
    if has_series and not has_movie:
        return "series"
    if has_movie and not has_series:
        return "movie"
    return None


def _extract_genres(norm: str) -> list[str]:
    found: list[str] = []
    for genre, terms in GENRE_TERMS.items():
        if any(_term_pattern(term).search(norm) for term in terms):
            found.append(genre)
    # "comedia romántica" debe devolver ambos géneros; el orden es estable.
    return found[:3]


def _extract_moods(norm: str) -> list[str]:
    found: list[str] = []
    for mood, terms in MOOD_TERMS.items():
        if any(_term_pattern(term).search(norm) for term in terms):
            found.append(mood)
    return found[:3]


def _extract_years(norm: str) -> tuple[int | None, int | None]:
    match = re.search(r"\b(18|19|20)\d{2}\b", norm)
    if match:
        year = int(match.group(0))
        return year, year
    decades = {
        "cincuenta": 1950, "sesenta": 1960, "setenta": 1970, "ochenta": 1980,
        "noventa": 1990, "dos mil": 2000,
    }
    for word, start in decades.items():
        if word in norm:
            return start, start + 9
    decade_match = re.search(r"\b(?:anos?|decada)\s*(?:de\s*)?(?:los\s*)?(\d{2})\b", norm)
    if decade_match:
        start = int(decade_match.group(1))
        start = 1900 + start if start >= 30 else 2000 + start
        return start, start + 9
    if any(term in norm for term in ("reciente", "recientes", "nueva", "nuevas", "nuevo", "nuevos", "actual", "actuales", "estreno")):
        return 2015, None
    if any(term in norm for term in ("clasica", "clasicas", "clasico", "clasicos", "antigua", "antiguas", "antiguo", "antiguos", "vintage")):
        return None, 1989
    if any(term in norm for term in ("moderna", "modernas", "moderno", "modernos")):
        return 2000, None
    return None, None


def _extract_count(norm: str) -> int | None:
    # "solo una", "una sola" sí expresan una cantidad explícita de 1.
    if any(pattern.search(norm) for pattern in ONE_ONLY_PATTERNS):
        return 1
    # Números explícitos, excluyendo años.
    for match in re.finditer(r"\b(\d{1,2})\b", norm):
        value = int(match.group(1))
        if 1900 <= value <= 2099:
            continue
        if 1 <= value <= 10:
            return value
    for word, value in COUNT_WORDS.items():
        if _term_pattern(word).search(norm):
            return value
    return None


def _extract_seasons(norm: str) -> int | None:
    match = re.search(r"\b(\d{1,2})\s*(?:temporadas?|seasons?)\b", norm)
    if match:
        return max(1, min(20, int(match.group(1))))
    if any(term in norm for term in ("corta", "cortas", "corto", "pocas temporadas", "una temporada", "miniserie")):
        return 2
    return None


def _extract_rating(norm: str) -> float | None:
    # "8/10", "nota 4.5", "le doy 4 estrellas", "califica con 5"
    match = re.search(
        r"(?:con|de|le doy|le pongo|pongo|puntuo|puntua|califico|califica|valoro|valora|nota|puntuacion|puntaje|estrellas?)\s*(?:un|una|de|con)?\s*(\d+(?:[.,]\d+)?)",
        norm,
    )
    if not match:
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*(?:estrellas?|puntos?|/ ?10|sobre 10)", norm)
    if not match:
        match = re.search(r"\b(\d(?:[.,]\d)?)\s*/\s*10\b", norm)
    value: float | None = None
    if match:
        value = float(match.group(1).replace(",", "."))
        if value > 5.0:
            value = value / 2.0
        value = round(max(0.5, min(5.0, value)) * 2) / 2
    if value is not None:
        return value
    if any(term in norm for term in ("no me gusto nada", "odio", "pesima", "horrible", "malisima")):
        return 1.0
    if any(term in norm for term in ("no me gusto", "no me convencio", "flojo", "aburrida", "mala")):
        return 2.0
    if any(term in norm for term in ("me encanto", "me fascino", "me gusto mucho", "excelente", "obra maestra")):
        return 5.0
    if any(term in norm for term in ("me gusto", "estuvo bien", "buena", "buenisima")):
        return 4.0
    if any(term in norm for term in ("regular", "normalita", "del monton")):
        return 3.0
    return None


# ---------------------------------------------------------------------------
# Refinamiento de intención con reglas
# ---------------------------------------------------------------------------

def _refine_intent(
    norm: str,
    predicted: str,
    confidence: float,
    title: TitleMatch | None,
    media_type: str | None,
    genres: list[str],
    moods: list[str],
    rating: float | None,
) -> tuple[str, float]:
    """Corrige errores del clasificador cuando hay evidencia léxica clara."""

    def choose(intent: str, boost: float = 0.0) -> tuple[str, float]:
        return intent, max(confidence, 0.82 + boost)

    if _contains_any(norm, RESET_MARKERS):
        return choose("resetear", 0.10)
    if _contains_any(norm, HISTORY_MARKERS):
        return choose("historial", 0.08)
    if _contains_any(norm, TOP_MARKERS):
        return choose("top_valoradas", 0.08)
    if title and _contains_any(norm, INFO_MARKERS):
        return choose("informacion", 0.12)
    if title and _contains_any(norm, RATING_MARKERS) and rating is not None:
        return choose("calificar", 0.12)
    if title and _contains_any(norm, SIMILAR_MARKERS):
        return choose("similar", 0.12)
    if _contains_any(norm, REJECT_MARKERS) and not title:
        return choose("rechazo", 0.05)
    if _contains_any(norm, ("calificar", "puntuar", "valorar", "poner una nota", "ponerle una nota")):
        return choose("calificar", 0.10)
    if _contains_any(norm, OTHER_MARKERS):
        return choose("otra_recomendacion", 0.05)
    if _contains_any(norm, THANKS_MARKERS):
        return choose("agradecimiento", 0.08)
    if _contains_any(norm, BYE_MARKERS):
        return choose("despedida", 0.06)
    if _contains_any(norm, HELP_MARKERS):
        return choose("ayuda", 0.06)
    if _contains_any(norm, GREETING_MARKERS) and len(norm.split()) <= 4:
        return choose("saludo", 0.05)
    if moods:
        return choose("recomendar_animo", 0.06)
    if genres:
        return choose("recomendar_genero", 0.06)
    if media_type == "series":
        return choose("recomendar_serie", 0.04)
    if media_type == "movie":
        return choose("recomendar_pelicula", 0.04)
    if norm in {"si", "sí", "dale", "ok", "vale", "claro", "perfecto", "de acuerdo"}:
        return choose("afirmacion", 0.08)
    return predicted, confidence

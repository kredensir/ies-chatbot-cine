"""Construye el catálogo curado de series a partir de metadatos de TVmaze.

TVmaze es una API pública y gratuita (no requiere clave) que aporta géneros,
año de estreno, cantidad de temporadas, episodios, cadena y rating promedio.
Las sinopsis en español fueron redactadas para este proyecto (no se copian
textos de TVmaze ni de TMDB).

Salida: data/curated/series_catalog.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW = BASE_DIR / "data" / "raw" / "tvmaze_selected_enriched.json"
RAW_IMAGES = BASE_DIR / "data" / "raw" / "tvmaze_selected_raw.json"
OUT = BASE_DIR / "data" / "curated" / "series_catalog.csv"

# Identificadores de TVmaze seleccionados para el catálogo docente.
SELECTED_IDS = [
    169, 618, 179, 527, 269, 5, 32, 335, 161, 10822, 13417, 2705, 27436,
    353, 69, 541, 172, 7, 157, 167, 123, 430, 158, 305, 2, 2993, 17861,
    44933, 1371, 38963, 52341, 210, 490, 180, 166, 204, 787, 82, 44778,
    28276, 15299, 30386, 555, 919, 563, 46562, 49041, 53277, 29, 716, 182,
    43, 30770, 465, 3594, 251, 385, 37336, 526, 431, 530, 66, 171, 532,
    318, 174, 216, 538, 83, 112, 184, 16149, 44458, 315, 143, 142, 73, 30,
    19, 53647, 43687, 41428, 42966, 16579, 23470, 54198, 51394, 48830, 45039,
    464,
]

TITLE_OVERRIDES = {
    10822: "Mindhunter",
    204: "Stargate SG-1",
    37336: "Shogun",
    27436: "La Casa de Papel",
}

# Géneros de TVmaze -> taxonomía del catálogo.
GENRE_MAP = {
    "Science-Fiction": "Sci-Fi",
    "Family": "Children",
    "Anime": "Animation",
    "Supernatural": "Fantasy",
    "Espionage": "Thriller",
    "Music": "Musical",
}
GENRE_ORDER = [
    "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
    "Adult", "Food", "History", "Legal", "Medical", "Sports",
]

# Sinopsis originales en español (una o dos frases por serie).
OVERVIEWS = {
    169: "Un profesor de química con cáncer terminal se asocia con un exalumno para fabricar metanfetamina y asegurar el futuro de su familia.",
    618: "Precuela de Breaking Bad que sigue la transformación del abogado Jimmy McGill en el astuto Saul Goodman.",
    179: "Retrato coral de Baltimore donde policías, narcotraficantes, sindicatos y periodistas revelan un sistema que falla.",
    527: "Un jefe de la mafia de Nueva Jersey intenta equilibrar sus negocios criminales con su familia y su terapia psicológica.",
    269: "La familia Shelby, una banda de Birmingham, escala en el hampa británica tras la Primera Guerra Mundial.",
    5: "Detectives atormentados investigan crímenes rituales que ponen a prueba su moral y su cordura.",
    32: "Historias criminales absurdas y sangrientas en el gélido Medio Oeste, inspiradas en la película de los Coen.",
    335: "Versión contemporánea del detective de Conan Doyle, con Sherlock Holmes y John Watson resolviendo casos en Londres.",
    161: "Un experto forense de Miami lleva una doble vida como justiciero que solo mata a asesinos.",
    10822: "Dos agentes del FBI entrevistan a asesinos en serie para fundar la ciencia del perfil criminal en los años setenta.",
    13417: "Un asesor financiero traslada a su familia a los Ozarks para lavar dinero de un cártel y salvar sus vidas.",
    2705: "Ascenso y caída de los cárteles colombianos, con la persecución a Pablo Escobar como eje central.",
    27436: "Un cerebro misterioso recluta a ocho delincuentes para el mayor atraco de la historia: la Fábrica de Moneda.",
    353: "Una unidad anticorrupción investiga a policías sospechosos de encubrir delitos dentro de su propia institución.",
    69: "Un ex criminal se entrega al FBI y ofrece una lista de peligrosos delincuentes a cambio de trabajar con una agente.",
    541: "Un ingeniero se hace encarcelar para ayudar a su hermano inocente a escapar de prisión.",
    172: "Un abogado brillante contrata a un joven sin título para litigar en uno de los bufetes más exigentes de Nueva York.",
    7: "Una analista de la CIA sospecha que un marine liberado fue captado por el enemigo y planea un atentado.",
    157: "Dos espías soviéticos se hacen pasar por un matrimonio estadounidense durante la Guerra Fría.",
    167: "Cada temporada narra un día completo en tiempo real, con un agente antiterrorista intentando evitar una crisis.",
    123: "Los sobrevivientes de un accidente aéreo quedan atrapados en una isla con secretos sobrenaturales.",
    430: "Dos agentes del FBI investigan fenómenos paranormales y conspiraciones extraterrestres.",
    158: "Un equipo del FBI explora fenómenos científicos imposibles ligados a un universo paralelo.",
    305: "Antología distópica sobre el impacto de la tecnología en la vida privada y en la condición humana.",
    2: "Un informático y un exagente usan una inteligencia artificial para prevenir crímenes antes de que ocurran.",
    2993: "En un pueblo de los ochenta, un grupo de niños enfrenta experimentos secretos y criaturas del otro lado.",
    17861: "La desaparición de dos niños en un pueblo alemán revela un enigma temporal que conecta a cuatro familias.",
    44933: "Empleados de una empresa se someten a una separación quirúrgica entre su vida laboral y personal.",
    1371: "Un parque temático de robots del Viejo Oeste empieza a fallar cuando sus androides toman conciencia.",
    38963: "Un cazarrecompensas solitario protege a un niño sensible a la Fuerza en la periferia de la galaxia.",
    52341: "Los orígenes de la rebelión contra el Imperio se cuentan a través de un ladrón convertido en revolucionario.",
    210: "Un viajero alienígena y sus acompañantes recorren el tiempo y el espacio en una cabina azul.",
    490: "La tripulación de la Enterprise explora nuevos mundos y defiende la diplomacia y la ciencia en la galaxia.",
    180: "Una tripulación de contrabandistas sobrevive en la frontera de un sistema solar tras una guerra perdida.",
    166: "Lo que queda de la humanidad huye por el espacio mientras enfrenta a los cylons que la exterminaron.",
    204: "Un equipo militar viaja por puertas estelares para explorar planetas y defender la Tierra.",
    787: "Antología de relatos fantásticos con finales sorprendentes y reflexiones sobre la naturaleza humana.",
    82: "Nobles familias se disputan el Trono de Hierro mientras una amenaza ancestral despierta en el norte.",
    44778: "La casa Targaryen se enfrenta a una guerra civil por la sucesión doscientos años antes de Game of Thrones.",
    28276: "Un cazador de monstruos con poderes mutantes busca su lugar en un continente marcado por la guerra.",
    15299: "Un grupo de civiles enfrenta a superhéroes corruptos y a la corporación que controla su imagen.",
    30386: "Hermanos adoptivos con poderes se reencuentran para evitar el fin del mundo y resolver la muerte de su padre.",
    555: "Un niño maestro de los cuatro elementos debe detener una guerra de cien años y restaurar el equilibrio.",
    919: "La humanidad sobrevive tras muros gigantes mientras soldados enfrentan titanes que devoran personas.",
    563: "Las batallas entre la República y la Confederación se ven a través de los jedis y sus clones.",
    46562: "Un contrabandista escolta a una adolescente inmune a una pandemia fúngica por unos Estados Unidos devastados.",
    49041: "Los habitantes de un refugio nuclear salen a un páramo postapocalíptico lleno de peligros y humor negro.",
    53277: "Un grupo de científicos enfrenta una amenaza extraterrestre que pone en jaque las leyes de la física.",
    29: "Las expediciones y guerras del legendario Ragnar Lothbrok y sus descendientes en la era vikinga.",
    716: "Un gladiador tracio lidera una rebelión de esclavos contra la República romana.",
    182: "Piratas, tesoros y poder colonial se cruzan en el Caribe antes de la historia de La isla del tesoro.",
    43: "Una enfermera de 1945 viaja al siglo XVIII escocés y queda atrapada entre dos épocas y dos amores.",
    30770: "Reconstrucción del accidente nuclear de 1986 y de la burocracia que intentó ocultar sus consecuencias.",
    465: "La compañía Easy de paracaidistas estadounidenses vive la Segunda Guerra Mundial desde el desembarco de Normandía.",
    3594: "La vida de la reina Isabel II y los cambios políticos y familiares de la monarquía británica.",
    251: "Una familia aristocrática y sus sirvientes atraviesan los cambios sociales de la Inglaterra de principios del siglo XX.",
    385: "Publicistas de Madison Avenue en los años sesenta lidian con el éxito, el alcohol y los cambios de época.",
    37336: "Un navegante inglés llega al Japón feudal y queda atrapado en la lucha por el poder entre señores de la guerra.",
    526: "Falso documental sobre la vida cotidiana de los empleados de una empresa papelera en Scranton.",
    431: "Seis amigos treintañeros comparten departamentos, trabajos y romances en Nueva York.",
    530: "Un comediante y sus amigos convierten las pequeñas incomodidades de la vida diaria en situaciones absurdas.",
    66: "Cuatro científicos y su vecina actriz conviven entre experimentos, cómics y relaciones torpes.",
    171: "Un padre cuenta a sus hijos la larga historia de cómo conoció a su madre.",
    532: "Un médico interno aprende a sobrevivir a un hospital caótico con humor y amistad.",
    318: "Un grupo dispar de estudiantes forma una comunidad en un colegio comunitario lleno de parodias.",
    174: "Una funcionaria optimista intenta mejorar su pueblo con proyectos que siempre se complican.",
    216: "Un científico alcohólico arrastra a su nieto a aventuras interdimensionales llenas de caos.",
    538: "Un repartidor de pizza despierta en el año 3000 y trabaja en una empresa de entregas espaciales.",
    83: "La familia amarilla más famosa de Springfield satiriza la vida estadounidense desde 1989.",
    112: "Cuatro niños de Colorado protagonizan una sátira irreverente sobre política, cultura y actualidad.",
    184: "Una estrella televisiva fracasada, mitad caballo y mitad hombre, intenta reconstruir su carrera y su vida.",
    16149: "Una mujer londinense usa el humor y el sexo para tapar el duelo y la soledad que la desbordan.",
    44458: "Un entrenador de fútbol americano dirige, sin saber nada del deporte, un equipo de fútbol inglés.",
    315: "Un espía narcisista y su disfuncional agencia de inteligencia viven misiones absurdas y peligrosas.",
    143: "Un programador tímido intenta levantar su startup mientras sobrevive a inversores y rivales.",
    142: "La vicepresidenta de Estados Unidos y su equipo intentan mantenerse en el poder entre egoísmos y torpezas.",
    73: "Un grupo de supervivientes busca un hogar en un mundo dominado por zombis y por otros humanos.",
    30: "Antología de terror que cambia de escenario y personajes en cada temporada.",
    19: "Dos hermanos cazan demonios, fantasmas y monstruos mientras buscan a su padre desaparecido.",
    53647: "Wednesday Addams estudia en una academia para raros y resuelve un misterio sobrenatural con humor negro.",
    43687: "Personas endeudadas compiten en juegos infantiles mortales por un premio millonario.",
    41428: "Una huérfana prodigio del ajedrez se abre camino en un mundo dominado por hombres y adicciones.",
    42966: "La alta sociedad londinense del siglo XIX se enreda en romances, rumores y matrimonios arreglados.",
    16579: "En una teocracia, las mujeres fértiles son obligadas a servir a familias poderosas.",
    23470: "Los hijos de un magnate de los medios compiten por heredar un imperio familiar lleno de traiciones.",
    54198: "Un chef de élite regresa a Chicago para salvar el restaurante de sándwiches de su familia.",
    51394: "Huéspedes y empleados de resorts de lujo exponen sus privilegios, secretos y miserias durante unas vacaciones.",
    48830: "Tres vecinos aficionados al true crime investigan un asesinato en su propio edificio.",
    45039: "Un equipo de espías británicos marginados investiga amenazas mientras lidia con su incompetencia.",
    464: "Marines estadounidenses viven la campaña del Pacífico durante la Segunda Guerra Mundial.",
}


def _order_genres(genres: list[str]) -> list[str]:
    mapped = {GENRE_MAP.get(g, g) for g in genres if g}
    return [g for g in GENRE_ORDER if g in mapped] + sorted(mapped - set(GENRE_ORDER))


def main() -> int:
    if not RAW.exists():
        print(
            f"No existe {RAW}. Ejecute primero scripts/download_data.py --series",
            file=sys.stderr,
        )
        return 1

    data = json.loads(RAW.read_text(encoding="utf-8"))
    by_id = {int(row["id"]): row for row in data}
    images: dict[int, dict] = {}
    if RAW_IMAGES.exists():
        raw_rows = json.loads(RAW_IMAGES.read_text(encoding="utf-8"))
        images = {
            int(row["id"]): (row.get("image") or {})
            for row in raw_rows.values()
            if row.get("id") is not None
        }
    rows = []
    for tvmaze_id in SELECTED_IDS:
        row = by_id.get(tvmaze_id)
        if row is None:
            print(f"[WARN] TVmaze id {tvmaze_id} no encontrado", file=sys.stderr)
            continue
        title = TITLE_OVERRIDES.get(tvmaze_id, row["name"]).strip()
        start_year = int(row.get("year") or 0)
        end_year = int(row["end"]) if row.get("end") else None
        seasons = int(row.get("seasons") or 0)
        episodes = int(row.get("episodes") or 0)
        runtime = row.get("average_runtime") or row.get("runtime") or 45
        rating = float(row.get("rating") or 0.0)
        weight = int(row.get("weight") or 0)
        genres = _order_genres(row.get("genres") or [])
        image = images.get(tvmaze_id) or {}
        poster_url = image.get("original") or image.get("medium") or ""

        keywords = " ".join(
            [
                title.lower(),
                "serie tv show television",
                " ".join(g.lower() for g in genres),
                str(start_year),
                (OVERVIEWS.get(tvmaze_id) or "").lower(),
            ]
        )

        rows.append(
            {
                "series_id": 900000 + tvmaze_id,
                "tvmaze_id": tvmaze_id,
                "title": title,
                "start_year": start_year,
                "end_year": end_year,
                "genres": "|".join(genres) if genres else "(no genres listed)",
                "seasons": seasons,
                "episodes": episodes,
                "runtime": runtime,
                "rating": round(rating, 1),
                "weight": weight,
                "network": row.get("network") or "",
                "country": row.get("country") or "",
                "keywords": keywords[:1200],
                "overview": OVERVIEWS.get(tvmaze_id, ""),
                "source_url": f"https://www.tvmaze.com/shows/{tvmaze_id}",
                "poster_url": poster_url,
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"[OK] {len(df)} series exportadas a {OUT}")
    print(df[["series_id", "title", "start_year", "seasons", "rating"]].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

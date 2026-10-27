"""Corpus sintético en español para entrenar el clasificador de intenciones.

El chatbot no usa un LLM externo: entrena un clasificador supervisado con
ejemplos etiquetados. Las reglas de entidades (géneros, títulos, años, notas)
complementan al clasificador para lograr un comportamiento robusto y
explicable, adecuado para un trabajo académico.
"""

from __future__ import annotations

INTENT_EXAMPLES: dict[str, list[str]] = {
    "saludo": [
        "hola", "hola cinebot", "buenas", "buen día", "buenas tardes", "buenas noches",
        "qué tal", "hey", "hola qué tal", "saludos", "hola, ¿cómo estás?", "qué onda",
        "hola amigo", "buenas, ¿qué me recomiendas?", "hola, necesito una recomendación",
        "qué hay", "hola de nuevo", "buenas noches cinebot", "hola, ¿estás ahí?",
        "qué tal todo", "hola, quiero ver algo",
    ],
    "despedida": [
        "adiós", "chau", "hasta luego", "nos vemos", "me voy", "bye", "hasta pronto",
        "me despido", "gracias, hasta luego", "buenas noches, chau", "corto",
        "me retiro", "hasta la próxima", "nos vemos luego", "chao pescao",
        "adiós cinebot", "ok, me voy", "suficiente por hoy", "terminemos", "hasta mañana",
    ],
    "agradecimiento": [
        "gracias", "muchas gracias", "te agradezco", "genial, gracias", "perfecto, gracias",
        "buenísimo, gracias", "gracias por la recomendación", "muy amable", "excelente",
        "me sirvió mucho", "gracias totales", "mil gracias", "te lo agradezco",
        "quedó claro, gracias", "gracias, muy útil", "buena esa", "me encantó la sugerencia",
        "gracias crack", "muy bueno", "de diez",
    ],
    "ayuda": [
        "ayuda", "qué puedes hacer", "cómo funciona", "dame instrucciones", "comandos",
        "opciones", "no sé qué pedir", "qué le puedo preguntar", "para qué sirves",
        "muéstrame ejemplos", "cómo te uso", "ayúdame a elegir", "qué funciones tienes",
        "no entiendo cómo usarte", "dime qué sabes hacer", "guía", "tutorial",
        "cómo pido una recomendación", "qué tipos de pedidos aceptas", "ayuda por favor",
    ],
    "recomendar_pelicula": [
        "recomiéndame una película", "quiero ver una peli", "dame una película",
        "qué película veo", "una peli para hoy", "sorpréndeme con una película",
        "necesito una película", "qué puedo ver de películas", "una buena peli",
        "recomienda una película para esta noche", "dame una peli para el fin de semana",
        "quiero una película nueva", "sugiere una película", "qué peli me recomiendas",
        "una película para ver solo", "busco una película", "pásame una peli",
        "quiero ver cine", "dame opciones de películas", "una película cualquiera",
    ],
    "recomendar_serie": [
        "recomiéndame una serie", "quiero empezar una serie", "dame series",
        "qué serie veo", "una serie corta", "series para maratón", "necesito una serie",
        "sugiere una serie", "qué serie me recomiendas", "una serie nueva",
        "series para ver en un fin de semana", "pásame una serie", "quiero maratonear",
        "dame opciones de series", "una serie para engancharme", "busco una serie",
        "recomienda series", "qué series hay buenas", "una serie de pocas temporadas",
        "quiero ver una serie hoy",
    ],
    "recomendar_genero": [
        "algo de terror", "una película de comedia", "series de ciencia ficción",
        "recomiéndame algo de acción", "quiero drama", "pelis románticas", "de misterio",
        "animación", "documentales", "una de suspenso", "cine negro", "algo de fantasía",
        "películas de guerra", "una comedia romántica", "series policiales",
        "quiero ver algo de crimen", "dame una peli de aventura", "algo histórico",
        "recomiéndame un western", "quiero un musical", "películas de miedo",
        "series de abogados", "algo de medicina", "una de deportes", "pelis familiares",
        "quiero cine de terror psicológico", "algo de mafia", "películas espaciales",
    ],
    "recomendar_animo": [
        "estoy triste, quiero algo para llorar", "algo para reír", "no quiero pensar mucho",
        "quiero adrenalina", "algo relajante", "tengo nostalgia", "algo para ver en familia",
        "quiero sorprenderme", "necesito desconectar", "estoy estresado, algo ligero",
        "quiero algo emotivo", "me siento solo", "algo que me haga sentir bien",
        "quiero una historia de amor", "algo para pensar", "quiero algo intenso",
        "necesito reírme", "algo tranquilo para dormir", "quiero algo épico",
        "estoy aburrido, sorpréndeme", "algo para llorar a moco tendido", "quiero pasar miedo",
        "dame algo feel good", "quiero una peli para el domingo lluvioso",
    ],
    "similar": [
        "algo como inception", "parecida a breaking bad", "más como interestelar",
        "similar a the office", "quiero algo tipo matrix", "otra como el padrino",
        "recomiéndame algo parecido a dark", "películas como pulp fiction",
        "series similares a lost", "algo en la línea de interstellar",
        "qué se parece a parasite", "dame algo como game of thrones",
        "quiero ver algo tipo the wire", "más pelis como alien",
        "una serie parecida a sherlock", "algo como back to the future",
        "recomiéndame algo del estilo de tarantino", "similar a la casa de papel",
        "otra película como el señor de los anillos", "algo como stranger things",
    ],
    "informacion": [
        "de qué trata inception", "sinopsis de dark", "cuéntame sobre the wire",
        "información de parasite", "háblame de la serie breaking bad",
        "quién actúa en matrix", "año de estreno de alien", "qué es the office",
        "dame la sinopsis de interstellar", "de qué se trata el padrino",
        "cuántas temporadas tiene lost", "cuánto dura inception",
        "información de la película pulp fiction", "resumen de game of thrones",
        "qué género es the shining", "detalles de la serie sherlock",
        "quiero saber sobre parasite", "cuéntame de qué va coco",
        "ficha de la película alien", "de qué trata la casa de papel",
    ],
    "calificar": [
        "califica inception con 5", "le doy 4 estrellas a matrix", "puntúa breaking bad con 9",
        "me gustó mucho the office", "no me gustó tal película", "valora alien con 4",
        "pongo 5 a parasite", "le doy un 3 a interstellar", "califico dark con 8",
        "mi nota para the wire es 5", "me encantó coco", "odio las películas de terror",
        "puntúo game of thrones con 10", "le doy 2 estrellas a sharknado",
        "califica la serie lost con 4", "mi puntuación para matrix es 5",
        "no me gustó nada sharknado", "me gustó el padrino", "le doy 4.5 a inception",
        "valoro the office con 5 estrellas",
    ],
    "historial": [
        "qué he valorado", "mi historial", "qué películas vi", "mis gustos",
        "qué he visto", "muéstrame mis notas", "mi perfil", "mis valoraciones",
        "qué le puse a inception", "cuáles son mis géneros favoritos",
        "qué series he puntuado", "mi actividad reciente", "qué recomendaciones me diste",
        "muéstrame mi historial", "cuántas películas he valorado", "mi resumen",
        "qué notas puse", "mis películas favoritas", "mi taste", "qué he calificado",
    ],
    "top_valoradas": [
        "las mejores películas", "top 10 series", "lo mejor de terror",
        "mejores valoradas", "ranking de ciencia ficción", "películas mejor puntuadas",
        "las series más aclamadas", "top de comedia", "dame el ranking",
        "cuáles son las mejores de drama", "lo mejor del cine", "top 5 películas",
        "las más populares", "mejores series de la historia", "ranking de suspenso",
        "qué películas tienen mejor nota", "las mejores de los 90", "top animación",
        "las series mejor valoradas", "dame lo más visto",
    ],
    "resetear": [
        "borra mis datos", "reinicia mi perfil", "elimina mis valoraciones",
        "quiero empezar de cero", "reset", "borra mi historial", "olvida mis gustos",
        "reinicia todo", "elimina mi cuenta", "borra las notas que te di",
        "empieza desde cero", "limpia mi perfil", "quiero borrar mis preferencias",
        "resetea el chatbot", "elimina todo lo que sabes de mí", "borra mi actividad",
        "pon mi perfil en blanco", "reinicia mis recomendaciones", "limpia el historial",
        "borra mis calificaciones",
    ],
    "otra_recomendacion": [
        "otra", "dame otra", "más opciones", "otra opción", "no me convence, dame otra",
        "recomiéndame más", "siguiente", "otras películas", "muéstrame más",
        "otra recomendación", "dame cinco más", "y otra?", "otra por favor",
        "quiero más sugerencias", "siguientes opciones", "no, otra", "otra distinta",
        "dame otras", "más resultados", "otra que no sea esa",
    ],
    "rechazo": [
        "no me gustó", "no me interesa", "esa no", "no quiero esa", "pésima recomendación",
        "no la vi", "no me llama la atención", "nada que ver", "no gracias",
        "no me convence", "esa película no", "no es lo que busco", "no era eso",
        "qué mala recomendación", "no me gusta ese género", "no quiero ver eso",
        "descártala", "no, gracias", "para nada", "no me sirve",
    ],
    "afirmacion": [
        "sí", "dale", "ok", "claro", "perfecto", "me gusta", "buena", "está bien",
        "acepto", "de acuerdo", "sí, por favor", "vale", "genial", "me parece bien",
        "adelante", "hagámoslo", "sí, muéstrame", "correcto", "eso quiero", "buenísimo",
    ],
    "desconocido": [
        "cuánto es 2+2", "qué hora es", "quién ganó el mundial", "cómo está el clima",
        "receta de pizza", "cuál es la capital de Francia", "ayúdame con la tarea de matemática",
        "dónde compro entradas", "quién es el presidente", "cómo se llama mi perro",
        "cuántos habitantes tiene Jujuy", "programa una alarma", "escríbeme un correo",
        "cuál es el sentido de la vida", "quién ganó las elecciones", "dime un chiste",
        "cómo llego al centro", "cuánto cuesta un auto", "qué día es hoy", "traduce esto al inglés",
    ],
}


def all_examples() -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for intent, examples in INTENT_EXAMPLES.items():
        for text in examples:
            texts.append(text)
            labels.append(intent)
    return texts, labels

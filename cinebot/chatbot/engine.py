"""Motor de diálogo de CineBot.

Coordina el NLU, el recomendador híbrido y la base de datos. Cada mensaje se
persiste en `conversations` / `messages` / `recommendations`, de modo que la
aplicación deja trazabilidad de lo que el usuario pidió y de lo que el sistema
sugirió (conceptos de bases de datos aplicados a un chatbot).

El motor no contiene código de interfaz: recibe texto y devuelve `ChatResponse`.
Eso permite probarlo por consola y reutilizarlo desde Flet.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from cinebot import config
from cinebot.chatbot.nlu import CineBotNLU, MOOD_LABELS
from cinebot.db.connection import Database
from cinebot.db.repository import CatalogRepository, ChatRepository, MetricsRepository, UserRepository
from cinebot.domain import (
    ChatResponse,
    GenreTaste,
    MediaItem,
    NLUResult,
    QueryFilters,
    Recommendation,
    UserProfile,
)
from cinebot.ml.recommender import HybridRecommender
from cinebot.text_utils import normalize_text, truncate

GREETINGS = [
    "recomiéndame una película",
    "series de ciencia ficción",
    "algo de terror para esta noche",
    "las mejores películas de los 90",
    "algo parecido a Inception",
]

HELP_TEXT = """**Esto es lo que puedo hacer**

- 🎬 **Recomendar** películas o series: *"recomiéndame una comedia"*.
- 🧠 **Entender tu ánimo**: *"quiero algo para reír"*, *"algo emotivo"*.
- 🔎 **Filtrar** por género, año, nota o cantidad de temporadas.
- 🎞️ **Buscar parecidos**: *"algo como Breaking Bad"*.
- 📝 **Calificar títulos**: *"califica Inception con 5"*.
- 📚 **Consultar tu historial** y tus gustos.
- 🏆 **Ver rankings**: *"top 10 series de drama"*.
- ♻️ **Reiniciar tu perfil**: *"borra mis valoraciones"*.

También puedes pedirme **información** de cualquier título del catálogo."""


class ChatEngine:
    """Motor conversacional con estado por usuario."""

    def __init__(
        self,
        db: Database | None = None,
        user_id: int | None = None,
        recommender: HybridRecommender | None = None,
        nlu: CineBotNLU | None = None,
    ) -> None:
        self.db = db or Database(config.DB_PATH)
        self.user_repo = UserRepository(self.db)
        self.catalog_repo = CatalogRepository(self.db)
        self.chat_repo = ChatRepository(self.db)
        self.metrics_repo = MetricsRepository(self.db)
        self.user_id = int(user_id or self.user_repo.get_or_create_guest())
        self.recommender = recommender or HybridRecommender(self.db)
        self.nlu = nlu or CineBotNLU(self.catalog_repo.all_titles())
        self.conversation_id: int | None = None
        self.last_filters = QueryFilters()
        self.last_recommendations: list[Recommendation] = []
        self.last_item_id: int | None = None
        self.pending: dict[str, Any] = {}
        self._welcome_sent = False

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------
    @property
    def profile(self) -> UserProfile:
        profile = self.user_repo.profile(self.user_id)
        if profile is None:
            raise ValueError(f"El usuario {self.user_id} no existe")
        return profile

    def start(self) -> ChatResponse:
        """Crea la conversación y devuelve el saludo inicial."""
        if self.conversation_id is None:
            self.conversation_id = self.chat_repo.create_conversation(
                self.user_id, title="Primera conversación"
            )
        profile = self.profile
        stats = self.catalog_repo.stats()
        text = (
            f"¡Hola, **{profile.username}**! Soy **CineBot**, tu copiloto cinéfilo. 🍿\n\n"
            f"Tengo **{stats['items']:,} títulos** para recomendarte: "
            f"{stats['movies']:,} películas y {stats['series']} series. "
            "Puedo filtrar por género, ánimo, año, nota o títulos parecidos."
        )
        if profile.is_cold_start:
            text += (
                "\n\nTodavía no conozco tus gustos. Cuéntame qué te apetece ver "
                "o pídeme un Ranking; también puedes elegir un perfil de demostración "
                "en la barra lateral para probar el filtrado colaborativo."
            )
        else:
            text += f"\n\n{profile.taste_summary} Ya tienes **{profile.n_ratings} valoraciones**."
        text += "\n\n¿Qué vemos hoy?"
        response = ChatResponse(
            text=text,
            intent="saludo",
            confidence=1.0,
            followups=GREETINGS,
            meta={"profile": self._profile_dict(profile), "catalog": stats},
            conversation_id=self.conversation_id,
        )
        self._persist_assistant(response)
        self._welcome_sent = True
        return response

    def welcome(self) -> ChatResponse:
        return self.start()

    def new_conversation(self) -> ChatResponse:
        if self.conversation_id is not None:
            self.chat_repo.end_conversation(self.conversation_id)
        self.conversation_id = self.chat_repo.create_conversation(
            self.user_id, title="Nueva conversación"
        )
        self.last_filters = QueryFilters()
        self.last_recommendations = []
        self.last_item_id = None
        self.pending = {}
        return self.start()

    # ------------------------------------------------------------------
    # Entrada principal
    # ------------------------------------------------------------------
    def handle(self, text: str) -> ChatResponse:
        text = (text or "").strip()
        if not text:
            return ChatResponse(
                text="Escríbeme qué quieres ver y te ayudo a elegir.",
                intent="desconocido",
                followups=GREETINGS,
                conversation_id=self.conversation_id,
            )
        if self.conversation_id is None:
            self.conversation_id = self.chat_repo.create_conversation(
                self.user_id, title=truncate(text, 60) or "Conversación"
            )
        nlu = self.nlu.parse(text)
        self.chat_repo.add_message(
            self.conversation_id,
            "user",
            text,
            intent=nlu.intent,
            confidence=nlu.confidence,
            entities=nlu.entities,
        )

        # Si el motor esperaba una nota o un título, se completa la acción.
        if self.pending.get("action") == "rate":
            nlu = self._complete_pending_rating(nlu)

        try:
            response = self._dispatch(nlu)
        except Exception as exc:  # noqa: BLE001
            response = ChatResponse(
                text=(
                    "Ups, tuve un problema al procesar tu pedido. "
                    f"Detalle técnico: `{type(exc).__name__}`. "
                    "Prueba con otra frase o reinicia la conversación."
                ),
                intent="error",
                confidence=0.0,
                followups=GREETINGS,
            )
        response.conversation_id = self.conversation_id
        response.intent = nlu.intent
        response.confidence = nlu.confidence
        response.filters = nlu.filters
        self._persist_assistant(response)
        return response

    # ------------------------------------------------------------------
    # Despacho por intención
    # ------------------------------------------------------------------
    def _dispatch(self, nlu: NLUResult) -> ChatResponse:
        intent = nlu.intent
        if intent == "saludo":
            return self._greet()
        if intent == "despedida":
            return self._bye()
        if intent == "agradecimiento":
            return ChatResponse(
                text="¡De nada! Si quieres seguir explorando, dime un género o un ánimo. 🍿",
                followups=["Dame otra recomendación", "Top 10 series", "Algo para reír"],
            )
        if intent == "ayuda":
            return ChatResponse(
                text=HELP_TEXT,
                followups=GREETINGS,
                meta={"help": True},
            )
        if intent in {
            "recomendar_pelicula",
            "recomendar_serie",
            "recomendar_genero",
            "recomendar_animo",
        }:
            return self._recommend_from_nlu(nlu)
        if intent == "similar":
            return self._handle_similar(nlu)
        if intent == "informacion":
            return self._handle_info(nlu)
        if intent == "calificar":
            return self._handle_rate(nlu)
        if intent == "historial":
            return self._handle_history()
        if intent == "top_valoradas":
            return self._handle_top(nlu)
        if intent == "resetear":
            return self._handle_reset()
        if intent == "otra_recomendacion":
            return self._repeat_recommendations(nlu)
        if intent == "rechazo":
            return self._handle_rejection(nlu)
        if intent == "afirmacion":
            if self.last_recommendations or self.last_item_id:
                return self._repeat_recommendations(nlu)
            return self._recommend_from_nlu(nlu)
        return self._fallback(nlu)

    # ------------------------------------------------------------------
    # Recomendaciones
    # ------------------------------------------------------------------
    def _recommend_from_nlu(self, nlu: NLUResult) -> ChatResponse:
        filters = copy.deepcopy(nlu.filters)
        if filters.is_empty():
            filters = copy.deepcopy(self.last_filters) if not self.last_filters.is_empty() else QueryFilters()
        filters.count = max(1, min(10, filters.count or config.DEFAULT_RECOMMENDATIONS))
        filters.exclude_seen = True
        self._inherit_context(filters, nlu)
        return self._recommend(filters, nlu=nlu)

    def _recommend(
        self,
        filters: QueryFilters,
        nlu: NLUResult | None = None,
        exclude_ids: set[int] | None = None,
        intro: str | None = None,
    ) -> ChatResponse:
        batch = self.recommender.recommend(
            self.user_id,
            filters,
            limit=filters.count,
            exclude_ids=exclude_ids,
        )
        recommendations = batch.recommendations
        if not recommendations:
            return ChatResponse(
                text=(
                    "No encontré títulos que cumplan **todos** esos filtros. "
                    "Prueba quitando el año, la nota mínima o cambiando de género."
                ),
                recommendations=[],
                followups=["Sorpréndeme", "Series populares", "Top 10 películas"],
                meta=batch.meta,
            )
        self.last_filters = copy.deepcopy(filters)
        self.last_recommendations = recommendations
        self.last_item_id = recommendations[0].item.item_id

        summary = self._describe_filters(filters)
        lines = [intro or f"Estas son mis apuestas{f' para {summary}' if summary else ''}:"]
        if batch.meta.get("relaxed"):
            relajados = ", ".join(batch.meta["relaxed"]).replace("_", " ")
            lines.append(
                f"_(Amplié algunos filtros —{relajados}— porque había pocas coincidencias.)_"
            )
        if batch.meta.get("cold_start"):
            lines.append("_Como aún no tengo tus gustos, prioricé calidad y afinidad temática._")
        elif batch.meta.get("cf_used"):
            lines.append("_Combiné tu historial con usuarios de gustos parecidos._")
        text = "\n\n".join(lines)
        followups = self._followups_for(filters)
        meta = dict(batch.meta)
        meta["profile_genres"] = meta.get("profile_genres", [])
        return ChatResponse(
            text=text,
            recommendations=recommendations,
            followups=followups,
            meta=meta,
        )

    def _repeat_recommendations(self, nlu: NLUResult) -> ChatResponse:
        filters = copy.deepcopy(self.last_filters)
        if filters.is_empty():
            filters = copy.deepcopy(nlu.filters)
        if filters.is_empty():
            filters = QueryFilters(count=config.DEFAULT_RECOMMENDATIONS)
        filters.exclude_seen = True
        excluded = {rec.item.item_id for rec in self.last_recommendations}
        batch = self.recommender.recommend(
            self.user_id, filters, limit=filters.count, exclude_ids=excluded
        )
        if not batch.recommendations and excluded:
            # Se agotaron las opciones con esos filtros: se amplía la búsqueda.
            filters = QueryFilters(count=filters.count, exclude_seen=True)
            batch = self.recommender.recommend(self.user_id, filters, limit=filters.count)
        if not batch.recommendations:
            return ChatResponse(
                text="Me quedé sin opciones nuevas con esos filtros. ¿Probamos otro género o año?",
                followups=["Sorpréndeme", "Series de comedia", "Películas de acción"],
            )
        self.last_filters = copy.deepcopy(filters)
        self.last_recommendations = batch.recommendations
        return ChatResponse(
            text="Aquí tienes otras opciones que no te había mostrado:",
            recommendations=batch.recommendations,
            followups=self._followups_for(filters),
            meta=batch.meta,
        )

    def _handle_rejection(self, nlu: NLUResult) -> ChatResponse:
        if self.last_recommendations:
            for rec in self.last_recommendations[:3]:
                self.user_repo.add_feedback(self.user_id, rec.item.item_id, "not_interested")
            return self._repeat_recommendations(nlu)
        return ChatResponse(
            text="Entendido. Dime qué género, ánimo o título te apetece y afinamos la búsqueda.",
            followups=GREETINGS,
        )

    def _handle_similar(self, nlu: NLUResult) -> ChatResponse:
        item = None
        if nlu.matched_item_id:
            item = self.catalog_repo.get_item(nlu.matched_item_id)
        if item is None:
            return ChatResponse(
                text=(
                    "¿De qué título quieres que busque parecidos? "
                    "Escríbelo, por ejemplo: _«algo como Inception»_."
                ),
                followups=["Algo como Inception", "Series parecidas a Dark", "Como Parasite"],
            )
        count = nlu.filters.count or 5
        batch = self.recommender.similar_to(self.user_id, item.item_id, limit=count)
        if not batch.recommendations:
            return ChatResponse(
                text=f"No encontré títulos parecidos a **{item.title}** que no hayas visto.",
                followups=[f"De qué trata {item.title}", "Otra recomendación"],
            )
        self.last_filters = QueryFilters(
            media_type=item.media_type, similar_to=item.item_id, count=count
        )
        self.last_recommendations = batch.recommendations
        self.last_item_id = item.item_id
        return ChatResponse(
            text=f"Si te gustó **{item.title}** ({item.year_label}), prueba con estas opciones:",
            recommendations=batch.recommendations,
            followups=[
                f"De qué trata {batch.recommendations[0].item.title}",
                f"Más como {item.title}",
                "Dame otras opciones",
            ],
            meta=batch.meta,
        )

    def _handle_top(self, nlu: NLUResult) -> ChatResponse:
        filters = copy.deepcopy(nlu.filters)
        filters.count = max(3, min(10, filters.count or 10))
        filters.exclude_seen = False
        if filters.min_rating is None:
            filters.min_rating = 4.0
        items = self.catalog_repo.candidates(filters, user_id=None)
        if not items:
            return ChatResponse(
                text="No hay títulos con esos filtros. Prueba con otro género o época.",
                followups=["Top 10 películas", "Top 10 series", "Mejores de terror"],
            )
        items.sort(key=lambda item: (item.bayes_rating, item.popularity), reverse=True)
        items = items[: filters.count]
        recommendations = [
            Recommendation(
                item=item,
                score=round(item.bayes_rating / 5.0, 4),
                quality_score=item.bayes_rating / 5.0,
                popularity_score=0.0,
                predicted_rating=item.bayes_rating,
                reason=self._quality_reason(item),
                rank=index,
            )
            for index, item in enumerate(items, start=1)
        ]
        self.last_filters = copy.deepcopy(filters)
        self.last_recommendations = recommendations
        summary = self._describe_filters(filters)
        return ChatResponse(
            text=f"Este es mi ranking{(' de ' + summary) if summary else ''} por valoración:",
            recommendations=recommendations,
            followups=["Dame otra recomendación", "Algo más reciente", "De qué trata la primera"],
            meta={"ranking": True, "filters": filters.as_dict()},
        )

    # ------------------------------------------------------------------
    # Información de títulos
    # ------------------------------------------------------------------
    def _handle_info(self, nlu: NLUResult) -> ChatResponse:
        item = None
        if nlu.matched_item_id:
            item = self.catalog_repo.get_item(nlu.matched_item_id)
        if item is not None:
            return ChatResponse(
                text=self._item_details(item),
                followups=[
                    f"Recomiéndame algo parecido a {item.title}",
                    f"Calificar {item.title}",
                    "Dame otra recomendación",
                ],
                meta={"item_id": item.item_id, "item": item.as_dict()},
            )
        results = self._search(nlu)
        if results:
            recommendations = [
                Recommendation(
                    item=found,
                    score=0.5,
                    quality_score=found.bayes_rating / 5.0,
                    predicted_rating=found.bayes_rating,
                    reason="Coincide con tu búsqueda",
                    rank=index,
                )
                for index, found in enumerate(results, start=1)
            ]
            return ChatResponse(
                text="Encontré estos títulos. Dime cuál te interesa y te doy la ficha:",
                recommendations=recommendations,
                followups=[f"De qué trata {results[0].title}", "Dame otra recomendación"],
            )
        return ChatResponse(
            text=(
                "No encontré ese título en el catálogo. Revisa la ortografía o pídeme "
                "una recomendación por género."
            ),
            followups=["Recomiéndame una película", "Series de drama", "Top 10 películas"],
        )

    # ------------------------------------------------------------------
    # Calificaciones
    # ------------------------------------------------------------------
    def _handle_rate(self, nlu: NLUResult) -> ChatResponse:
        item = None
        if nlu.matched_item_id:
            item = self.catalog_repo.get_item(nlu.matched_item_id)
        if item is None:
            self.pending = {"action": "rate"}
            return ChatResponse(
                text="¿A qué título le pongo la nota? Escríbelo tal como aparece en el catálogo.",
                followups=["Calificar Inception", "Calificar Breaking Bad"],
            )
        if nlu.rating_value is None:
            self.pending = {"action": "rate", "item_id": item.item_id}
            return ChatResponse(
                text=(
                    f"¿Qué nota le das a **{item.title}**? "
                    "Puedes decir _«4»_, _«4.5»_ o _«8 de 10»_."
                ),
                followups=[f"Calificar {item.title} con 5", f"Calificar {item.title} con 3"],
                meta={"item_id": item.item_id},
            )
        return self._save_rating(item, float(nlu.rating_value))

    def _complete_pending_rating(self, nlu: NLUResult) -> NLUResult:
        if self.pending.get("action") != "rate":
            return nlu
        pending_item = self.pending.get("item_id")
        item_id = pending_item or nlu.matched_item_id
        if item_id is None:
            return nlu
        item = self.catalog_repo.get_item(int(item_id))
        if item is None:
            return nlu
        nlu.matched_item_id = int(item_id)
        nlu.title_query = item.title
        nlu.intent = "calificar"
        nlu.confidence = max(nlu.confidence, 0.9)
        if nlu.rating_value is None:
            match = re.fullmatch(r"\s*(\d(?:[.,]\d)?)\s*", nlu.raw_text)
            if match:
                value = float(match.group(1).replace(",", "."))
                nlu.rating_value = round(max(0.5, min(5.0, value)) * 2) / 2
        return nlu

    def _save_rating(self, item: MediaItem, rating: float) -> ChatResponse:
        self.user_repo.upsert_rating(self.user_id, item.item_id, rating, source="app")
        if rating >= 4.0:
            self.user_repo.add_feedback(self.user_id, item.item_id, "like")
        elif rating <= 2.0:
            self.user_repo.add_feedback(self.user_id, item.item_id, "dislike")
        self.pending = {}
        profile = self.profile
        stars = "★" * int(round(rating)) + "☆" * (5 - int(round(rating)))
        text = (
            f"Anotado: **{item.title}** {stars} ({rating:.1f}/5).\n\n"
            f"Tu perfil ya tiene **{profile.n_ratings} valoraciones**"
            + (f" y una nota media de **{profile.avg_rating:.2f}**." if profile.n_ratings else ".")
        )
        if profile.top_genres:
            text += f"\n\n{profile.taste_summary}"
        return ChatResponse(
            text=text,
            followups=[
                "Recomiéndame algo según mis gustos",
                f"Algo parecido a {item.title}",
                "Mi historial",
            ],
            meta={"item_id": item.item_id, "rating": rating, "profile": self._profile_dict(profile)},
        )

    def rate(self, item_id: int, rating: float) -> ChatResponse:
        """Atajo para la interfaz: valorar un título desde una tarjeta."""
        item = self.catalog_repo.get_item(item_id)
        if item is None:
            return ChatResponse(text="No encontré ese título.")
        if self.conversation_id is None:
            self.conversation_id = self.chat_repo.create_conversation(self.user_id)
        response = self._save_rating(item, rating)
        self.chat_repo.add_message(
            self.conversation_id,
            "user",
            f"califica {item.title} con {rating}",
            intent="calificar",
            confidence=1.0,
            entities={"title": item.title, "rating": rating},
        )
        response.conversation_id = self.conversation_id
        response.intent = "calificar"
        response.confidence = 1.0
        self._persist_assistant(response)
        return response

    def feedback(self, item_id: int, action: str) -> str:
        """Registra like/dislike/visto desde la interfaz."""
        self.user_repo.add_feedback(self.user_id, item_id, action)
        messages = {
            "like": "¡Bien! Tendré en cuenta que te gustó.",
            "dislike": "Entendido, bajaré el peso de títulos parecidos.",
            "seen": "Marcado como visto. No volveré a recomendártelo.",
            "not_interested": "No te la volveré a recomendar.",
        }
        return messages.get(action, "Feedback registrado.")

    # ------------------------------------------------------------------
    # Historial / perfil / reset
    # ------------------------------------------------------------------
    def _handle_history(self) -> ChatResponse:
        profile = self.profile
        ratings = self.user_repo.ratings(self.user_id)
        if not ratings:
            return ChatResponse(
                text=(
                    f"**{profile.username}** todavía no tiene valoraciones. "
                    "Cuando califiques títulos podré explicarte tus gustos y mejorar las recomendaciones."
                ),
                followups=GREETINGS,
                meta={"profile": self._profile_dict(profile)},
            )
        lines = [
            f"**Perfil de {profile.username}**",
            f"- Valoraciones: **{profile.n_ratings}** · nota media: **{profile.avg_rating:.2f}/5** · "
            f"favoritas: **{profile.n_likes}**",
        ]
        if profile.top_genres:
            genres = ", ".join(
                f"{g.genre} ({g.avg_rating:.1f}★)" for g in profile.top_genres[:5]
            )
            lines.append(f"- Afinidad por género: {genres}")
        if profile.cluster_name:
            lines.append(f"- Segmento: **{profile.cluster_name}**")
        lines.append("\n**Últimas valoraciones**")
        for row in ratings[:8]:
            stars = "★" * int(round(float(row["rating"])))
            kind = "🎬" if row.get("media_type") == "movie" else "📺"
            source = "app" if row.get("source") == "app" else "MovieLens"
            lines.append(
                f"- {kind} {row['title']} — {stars} ({float(row['rating']):.1f}) · _{source}_"
            )
        return ChatResponse(
            text="\n".join(lines),
            followups=[
                "Recomiéndame algo según mis gustos",
                "Borra mis valoraciones",
                "Mis géneros favoritos",
            ],
            meta={"profile": self._profile_dict(profile), "ratings": ratings[:20]},
        )

    def _handle_reset(self) -> ChatResponse:
        with self.db.transaction() as connection:
            deleted = connection.execute(
                "DELETE FROM ratings WHERE user_id = ? AND source = 'app'", (self.user_id,)
            ).rowcount
            connection.execute("DELETE FROM feedback WHERE user_id = ?", (self.user_id,))
            connection.execute("DELETE FROM interactions WHERE user_id = ?", (self.user_id,))
        self.last_filters = QueryFilters()
        self.last_recommendations = []
        self.last_item_id = None
        self.pending = {}
        profile = self.profile
        return ChatResponse(
            text=(
                f"Listo. Borré **{deleted or 0} valoraciones** de la app y tu feedback. "
                "Las valoraciones históricas de MovieLens de los perfiles de demostración "
                "se conservan porque son parte del conjunto de datos.\n\n"
                "Volvemos a empezar: ¿qué te apetece ver?"
            ),
            followups=GREETINGS,
            meta={"deleted": int(deleted or 0), "profile": self._profile_dict(profile)},
        )

    # ------------------------------------------------------------------
    # Saludos / ayuda / fallback
    # ------------------------------------------------------------------
    def _greet(self) -> ChatResponse:
        profile = self.profile
        if profile.n_ratings:
            line = f"¡Hola de nuevo, **{profile.username}**! {profile.taste_summary}"
        else:
            line = (
                f"¡Hola, **{profile.username}**! Cuéntame qué te apetece: "
                "un género, un ánimo o un título que te haya gustado."
            )
        return ChatResponse(
            text=line + "\n\n¿Qué vemos hoy?",
            followups=GREETINGS,
            meta={"profile": self._profile_dict(profile)},
        )

    def _bye(self) -> ChatResponse:
        if self.conversation_id is not None:
            self.chat_repo.end_conversation(self.conversation_id)
        return ChatResponse(
            text="¡Hasta la próxima! Que disfrutes la película o la serie. 🍿",
            followups=[],
        )

    def _fallback(self, nlu: NLUResult) -> ChatResponse:
        return ChatResponse(
            text=(
                "No estoy seguro de haber entendido. Puedo recomendarte películas y series, "
                "buscar parecidos, darte información de un título o registrar tus notas.\n\n"
                "Prueba con: _«recomiéndame una serie de misterio»_ o _«de qué trata Parasite»_."
            ),
            followups=GREETINGS,
            meta={"predicted_intent": nlu.intent},
        )

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------
    def _inherit_context(self, filters: QueryFilters, nlu: NLUResult) -> None:
        """Mantiene el tipo (película/serie) cuando el usuario cambia de género."""
        if self.last_filters.is_empty():
            return
        if (
            filters.media_type is None
            and (filters.genres or filters.moods)
            and self.last_filters.media_type
        ):
            filters.media_type = self.last_filters.media_type
        if (
            filters.media_type == "series"
            and filters.max_seasons is None
            and self.last_filters.max_seasons is not None
        ):
            filters.max_seasons = self.last_filters.max_seasons

    def _describe_filters(self, filters: QueryFilters) -> str:
        parts: list[str] = []
        if filters.media_type == "movie":
            parts.append("películas")
        elif filters.media_type == "series":
            parts.append("series")
        if filters.genres:
            parts.append(", ".join(filters.genres))
        if filters.moods:
            parts.append(", ".join(MOOD_LABELS.get(m, m) for m in filters.moods))
        if filters.year_min and filters.year_max and filters.year_min == filters.year_max:
            parts.append(f"de {filters.year_min}")
        elif filters.year_min and filters.year_max:
            parts.append(f"de {filters.year_min}–{filters.year_max}")
        elif filters.year_min:
            parts.append(f"desde {filters.year_min}")
        elif filters.year_max:
            parts.append(f"hasta {filters.year_max}")
        if filters.max_seasons:
            parts.append(f"de máximo {filters.max_seasons} temporadas")
        if filters.min_rating:
            parts.append(f"con nota ≥ {filters.min_rating:.1f}")
        return " ".join(parts) if parts else "tus gustos"

    def _followups_for(self, filters: QueryFilters) -> list[str]:
        options = ["Dame otras opciones"]
        if filters.media_type == "movie":
            options.append("Series parecidas")
        elif filters.media_type == "series":
            options.append("Películas parecidas")
        else:
            options.append("Ahora algo de terror")
        if filters.genres:
            options.append("Mejores valoradas de ese género")
        else:
            options.append("Top 10 series")
        return options

    def _quality_reason(self, item: MediaItem) -> str:
        if item.media_type == "series":
            return (
                f"Serie de {item.seasons or '?'} temporadas · "
                f"valoración {item.rating_avg:.1f}/5"
            )
        return f"Nota media ponderada {item.bayes_rating:.2f}/5 · {item.rating_count:,} votos"

    def _item_details(self, item: MediaItem) -> str:
        icon = "🎬" if item.media_type == "movie" else "📺"
        lines = [
            f"{icon} **{item.title}** ({item.year_label}) · {item.type_label}",
            f"**Géneros:** {', '.join(item.genres) if item.genres else 'sin clasificar'}",
            f"**Valoración:** {item.rating_label}"
            + (f" · {item.rating_count:,} votos" if item.rating_count else ""),
            f"**Duración:** {item.duration_label}",
        ]
        if item.overview:
            lines.append(f"\n**Sinopsis:** {item.overview}")
        else:
            lines.append(
                "\n_No tengo sinopsis de este título en el catálogo, "
                "pero puedo recomendarte otros parecidos._"
            )
        if item.external_url:
            lines.append(f"\n[Ver ficha externa]({item.external_url})")
        return "\n".join(lines)

    def _search(self, nlu: NLUResult) -> list[MediaItem]:
        if nlu.title_query:
            results = self.catalog_repo.find_title(nlu.title_query, media_type=nlu.filters.media_type)
            if results:
                return results[:5]
        cleaned = re.sub(
            r"\b(de|que|trata|trata|sinopsis|cuentame|hablame|informacion|ficha|resumen|"
            r"detalles|pelicula|peliculas|serie|series|sobre|del|la|el|los|las|un|una)\b",
            " ",
            normalize_text(nlu.raw_text),
        )
        return self.catalog_repo.search_items(cleaned.strip(), limit=5)

    def _profile_dict(self, profile: UserProfile) -> dict[str, Any]:
        return {
            "user_id": profile.user_id,
            "username": profile.username,
            "n_ratings": profile.n_ratings,
            "avg_rating": profile.avg_rating,
            "n_likes": profile.n_likes,
            "cluster_name": profile.cluster_name,
            "top_genres": [
                {
                    "genre": genre.genre,
                    "n_ratings": genre.n_ratings,
                    "avg_rating": genre.avg_rating,
                    "affinity": genre.affinity,
                }
                for genre in profile.top_genres
            ],
        }

    def _persist_assistant(self, response: ChatResponse) -> None:
        if self.conversation_id is None:
            return
        message_id = self.chat_repo.add_message(
            self.conversation_id,
            "assistant",
            response.text,
            intent=response.intent,
            confidence=response.confidence,
            entities={
                "recommendations": [rec.as_dict() for rec in response.recommendations],
                **({"meta": response.meta} if response.meta else {}),
            },
        )
        response.message_id = message_id
        if response.recommendations:
            self.chat_repo.add_recommendations(message_id, self.user_id, response.recommendations)

    # ------------------------------------------------------------------
    # Consultas para la interfaz
    # ------------------------------------------------------------------
    def catalog_stats(self) -> dict[str, Any]:
        return self.catalog_repo.stats()

    def recent_ratings(self, limit: int = 10) -> list[dict[str, Any]]:
        return self.user_repo.ratings(self.user_id)[:limit]

    def genre_taste(self, limit: int = 6) -> list[GenreTaste]:
        return self.user_repo.genre_taste(self.user_id, limit=limit)

    def model_metrics(self) -> dict[str, Any]:
        """Métricas legibles para el panel lateral."""
        import json

        metadata_path = config.model_path("metadata")
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return metadata

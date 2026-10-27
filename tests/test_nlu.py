"""Pruebas del clasificador de intenciones y de la extracción de entidades."""

from __future__ import annotations

import unittest

from cinebot.bootstrap import ensure_database
from cinebot.chatbot.nlu import CineBotNLU
from cinebot.db.repository import CatalogRepository


class NLUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        db = ensure_database()
        rows = CatalogRepository(db).all_titles()
        cls.nlu = CineBotNLU(rows)

    def test_genre_and_media_type(self) -> None:
        result = self.nlu.parse("series de ciencia ficción cortas")
        self.assertEqual(result.filters.media_type, "series")
        self.assertIn("Sci-Fi", result.filters.genres)
        self.assertEqual(result.filters.max_seasons, 2)
        self.assertIn(result.intent, {"recomendar_genero", "recomendar_serie"})

    def test_similar_title(self) -> None:
        result = self.nlu.parse("algo parecido a Inception")
        self.assertEqual(result.intent, "similar")
        self.assertEqual(result.title_query, "Inception")
        self.assertEqual(result.filters.similar_to, 79132)

    def test_information_title(self) -> None:
        result = self.nlu.parse("de qué trata Breaking Bad")
        self.assertEqual(result.intent, "informacion")
        self.assertEqual(result.title_query, "Breaking Bad")
        self.assertEqual(result.filters.media_type, "series")

    def test_rating_entities(self) -> None:
        result = self.nlu.parse("califica Inception con 5")
        self.assertEqual(result.intent, "calificar")
        self.assertEqual(result.rating_value, 5.0)
        self.assertEqual(result.matched_item_id, 79132)

    def test_article_is_not_quantity(self) -> None:
        # "una película" es un artículo, no una cantidad: debe pedir el
        # número por defecto de recomendaciones.
        result = self.nlu.parse("recomiéndame una película")
        self.assertEqual(result.intent, "recomendar_pelicula")
        self.assertEqual(result.filters.count, 5)
        explicit = self.nlu.parse("recomiéndame una sola película")
        self.assertEqual(explicit.filters.count, 1)

    def test_top_count(self) -> None:
        result = self.nlu.parse("top 10 series de drama")
        self.assertEqual(result.intent, "top_valoradas")
        self.assertEqual(result.filters.count, 10)
        self.assertEqual(result.filters.media_type, "series")
        self.assertIn("Drama", result.filters.genres)

    def test_mood(self) -> None:
        result = self.nlu.parse("quiero algo para reír")
        self.assertEqual(result.intent, "recomendar_animo")
        self.assertIn("animo_divertido", result.filters.moods)

    def test_alias(self) -> None:
        result = self.nlu.parse("de qué trata el padrino")
        self.assertEqual(result.title_query, "Godfather, The")

    def test_history(self) -> None:
        result = self.nlu.parse("muéstrame mi historial")
        self.assertEqual(result.intent, "historial")

    def test_reset(self) -> None:
        result = self.nlu.parse("borra mis valoraciones")
        self.assertEqual(result.intent, "resetear")


if __name__ == "__main__":
    unittest.main()

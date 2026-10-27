"""Pruebas de integración del motor conversacional."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from cinebot import config
from cinebot.chatbot.engine import ChatEngine
from cinebot.db.connection import Database
from cinebot.db.repository import UserRepository
from cinebot.ml.recommender import HybridRecommender


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        target = Path(self.tmp.name) / "engine.db"
        source = sqlite3.connect(config.DB_PATH)
        destination = sqlite3.connect(target)
        try:
            source.backup(destination)
        finally:
            destination.close()
            source.close()
        self.db = Database(target)
        self.engine = ChatEngine(
            self.db,
            user_id=100001,
            recommender=HybridRecommender(self.db),
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_start_and_greeting(self) -> None:
        response = self.engine.start()
        self.assertIn("CineBot", response.text)
        self.assertEqual(response.intent, "saludo")
        self.assertIsNotNone(response.conversation_id)

    def test_recommend_movies(self) -> None:
        self.engine.start()
        response = self.engine.handle("recomiéndame una película de ciencia ficción")
        self.assertIn(response.intent, {"recomendar_pelicula", "recomendar_genero"})
        self.assertEqual(len(response.recommendations), 5)
        self.assertTrue(all(rec.item.media_type == "movie" for rec in response.recommendations))

    def test_plain_movie_request_returns_full_batch(self) -> None:
        # Regresión: "una película" no debe interpretarse como count=1.
        self.engine.start()
        response = self.engine.handle("recomiéndame una película")
        self.assertEqual(response.intent, "recomendar_pelicula")
        self.assertEqual(len(response.recommendations), 5)

    def test_recommend_series(self) -> None:
        self.engine.start()
        response = self.engine.handle("series de comedia")
        self.assertTrue(response.recommendations)
        self.assertTrue(all(rec.item.media_type == "series" for rec in response.recommendations))

    def test_information(self) -> None:
        self.engine.start()
        response = self.engine.handle("de qué trata Inception")
        self.assertEqual(response.intent, "informacion")
        self.assertIn("Inception", response.text)
        self.assertEqual(response.meta.get("item_id"), 79132)

    def test_rating_and_history(self) -> None:
        self.engine.start()
        response = self.engine.handle("califica Inception con 5")
        self.assertEqual(response.intent, "calificar")
        rating = UserRepository(self.db).rating_for(100001, 79132)
        self.assertIsNotNone(rating)
        assert rating is not None
        self.assertEqual(float(rating["rating"]), 5.0)
        history = self.engine.handle("mi historial")
        self.assertEqual(history.intent, "historial")
        self.assertIn("Inception", history.text)

    def test_pending_rating_flow(self) -> None:
        self.engine.start()
        first = self.engine.handle("quiero calificar una película")
        self.assertIn("título", first.text.lower())
        second = self.engine.handle("Inception")
        self.assertEqual(second.intent, "calificar")
        third = self.engine.handle("5")
        self.assertIn("Anotado", third.text)

    def test_rejection_and_more(self) -> None:
        self.engine.start()
        self.engine.handle("recomiéndame una película")
        more = self.engine.handle("dame otra")
        self.assertTrue(more.recommendations)
        rejection = self.engine.handle("no me gustó")
        self.assertTrue(rejection.recommendations)

    def test_reset(self) -> None:
        self.engine.start()
        self.engine.handle("califica Inception con 4")
        response = self.engine.handle("borra mis valoraciones")
        self.assertEqual(response.intent, "resetear")
        self.assertIsNone(UserRepository(self.db).rating_for(100001, 79132))

    def test_ui_rate_shortcut(self) -> None:
        self.engine.start()
        response = self.engine.rate(79132, 4)
        self.assertIn("Anotado", response.text)
        rating = UserRepository(self.db).rating_for(100001, 79132)
        assert rating is not None
        self.assertEqual(float(rating["rating"]), 4.0)


if __name__ == "__main__":
    unittest.main()

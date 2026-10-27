"""Pruebas de la capa de datos y del esquema SQLite."""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from cinebot import config
from cinebot.db.connection import Database
from cinebot.db.repository import CatalogRepository, UserRepository
from cinebot.domain import QueryFilters
from cinebot.bootstrap import ensure_database


class DatabaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.db = ensure_database()
        cls.catalog = CatalogRepository(cls.db)
        cls.users = UserRepository(cls.db)

    def test_health_and_counts(self) -> None:
        health = self.db.health_check()
        self.assertTrue(health["ok"])
        self.assertEqual(health["counts"]["items"], 9_832)
        self.assertGreaterEqual(health["counts"]["ratings"], 100_000)

    def test_catalog_has_movies_and_series(self) -> None:
        stats = self.catalog.stats()
        self.assertEqual(stats["movies"], 9_742)
        self.assertEqual(stats["series"], 90)
        self.assertGreaterEqual(stats["genres"], 20)

    def test_fts_search(self) -> None:
        items = self.catalog.search_items("inception", limit=5)
        self.assertTrue(items)
        self.assertEqual(items[0].title, "Inception")

    def test_title_normalization(self) -> None:
        items = self.catalog.find_title("godfather")
        self.assertTrue(items)
        self.assertEqual(items[0].title, "Godfather, The")

    def test_demo_profile(self) -> None:
        profile = self.users.profile(414)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertGreater(profile.n_ratings, 100)
        self.assertTrue(profile.top_genres)

    def test_candidates_filters(self) -> None:
        filters = QueryFilters(media_type="series", genres=["Drama"], count=5)
        items = self.catalog.candidates(filters)
        self.assertTrue(items)
        self.assertTrue(all(item.media_type == "series" for item in items))
        self.assertTrue(any("Drama" in item.genres for item in items))

    def test_rating_triggers_in_copy(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "test.db"
            source = sqlite3.connect(config.DB_PATH)
            destination = sqlite3.connect(target)
            try:
                source.backup(destination)
            finally:
                destination.close()
                source.close()

            db = Database(target)
            users = UserRepository(db)
            catalog = CatalogRepository(db)
            # El usuario invitado puede tener valoraciones de una ejecución
            # anterior de la app; se limpian para que la prueba sea aislada.
            users.reset_ratings(100001)
            before = catalog.get_item(79132)
            assert before is not None
            users.upsert_rating(100001, 79132, 4.5)
            after = catalog.get_item(79132)
            assert after is not None
            self.assertGreater(after.local_rating_count, before.local_rating_count)
            self.assertAlmostEqual(after.local_rating_avg, 4.5, places=2)
            users.reset_ratings(100001)
            cleared = catalog.get_item(79132)
            assert cleared is not None
            self.assertEqual(cleared.local_rating_count, before.local_rating_count)


if __name__ == "__main__":
    unittest.main()

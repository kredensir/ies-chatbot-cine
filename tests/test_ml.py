"""Pruebas de los modelos de machine learning y del recomendador híbrido."""

from __future__ import annotations

import unittest

from cinebot.bootstrap import ensure_database, ensure_models
from cinebot.domain import QueryFilters
from cinebot.ml import ContentModel, HybridRecommender, IntentClassifier, SVDModel, UserClustering


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_models()
        cls.db = ensure_database()
        cls.content = ContentModel.load()
        cls.svd = SVDModel.load()
        cls.clusters = UserClustering.load()
        cls.intent = IntentClassifier.load()
        cls.recommender = HybridRecommender(cls.db)

    def test_content_neighbors(self) -> None:
        similar = self.content.similar_items(79132, limit=5)
        self.assertEqual(len(similar), 5)
        self.assertTrue(all(score >= 0 for _, score in similar))

    def test_content_scores_range(self) -> None:
        scores = self.content.score_candidates(
            [79132, 858, 2571], genres=["Sci-Fi"], moods=["animo_reflexivo"]
        )
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in scores.values()))

    def test_svd_prediction_range(self) -> None:
        prediction = self.svd.predict(414, 79132)
        self.assertGreaterEqual(prediction, 0.5)
        self.assertLessEqual(prediction, 5.0)

    def test_cluster_assignment(self) -> None:
        profile = self.clusters.profile_for_user(414)
        self.assertIsNotNone(profile)
        cluster_id, name, description = profile
        self.assertIsInstance(cluster_id, int)
        self.assertTrue(name)

    def test_intent_classifier(self) -> None:
        intent, confidence, top = self.intent.predict("recomiéndame una serie de terror")
        self.assertEqual(intent, "recomendar_serie")
        self.assertGreater(confidence, 0.3)
        self.assertEqual(len(top), 3)

    def test_recommender_movies(self) -> None:
        batch = self.recommender.recommend(
            414, QueryFilters(media_type="movie", count=5), limit=5
        )
        self.assertEqual(len(batch.recommendations), 5)
        seen = self.recommender.user_repo.seen_item_ids(414)
        self.assertFalse({rec.item.item_id for rec in batch.recommendations} & seen)
        self.assertTrue(all(rec.reason for rec in batch.recommendations))

    def test_recommender_series(self) -> None:
        batch = self.recommender.recommend(
            414, QueryFilters(media_type="series", genres=["Drama"], count=4), limit=4
        )
        self.assertTrue(batch.recommendations)
        self.assertTrue(all(rec.item.media_type == "series" for rec in batch.recommendations))

    def test_similar_items(self) -> None:
        batch = self.recommender.similar_to(414, 79132, limit=4)
        self.assertTrue(batch.recommendations)
        self.assertTrue(all(rec.item.item_id != 79132 for rec in batch.recommendations))


if __name__ == "__main__":
    unittest.main()

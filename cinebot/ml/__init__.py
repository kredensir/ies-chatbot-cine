"""Modelos de machine learning de CineBot.

Este paquete agrupa los cuatro componentes entrenados:

- `content`: recomendación por contenido con TF-IDF y similitud coseno.
- `collaborative`: filtrado colaborativo con FunkSVD y sesgos.
- `clustering`: segmentación de usuarios con K-Means y PCA.
- `intent`: clasificación de intenciones del chatbot con regresión logística.
"""

from cinebot.ml.clustering import UserClustering
from cinebot.ml.collaborative import SVDModel, UserFactor, split_ratings
from cinebot.ml.content import ContentModel
from cinebot.ml.intent import IntentClassifier
from cinebot.ml.recommender import HybridRecommender, RecommendationBatch

__all__ = [
    "ContentModel",
    "HybridRecommender",
    "IntentClassifier",
    "RecommendationBatch",
    "SVDModel",
    "UserClustering",
    "UserFactor",
    "split_ratings",
]
